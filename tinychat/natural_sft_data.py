"""Prepare a reviewed natural Chinese SFT mixture without touching frozen old data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from tokenizers import Tokenizer

from tinychat.chat_format import encode_conversation, SPECIAL_IDS, IGNORE_LABEL
from tinychat.comparison_sft_data import ConflictIndex
from tinychat.stage1_data import normalized_content
from tinychat.training_state import file_identity
from tinychat.verified_sft_checks import verify_encoded_row

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train','validation','test')
CATEGORIES = ('grounded','natural','multi','verified')
QUOTAS = {'train':dict(zip(CATEGORIES,(8000,5000,4000,3000))),
          'validation':dict(zip(CATEGORIES,(200,125,100,75))),
          'test':dict(zip(CATEGORIES,(200,125,100,75)))}


def stable(value, purpose='order'):
    return hashlib.sha256(f'42:{purpose}:{value}'.encode('utf-8')).hexdigest()


def json_write(path, value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def json_rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def public_split(group):
    bucket=int(stable(group,'split')[:8],16)%10000
    return 'validation' if bucket<250 else ('test' if bucket<500 else 'train')


def legacy_candidates():
    """Use only original train for new train, and preserve its established holdouts."""
    selected=[]
    for split in SPLITS:
        groups=defaultdict(list)
        for row in json_rows(ROOT/'data/sft_verified_v2'/f'{split}.jsonl'):
            if len(row['messages'])==2:
                groups[row['task_family']].append(row)
        for index,family in enumerate(sorted(groups)):
            quota=150 if split=='train' else (4 if index<15 else 3)
            pool=sorted(groups[family],key=lambda r:stable(r['id'],'verified'))
            # Keep reserve rows for the same family in case evaluation screening rejects one.
            for row in pool:
                row=dict(row)
                row.update(category='verified',subtype=family,legacy_split=split,
                           source_group_id='verified:'+row['scene_id'],
                           source_revision='frozen_sft_verified_v2',source_license='locally_generated',
                           source_file=f'data/sft_verified_v2/{split}.jsonl',source_row_id=row['id'],
                           transforms=['unchanged_verified_single_turn_anchor'],family_quota=quota)
                selected.append(row)
    return selected


def external_index(paths):
    index=ConflictIndex(containment=True,minimum=12,substring_minimum=24)
    records=[]
    for path in paths:
        path=path.resolve()
        count=0
        for row in json_rows(path):
            users=[normalized_content(m['content']) for m in row['messages'] if m['role']=='user']
            for text in users:
                index.add(text)
            if len(users)>1:
                index.add('|'.join(users))
            count+=1
        records.append({'path':str(path),'records':count,**file_identity(path)})
    return index,records


def audit_cmrc(row):
    e=row['source_evidence']
    c=e['context']; answer=e['gold_answer']; start=e['gold_start']
    if c[start:start+len(answer)]!=answer:
        raise ValueError('CMRC gold offset mismatch')
    p0,p1=e['passage_start'],e['passage_end']
    if not (0<=p0<=start and start+len(answer)<=p1<=len(c)):
        raise ValueError('CMRC passage excludes answer')
    passage=c[p0:p1].strip()
    question=e['question']
    if passage not in row['messages'][0]['content'] or not row['messages'][0]['content'].endswith('\n\n'+question):
        raise ValueError('CMRC visible passage differs from evidence')
    expected_answers={answer.strip()}
    if row['subtype']=='quote':
        s0,s1=e['evidence_sentence_start'],e['evidence_sentence_end']
        if not (p0<=s0<=start and start+len(answer)<=s1<=p1):
            raise ValueError('CMRC quote outside visible passage')
        sentence=c[s0:s1].strip()
        expected_answers={f'{answer.strip()}\n原文写道：“{sentence}”',
                          f'{answer.strip()}\n依据是：“{sentence}”',
                          f'{answer.strip()}\n材料里对应的一句是：“{sentence}”'}
    if row['messages'][-1]['content'] not in expected_answers:
        raise ValueError('CMRC complete response differs from answer and original evidence')


def summary(rows):
    lengths=[len(r['input_ids']) for r in rows]
    answer_lengths=[]
    for row in rows:
        ids=row['input_ids']
        for pos,token in enumerate(ids):
            if token==5:
                answer_lengths.append(ids.index(6,pos+1)-pos-1)
    return {'documents':len(rows),'tokens':sum(lengths),
            'assistant_tokens':sum(t!=IGNORE_LABEL for r in rows for t in r['labels']),
            'assistant_content_tokens':sum(t>6 for r in rows for t in r['labels']),
            'assistant_turns':len(answer_lengths),'multi':sum(len(r['messages'])>2 for r in rows),
            'max_context':max(lengths),'min_context':min(lengths),
            'mean_assistant_content_tokens':statistics.mean(answer_lengths),
            'median_assistant_content_tokens':statistics.median(answer_lengths),
            'one_token_answers':answer_lengths.count(1),
            'category':dict(Counter(r['category'] for r in rows)),
            'source':dict(Counter(r['source'] for r in rows)),
            'subtype':dict(Counter(r.get('subtype','unknown') for r in rows))}


def review_sample(rows):
    sample=[]
    targets=dict(zip(CATEGORIES,(80,50,40,30)))
    for category,target in targets.items():
        strata=defaultdict(list)
        for r in rows:
            if r['category']!=category:
                continue
            n=r['assistant_content_tokens']
            band='short' if n<=32 else ('medium' if n<=96 else 'long')
            strata[(r['source'],r.get('subtype',''),band)].append(r)
        for values in strata.values():
            values.sort(key=lambda r:stable(r['id'],'quality_sample'))
        keys=sorted(strata)
        count=0
        while count<target:
            moved=False
            for key in keys:
                if strata[key] and count<target:
                    r=strata[key].pop(0)
                    sample.append({k:v for k,v in r.items() if k not in ('input_ids','labels')})
                    count+=1; moved=True
            if not moved:
                raise ValueError('Not enough rows for stratified quality review')
    return sorted(sample,key=lambda r:stable(r['id'],'review_order'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-files',type=Path,nargs='+',required=True)
    p.add_argument('--evaluation-prompts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing candidate selection')
    args.output.mkdir(parents=True)
    args.reports.mkdir(parents=True,exist_ok=True)
    base=json.loads((ROOT/'data/stage1_v1/manifest.json').read_text(encoding='utf-8'))
    tokenizer=Tokenizer.from_file(base['tokenizer_path'])
    external,external_records=external_index([args.evaluation_prompts,
        ROOT/'evaluation/comparison_v2/prompts.jsonl',ROOT/'evaluation/stage1_v1/prompts.jsonl'])
    old_verified_heldout,old_verified_records=external_index([
        ROOT/'data/sft_verified_v2/validation.jsonl',ROOT/'data/sft_verified_v2/test.jsonl'])
    candidates=[]
    for path in args.public_files:
        for r in json_rows(path):
            if r['category'] not in CATEGORIES[:3]:
                raise ValueError('Unexpected public category')
            r=dict(r)
            r['id']='natural-v3:'+stable(r['source']+'|'+r['source_file']+'|'+str(r['source_row_id']),'id')
            candidates.append(r)
    candidates.extend(legacy_candidates())
    candidates.sort(key=lambda r:stable(r['id'],'selection'))
    selected={s:[] for s in SPLITS}
    counts={s:Counter() for s in SPLITS}
    rejects=Counter(); family_counts=Counter(); seen_ids=set()
    moss_near=ConflictIndex(minimum=12)
    cmrc_near=defaultdict(lambda:ConflictIndex(minimum=12))
    cross_split_indices={s:ConflictIndex(minimum=12) for s in SPLITS}
    group_splits={}; group_aliases={}; context_aliases={}; retained_groups=defaultdict(set)
    filtered=0
    for number,row in enumerate(candidates,1):
        category=row['category']
        if row['id'] in seen_ids:
            rejects['duplicate_source_row']+=1; continue
        seen_ids.add(row['id'])
        if row['source']=='hfl/cmrc2018':
            # Canonicalize original document text, not cropped passages or source split labels.
            group='cmrc:'+stable(normalized_content(row['source_evidence']['context']),'document')
            if row['source_group_id'] in group_aliases and group_aliases[row['source_group_id']]!=group:
                raise ValueError('One CMRC source document ID has multiple full contexts; must group connected aliases')
            if group in context_aliases and context_aliases[group]!=row['source_group_id']:
                raise ValueError('CMRC full context has multiple source IDs; must group connected aliases')
            group_aliases[row['source_group_id']]=group
            context_aliases[group]=row['source_group_id']
        else:
            group=row['source_group_id']
        split=row.get('legacy_split') or public_split(group)
        if group in group_splits and group_splits[group]!=split:
            raise ValueError('Source group crosses partitions')
        group_splits[group]=split
        if counts[split][category]>=QUOTAS[split][category]:
            continue
        if category=='verified' and family_counts[(split,row['task_family'])]>=row['family_quota']:
            continue
        users=[normalized_content(m['content']) for m in row['messages'] if m['role']=='user']
        if any(external.match(u) for u in users) or (len(users)>1 and external.match('|'.join(users))):
            rejects['external_evaluation_overlap']+=1; continue
        if split=='train' and category!='verified' and any(old_verified_heldout.match(u) for u in users):
            rejects['old_verified_holdout_overlap']+=1; continue
        visible_keys=[users[0],'|'.join(m['role']+':'+normalized_content(m['content']) for m in row['messages'])]
        if any(index.match(key) for peer,index in cross_split_indices.items() if peer!=split for key in visible_keys):
            rejects['cross_source_first_turn_or_trajectory_near_duplicate']+=1; continue
        if row['source']=='hfl/cmrc2018':
            key=normalized_content(row['source_evidence']['question'])
            if cmrc_near[group].match(key):
                rejects['same_document_question_near_duplicate']+=1; continue
        elif category!='verified':
            key='|'.join(users)
            if moss_near.match(key):
                rejects['public_conversation_near_duplicate']+=1; continue
        if category=='verified':
            verify_encoded_row(row,tokenizer,512)
        elif row['source']=='hfl/cmrc2018':
            audit_cmrc(row)
        encoded=encode_conversation(tokenizer,row['messages'])
        if len(encoded['input_ids'])>512:
            rejects['over_context']+=1; continue
        if category=='multi' and len(row['messages'])<4:
            raise ValueError('Multi category contains a single-turn example')
        row.update(encoded)
        row['split']=split; row['split_group_id']=group
        row['tokens']=len(encoded['input_ids'])
        row['assistant_tokens']=sum(t!=IGNORE_LABEL for t in encoded['labels'])
        row['assistant_content_tokens']=sum(t>6 for t in encoded['labels'])
        if row['source']=='hfl/cmrc2018':
            cmrc_near[group].add(key)
        elif category!='verified':
            moss_near.add(key)
        selected[split].append(row); counts[split][category]+=1
        retained_groups[split].add(group)
        for key in visible_keys:
            cross_split_indices[split].add(key)
        if category=='verified':
            family_counts[(split,row['task_family'])]+=1
        filtered+=1
        if filtered%1000==0:
            print(json.dumps({'selected':filtered,'scanned':number,'counts':counts},ensure_ascii=False),flush=True)
    missing={s:{c:QUOTAS[s][c]-counts[s][c] for c in CATEGORIES if counts[s][c]!=QUOTAS[s][c]} for s in SPLITS}
    intake={'candidate_rows':len(candidates),'selected_counts':counts,'missing':missing,'rejections':rejects,
            'external_screening':external_records,'old_verified_holdout_train_screening':old_verified_records,
            'cmrc_one_to_one_doc_context_groups':len(group_aliases),
            'group_overlap':{f'{a}/{b}':len(retained_groups[a]&retained_groups[b])
            for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]},
            'near_duplicate_scope':'Across partitions and sources, rendered first user and full visible trajectory exact/Jaccard>=0.8 screening. Within selected MOSS, complete user sequence Jaccard>=0.8; CMRC same original document questions Jaccard>=0.8. CMRC source document IDs and canonical full contexts verified one-to-one before grouping. Verified anchors keep original disjoint scenario/entity/template partitions and are semantically recomputed; public train also excludes old verified heldout user prompts.',
            'limitations':'Lexical near matching does not prove no semantic paraphrases. Source group splitting is deterministic. New experiment is not a CMRC benchmark; original train/validation source documents are pooled then grouped. Verified train anchors were present in baseline SFT.'}
    json_write(args.reports/'selection_intake.json',intake)
    if any(missing.values()):
        raise ValueError('Candidate quality/quantity cannot fill frozen quotas; inspect selection_intake.json')
    if any(intake['group_overlap'].values()):
        raise ValueError('Selected source groups overlap partitions')
    stats={s:summary(selected[s]) for s in SPLITS}
    files={'tokenizer.json':base['files']['tokenizer.json']}
    for split in SPLITS:
        rows=sorted(selected[split],key=lambda r:stable(r['id'],'global_shuffle'))
        path=args.output/f'{split}.jsonl'
        with path.open('x',encoding='utf-8',newline='\n') as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        files[path.name]=file_identity(path)
    manifest={'schema_version':1,'corpus_id':'sft_natural_v3','frozen':True,'vocab_size':base['vocab_size'],
              'seed':42,'max_tokens':512,'tokenizer_path':base['tokenizer_path'],
              'tokenizer_sha256':base['tokenizer_sha256'],'special_token_ids':SPECIAL_IDS,
              'counts':{s:sum(QUOTAS[s].values()) for s in SPLITS},'splits':stats,'files':files,
              'sources':sorted({(r['source'],r['source_revision'],r['source_license']) for rows in selected.values() for r in rows}),
              'quota':QUOTAS,'screened_inputs':external_records,'supervision':'unchanged assistant content/end_turn/final EOS',
              'usage':'Source attribution and per-row provenance retained; CMRC CC-BY-SA-4.0, MOSS CC-BY-4.0, local verified anchors. No claim of legal review or universal dialogue ability.',
              'split_policy':intake['near_duplicate_scope'],'limitations':intake['limitations']}
    json_write(args.output/'manifest.prepared.json',manifest)
    json_write(args.reports/'data_stats.json',stats)
    json_write(args.reports/'quality_sample_200.json',review_sample(selected['train']))
    print(json.dumps({'event':'prepared_for_independent_review','output':str(args.output),'stats':stats},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
