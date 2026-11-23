"""Read-only causal probes of saved models; never train or update weights."""
import argparse
import collections
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinychat.chat_format import encode_conversation, IGNORE_LABEL
from tinychat.evaluate_stage1 import diagnostics
from tinychat.inference import load_model

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['arithmetic', 'extraction', 'lookup', 'clarification', 'correction', 'count', 'classify', 'reference']
MODELS = {'old_base': 'base_100m_v1', 'clean_base': 'base_clean_100m_v2',
          'old_sft': 'chat_oldbase_v2', 'clean_sft': 'chat_cleanbase_v2'}
PARAPHRASES = [
    ['425 减去 30 是多少？只写结果。'],
    ['记录写着：所用纸张：铜版纸；归档点：绛坊资料柜；条目名称：柳坡展览记录；代号：莓潭。请原样抄出所用纸张的值，只写这个值。'],
    ['有张表，列按顺序是鸿坊、酥岭、铃浦；行按顺序是杉坊、绛梁、柳坡。三行数值依次是 [-61,-27,79]、[-67,-60,-54]、[72,31,53]。绛梁这一行、铃浦这一列填了什么？只写整数。'],
    ['柚潭小组登记寄存，必须填写寄存人、柜号、件数。现在寄存人为空，柜号是络田柜4号，件数是3。空字符串算缺项。请按上述顺序列缺项，用“请补充甲、乙。”这样的格式回答；若都填了则回答“资料已齐。”。'],
    ['原记录中存放处是白梁室，负责人是杉岭，记录代号是珀坊。现在只把存放处更正为杏坊室，其他字段不变。请只说现在的存放处。'],
    ['数列是 [1,44,5,13,49,55,15,10,1,8,38,14,41]。其中从11到29（两端都含）的偶数有几项？重复的逐个数，只写整数。', '现在不限制奇偶，其他条件沿用上轮。只写数量。'],
    ['测量值是38。分组规则是：21到26叫杏坡，26到42叫柳坡，42到57叫绛涧，每段都含左端、不含右端。这个值属于哪组？只写标签。'],
    ['白闸说：“我先提到玄麓展签，随后提到玄坎图册。”缃岛说：“我先提到玄坡木盒，随后提到桃梁色卡。”前者指先提到的，后者指随后提到的。白闸的前者是什么？只写物件。'],
]
NATURAL = [
    ('greet', [{'role': 'user', 'content': '你好，我刚搬来这里，想和你打个招呼。'}], '你好！很高兴认识你。', '回应问候，内容与初次认识有关。'),
    ('introduce', [{'role': 'user', 'content': '你好，请简单介绍一下你自己。'}], '你好，我是一个中文助手。', '简短介绍助手身份，不虚构现实经历。'),
    ('comfort', [{'role': 'user', 'content': '今天面试没通过，有点失落。你能安慰我两句吗？'}], '没通过会失落很正常，这次结果不代表你的全部能力。', '回应失落并给适度安慰。'),
    ('thanks', [{'role': 'user', 'content': '谢谢你帮我整理资料！'}], '不客气，很高兴能帮上忙。', '对感谢做自然简短回应。'),
    ('steam', [{'role': 'user', 'content': '请用一句简单的话解释：湿衣服为什么晾着会变干？'}], '衣服上的水会蒸发到空气中，所以衣服会变干。', '提到水蒸发，因果正确且一句话。'),
    ('grounded', [{'role': 'user', 'content': '通知：读书会改到周六上午十点，在三楼会议室举行。读书会在哪里开？请只根据通知回答。'}], '三楼会议室。', '回答三楼会议室，不编造地址。'),
    ('memory', [{'role': 'user', 'content': '我给我的小猫取名叫团团，请记住。'}, {'role': 'assistant', 'content': '好的，你的小猫叫团团。'}, {'role': 'user', 'content': '我的小猫叫什么？'}], '团团。', '准确回答团团。'),
    ('clarify', [{'role': 'user', 'content': '帮我把出行路线安排一下。'}], '你从哪里出发，准备去哪里？', '缺目的地和出发地时先澄清，不编造路线。'),
]
PLAIN = [
    '今天天气晴朗，小林吃过早饭，背上书包，',
    '衣服晾干是因为水分蒸发。水从液态变成气态后，',
    '通知：读书会改到周六上午十点，在三楼会议室举行。因此，读书会的地点是',
    '朋友在面试后感到失落，我想安慰他说：',
]


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_rows(split):
    path = ROOT / 'data/sft_verified_v2' / (split + '.jsonl')
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def probes(validation):
    result = []
    for i, family in enumerate(FAMILIES):
        source = next(r for r in validation if r['task_family'] == family)
        reference = source['messages'][-1]['content']
        messages = source['messages'][:-1]
        result.append({'id': f'id_{i+1:02d}', 'group': 'in_domain', 'family': family,
                       'source_id': source['id'], 'messages': messages, 'reference': reference})
        revised = [dict(m) for m in messages]
        user_index = 0
        for m in revised:
            if m['role'] == 'user':
                m['content'] = PARAPHRASES[i][user_index]
                user_index += 1
        assert user_index == len(PARAPHRASES[i])
        result.append({'id': f'para_{i+1:02d}', 'group': 'paraphrase', 'family': family,
                       'source_id': source['id'], 'messages': revised, 'reference': reference})
    for name, messages, reference, rubric in NATURAL:
        result.append({'id': name, 'group': 'natural', 'messages': messages,
                       'reference': reference, 'rubric': rubric})
    return result


def data_audit(tokenizer, validation):
    report = {}
    for split, rows in [('train', read_rows('train')), ('validation', validation)]:
        lengths, firsts = [], collections.Counter()
        hist = collections.Counter()
        mismatches = []
        prefix_count = 0
        for row in rows:
            ids, labels = row['input_ids'], row['labels']
            hist.update(t for t in labels if t != IGNORE_LABEL)
            positions = [i for i, t in enumerate(ids) if t == 5]
            turns = [i for i, m in enumerate(row['messages']) if m['role'] == 'assistant']
            assert len(positions) == len(turns)
            for position, turn in zip(positions, turns):
                expected = encode_conversation(tokenizer, row['messages'][:turn], generation_prompt=True)['input_ids']
                prefix_count += 1
                if ids[:position+1] != expected:
                    mismatches.append([row['id'], turn])
                length = ids.index(6, position+1)-position-1
                lengths.append(length)
                firsts[labels[position+1]] += 1
        ordered = sorted(lengths)
        report[split] = {'rows': len(rows), 'assistant_turns': len(lengths), 'targets': sum(hist.values()),
                         'content_tokens': sum(v for k, v in hist.items() if k > 6),
                         'eos_targets': hist[2], 'end_turn_targets': hist[6],
                         'stop_fraction': (hist[2]+hist[6])/sum(hist.values()),
                         'mean_content_length': statistics.mean(lengths), 'median_content_length': statistics.median(lengths),
                         'p90_content_length': ordered[int(.90*len(ordered))], 'p95_content_length': ordered[int(.95*len(ordered))],
                         'one_token_answers': lengths.count(1), 'empty_answers': lengths.count(0),
                         'first_target_stop_count': firsts[2]+firsts[6],
                         'prefix_checks': prefix_count, 'prefix_mismatches': mismatches}
    return report


@torch.inference_mode()
def validation_ce(model, rows):
    sums = collections.defaultdict(float)
    counts = collections.Counter()
    for off in range(0, len(rows), 8):
        batch = rows[off:off+8]
        width = max(len(r['input_ids'])-1 for r in batch)
        x = torch.zeros((len(batch), width), dtype=torch.long, device='cuda')
        y = torch.full_like(x, -100)
        for i, row in enumerate(batch):
            n = len(row['input_ids'])-1
            x[i, :n] = torch.tensor(row['input_ids'][:-1], device='cuda')
            y[i, :n] = torch.tensor(row['labels'][1:], device='cuda')
        logits = model(x).float()
        losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none', ignore_index=-100).view_as(y)
        masks = {'all': y != -100, 'content': y > 6, 'stop': (y == 2) | (y == 6),
                 'first': (x == 5) & (y != -100)}
        for key, mask in masks.items():
            counts[key] += int(mask.sum())
            sums[key] += float(losses[mask].sum())
        del logits, losses, x, y
    return {key: {'ce': sums[key]/counts[key], 'targets': counts[key]} for key in counts}


def first_metrics(logits, tokenizer, reference):
    logp = logits.log_softmax(-1)
    p = logp.exp()
    candidates = logits.topk(5).indices.tolist()
    result = {'top5': [{'id': i, 'token': tokenizer.id_to_token(i), 'probability': float(p[i])} for i in candidates],
              'entropy': float(-(p*logp).sum())}
    for name, i in [('eos', 2), ('end_turn', 6)]:
        result[name] = {'probability': float(p[i]), 'rank': 1+int((logits > logits[i]).sum())}
    if reference:
        i = tokenizer.encode(reference, add_special_tokens=False).ids[0]
        result['reference_first'] = {'id': i, 'probability': float(p[i]), 'rank': 1+int((logits > logits[i]).sum())}
    return result


@torch.inference_mode()
def decode(model, tokenizer, prompt, variant, seed, reference):
    assert len(prompt)+96 <= model.config.max_seq_len
    ids, steps = [], []
    generator = torch.Generator(device='cuda').manual_seed(seed)
    first = None
    for index in range(96):
        x = torch.tensor([prompt+ids], dtype=torch.long, device='cuda')
        logits = model(x)[0, -1].float()
        p = logits.softmax(-1)
        if first is None:
            first = first_metrics(logits, tokenizer, reference)
        modified = logits.clone()
        if variant == 'minimum4' and index < 4:
            modified[[2, 6]] = -torch.inf
        if variant == 'no_repeat4' and len(ids) >= 3:
            for start in range(len(ids)-3):
                if ids[start:start+3] == ids[-3:]:
                    modified[ids[start+3]] = -torch.inf
        if variant.startswith('sample'):
            ordered, indices = (modified/.8).sort(descending=True)
            probabilities = ordered.softmax(-1)
            probabilities[probabilities.cumsum(-1)-probabilities > .9] = 0
            selected = int(indices[torch.multinomial(probabilities, 1, generator=generator)])
        else:
            selected = int(modified.argmax())
        steps.append({'id': selected, 'original_probability': float(p[selected]),
                      'original_stop_probability': float(p[2]+p[6]),
                      'entropy': float(-(p*logits.log_softmax(-1)).sum())})
        ids.append(selected)
        if selected in (2, 6):
            break
    text = tokenizer.decode(ids, skip_special_tokens=True)
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    return {'text': text, 'raw_text': raw, 'generated_ids': ids, 'input_tokens': len(prompt),
            'output_tokens': len(ids), 'stop_reason': 'end_token' if ids[-1] in (2, 6) else 'max_new_tokens',
            'first': first, 'trajectory': steps, 'diagnostics': diagnostics(text, raw, reference),
            'exact_reference': text.strip() == reference.strip() if reference else None}


def aggregate(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        for group in ['all', row['group']]:
            grouped[(row['model'], row['variant'], group)].append(row)
    result = []
    for (model, variant, group), values in grouped.items():
        result.append({'model': model, 'variant': variant, 'group': group, 'n': len(values),
                       'empty': sum(r['diagnostics']['empty'] for r in values),
                       'severe_repetition': sum(r['diagnostics']['severe_repetition'] for r in values),
                       'max_length': sum(r['stop_reason']=='max_new_tokens' for r in values),
                       'immediate_eos': sum(r['generated_ids'][0]==2 for r in values),
                       'immediate_end_turn': sum(r['generated_ids'][0]==6 for r in values),
                       'exact_reference': sum(r['exact_reference'] is True for r in values),
                       'exact_reference_denominator': sum(r['exact_reference'] is not None for r in values),
                       'mean_output_tokens': statistics.mean(r['output_tokens'] for r in values),
                       'mean_first_eos_probability': statistics.mean(r['first']['eos']['probability'] for r in values)})
    return result


def normalize_reference_metrics(row):
    # Natural answers have many valid forms; their reference is for logit probes only.
    if row['group'] not in {'in_domain', 'paraphrase'}:
        row['exact_reference'] = None
        row['diagnostics']['reference_substring_present'] = None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    validation = read_rows('validation')
    cases = probes(validation)
    write_json(out/'probes.json', cases)
    source_files = [ROOT/'artifacts'/v/'model.safetensors' for v in MODELS.values()]
    before = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in source_files}
    rows, ce = [], {}
    if args.resume_from:
        previous = args.resume_from.resolve()
        prior_run = json.loads((previous/'run.json').read_text(encoding='utf-8'))
        assert prior_run['source_state_before'] == before, 'Source artifacts changed since previous run'
        assert json.loads((previous/'probes.json').read_text(encoding='utf-8')) == cases, 'Probe inputs changed'
        rows = [normalize_reference_metrics(json.loads(line)) for line in
                (previous/'outputs.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        write_json(out/'data_audit.json', json.loads((previous/'data_audit.json').read_text(encoding='utf-8')))
        for label in MODELS:
            path = previous/f'validation_ce_{label}.json'
            if path.exists():
                ce[label] = json.loads(path.read_text(encoding='utf-8'))
    completed = {(r['model'], r['id'], r['variant']) for r in rows}
    assert len(completed) == len(rows), 'Duplicate inherited results'
    write_json(out/'run.json', {'pid': os.getpid(), 'models': MODELS, 'variants': ['greedy','minimum4','sample0','sample1','no_repeat4'],
                              'seeds': [20261122, 20261123], 'max_new_tokens': 96, 'read_only_inference': True,
                              'diagnostic_not_independent_benchmark': True, 'source_state_before': before,
                              'resume_from': str(args.resume_from.resolve()) if args.resume_from else None,
                              'inherited_generations': len(rows)})
    with (out/'outputs.jsonl').open('x', encoding='utf-8', newline='\n') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        output.flush()
        for label, source in MODELS.items():
            model, tokenizer = load_model(ROOT/'artifacts'/source, device='cuda')
            model.requires_grad_(False)
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            if label == 'old_base' and not args.resume_from:
                write_json(out/'data_audit.json', data_audit(tokenizer, validation))
            if label not in ce:
                ce[label] = validation_ce(model, validation)
            write_json(out/f'validation_ce_{label}.json', ce[label])
            variants = ['greedy'] if label.endswith('base') else ['greedy','minimum4','sample0','sample1','no_repeat4']
            for case in cases:
                prompt = encode_conversation(tokenizer, case['messages'], generation_prompt=True)['input_ids']
                for variant in variants:
                    if (label, case['id'], variant) in completed:
                        continue
                    seed = 20261123 if variant == 'sample1' else 20261122
                    result = decode(model, tokenizer, prompt, variant, seed, case['reference'])
                    row = {'model': label, 'id': case['id'], 'group': case['group'], 'variant': variant, 'seed': seed, **result}
                    normalize_reference_metrics(row)
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    output.flush()
                    rows.append(row)
                print(json.dumps({'model': label, 'case': case['id'], 'completed_generations': len(rows)},ensure_ascii=False), flush=True)
            if label.endswith('base'):
                for index, text in enumerate(PLAIN):
                    if (label, f'plain_{index+1}', 'plain_greedy') in completed:
                        continue
                    prompt = tokenizer.encode(text, add_special_tokens=False).ids
                    result = decode(model, tokenizer, prompt, 'greedy', 20261122, None)
                    row = {'model': label, 'id': f'plain_{index+1}', 'group': 'plain', 'variant': 'plain_greedy',
                           'prompt': text, **result}
                    normalize_reference_metrics(row)
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    output.flush()
                    rows.append(row)
            del model
            torch.cuda.empty_cache()
    after = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in source_files}
    assert before == after, 'Source artifacts were modified'
    write_json(out/'summary.json', {'generations': len(rows), 'elapsed_seconds': time.perf_counter()-started,
                                  'metrics': aggregate(rows), 'validation_ce': ce, 'source_files_unchanged': before == after,
                                  'grad_enabled': torch.is_grad_enabled(), 'parameters_requires_grad': False})
    print(json.dumps({'event': 'complete', 'generations': len(rows), 'elapsed_seconds': time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    main()
