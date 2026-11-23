"""Summarize existing inference evidence and measured training-time baselines."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/diagnosis_v2_resume'
rows = [json.loads(s) for s in (OUT/'outputs.jsonl').read_text(encoding='utf-8').splitlines()]
facts = {'sample_scope':'24 diagnostic chat inputs plus 4 plain continuations; not an independent benchmark',
         'empty_crosscheck':[], 'introduction_first_eos':[], 'natural_examples':[], 'exact_metric_note':'Natural/plain exact metrics are N/A (denominator 0), never a correctness rate.'}
for model in ['old_base','clean_base','old_sft','clean_sft']:
    selected = [r for r in rows if r['model']==model and r['variant']=='greedy']
    empty = [r for r in selected if r['diagnostics']['empty']]
    facts['empty_crosscheck'].append({'model':model,'n':len(selected),'empty':len(empty),
                                    'empty_immediate_eos':sum(r['generated_ids']==[2] for r in empty),
                                    'empty_other':sum(r['generated_ids']!=[2] for r in empty)})
    introduction = next(r for r in selected if r['id']=='introduce')
    facts['introduction_first_eos'].append({'model':model,**introduction['first']['eos']})
for row in rows:
    if row['model'].endswith('sft') and row['id'] in ['introduce','thanks','grounded','memory'] and row['variant'] in ['greedy','minimum4','sample0']:
        facts['natural_examples'].append({k:row[k] for k in ['model','id','variant','text']})
timing = {}
for name in ['pretrain_100m_v1','pretrain_clean_100m_v2','sft_oldbase_v2','sft_cleanbase_v2']:
    m=json.loads((ROOT/'runs'/name/'metrics.json').read_text(encoding='utf-8'))
    keys=['cumulative_valid_loss_tokens','cumulative_input_tokens','cumulative_padding_tokens','elapsed_seconds','training_seconds','completed_updates','cumulative_examples']
    timing[name]={k:m[k] for k in keys}
pt_seconds=statistics.mean(timing[n]['elapsed_seconds'] for n in ['pretrain_100m_v1','pretrain_clean_100m_v2'])
train_seconds=statistics.mean(timing[n]['training_seconds'] for n in ['sft_oldbase_v2','sft_cleanbase_v2'])
padded=timing['sft_cleanbase_v2']['cumulative_input_tokens']+timing['sft_cleanbase_v2']['cumulative_padding_tokens']
timing['estimation']={'pt_seconds_per_100m':pt_seconds,'pt_tokens_per_second':100_000_000/pt_seconds,
                      'pt_hours_100m':pt_seconds/3600,'pt_hours_300m':pt_seconds*3/3600,'pt_hours_1b':pt_seconds*10/3600,
                      'sft_padded_input_tokens':padded,'sft_training_padded_tokens_per_second':padded/train_seconds,
                      'sft_6m_padded_raw_minutes':6_000_000/padded*train_seconds/60,
                      'sft_10m_padded_raw_minutes':10_000_000/padded*train_seconds/60,
                      'proposed_sft_20k_one_epoch_minutes_range':[10,25],
                      'pt_300m_hours_with_margin':[4,5],'pt_1b_hours_with_margin':[14,16],
                      'assumptions':'Same laptop, power mode, architecture, context512, BF16, B8A8. SFT estimate assumes 6-10M padded input tokens; length distribution and attention cost need recalibration. Data preparation and content review excluded.'}
for name,value in [('mechanism_evidence.json',facts),('time_estimates.json',timing)]:
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
print(json.dumps(timing['estimation'],ensure_ascii=False,indent=2))
