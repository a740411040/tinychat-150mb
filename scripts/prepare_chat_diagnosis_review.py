"""Prepare a compact blinded content review; no model inference or training."""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/diagnosis_v2_resume'
rows = [json.loads(s) for s in (OUT/'outputs.jsonl').read_text(encoding='utf-8').splitlines()]
cases = json.loads((OUT/'probes.json').read_text(encoding='utf-8'))
rng = random.Random(20261122)
key = {}
packet = []
for case in cases:
    if case['group'] != 'natural':
        continue
    candidates = [r for r in rows if r['group']=='natural' and r['id']==case['id'] and r['model'].endswith('sft')]
    rng.shuffle(candidates)
    answers = []
    for i, row in enumerate(candidates):
        cid = f'{case["id"]}_{i+1:02d}'
        key[cid] = {'model': row['model'], 'id': row['id'], 'variant': row['variant']}
        answers.append({'id': cid, 'answer': row['text']})
    packet.append({'messages':case['messages'], 'rubric':case['rubric'], 'candidates':answers})
for name, value in [('quality_blind_input.json',packet),('quality_private_key.json',key)]:
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
print(json.dumps({'cases':len(packet),'candidates':len(key),'input_bytes':(OUT/'quality_blind_input.json').stat().st_size}))
