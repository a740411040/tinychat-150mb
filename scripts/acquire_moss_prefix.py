"""Bounded public-source sampling: at most 64 MiB, keep complete original records only."""
import codecs
import json
from pathlib import Path
import time
from huggingface_hub.utils import get_session

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/public_sources/natural_sft_v3/moss-002-sft-data'
REV='d530c921add4c7551b4e2309e0e7fe9468278311'
URL=f'https://huggingface.co/datasets/OpenMOSS-Team/moss-002-sft-data/resolve/{REV}/zh_helpfulness.json'
LIMIT=64*1024*1024
CHUNK=8*1024*1024

def main():
    raw_path=OUT/'zh_helpfulness.prefix64m.raw'
    started=time.perf_counter()
    fetched=0
    with raw_path.open('xb') as raw:
        for start in range(0,LIMIT,CHUNK):
            end=min(LIMIT,start+CHUNK)-1
            with get_session().stream('GET',URL,headers={'Range':f'bytes={start}-{end}'},
                                      follow_redirects=True,timeout=60) as response:
                expected=f'bytes {start}-{end}/1040212294'
                if response.status_code!=206 or response.headers.get('content-range')!=expected:
                    raise RuntimeError(f'Range response mismatch: {response.status_code}, {response.headers.get("content-range")}')
                payload=bytearray()
                for chunk in response.iter_bytes(chunk_size=65536):
                    payload.extend(chunk)
                    if len(payload)>end-start+1:
                        raise RuntimeError('Response exceeded requested range')
                if len(payload)!=end-start+1:
                    raise RuntimeError('Incomplete range response')
                raw.write(payload); raw.flush(); fetched+=len(payload)
                print(json.dumps({'event':'range_saved','bytes':fetched,'limit':LIMIT,'elapsed_seconds':time.perf_counter()-started}),flush=True)
    # A prefix may end in a UTF-8 character or JSON object; only complete records are retained.
    text=codecs.getincrementaldecoder('utf-8')().decode(raw_path.read_bytes(),final=False)
    decoder=json.JSONDecoder()
    pos=len(text)-len(text.lstrip())
    if text[pos]!='[':
        raise ValueError('Expected the original top-level JSON array')
    pos+=1; records=0
    output=OUT/'zh_helpfulness.prefix64m.jsonl'
    with output.open('x',encoding='utf-8',newline='\n') as f:
        while pos<len(text):
            while pos<len(text) and text[pos] in ' \t\r\n,':
                pos+=1
            if pos==len(text) or text[pos]==']':
                break
            try:
                value,next_pos=decoder.raw_decode(text,pos)
            except json.JSONDecodeError:
                break
            if not isinstance(value,dict):
                raise ValueError('Non-object source record')
            f.write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
            records+=1; pos=next_pos
    if not records:
        raise ValueError('No complete source record in prefix')
    metadata={'repo':'OpenMOSS-Team/moss-002-sft-data','revision':REV,'source_file':'zh_helpfulness.json',
              'url':URL,'license_from_card':'cc-by-4.0','range_start':0,'range_end_inclusive':fetched-1,
              'bytes_fetched':fetched,'maximum_bytes':LIMIT,'full_source_bytes':1040212294,
              'complete_records':records,'output':str(output),'elapsed_seconds':time.perf_counter()-started,
              'selection_bias':'Contiguous file prefix, not random coverage of entire source',
              'incomplete_final_record':'Excluded without repairing or fabricating text'}
    with (OUT/'zh_helpfulness.prefix64m.metadata.json').open('x',encoding='utf-8') as f:
        json.dump(metadata,f,ensure_ascii=False,indent=2)
    print(json.dumps(metadata,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
