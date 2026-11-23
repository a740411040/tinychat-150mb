"""Download only pinned, public Chinese source files into this workspace."""
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
from huggingface_hub import hf_hub_download

SOURCES=[
    ('OpenMOSS-Team/moss-002-sft-data','d530c921add4c7551b4e2309e0e7fe9468278311','cc-by-4.0',
     ['README.md','zh_helpfulness.json','zh_honesty.json']),
    ('hfl/cmrc2018','137f2c45a24275fb68f6961c4d357f46288886aa','cc-by-sa-4.0',
     ['README.md','data/train-00000-of-00001.parquet','data/validation-00000-of-00001.parquet']),
]

def acquire(item):
    repo,rev,license_name,filename=item
    local_dir=ROOT/'data/public_sources/natural_sft_v3'/repo.split('/')[-1]
    path=hf_hub_download(repo,filename,repo_type='dataset',revision=rev,local_dir=local_dir)
    result={'repo':repo,'revision':rev,'license_from_card':license_name,'filename':filename,
            'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,
            'url':f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{filename}'}
    print(json.dumps({'event':'downloaded','repo':repo,'filename':filename,'bytes':result['bytes']},ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    items=[(repo,rev,lic,f) for repo,rev,lic,files in SOURCES for f in files]
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(acquire,items))
    out=ROOT/'planning/train_150mb/natural_sft_v3_sources.json'
    with out.open('x',encoding='utf-8') as f:
        json.dump({'purpose':'candidate source acquisition, not quality acceptance','files':records},f,ensure_ascii=False,indent=2)
    print(json.dumps({'event':'complete','files':len(records),'manifest':str(out)}),flush=True)
