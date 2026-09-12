"""Stage immutable P04 source and verified shards; never launch GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from manage import ART,PROJECT,RUN,CODE,PY,ssh

CACHE=PROJECT+'/data/rsna-knee-uint8-224-9-c130-w10-90'

def remote(code,args=()):
    return ssh('nsu-a100',['python3','-c',code]+list(args))

def transfer(local,dest):
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=10',str(local),
        'nsu-a100:'+dest],check=True,timeout=300)

def stage_source():
    source=ART/'source'
    remote("import json,sys; from pathlib import Path; [Path(p).mkdir(parents=True,exist_ok=True) for p in sys.argv[1:]]; print('{}')",
        [CODE,RUN+'/tmp'])
    existing=remote("import json,sys,hashlib; from pathlib import Path; p=Path(sys.argv[1]); print(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()}))",[CODE])
    for f in sorted(source.iterdir()):
        if not f.is_file():continue
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        if f.name in existing:
            if existing[f.name]!=digest:raise RuntimeError(f'Frozen remote source differs: {f.name}')
        else:transfer(f,CODE+'/'+f.name)
    manifest=dict(owner='rsna-codex-p04',project='rsna-knee-abnormality-detection',
        code=CODE,cache=CACHE,source_manifest_sha256=hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest(),
        peak_new_storage_gib=16,cache_gib=11.12,raw_dicom_staged=False,reports_staged=False,
        environment=PY,state='WAITING_CACHE',retention=dict(owner='rsna-codex-p04',
            reason='Fixed P04 window comparison and conditional refit',review_date='2026-09-14'))
    local=ART/'run_manifest.json';local.write_text(json.dumps(manifest,indent=2))
    transfer(local,RUN+'/run.json')
    print(json.dumps(dict(status='SOURCE_STAGED',code=CODE,run=RUN)))

def stage_cache(local):
    local=Path(local);spec=json.loads((local/'SPEC.json').read_text())
    if spec['window']!=[.1,.9] or spec['n_studies']!=4407:raise ValueError('Wrong cache')
    remote("import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); print('{}')",[CACHE])
    for name,digest in spec['sha256'].items():
        if Path(name).name!=name:raise ValueError('Non-local cache filename')
        f=local/name;h=hashlib.sha256()
        with f.open('rb') as stream:
            for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
        if h.hexdigest()!=digest:raise ValueError(f'Local cache hash differs: {name}')
        info=remote("import json,sys,hashlib; from pathlib import Path; p=Path(sys.argv[1]); h=hashlib.sha256(); f=p.open('rb') if p.exists() else None; [h.update(b) for b in iter(lambda:f.read(8<<20),b'')] if f else None; print(json.dumps({'exists':f is not None,'sha256':h.hexdigest()}))",[CACHE+'/'+name])
        if info['exists']:
            if info['sha256']!=digest:raise RuntimeError(f'Remote partial/different shard: {name}; inspect before replacement')
        else:transfer(f,CACHE+'/'+name)
        print(f'Staged {name}',flush=True)
    for name in ['SPEC.json','NOTICE.md','LICENSE-APACHE-2.0.txt']:
        if (local/name).exists():transfer(local/name,CACHE+'/'+name)
    check=ssh('nsu-a100',[PY,CODE+'/remote_preflight.py'])
    (ART/'REMOTE_READY.json').write_text(json.dumps(check,indent=2))
    print(json.dumps(check,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['source','cache']);p.add_argument('--cache')
    a=p.parse_args()
    if a.action=='source':stage_source()
    else:stage_cache(a.cache)
