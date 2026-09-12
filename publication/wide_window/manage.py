"""Own P04 reservation and process management; never submit to Kaggle."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import uuid

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/wide_window'
PROJECT='/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection'
RUN=PROJECT+'/runs/20260912-codex-wide-window'
CODE=PROJECT+'/code/coatnet-wide-window-v2'
QUEUE='/home/scientists/gluz_d_s/kaggle/_control/resource_queue.py'
PY=PROJECT+'/envs/ngpu01/py3.11-torch-cu124-v1/bin/python'

def ssh(host,args,ok=(0,)):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)],
        capture_output=True,text=True,timeout=45)
    if r.returncode not in ok:raise RuntimeError(f'SSH exit {r.returncode}: {r.stderr[:500]}')
    return json.loads(r.stdout)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['request','launch','status','release'])
    p.add_argument('--mode',choices=['smoke','pilot','full'],required=True);a=p.parse_args()
    if a.mode=='full' and a.action in ['request','launch']:
        comparison=json.loads((ART/'comparison.json').read_text())
        if (comparison['status']!='PROMOTE_TO_FULL_REFIT' or comparison['delta']<.005
            or comparison['bootstrap_95'][0]<=0 or comparison['targets_down_more_than_002']>1):
            raise RuntimeError('Full refit requires the predeclared comparison gate to pass')
    path=ART/f'{a.mode}_lease.json'
    if path.exists():lease=json.loads(path.read_text())
    else:
        if a.action!='request':raise RuntimeError('Request first')
        receipt=ART/'REMOTE_READY.json'
        if not receipt.is_file() or json.loads(receipt.read_text())['status']!='READY':
            raise RuntimeError('Complete input/hash/environment preflight before requesting GPU')
        lease=dict(id=f'rsna-codex-p04-{a.mode}-20260912',token=str(uuid.uuid4()))
        path.write_text(json.dumps(lease))
    def queue(action,extras=(),ok=(0,)):
        return ssh('nsu-quadro',['python3',QUEUE,action,'--id',lease['id'],'--token',lease['token']]+list(extras),ok)
    def request():
        return queue('request',['--owner','rsna-codex-p04','--project','rsna-knee-abnormality-detection',
            '--run-path',RUN+'/'+a.mode,'--pool','a100','--count','1','--cpu','8','--ram-gib','48',
            '--disk-gib','3','--minutes','10' if a.mode=='smoke' else '120'],ok=(0,3))
    if a.action=='request':
        r=request();lease['request']=r;path.write_text(json.dumps(lease))
        print(json.dumps({k:v for k,v in r.items() if 'token' not in k.lower()}));return
    if a.action=='launch':
        if lease.get('dispatch_attempted'):raise RuntimeError('Already attempted; reconcile existing process')
        current=ssh('nsu-a100',[PY,CODE+'/remote_preflight.py'])
        previous=json.loads((ART/'REMOTE_READY.json').read_text())
        for key in ['cache_spec_sha256','initialization_sha256','source_manifest_sha256','versions']:
            if current[key]!=previous[key]:raise RuntimeError(f'Input/runtime changed since preflight: {key}')
        r=request()
        if r.get('state')!='RESERVED' or r.get('health')!='CURRENT':raise RuntimeError('Fresh reservation required')
        gpu=r['gpus'][0]
        apps=subprocess.run(['ssh','-o','BatchMode=yes','nsu-a100','nvidia-smi',
            '--query-compute-apps=gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=30)
        if gpu in apps.stdout:raise RuntimeError('Reserved GPU has a process; do not launch')
        lease['dispatch_attempted']=True;path.write_text(json.dumps(lease))
        job=ssh('nsu-a100',[PY,CODE+'/remote_job.py','launch','--mode',a.mode,'--gpu',gpu])
        lease['process']=job;path.write_text(json.dumps(lease))
        queue('started',['--pid',str(job['pid']),'--process-start',job['start']])
        print(json.dumps(job));return
    if a.action=='release' and not lease.get('dispatch_attempted'):
        waiting=lease.get('request',{}).get('state')=='WAITING_RESOURCE'
        released=queue('cancel' if waiting else 'release',[] if waiting else ['--verified-stopped'])
        print(json.dumps(dict(released=released['state'],
            evidence='No dispatch was attempted for this owned lease')));return
    status=ssh('nsu-a100',[PY,CODE+'/remote_job.py','status','--mode',a.mode])
    if status['alive']:
        if a.action=='release':raise RuntimeError('Cannot release a live supervisor')
        queue('heartbeat')
    elif a.action=='release':
        if 'result' not in status:raise RuntimeError('Missing supervisor result: reconcile child processes')
        apps=subprocess.run(['ssh','-o','BatchMode=yes','nsu-a100','nvidia-smi',
            '--query-compute-apps=gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=30)
        if status['gpu'] in apps.stdout:raise RuntimeError('GPU still has process; cannot verify release')
        status['released']=queue('release',['--verified-stopped'])['state']
    print(json.dumps(status))

if __name__=='__main__':main()
