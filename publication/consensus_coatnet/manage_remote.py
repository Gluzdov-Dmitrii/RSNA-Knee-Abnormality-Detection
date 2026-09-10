"""Local coordinator client. Own lease tokens live only in ignored artifacts."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import uuid

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/consensus_coatnet'
PROJECT='/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection'
RUN=PROJECT+'/runs/20260910T0913Z-codex-consensus'
QUEUE='/home/scientists/gluz_d_s/kaggle/_control/resource_queue.py'
PY=PROJECT+'/envs/ngpu01/py3.11-torch-cu124-v1/bin/python'
JOB=PROJECT+'/code/consensus-coatnet-v4/remote_job.py'

def ssh(host,args,ok=(0,)):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)],capture_output=True,text=True,timeout=45)
    if r.returncode not in ok: raise RuntimeError(f'SSH exit{r.returncode}: {r.stderr[:500]}')
    return json.loads(r.stdout)

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['request','launch','status','release'])
    p.add_argument('--mode',choices=['smoke','smoke2','pair','verified_pair','full'],required=True); a=p.parse_args()
    ART.mkdir(parents=True,exist_ok=True); lease_path=ART/f'{a.mode}_lease.json'
    if lease_path.exists(): lease=json.loads(lease_path.read_text())
    else:
        if a.action!='request': raise RuntimeError('Request first')
        lease=dict(id=f'rsna-codex-consensus-{a.mode}-20260910',token=str(uuid.uuid4()))
        lease_path.write_text(json.dumps(lease))
    base=['python3',QUEUE]
    def queue(action,extras=(),ok=(0,)):
        return ssh('nsu-quadro',base+[action,'--id',lease['id'],'--token',lease['token']]+list(extras),ok)
    def request():
        return queue('request',['--owner','rsna-codex-publication','--project','rsna-knee-abnormality-detection',
            '--run-path',RUN+'/'+a.mode,'--pool','a100','--count','1','--cpu','8','--ram-gib','48','--disk-gib','6',
            '--minutes','10' if a.mode.startswith('smoke') else '120'],ok=(0,3))
    if a.action=='request':
        result=request(); lease['request']=result; lease_path.write_text(json.dumps(lease))
        print(json.dumps({k:v for k,v in result.items() if k not in ['token']})); return
    if a.action=='launch':
        if 'dispatch_attempted' in lease: raise RuntimeError('Already attempted; reconcile status, never relaunch')
        result=request()
        if result.get('state')!='RESERVED' or result.get('health')!='CURRENT': raise RuntimeError('No fresh lease')
        gpu=result['gpus'][0]
        check=subprocess.run(['ssh','-o','BatchMode=yes','nsu-a100','nvidia-smi',
            '--query-compute-apps=gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=30,check=True)
        if gpu in check.stdout: raise RuntimeError('Reserved GPU already has a process; do not launch')
        lease['dispatch_attempted']=True; lease_path.write_text(json.dumps(lease))
        job=ssh('nsu-a100',[PY,JOB,'launch','--mode',a.mode,'--gpu',gpu])
        lease['process']=job; lease_path.write_text(json.dumps(lease))
        result=queue('started',['--pid',str(job['pid']),'--process-start',job['start']])
        print(json.dumps(job)); return
    version='v1' if a.mode=='smoke' else 'v2' if a.mode in ['smoke2','pair'] else 'v3' if a.mode=='verified_pair' else 'v4'
    job_path=PROJECT+f'/code/consensus-coatnet-{version}/remote_job.py'
    status=ssh('nsu-a100',[PY,job_path,'status','--mode',a.mode])
    if status['alive']:
        if a.action=='release': raise RuntimeError('Refusing release of live supervisor')
        queue('heartbeat')
    elif a.action=='release':
        gpu=status['gpu']
        check=subprocess.run(['ssh','-o','BatchMode=yes','nsu-a100','nvidia-smi',
            '--query-compute-apps=gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=30,check=True)
        if gpu in check.stdout: raise RuntimeError('GPU still has process; cannot verify stopped')
        if 'result' not in status: raise RuntimeError('Supervisor result missing; reconcile children manually')
        result=queue('release',['--verified-stopped']); status['released']=result.get('state')
    print(json.dumps(status))

if __name__=='__main__': main()
