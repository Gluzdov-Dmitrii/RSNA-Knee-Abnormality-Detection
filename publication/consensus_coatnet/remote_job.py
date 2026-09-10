"""Remote bounded supervisor/launcher. Called only after a common GPU lease."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import hashlib

PROJECT=Path('/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection')
RUN=PROJECT/'runs/20260910T0913Z-codex-consensus'
CODE=Path(__file__).resolve().parent
PY=PROJECT/'envs/ngpu01/py3.11-torch-cu124-v1/bin/python'

def identity(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists(): return None
    fields=path.read_text().rsplit(')',1)[1].split()
    return None if fields[0]=='Z' else fields[19]

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['launch','work','status'])
    p.add_argument('--mode',choices=['smoke','smoke2','pair','verified_pair','full'],required=True); p.add_argument('--gpu')
    a=p.parse_args(); assert socket.gethostname()=='ngpu01'
    base=RUN/a.mode; base.mkdir(parents=True,exist_ok=True); process_path=base/'process.json'
    if a.action=='status':
        meta=json.loads(process_path.read_text()); meta['alive']=identity(meta['pid'])==meta['start']
        result=base/'supervisor_result.json'
        if result.exists(): meta['result']=json.loads(result.read_text())
        print(json.dumps(meta)); return
    if a.action=='launch':
        if not a.gpu or not a.gpu.startswith('GPU-'): raise ValueError('Explicit reserved GPU UUID required')
        # Atomic refusal to redispatch even if the previous response was lost.
        fd=os.open(base/'launch.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
        env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            HF_HOME=str(PROJECT/'cache/consensus-coatnet-v1/hf'),HF_HUB_OFFLINE='1',
            TORCH_HOME=str(PROJECT/'cache/consensus-coatnet-v1/torch'),
            XDG_CACHE_HOME=str(PROJECT/'cache/consensus-coatnet-v1/xdg'),TMPDIR=str(RUN/'tmp'))
        with (base/'supervisor.log').open('a') as log:
            child=subprocess.Popen([str(PY),str(CODE/'remote_job.py'),'work','--mode',a.mode],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=CODE)
        meta=dict(pid=child.pid,start=identity(child.pid),mode=a.mode,gpu=a.gpu)
        process_path.write_text(json.dumps(meta)); print(json.dumps(meta)); return
    started=time.monotonic(); codes=[]; child=None
    def stop_child(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
        if _: raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop_child); signal.signal(signal.SIGINT,stop_child)
    frozen=json.loads((CODE/'source_manifest.json').read_text())
    arms=['pilkwang'] if a.mode.startswith('smoke') or a.mode=='full' else ['smoke','pilkwang','median']
    for arm in arms:
        for name,digest in frozen.items():
            if hashlib.sha256((CODE/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Frozen source changed: {name}')
        cmd=[str(PY),'-u',str(CODE/'train.py'),'--cache',str(PROJECT/'data/rsna-knee-uint8-224-9-c130'),
            '--folds',str(PROJECT/'data/FOLDS_V1/folds.csv'),'--labels',str(PROJECT/'data/labels'),
            '--gold-uids',str(RUN/'inputs/gold_uids.csv'),'--initialization',str(RUN/'prepared_correctnorm/initialization.pt'),
            '--arm','pilkwang' if arm=='smoke' else arm,'--fold','0','--epochs','4','--batch-size','4','--workers','0','--out',str(base/arm)]
        if a.mode.startswith('smoke') or arm=='smoke': cmd+=['--smoke-steps','8']
        if a.mode=='full': cmd+=['--full-data']
        with (base/f'{arm}.log').open('a') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=child.wait(timeout=max(1,(10 if a.mode.startswith('smoke') else 110)*60-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:
                stop_child()
                code=124
        codes.append(dict(arm=arm,exit_code=code))
        if code: break
    (base/'supervisor_result.json').write_text(json.dumps(dict(arms=codes,elapsed_seconds=time.monotonic()-started)))

if __name__=='__main__': main()
