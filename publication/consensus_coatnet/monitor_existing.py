"""Monitor one existing run, collect diagnostics and release its finished lease.

Never launches training, publishes artifacts, submits predictions, or touches
another agent's lease. This finite process is not a scheduled automation.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import argparse
from manage_remote import ROOT,ART,RUN

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['pair','verified_pair'],default='verified_pair')
    mode=p.parse_args().mode
    runtime=ART/f'monitor_{mode}'; runtime.mkdir(parents=True,exist_ok=True)
    manager=Path(__file__).with_name('manage_remote.py')
    deadline=time.monotonic()+3*3600
    with (runtime/'events.jsonl').open('a',encoding='utf-8') as log:
        while time.monotonic()<deadline:
            try:
                r=subprocess.run([sys.executable,str(manager),'status','--mode',mode],
                    capture_output=True,text=True,timeout=55,check=True)
                state=json.loads(r.stdout); state['checked_unix']=time.time()
                (runtime/'status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
                log.write(json.dumps(state)+'\n'); log.flush()
                if not state['alive']:
                    dest=ART/mode; dest.mkdir(exist_ok=True)
                    files=['supervisor_result.json','supervisor.log','pilkwang.log','median.log']
                    for arm in ['pilkwang','median']:
                        (dest/arm).mkdir(exist_ok=True)
                        files.extend(f'{arm}/{name}' for name in ['config.json','history.json','result.json','validation_predictions.csv'])
                    for name in files:
                        fetched=subprocess.run(['scp','-o','BatchMode=yes',f'nsu-a100:{RUN}/{mode}/{name}',str(dest/name)],
                            capture_output=True,text=True,timeout=45)
                        if fetched.returncode: log.write(json.dumps({'fetch_failed':name,'error':fetched.stderr[:300]})+'\n')
                    released=subprocess.run([sys.executable,str(manager),'release','--mode',mode],
                        capture_output=True,text=True,timeout=55,check=True)
                    log.write(released.stdout); log.flush()
                    if all((dest/arm/'validation_predictions.csv').exists() for arm in ['pilkwang','median']):
                        from compare import compare
                        result=compare(dest/'pilkwang/validation_predictions.csv',dest/'median/validation_predictions.csv',
                            ROOT/'tmp/labels/pilkwang/report_labels_v2.csv',dest/'comparison.json')
                        print(json.dumps(result),flush=True)
                    else: print('Run ended without two complete prediction artifacts; inspect logs.',flush=True)
                    return
            except Exception as error:
                log.write(json.dumps({'checked_unix':time.time(),'error':str(error)[:500]})+'\n'); log.flush()
            time.sleep(55)
    raise RuntimeError('Monitor deadline reached; inspect the existing process and lease manually')

if __name__=='__main__': main()
