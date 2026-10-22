"""Bounded foreground launch retry for the explicit Kaggle GPU-slot wait.

Retries only the exact concurrency-cap response. Stops immediately after one
accepted launch or any other error. No callbacks, background services or jobs
are created. Use within the current authorized execution window.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def main(minutes, interval):
    deadline=time.monotonic()+minutes*60
    attempts=[]
    while time.monotonic()<deadline:
        proc=subprocess.run(['kaggle','kernels','push','-p',str(ROOT/'kaggle')],
                            capture_output=True,text=True,timeout=120)
        output=(proc.stdout+'\n'+proc.stderr).strip()
        now=datetime.now(timezone.utc).isoformat()
        (ROOT/'launch_last.log').write_text(output+'\n',encoding='utf-8')
        match=re.search(r'Kernel version (\d+) successfully pushed',output)
        if match:
            state={'status':'launched','kernel_version':int(match.group(1)),
                   'time_utc':now,'attempts':len(attempts)+1,'message':output}
            (ROOT/'launch_receipt.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
            print(json.dumps(state),flush=True)
            return 0
        quota='Maximum batch GPU session count of 2 reached' in output
        attempts.append({'time_utc':now,'quota_busy':quota,'exit_code':proc.returncode})
        state={'status':'waiting_gpu_slot' if quota else 'launch_error',
               'attempts':attempts,'last_message':output}
        (ROOT/'launch_receipt.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        print(json.dumps({'status':state['status'],'attempt':len(attempts),'time_utc':now}),flush=True)
        if not quota:
            print(output,flush=True)
            return 1
        left=deadline-time.monotonic()
        if left<=0:
            break
        time.sleep(min(interval,left))
    state['status']='timed_out_waiting_gpu_slot'
    (ROOT/'launch_receipt.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('Slot wait expired; no job launched.',flush=True)
    return 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes',type=float,default=25)
    p.add_argument('--interval',type=float,default=120)
    a=p.parse_args()
    if a.minutes<=0 or a.interval<60:
        p.error('Positive timeout and interval >=60 seconds required')
    raise SystemExit(main(a.minutes,a.interval))
