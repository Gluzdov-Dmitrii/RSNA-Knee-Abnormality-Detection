"""Read-only wait for the existing P04 CPU kernel; no pushes or submissions."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

DEST=Path(__file__).resolve().parents[2]/'artifacts/wide_window'
CLI=Path(sys.executable).with_name('kaggle.exe')
KERNEL='dmitriigluzdov/knee-mri-private-wide-cache'
start=time.monotonic()
while time.monotonic()-start<9*3600:
    try:
        p=subprocess.run([str(CLI),'kernels','status',KERNEL],capture_output=True,text=True,timeout=40)
        if p.returncode:raise RuntimeError(f'read-only status exit {p.returncode}')
        state=next((s for s in ['COMPLETE','ERROR','CANCELLED','RUNNING','QUEUED']
                    if 'KernelWorkerStatus.'+s in p.stdout),'UNKNOWN')
        receipt=dict(kernel=KERNEL,version=1,state=state,checked_at_utc=datetime.now(timezone.utc).isoformat())
        (DEST/'cache_status.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
        if state in ['COMPLETE','ERROR','CANCELLED']:break
    except (subprocess.TimeoutExpired,RuntimeError) as e:
        print(json.dumps(dict(read_only_poll_error=str(e))),flush=True)
    time.sleep(45)
