"""Read-only watcher for the already submitted P03; never calls submit."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'artifacts/consensus_coatnet'
intent=json.loads((DEST/'submission_intent.json').read_text())
assert intent['attempts_executed']==1
ref=str(intent['submission_ref'])
cli=Path(sys.executable).with_name('kaggle.exe')
start=time.monotonic()
while time.monotonic()-start<10*3600:
    result=subprocess.run([str(cli),'competitions','submissions','-c',
        'rsna-knee-abnormality-detection','--csv'],capture_output=True,text=True,timeout=45)
    if result.returncode==0:
        rows=list(csv.DictReader(io.StringIO(result.stdout)))
        selected=[r for r in rows if r.get('ref')==ref]
        if len(selected)!=1:raise RuntimeError('Expected exact submitted ref')
        row=selected[0]
        assert row['description']==intent['message']
        row['checked_at_utc']=datetime.now(timezone.utc).isoformat()
        (DEST/'submission_status.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
        state=row['status'].split('.')[-1].upper()
        if state not in {'PENDING','RUNNING','SUBMITTED'}:
            break
    else:
        print(json.dumps({'read_only_poll_failed':result.returncode}),flush=True)
    time.sleep(45)
