"""Read-only project/resource inventory, with ownership tokens excluded."""
import json
from pathlib import Path
import shutil
import socket
import subprocess

P=Path('/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection')
q=json.loads((P.parents[1]/'_control/resource_queue_state.json').read_text())
def scrub(o):
    if isinstance(o,dict):return {k:scrub(v) for k,v in o.items() if 'token' not in k.lower()}
    if isinstance(o,list):return [scrub(x) for x in o]
    return o
def active(o):
    if isinstance(o,dict):
        if o.get('state') in {'RUNNING','RESERVED','WAITING_RESOURCE','WAITING'}:return [scrub(o)]
        return [r for v in o.values() for r in active(v)]
    if isinstance(o,list):return [r for v in o for r in active(v)]
    return []
prior=P/'runs/20260910T0913Z-codex-consensus'
files=[(str(f.relative_to(prior)),f.stat().st_size) for f in prior.rglob('*') if f.is_file()]
print(json.dumps(dict(host=socket.gethostname(),active_queue=active(q),
    prior_run_bytes=sum(n for _,n in files),prior_large_files=sorted(files,key=lambda x:-x[1])[:14],
    disk_free_bytes=shutil.disk_usage(P).free,
    nvidia_smi=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader'],capture_output=True,text=True).stdout),indent=2))
