"""Read-only retirement inventory. Deletion was rejected and is not performed."""
import json
from pathlib import Path
import socket

P=Path('/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection')
BASE=(P/'runs/20260910T0913Z-codex-consensus').resolve()
assert socket.gethostname()=='ngpu01'
def identity(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return None
    f=p.read_text().rsplit(')',1)[1].split()
    return None if f[0]=='Z' else f[19]
for meta in BASE.rglob('process.json'):
    r=json.loads(meta.read_text())
    if identity(r['pid'])==r['start']:raise RuntimeError(f'Previous supervisor still alive: {meta}')
q=json.loads((P.parents[1]/'_control/resource_queue_state.json').read_text())
def check_queue(x):
    if isinstance(x,dict):
        if x.get('state') in {'RUNNING','RESERVED','WAITING_RESOURCE','WAITING'} and str(BASE) in x.get('run_path',''):
            raise RuntimeError('Old run has active reservation')
        for v in x.values():check_queue(v)
    elif isinstance(x,list):
        for v in x:check_queue(v)
check_queue(q)
targets=['verified_pair/median/latest.pt','verified_pair/pilkwang/latest.pt',
         'pair/pilkwang/latest.pt','full/pilkwang/latest.pt','verified_pair/smoke/smoke.pt']
removed=[]
for rel in targets:
    path=BASE/rel
    if not path.exists():continue
    resolved=path.resolve()
    if path.is_symlink() or BASE not in resolved.parents:raise RuntimeError('Unsafe resolved target')
    if 'pair/pilkwang/latest.pt'!=rel and 'smoke' not in rel:
        if not path.with_name('weights.pt').is_file() or not path.with_name('result.json').is_file():
            raise RuntimeError('Final checkpoint/evidence missing')
    removed.append(dict(path=str(resolved),bytes=path.stat().st_size))
receipt=dict(reason='Potential retirement only; automatic review rejected deletion. No files changed.',
    candidates=removed,potential_bytes=sum(r['bytes'] for r in removed),reclaimed_bytes=0,deletion_performed=False,
    retained='All final model weights, configs, predictions, logs; exact generic initialization pinned by P04')
print(json.dumps(receipt,indent=2))
