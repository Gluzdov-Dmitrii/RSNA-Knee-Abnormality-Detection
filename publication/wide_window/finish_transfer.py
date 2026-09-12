"""Remove only byte-identical temporary downloads created by direct_transfer.py."""
import json
from manage import PROJECT,ssh

CODE=r'''
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); spec=json.loads((root/'SPEC.json').read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
removed=[]
for name,expected in spec['sha256'].items():
    if not name.startswith('pixels-'):continue
    assert Path(name).name==name
    target=(root/name).resolve();tmp=(root/(name+'.direct-part')).resolve()
    assert target.parent==root and tmp.parent==root
    if tmp.exists():
        assert digest(target)==expected and digest(tmp)==expected
        n=tmp.stat().st_size;tmp.unlink();removed.append(dict(name=tmp.name,bytes=n))
print(json.dumps(dict(removed_identical_transfer_temps=removed)))
'''
if __name__=='__main__':
    print(json.dumps(ssh('nsu-a100',['python3','-c',CODE,
        PROJECT+'/data/rsna-knee-uint8-224-9-c130-w10-90']),indent=2))
