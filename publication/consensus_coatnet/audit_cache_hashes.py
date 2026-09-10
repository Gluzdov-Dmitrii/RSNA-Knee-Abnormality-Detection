"""Export only pixel fingerprints for gold58 to check live/cache parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['code','cache','uids','out']:p.add_argument('--'+name,required=True)
    a=p.parse_args(); sys.path.insert(0,a.code)
    from common import Cache,UID
    cache=Cache(a.cache); wanted=pd.read_csv(a.uids)[UID]; indices=cache.index.set_index(UID).index
    hashes={}
    for uid in wanted:
        x,mask=cache.get(indices.get_loc(uid))
        hashes[uid]=hashlib.sha256(x.tobytes()+mask.tobytes()).hexdigest()
    Path(a.out).write_text(json.dumps(hashes,indent=2));print(f'Exported{len(hashes)} fingerprints; no images or diagnoses.')
