"""Independent CPU FP32 reference, without labels or any model tuning."""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from common import Cache, UID, TARGETS, windows, save_json, sha256
from model import KneeModel

@torch.inference_mode()
def main(a):
    torch.set_num_threads(8)
    cache=Cache(a.cache)
    ids=pd.read_csv(a.uids,dtype={UID:str})[UID].tolist()
    lookup={uid:i for i,uid in enumerate(cache.index[UID])}
    model=KneeModel(pretrained=False)
    ck=torch.load(a.weights,map_location='cpu',weights_only=True)
    model.load_state_dict(ck['model'],strict=True); del ck
    model.eval(); preds=[]; start=time.monotonic()
    for i,uid in enumerate(ids):
        x,m=windows(*cache.get(lookup[uid]))
        p=model(torch.from_numpy(x)[None],torch.from_numpy(m)[None]).float().sigmoid()
        preds.append(p.numpy()[0])
        print(f'CPU FP32 {i+1}/{len(ids)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    frame=pd.DataFrame(np.stack(preds),columns=TARGETS); frame.insert(0,UID,ids)
    frame.to_csv(a.out,index=False)
    save_json(Path(a.out).with_suffix('.json'),dict(weights_sha256=sha256(a.weights),
        elapsed_seconds=time.monotonic()-start,torch_version=torch.__version__,dtype='CPU FP32',n=len(ids)))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['cache','uids','weights','out']:p.add_argument('--'+key,required=True)
    main(p.parse_args())
