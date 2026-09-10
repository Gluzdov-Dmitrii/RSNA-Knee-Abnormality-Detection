"""CPU-only input audit and shared ImageNet initialization; never trains."""
import argparse
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import *
from model import KneeModel

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True); p.add_argument('--folds',required=True)
    p.add_argument('--labels',required=True); p.add_argument('--gold-uids',required=True)
    p.add_argument('--out',required=True); p.add_argument('--verify-pixels',action='store_true')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    cache=Cache(a.cache); table=load_table(cache,a.folds)
    y,ym,hashes=load_labels(a.labels,table[UID])
    gold=pd.read_csv(a.gold_uids,dtype={UID:str})
    if len(gold)!=58 or gold[UID].duplicated().any() or not set(gold[UID])<=set(table[UID]):
        raise ValueError('Expected exactly58 official gold UIDs contained in folds')
    for name in ['studies.csv','slot_mask.npy']:
        if sha256(cache.root/name)!=cache.spec['sha256'][name]: raise ValueError(f'Hash differs: {name}')
    if a.verify_pixels:
        for name,digest in cache.spec['sha256'].items():
            if name.startswith('pixels-'):
                if sha256(cache.root/name)!=digest: raise ValueError(f'Hash differs: {name}')
    eligible=np.isfinite(y).any(axis=1)&~table[UID].isin(gold[UID]).to_numpy()
    manifest=dict(geometry=GEOMETRY,arch=ARCH,seed=2026,targets=TARGETS,
        label_hashes=hashes,folds_sha256=sha256(a.folds),gold_uids_sha256=sha256(a.gold_uids),
        cache_spec_sha256=sha256(cache.root/'SPEC.json'),n_studies=len(table),
        n_eligible=int(eligible.sum()),n_gold_excluded=58,
        fold_eligible={str(f):int((eligible&(table.fold.to_numpy()==f)).sum()) for f in range(5)},
        median_mean_absolute_change=float(np.nanmean(np.abs(y[eligible]-ym[eligible]))),
        verified_pixel_hashes=a.verify_pixels,
        versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','timm','numpy','pandas','scikit-learn']})
    save_json(out/'input_manifest.json',manifest)
    # Identical initialization (including head) is reused by both arms.
    init=out/'initialization.pt'
    if init.exists(): raise FileExistsError(f'Refusing to replace shared initialization: {init}')
    torch.set_num_threads(2); torch.manual_seed(2026)
    model=KneeModel(pretrained=True).cpu().eval()
    with torch.inference_mode():
        pixels,mask=windows(*cache.get(0))
        # One valid window is enough for a CPU shape/relative-position check.
        index=int(np.flatnonzero(mask)[0])
        output=model(torch.from_numpy(pixels[index:index+1])[None],torch.ones(1,1,dtype=torch.bool))
        if output.shape!=(1,12) or not torch.isfinite(output).all(): raise ValueError('CPU model smoke failed')
    manifest['normalization']={'mean':model.mean.flatten().tolist(),'std':model.std.flatten().tolist()}
    torch.save({'model':model.state_dict(),'arch':ARCH,'res':224,'targets':TARGETS,
                'geometry':GEOMETRY,'source':'timm generic ImageNet pretrained; never knee-trained'},init)
    manifest.update(initialization_sha256=sha256(init),n_parameters=sum(x.numel() for x in model.parameters()),cpu_forward='PASS')
    save_json(out/'input_manifest.json',manifest); print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()
