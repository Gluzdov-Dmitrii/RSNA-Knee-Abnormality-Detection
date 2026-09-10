"""Compare paired final-epoch predictions on the same locked reference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import TARGETS,UID,read_scores,auc_metrics,save_json

def compare(p01,p02,pilkwang,out):
    ca=json.loads(Path(p01).with_name('config.json').read_text())
    cb=json.loads(Path(p02).with_name('config.json').read_text())
    if ca['arm']!='pilkwang' or cb['arm']!='median': raise ValueError('Comparison arm order differs')
    for key in ['fold','epochs','seed','batch_size','accum','workers','initialization_sha256',
                'label_hashes','source_hashes','folds_sha256','gold_uids_sha256','cache_spec_sha256',
                'n_train','n_validation','arch','geometry','targets']:
        if ca[key]!=cb[key]: raise ValueError(f'A/B contract changed: {key}')
    from common import sha256,PILKWANG_SHA
    if sha256(pilkwang)!=PILKWANG_SHA: raise ValueError('Evaluation reference hash differs')
    a=pd.read_csv(p01,dtype={UID:str}); b=pd.read_csv(p02,dtype={UID:str})
    if a[UID].duplicated().any() or b[UID].duplicated().any() or set(a[UID])!=set(b[UID]):
        raise ValueError('Prediction UID sets differ')
    b=b.set_index(UID).loc[a[UID]].reset_index()
    y=read_scores(pilkwang,a[UID]); pa=a[TARGETS].to_numpy(); pb=b[TARGETS].to_numpy()
    ma,mb=auc_metrics(y,pa),auc_metrics(y,pb)
    result=dict(evidence='One-fold fixed-final-epoch screening, not LB or full OOF',
        pilkwang=ma,median=mb,delta=mb['macro_auc']-ma['macro_auc'],
        per_class_delta={k:mb['per_class'][k]-ma['per_class'][k] for k in TARGETS
                         if ma['per_class'][k] is not None and mb['per_class'][k] is not None})
    Path(out).parent.mkdir(parents=True,exist_ok=True); save_json(out,result); return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['p01','p02','pilkwang','out']: p.add_argument('--'+key,required=True)
    a=p.parse_args(); print(compare(a.p01,a.p02,a.pilkwang,a.out))
