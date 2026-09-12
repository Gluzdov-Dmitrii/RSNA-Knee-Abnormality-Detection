"""Predeclared P04 coverage screening against frozen P01 predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'publication/consensus_coatnet'))
from common import UID,TARGETS,read_scores,auc_metrics,sha256,PILKWANG_SHA,save_json

def compare(control,candidate,labels,out):
    control,candidate=map(Path,[control,candidate])
    ca=json.loads((control/'config.json').read_text());cb=json.loads((candidate/'config.json').read_text())
    for key in ['fold','epochs','seed','batch_size','accum','workers','initialization_sha256',
                'label_hashes','folds_sha256','gold_uids_sha256','n_train','n_validation','arch','targets','arm']:
        if ca[key]!=cb[key]:raise ValueError(f'Comparison contract changed: {key}')
    if ca.get('full_data',False) or cb.get('full_data',False):raise ValueError('Screening must be held-out fold training')
    if ca['source_hashes']['model.py']!=cb['source_hashes']['model.py']:raise ValueError('Model implementation differs')
    if ca['geometry']!=dict(img=224,n_slices=9,crop_mm=130,window=[.35,.65]):raise ValueError('Wrong control window')
    if cb['geometry']!=dict(img=224,n_slices=9,crop_mm=130,window=[.1,.9]):raise ValueError('Wrong candidate window')
    for file in ['train_uids.csv','validation_uids.csv']:
        a=pd.read_csv(control/file,dtype={UID:str});b=pd.read_csv(candidate/file,dtype={UID:str})
        if not a.equals(b):raise ValueError(f'UID order differs: {file}')
    if sha256(labels)!=PILKWANG_SHA:raise ValueError('Reference labels differ')
    a=pd.read_csv(control/'validation_predictions.csv',dtype={UID:str})
    b=pd.read_csv(candidate/'validation_predictions.csv',dtype={UID:str})
    if len(a)!=875 or a[UID].duplicated().any() or b[UID].duplicated().any() or a[UID].tolist()!=b[UID].tolist():
        raise ValueError('Expected same ordered 875 validation studies')
    y=read_scores(labels,a[UID]);pa=a[TARGETS].to_numpy();pb=b[TARGETS].to_numpy()
    ma,mb=auc_metrics(y,pa),auc_metrics(y,pb)
    delta=mb['macro_auc']-ma['macro_auc']
    class_delta={t:mb['per_class'][t]-ma['per_class'][t] for t in TARGETS}
    rng=np.random.default_rng(2026);boot=[]
    for _ in range(1000):
        idx=rng.integers(0,len(y),len(y));diff=[]
        for j in range(len(TARGETS)):
            valid=np.isfinite(y[idx,j]);truth=y[idx,j][valid]>=.5
            if np.unique(truth).size==2:
                diff.append(roc_auc_score(truth,pb[idx,j][valid])-roc_auc_score(truth,pa[idx,j][valid]))
        boot.append(float(np.mean(diff)))
    interval=np.quantile(boot,[.025,.975]).tolist()
    declines=sum(d<-.02 for d in class_delta.values())
    passed=delta>=.005 and interval[0]>0 and declines<=1
    result=dict(status='PROMOTE_TO_FULL_REFIT' if passed else 'REJECT',control=ma,candidate=mb,
        delta=delta,per_class_delta=class_delta,bootstrap_95=interval,bootstrap_replicates=1000,
        targets_down_more_than_002=declines,gate='delta >= .005; bootstrap lower >0; at most1 target down >.02',
        evidence='Conditional single-fold screening; not full OOF or LB',
        control_predictions_sha256=sha256(control/'validation_predictions.csv'),
        candidate_predictions_sha256=sha256(candidate/'validation_predictions.csv'))
    Path(out).parent.mkdir(parents=True,exist_ok=True);save_json(out,result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['control','candidate','labels','out']:p.add_argument('--'+key,required=True)
    a=p.parse_args();compare(a.control,a.candidate,a.labels,a.out)
