"""Paired study bootstrap for the completed, fixed fold0 label comparison."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import UID,TARGETS,read_scores,auc_metrics,save_json

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];folder=root/'artifacts/consensus_coatnet/verified_pair'
    a=pd.read_csv(folder/'pilkwang/validation_predictions.csv').set_index(UID)
    b=pd.read_csv(folder/'median/validation_predictions.csv').set_index(UID).loc[a.index]
    y=read_scores(root/'tmp/labels/pilkwang/report_labels_v2.csv',a.index)
    pa,pb=a[TARGETS].to_numpy(),b[TARGETS].to_numpy();rng=np.random.default_rng(2026);delta=[]
    for _ in range(300):
        ix=rng.integers(0,len(y),len(y))
        ma,mb=auc_metrics(y[ix],pa[ix]),auc_metrics(y[ix],pb[ix])
        if ma['n_scored_classes']==12 and mb['n_scored_classes']==12:delta.append(mb['macro_auc']-ma['macro_auc'])
    report=dict(n_studies=len(y),n_bootstrap=len(delta),seed=2026,
        median_minus_pilkwang=auc_metrics(y,pb)['macro_auc']-auc_metrics(y,pa)['macro_auc'],
        paired_study_bootstrap_95_interval=np.percentile(delta,[2.5,97.5]).tolist(),
        limitation='Conditional on this one fold and these fitted models; not training-seed uncertainty, full OOF or LB.')
    save_json(folder/'bootstrap.json',report);print(report)
