"""Shared, report-free data contract for the publication lane."""
from __future__ import annotations
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA',
           'Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
UID = 'StudyInstanceUID'
FOLDS_SHA = '3086df3341333f44adb883292da386857c3230eaa2d501514ddf827a2da11b1a'
PILKWANG_SHA = '6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920'
ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
GEOMETRY = dict(img=224,n_slices=9,crop_mm=130,window=[.35,.65])

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def read_scores(path,uids):
    # Selecting columns also prevents reports/verdict text entering output artifacts.
    frame=pd.read_csv(path,usecols=[UID]+TARGETS,dtype={UID:str})
    if frame[UID].duplicated().any(): raise ValueError(f'Duplicate UIDs: {path}')
    if not set(frame[UID]).intersection(uids): raise ValueError(f'No matching study UIDs: {path}')
    values=frame.set_index(UID).reindex(uids)[TARGETS].to_numpy(dtype=np.float32)
    if np.isinf(values).any() or ((values[np.isfinite(values)]<0)|(values[np.isfinite(values)]>1)).any():
        raise ValueError(f'Scores outside [0,1]: {path}')
    return values

def median_scores(arrays):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmedian(np.stack(arrays),axis=0).astype(np.float32)

def load_labels(label_root,uids):
    root=Path(label_root)
    paths={'pilkwang':root/'pilkwang/report_labels_v2.csv',
           'steven':root/'steven/llm_labels_v2.csv',
           'lixin':root/'lixin/labels_llm_gpt56sol.csv'}
    hashes={name:sha256(path) for name,path in paths.items()}
    if hashes['pilkwang']!=PILKWANG_SHA: raise ValueError('Locked Pilkwang hash differs')
    arrays={name:read_scores(path,uids) for name,path in paths.items()}
    return arrays['pilkwang'],median_scores(list(arrays.values())),hashes

class Cache:
    def __init__(self,root):
        self.root=Path(root)
        self.spec=json.loads((self.root/'SPEC.json').read_text())
        for key,value in GEOMETRY.items():
            if self.spec.get(key)!=value: raise ValueError(f'Cache geometry differs: {key}')
        self.index=pd.read_csv(self.root/'studies.csv',dtype={UID:str})
        if self.index[UID].duplicated().any(): raise ValueError('Duplicate cache UID')
        self.mask=np.load(self.root/'slot_mask.npy',mmap_mode='r',allow_pickle=False)
        if self.mask.shape!=(len(self.index),6): raise ValueError('Wrong slot mask')
        if not np.isin(self.mask,[0,1]).all(): raise ValueError('Nonbinary slot mask')
        if not self.mask.any(axis=1).all(): raise ValueError('Study with no available slots')
        self.shards={}
    def get(self,index):
        row=self.index.iloc[index]; name=row['shard']
        if name not in self.shards:
            self.shards[name]=np.load(self.root/name,mmap_mode='r',allow_pickle=False)
        pixels=self.shards[name][int(row['row'])]
        if pixels.shape!=(6,9,224,224) or pixels.dtype!=np.uint8: raise ValueError('Wrong pixels')
        return pixels,np.asarray(self.mask[index],dtype=bool)

def windows(pixels,mask):
    """Each of18 RGB triplets stays within its original three-slice group."""
    if pixels.shape!=(6,9,224,224): raise ValueError('Expected six slots of nine slices')
    return np.array(pixels.reshape(18,3,224,224),copy=True),np.repeat(mask,3).copy()

def load_table(cache,folds_path):
    if sha256(folds_path)!=FOLDS_SHA: raise ValueError('FOLDS_V1 hash differs')
    folds=pd.read_csv(folds_path,dtype={UID:str})
    joined=cache.index[[UID]].merge(folds,on=UID,how='left',validate='one_to_one')
    if len(joined)!=4407 or joined.fold.isna().any() or not joined.fold.isin(range(5)).all():
        raise ValueError('Incomplete locked study/fold table')
    return joined

def auc_metrics(y,p):
    from sklearn.metrics import roc_auc_score
    if p.shape!=y.shape or not np.isfinite(p).all(): raise ValueError('Invalid prediction array')
    scores={}
    for j,name in enumerate(TARGETS):
        valid=np.isfinite(y[:,j]); truth=y[valid,j]>=.5
        scores[name]=float(roc_auc_score(truth,p[valid,j])) if len(np.unique(truth))==2 else None
    valid_scores=[s for s in scores.values() if s is not None]
    return {'macro_auc':float(np.mean(valid_scores)) if valid_scores else None,
            'per_class':scores,'n_studies':len(y),'n_scored_classes':len(valid_scores)}
