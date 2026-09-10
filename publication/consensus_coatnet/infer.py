"""Offline inference from current test DICOM, arbitrary study count/order."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from common import TARGETS,UID,ARCH,GEOMETRY,windows,sha256,save_json
from geometry import SLOTS,load_series,render_slot
from model import KneeModel

def study_pixels(root,frame,audit,series_directory):
    records=load_series(root,frame,audit,[GEOMETRY],series_directory=series_directory)
    pixels=np.zeros((6,9,224,224),dtype=np.uint8); mask=np.zeros(6,dtype=bool)
    for slot,(name,_,_) in enumerate(SLOTS):
        if name in records:
            pixels[slot]=render_slot(records[name],GEOMETRY,audit); mask[slot]=True
    if not mask.any(): raise ValueError('Study has no usable DICOM series')
    return pixels,mask

@torch.inference_mode()
def run(root,checkpoint,out,device='cuda',series_csv='test_series.csv',
        series_directory='test_series',sample_csv='sample_submission.csv',test_csv='test.csv'):
    root=Path(root); out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck['geometry']!=GEOMETRY or ck['targets']!=TARGETS or ck['arch']!=ARCH or ck['res']!=224:
        raise ValueError('Checkpoint architecture/geometry/target contract differs')
    model=KneeModel(pretrained=False); model.load_state_dict(ck['model'],strict=True); del ck
    model.to(device).eval()
    series=pd.read_csv(root/series_csv,dtype={UID:str,'SeriesInstanceUID':str})
    sample=pd.read_csv(root/sample_csv,dtype={UID:str})
    if set(sample.columns)!=set([UID]+TARGETS): raise ValueError('Unexpected sample schema')
    test=pd.read_csv(root/test_csv,usecols=[UID],dtype={UID:str})
    if test[UID].isna().any() or test[UID].duplicated().any() or not len(test):
        raise ValueError('Unexpected live test UIDs')
    if not set(test[UID])<=set(series[UID]): raise ValueError('Test metadata missing live test study')
    groups={uid:frame for uid,frame in series.groupby(UID,sort=False)}
    audit=Counter(); times=[]; predictions=[]; start=time.monotonic()
    for i,uid in enumerate(test[UID]):
        t=time.monotonic(); x,mask=windows(*study_pixels(root,groups[uid],audit,series_directory))
        with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,enabled=str(device).startswith('cuda')):
            prediction=model(torch.from_numpy(x)[None].to(device),torch.from_numpy(mask)[None].to(device)).float().sigmoid()
        predictions.append(prediction.cpu().numpy()[0]); times.append(time.monotonic()-t)
        if (i+1)%25==0: print(f'Predicted {i+1}/{len(test)} studies',flush=True)
    pred=np.stack(predictions)
    if pred.shape!=(len(test),12) or not np.isfinite(pred).all() or ((pred<0)|(pred>1)).any():
        raise ValueError('Invalid predictions')
    result=test.copy(); result[TARGETS]=pred; result=result[sample.columns]; result.to_csv(out,index=False)
    save_json(out.with_suffix('.audit.json'),dict(n_studies=len(test),elapsed_seconds=time.monotonic()-start,
        median_study_seconds=float(np.median(times)),p95_study_seconds=float(np.percentile(times,95)),
        weights_sha256=sha256(checkpoint),geometry=GEOMETRY,decode_audit=dict(audit),
        evidence='Live DICOM execution; not a leaderboard result'))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--competition',required=True); p.add_argument('--weights',required=True)
    p.add_argument('--out',default='submission.csv'); p.add_argument('--device',default='cuda')
    a=p.parse_args(); run(a.competition,a.weights,a.out,a.device)
