"""Fixed-epoch, study-fold training. No submissions; final epoch is primary."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from common import *
from model import KneeModel,masked_bce

class Studies(Dataset):
    def __init__(self,cache,indices,targets): self.cache,self.indices,self.targets=cache,np.array(indices),targets
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        index=self.indices[i]; x,mask=windows(*self.cache.get(index))
        return torch.from_numpy(x),torch.from_numpy(mask),torch.from_numpy(self.targets[index]),int(index)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

@torch.inference_mode()
def predict(model,loader,device):
    model.eval(); predictions=[]; indices=[]
    for x,mask,_,idx in loader:
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            p=model(x.to(device),mask.to(device)).float().sigmoid()
        predictions.append(p.cpu().numpy()); indices.extend(idx.tolist())
    return np.concatenate(predictions),np.array(indices)

def main():
    parser=argparse.ArgumentParser()
    for key in ['cache','folds','labels','gold-uids','initialization','out']: parser.add_argument('--'+key,required=True)
    parser.add_argument('--arm',choices=['pilkwang','median'],required=True)
    parser.add_argument('--fold',type=int,default=0); parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=4); parser.add_argument('--accum',type=int,default=2)
    parser.add_argument('--workers',type=int,default=0); parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--full-data',action='store_true',help='Refit selected recipe on all non-gold studies; no CV metrics')
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.resume: raise FileExistsError(f'Output not empty: {out}')
    out.mkdir(parents=True,exist_ok=True); seed_all(a.seed); torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('Training requires a reserved CUDA GPU')
    device=torch.device('cuda'); cache=Cache(a.cache); table=load_table(cache,a.folds)
    reference,median,hashes=load_labels(a.labels,table[UID]); targets=reference if a.arm=='pilkwang' else median
    gold=pd.read_csv(a.gold_uids,dtype={UID:str})
    if len(gold)!=58 or gold[UID].duplicated().any() or not set(gold[UID])<=set(table[UID]): raise ValueError('Gold UIDs differ')
    eligible=np.isfinite(reference).any(axis=1)&~table[UID].isin(gold[UID]).to_numpy()
    train_idx=np.flatnonzero(eligible if a.full_data else eligible&(table.fold.to_numpy()!=a.fold))
    val_idx=np.array([],dtype=int) if a.full_data else np.flatnonzero(eligible&(table.fold.to_numpy()==a.fold))
    if set(train_idx)&set(val_idx): raise ValueError('Training/validation overlap')
    init=torch.load(a.initialization,map_location='cpu',weights_only=True)
    if init['geometry']!=GEOMETRY or init['targets']!=TARGETS or init['arch']!=ARCH: raise ValueError('Initialization contract differs')
    model=KneeModel(grad_checkpointing=True); model.load_state_dict(init['model'],strict=True); del init
    model.to(device)
    head=[p for n,p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':3e-5},
                                 {'params':head,'lr':1e-3}],weight_decay=.01)
    generator=torch.Generator().manual_seed(a.seed)
    train_loader=DataLoader(Studies(cache,train_idx,targets),batch_size=a.batch_size,shuffle=True,
        num_workers=a.workers,pin_memory=True,generator=generator)
    val_loader=DataLoader(Studies(cache,val_idx,reference),batch_size=a.batch_size,shuffle=False,
        num_workers=a.workers,pin_memory=True)
    steps_per_epoch=math.ceil(len(train_loader)/a.accum)
    total_steps=steps_per_epoch*a.epochs
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:
        min((step+1)/max(1,total_steps*.1),1.)*.5*(1+math.cos(math.pi*step/max(1,total_steps))))
    config=vars(a)|dict(arch=ARCH,geometry=GEOMETRY,targets=TARGETS,label_hashes=hashes,
        initialization_sha256=sha256(a.initialization),folds_sha256=sha256(a.folds),
        gold_uids_sha256=sha256(a.gold_uids),cache_spec_sha256=sha256(cache.root/'SPEC.json'),
        n_train=len(train_idx),n_validation=len(val_idx),primary_checkpoint='fixed final epoch',
        gpu_name=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    if a.resume:
        old=json.loads((out/'config.json').read_text())
        for key in ['arm','fold','epochs','seed','full_data','batch_size','accum','workers','initialization_sha256',
                    'label_hashes','source_hashes','folds_sha256','gold_uids_sha256','cache_spec_sha256']:
            if old[key]!=config[key]: raise ValueError(f'Resume config changed: {key}')
    else: save_json(out/'config.json',config)
    table.iloc[train_idx][[UID,'fold']].to_csv(out/'train_uids.csv',index=False)
    table.iloc[val_idx][[UID,'fold']].to_csv(out/'validation_uids.csv',index=False)
    start_epoch=0; history=[]
    if a.resume:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        generator.set_state(ck['generator']); torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start_epoch=ck['epoch']; history=ck['history']; del ck
    start=time.monotonic(); torch.cuda.reset_peak_memory_stats(); total_batches=0
    for epoch in range(start_epoch,a.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); epoch_start=time.monotonic(); loss_sum=0.; seen=0
        for step,(x,mask,y,_) in enumerate(train_loader):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(x.to(device,non_blocking=True),mask.to(device,non_blocking=True))
                loss=masked_bce(logits,y.to(device,non_blocking=True))
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            group_start=(step//a.accum)*a.accum
            group_size=min(a.accum,len(train_loader)-group_start)
            (loss/group_size).backward(); loss_sum+=float(loss.detach())*len(x); seen+=len(x); total_batches+=1
            if (step+1)%a.accum==0 or step+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step%50==0:
                print(json.dumps(dict(arm=a.arm,epoch=epoch+1,batch=step+1,batches=len(train_loader),
                    loss=float(loss.detach()),elapsed_s=round(time.monotonic()-start,1))),flush=True)
            if a.smoke_steps and total_batches>=a.smoke_steps:
                torch.cuda.synchronize()
                model.eval()
                with torch.inference_mode():
                    pred=model(x.to(device),mask.to(device)).float().cpu()
                weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
                torch.save({'model':weights,'arch':ARCH,'res':224,'targets':TARGETS,'geometry':GEOMETRY},out/'smoke.pt')
                model.load_state_dict(torch.load(out/'smoke.pt',weights_only=True,map_location='cpu')['model'])
                with torch.inference_mode(): again=model(x.to(device),mask.to(device)).float().cpu()
                if not torch.allclose(pred,again,atol=1e-5,rtol=1e-5): raise ValueError('Checkpoint reload mismatch')
                report=dict(status='SMOKE_PASS_NOT_QUALITY_EVIDENCE',batches=total_batches,
                    elapsed_s=time.monotonic()-start,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,
                    checkpoint_reload_max_error=float((pred-again).abs().max()))
                save_json(out/'result.json',report); print(json.dumps(report),flush=True); return
        if a.full_data:
            metrics={'validation':'disabled_for_full_data'}
        else:
            predictions,indices=predict(model,val_loader,device)
            metrics=auc_metrics(reference[indices],predictions)
        record=dict(epoch=epoch+1,train_loss=loss_sum/seen,epoch_seconds=time.monotonic()-epoch_start,**metrics)
        history.append(record); print(json.dumps(record),flush=True)
        save_json(out/'history.json',history)
        ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),epoch=epoch+1,history=history,generator=generator.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(ck,out/'latest.tmp'); (out/'latest.tmp').replace(out/'latest.pt'); del ck
    if start_epoch>=a.epochs and not a.full_data:
        predictions,indices=predict(model,val_loader,device)
        metrics=auc_metrics(reference[indices],predictions)
    if a.full_data:
        metrics={'validation':'disabled_for_full_data'}
        gold_idx=np.flatnonzero(table[UID].isin(gold[UID]).to_numpy())
        gold_loader=DataLoader(Studies(cache,gold_idx,reference),batch_size=a.batch_size,shuffle=False,num_workers=0)
        predictions,indices=predict(model,gold_loader,device)
    result=dict(status='COMPLETE_FULL_DATA' if a.full_data else 'COMPLETE_SCREENING' if a.fold==0 else 'COMPLETE_FOLD',arm=a.arm,fold=a.fold,
        elapsed_s=time.monotonic()-start,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,**metrics)
    pred_frame=pd.DataFrame(predictions,columns=TARGETS); pred_frame.insert(0,UID,table.iloc[indices][UID].to_numpy())
    pred_frame.to_csv(out/('gold_predictions.csv' if a.full_data else 'validation_predictions.csv'),index=False)
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},arch=ARCH,res=224,
        targets=TARGETS,geometry=GEOMETRY,epoch=a.epochs,arm=a.arm,seed=a.seed,fold=None if a.full_data else a.fold,
        full_data=a.full_data,n_train=len(train_idx),
        initialization_sha256=config['initialization_sha256']),out/'weights.pt')
    result['weights_sha256']=sha256(out/'weights.pt'); save_json(out/'result.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
