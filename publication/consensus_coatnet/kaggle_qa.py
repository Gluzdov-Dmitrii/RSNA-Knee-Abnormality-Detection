"""Pre-submit live DICOM, precision, gold and runtime checks. No tuning."""
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,JPEG2000Lossless,JPEGLSLossless
import torch
from common import UID,TARGETS,GEOMETRY,ARCH,windows,auc_metrics,save_json
from infer import study_pixels
from model import KneeModel

def codec_checks():
    checks={}
    pixels=(np.arange(64*64,dtype=np.uint16).reshape(64,64)*7)%4096
    for syntax in [JPEG2000Lossless,JPEGLSLossless]:
        ds=Dataset(); ds.file_meta=FileMetaDataset(); ds.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
        ds.Rows,ds.Columns=pixels.shape; ds.SamplesPerPixel=1; ds.PhotometricInterpretation='MONOCHROME2'
        ds.BitsAllocated=16; ds.BitsStored=12; ds.HighBit=11; ds.PixelRepresentation=0; ds.PixelData=pixels.tobytes()
        ds.compress(syntax)
        if not np.array_equal(ds.pixel_array,pixels): raise ValueError('Lossless codec roundtrip differs')
        checks[str(syntax)]='lossless roundtrip PASS'
    from pydicom.pixels import get_decoder
    decoder=get_decoder('1.2.840.10008.1.2.4.70')
    if not decoder.is_available: raise RuntimeError('JPEG Lossless decoder unavailable')
    checks['1.2.840.10008.1.2.4.70']=f'available plugins: {decoder.available_plugins}'
    return checks

@torch.inference_mode()
def quality_checks(root,package,out,n_benchmark=32):
    root,package,out=map(Path,[root,package,out]);torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for T4 verification')
    ck=torch.load(package/'weights.pt',map_location='cpu',weights_only=True)
    model=KneeModel();model.load_state_dict(ck['model'],strict=True);del ck;model.cuda().eval()
    gold=pd.read_csv(root/'train.csv',usecols=[UID]+TARGETS,dtype={UID:str})
    gold=gold[gold[TARGETS].notna().any(axis=1)]
    if len(gold)!=58: raise ValueError('Gold cohort changed')
    series=pd.read_csv(root/'train_series.csv',dtype={UID:str,'SeriesInstanceUID':str})
    groups={uid:frame for uid,frame in series.groupby(UID,sort=False)}
    counts=series.groupby(UID).size().sort_values(kind='stable')
    remaining=counts[~counts.index.isin(gold[UID])]
    benchmark=list(remaining.iloc[np.linspace(0,len(remaining)-1,n_benchmark).astype(int)].index)
    hashes=json.loads((package/'training_pixel_hashes.json').read_text())
    if set(hashes)!=set(gold[UID]):raise ValueError('Expected exact58 gold pixel fingerprints')
    parity_count=0
    audit=Counter();predictions={};times=[];precision_errors=[];start=time.monotonic()
    codec_report=codec_checks();torch.cuda.reset_peak_memory_stats()
    ids=list(gold[UID])+benchmark
    for i,uid in enumerate(ids):
        before=time.monotonic();pixels,mask=study_pixels(root,groups[uid],audit,'train_series')
        if uid in hashes:
            actual=hashlib.sha256(pixels.tobytes()+mask.tobytes()).hexdigest()
            if actual!=hashes[uid]:raise ValueError(f'Train/cache/live pixel parity failed for {uid}')
            parity_count+=1
        x,m=windows(pixels,mask);x=torch.from_numpy(x)[None].cuda();m=torch.from_numpy(m)[None].cuda()
        with torch.autocast('cuda',dtype=torch.float16):p=model(x,m).float().sigmoid()
        if i<3:
            reference=model(x,m).float().sigmoid()
            error=float((p-reference).abs().max());precision_errors.append(error)
            if error>.01:raise ValueError(f'FP16/FP32 probability error too large: {error}')
        predictions[uid]=p.cpu().numpy()[0];times.append(time.monotonic()-before)
        if (i+1)%10==0:print(f'QA: {i+1}/{len(ids)} studies',flush=True)
    gold_pred=np.stack([predictions[uid] for uid in gold[UID]])
    report=dict(evidence='Held-out gold58 audit of a frozen candidate; not a tuned selection score',
        gold=auc_metrics(gold[TARGETS].to_numpy(dtype=float),gold_pred),n_benchmark=n_benchmark,
        geometry=GEOMETRY,cache_pixel_parity=f'{parity_count}/58 exact SHA256',codec_checks=codec_report,
        fp16_fp32_max_error=max(precision_errors),gpu=torch.cuda.get_device_name(),
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.monotonic()-start,
        median_study_seconds=float(np.median(times)),p95_study_seconds=float(np.percentile(times,95)),
        python=platform.python_version(),versions={name:metadata.version(name) for name in
        ['torch','torchvision','timm','numpy','pandas','scipy','pydicom','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg','pyjpegls']})
    frame=pd.DataFrame(gold_pred,columns=TARGETS);frame.insert(0,UID,gold[UID].to_numpy())
    frame.to_csv(out/'gold_predictions.csv',index=False);save_json(out/'qa.json',report)
    print(json.dumps(report,indent=2),flush=True)
