"""Focused geometry, paired-statistics, and actual shard round-trip checks."""
import importlib.util
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tmp/cache_budget_deps'))
sys.path.insert(0,str(Path(__file__).parent))
import pipeline as p
import numpy as np
import pandas as pd
from collections import Counter
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,generate_uid

def make_dicom(path,z):
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=generate_uid();meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
    ds.ImagePositionPatient=[0,0,float(z)];ds.ImageOrientationPatient=[1,0,0,0,1,0]
    ds.SliceLocation=z;ds.InstanceNumber=z+1;ds.PixelSpacing=[2,2]
    ds.Rows=96;ds.Columns=96;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
    ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0
    ds.PixelData=(np.arange(96*96).reshape(96,96)+z*50).astype(np.uint16).tobytes()
    ds.save_as(path,enforce_file_format=True)

def main():
    assert p.sample_indices(21,3,(.35,.65))==[9,10,11]
    assert p.sample_indices(21,9,(.35,.65))==[6,7,8,9,10,11,12,13,14]
    assert p.sample_indices(1,9,(.35,.65))==[0]*9
    v=np.tile(np.arange(224,dtype=np.uint8),(9,224,1))
    a=p.encode_slot(v);b=p.encode_slot(v.transpose(0,2,1))
    assert a.shape==(1728,) and np.isfinite(a).all() and not np.allclose(a,b)
    rng=np.random.default_rng(71);y=rng.integers(0,2,(80,12));pred=rng.normal(size=(80,12))
    statistics=p.paired_statistics(y,{p.DEFAULT:pred,'same':pred.copy()})
    assert statistics['same']['delta_ci95_lo']==statistics['same']['delta_ci95_hi']==0
    assert statistics['same']['holm_p']==1
    # Check optimized bootstrap's quantiles against sklearn using the same draw weights.
    weights=np.random.default_rng(p.SEED+1).multinomial(len(y),np.ones(len(y))/len(y),size=800)
    ref=[]
    for weight in weights:
        ids=np.repeat(np.arange(len(y)),weight)
        ref.append(p.macro_auc(y[ids],pred[ids]))
    assert np.allclose([statistics[p.DEFAULT]['ci95_lo'],statistics[p.DEFAULT]['ci95_hi']],np.percentile(ref,[2.5,97.5]))
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as td:
        root=Path(td);uids=[f'1.2.3.{i:04}' for i in range(4407)]
        pd.DataFrame({'StudyInstanceUID':uids,'Report':['DO NOT READ OR SHIP']*4407}).to_csv(root/'train.csv',index=False)
        rows=[]
        for uid in uids[:2]:
            d=root/'train_series'/uid/'1.2.4';d.mkdir(parents=True)
            for z in range(21):make_dicom(d/f'{20-z:03}.dcm',z)
            rows.append(dict(StudyInstanceUID=uid,SeriesInstanceUID='1.2.4',Anatomical_Plane='Sagittal',Fluid_Sensitive=1))
        pd.DataFrame(rows).to_csv(root/'train_series.csv',index=False)
        audit=Counter();paths=list((root/'train_series'/uids[0]/'1.2.4').glob('*.dcm'))
        ordered,px=p.order_files(paths,audit)
        assert ordered[0].stem=='020' and ordered[-1].stem=='000' and px==2
        spec=p.materialize(root,root/'cache',study_limit=2)
        assert spec['n_studies']==2 and spec['pixel_bytes']==2*6*9*224*224
        cache=np.load(root/'cache/pixels-000.npy',mmap_mode='r',allow_pickle=False)
        assert cache.shape==(2,6,9,224,224) and cache.dtype==np.uint8
        assert cache[:,0].max()>0 and cache[:,1:].max()==0
        mask=np.load(root/'cache/slot_mask.npy');assert np.all(mask[:,0]==1) and mask.sum()==2
        for filename,digest in spec['sha256'].items():assert p.sha256(root/'cache'/filename)==digest
        assert not any('Report' in file.name for file in (root/'cache').iterdir())
        del cache
    print('PASS: ordered geometry, slice coverage, spatial features, paired bootstrap, cache round-trip and hashes')

if __name__=='__main__':main()
