"""Audit every downloaded pixel shard before uploading the private Dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def validate(directory):
    directory=Path(directory)
    spec=json.loads((directory/'SPEC.json').read_text())
    assert (spec['img'],spec['n_slices'],spec['crop_mm'],spec['dtype'],spec['n_studies'])==(224,9,130,'uint8',4407)
    assert spec['window']==[.35,.65]
    ids=pd.read_csv(directory/'studies.csv')
    assert list(ids.columns)==['StudyInstanceUID','shard','row']
    official=pd.read_csv(ROOT/'data/official/train.csv',usecols=['StudyInstanceUID'])
    assert len(ids)==4407 and ids.StudyInstanceUID.is_unique
    assert set(ids.StudyInstanceUID)==set(official.StudyInstanceUID)
    mask=np.load(directory/'slot_mask.npy',allow_pickle=False)
    assert mask.shape==(4407,6) and np.isin(mask,[0,1]).all() and (mask.sum(axis=1)>0).all()
    expected={'SPEC.json','studies.csv','slot_mask.npy','AUDIT.json','NOTICE.md','LICENSE-APACHE-2.0.txt'}
    expected.update(s['file'] for s in spec['shards'])
    actual={p.name for p in directory.iterdir()}
    assert actual-{'dataset-metadata.json'}==expected, f'Unexpected files: {actual-expected}'
    pixel_bytes=0;allzero_present=0
    for shard in spec['shards']:
        path=directory/shard['file']
        assert path.stat().st_size==shard['bytes'] and digest(path)==shard['sha256']
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        assert a.dtype==np.uint8 and a.shape==(shard['n_studies'],6,9,224,224)
        pixel_bytes+=a.nbytes
        group=ids[ids.shard==path.name]
        assert sorted(group.row)==list(range(len(a)))
        for ix,row in group.iterrows():
            study=a[int(row['row'])]
            maxima=study.max(axis=(1,2,3))
            assert np.all(maxima[mask[ix]==0]==0)
            allzero_present+=int(np.sum(maxima[mask[ix]==1]==0))
        del a
        print('Validated',path.name,flush=True)
    assert pixel_bytes==4407*6*9*224**2==spec['pixel_bytes']
    assert allzero_present==0, f'{allzero_present} present slots contain no intensity signal'
    for name,sha in spec['sha256'].items():assert digest(directory/name)==sha
    receipt=dict(status='validated',n_studies=4407,n_shards=len(spec['shards']),
        pixel_bytes=pixel_bytes,pixel_gib=pixel_bytes/1024**3,
        total_bytes=sum(p.stat().st_size for p in directory.iterdir() if p.name!='dataset-metadata.json'),
        available_slots=int(mask.sum()),absent_slots=int(mask.size-mask.sum()),
        spec_sha256=digest(directory/'SPEC.json'),all_shards_sha256_verified=True,
        official_study_set_verified=True,reports_included=False)
    out=ROOT/'artifacts/cache_budget_recheck';out.mkdir(parents=True,exist_ok=True)
    (out/'cache_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    meta=dict(id='dmitriigluzdov/rsna-knee-uint8-224-9-c130',title='RSNA Knee uint8 224 x 9 crop 130',
        subtitle='Private derived MRI cache for registered competition participants',
        licenses=[{'name':'other'}],
        description=(directory/'NOTICE.md').read_text()+'\n\n'
        'All 4,407 training studies; 11.12072 GiB pixel payload, 35 NumPy shards. '
        'Shape per study: (6,9,224,224), dtype uint8; crop 130 mm, window .35-.65. '
        'Read studies.csv to locate a study and np.load(shard,mmap_mode="r",allow_pickle=False)[row]. '
        'slot_mask.npy marks present public slots. SPEC.json contains SHA-256 checksums. '
        'PRIVATE: derived competition MRI governed by competition rules and RSNA MIRA; '
        'Apache-2.0 applies only to geometry code. Not for non-participants; '
        'not a substitute for the official train zip; not lossless DICOM. '
        'No reports, report lexicon, train.csv, or labels.\n')
    (directory/'dataset-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    validate(args.directory)
