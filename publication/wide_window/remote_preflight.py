"""Read-only full input verification, run beside frozen P04 training source."""
import json
from pathlib import Path
import platform
import shutil
import socket
import importlib.metadata as metadata
import numpy as np
import pandas as pd
from common import Cache,UID,GEOMETRY,FOLDS_SHA,PILKWANG_SHA,sha256

P=Path('/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection')
C=P/'data/rsna-knee-uint8-224-9-c130-w10-90'
BASE=P/'runs/20260910T0913Z-codex-consensus'
assert socket.gethostname()=='ngpu01'
cache=Cache(C)
assert GEOMETRY==dict(img=224,n_slices=9,crop_mm=130,window=[.1,.9])
assert len(cache.index)==4407 and cache.spec['n_studies']==4407
for name,digest in cache.spec['sha256'].items():
    if Path(name).name!=name or sha256(C/name)!=digest:raise ValueError(f'Cache hash: {name}')
baseline=C.parent/'rsna-knee-uint8-224-9-c130'
assert cache.index[UID].tolist()==pd.read_csv(baseline/'studies.csv',dtype={UID:str})[UID].tolist()
assert np.array_equal(cache.mask,np.load(baseline/'slot_mask.npy',allow_pickle=False))
assert sha256(P/'data/FOLDS_V1/folds.csv')==FOLDS_SHA
assert sha256(P/'data/labels/pilkwang/report_labels_v2.csv')==PILKWANG_SHA
init=BASE/'prepared_correctnorm/initialization.pt'
assert sha256(init)=='a7d32ac8c3f006d2b60d1837a90a27c1cf6ed7ad80bdef018c8de52decc4a8c3'
here=Path(__file__).resolve().parent
manifest=json.loads((here/'source_manifest.json').read_text())
for name,digest in manifest.items():
    if sha256(here/name)!=digest:raise ValueError(f'Source hash: {name}')
versions={n:metadata.version(n) for n in ['torch','torchvision','timm','numpy','pandas','scikit-learn']}
if versions['timm']!='1.0.19' or not versions['torch'].startswith('2.6.0'):raise ValueError('Reference runtime differs')
print(json.dumps(dict(status='READY',host=socket.gethostname(),python=platform.python_version(),versions=versions,
    cache_spec_sha256=sha256(C/'SPEC.json'),cache_files_verified=len(cache.spec['sha256']),
    initialization_sha256=sha256(init),source_manifest_sha256=sha256(here/'source_manifest.json'),
    row_order_and_masks='same as P01',n_studies=4407,free_bytes=shutil.disk_usage(P).free,
    planned_additional_project_growth_gib=16,personal_filesystem_quota='not reported; 16GiB task budget, not export capacity'),indent=2))
