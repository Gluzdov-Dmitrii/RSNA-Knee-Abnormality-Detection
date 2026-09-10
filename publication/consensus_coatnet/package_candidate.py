"""Build a private weights package and runnable Kaggle notebook; no uploads."""
import argparse
import json
from pathlib import Path
import shutil
from common import ARCH,GEOMETRY,TARGETS,sha256,save_json

SLUG='dmitriigluzdov/rsna-knee-compact-coatnet'
KERNEL='dmitriigluzdov/knee-mri-compact-coatnet'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def cell(kind,text,hidden=False):
    item=dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True))
    if kind=='code':
        item.update(execution_count=None,outputs=[])
        if hidden:item['metadata']={'jupyter':{'source_hidden':True},'tags':['hide-input']}
    return item

def build(weights,package,notebook,skip_qa=False):
    package=Path(package); notebook=Path(notebook)
    package.mkdir(parents=True,exist_ok=True); notebook.mkdir(parents=True,exist_ok=True)
    if not (package/'training_pixel_hashes.json').is_file():
        raise FileNotFoundError('Stage the58 training pixel fingerprints before packaging')
    # Kaggle's default uploader skips directories: keep all offline wheels flat.
    wheel_dir=package/'wheels'
    if wheel_dir.exists():
        for wheel in wheel_dir.glob('*.whl'):wheel.replace(package/wheel.name)
        wheel_dir.rmdir()
    shutil.copyfile(weights,package/'weights.pt')
    for name in ['common.py','model.py','geometry.py','infer.py','kaggle_qa.py','train.py','prepare.py',
                 'export_gold_uids.py','NOTICE.md','LICENSE-APACHE-2.0.txt','PROTOCOL.md','README.md']:
        shutil.copyfile(HERE/name,package/name)
    shutil.copyfile(ROOT/'ops/assets/FOLDS_V1/folds.csv',package/'folds.csv')
    comparison=json.loads((ROOT/'artifacts/consensus_coatnet/verified_pair/comparison.json').read_text())
    save_json(package/'ablation.json',comparison)
    save_json(package/'MODEL_CONFIG.json',dict(arch=ARCH,geometry=GEOMETRY,targets=TARGETS,
        normalization={'mean':[.5]*3,'std':[.5]*3},dtype='float32 weights; FP16 CUDA inference',
        epochs=4,seed=2026,n_train=4349,initialization='generic ImageNet; never a knee-trained public checkpoint',
        selected_arm='Pilkwang supplied scores',weights_sha256=sha256(package/'weights.pt')))
    card='''# Compact Knee CoAtNet

Development weights for an independent RSNA knee MRI reader. No leaderboard
score is claimed until the exact accompanying notebook has been scored.

73.1M-parameter CoAtNet with per-finding attention over18 within-series triplets.
Six slots,224²,nine slices per slot,crop130mm,central window.35–.65. Normalization
matches this exact timm checkpoint: mean/std0.5/0.5. The cache is private and lossy.

Controlled fold0 screening: Pilkwang0.837165 versus supplied-score median0.826699
on the SAME875studies and weak-label reference; fixed final epoch4. Median did
not improve this comparison. These are not LB scores or full five-fold OOF.
Final weights refit the selected recipe on4,349 non-gold studies. Gold58 excluded.

Use the accompanying offline inference notebook, or `python infer.py --competition
COMPETITION_ROOT --weights weights.pt --out submission.csv`. Internet is not
required; small dependency wheels are supplied. The final notebook records actual
runtime, library versions, gold audit and precision/remount checks.

Training scripts and folds are included. Obtain label sources from NOTICE.md
and generate a private cache from official MRI; do not redistribute MRI/reports.
This package contains model parameters, code, fold assignments and aggregate
results, not MRI pixels, reports, diagnosis-label tables or cached submissions.

Credits: dreaddevelopment's Raptor attention recipe, Steven Lee geometry and
timm CoAtNet. This compact adaptation does not reproduce Raptor's wider384 recipe
or inherit its score. See NOTICE and Apache2.0 license. Upstream wheel licenses
remain inside the wheels. Generic pretrained data and competition terms apply.
'''
    (package/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    metadata=dict(id=SLUG,title='RSNA Knee Compact CoAtNet',subtitle='Reproducible MRI model weights and offline inference',
        description=card,licenses=[{'name':'other'}],keywords=['computer vision','deep learning','healthcare','pytorch'])
    save_json(package/'dataset-metadata.json',metadata)
    files={str(p.relative_to(package)).replace('\\','/'):sha256(p) for p in package.rglob('*')
           if p.is_file() and p.name not in ['dataset-metadata.json','PACKAGE_SHA256.json']}
    save_json(package/'PACKAGE_SHA256.json',files)
    cells=[cell('markdown','''# Knee MRI: Compact CoAtNet

One MRI reader, one model package, offline inference. The model predicts twelve
findings from eighteen three-slice windows. Each finding has its own attention
weights — it can focus on different slices.

**Development candidate: no leaderboard claim yet.** The controlled experiment
did not support median labels: AUC0.8267 versus0.8372 for Pilkwang on one locked
validation fold. Final weights use the winning control and all4,349 non-gold
training studies. This is not a reproduction of Raptor's published score.
'''),cell('markdown',f'''Attach the official competition and [model package](https://www.kaggle.com/datasets/{SLUG}).
Choose GPU and keep internet off. Run All creates `submission.csv` from the
current `test.csv` and DICOM mount; it never submits automatically.
'''),cell('code','''from pathlib import Path
import os,sys,subprocess,json
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
COMPETITION=next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')] if (p/'test.csv').is_file())
PACKAGE=next(p for p in [Path('/kaggle/input/rsna-knee-compact-coatnet'),
    Path('/kaggle/input/datasets/dmitriigluzdov/rsna-knee-compact-coatnet'),
    Path('/kaggle/input/dmitriigluzdov/rsna-knee-compact-coatnet')] if (p/'MODEL_CONFIG.json').is_file())
RUN_QA=True
QA_STUDIES=32
OUTPUT=Path('/kaggle/working/submission.csv')
'''),cell('code','''import hashlib
manifest=json.loads((PACKAGE/'PACKAGE_SHA256.json').read_text())
for name,digest in manifest.items():
    h=hashlib.sha256()
    with (PACKAGE/name).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    if h.hexdigest()!=digest:raise ValueError(f'Package hash mismatch: {name}')
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
    '--find-links',str(PACKAGE),'timm==1.0.19','pydicom==3.0.2',
    'pylibjpeg==2.1.0','pylibjpeg-libjpeg==2.3.0','pylibjpeg-openjpeg==2.5.0',
    'pyjpegls==1.5.1'],check=True)
sys.path.insert(0,str(PACKAGE))
from kaggle_qa import quality_checks
from infer import run
if RUN_QA: quality_checks(COMPETITION,PACKAGE,Path('/kaggle/working'),QA_STUDIES)
submission=run(COMPETITION,PACKAGE/'weights.pt',OUTPUT,device='cuda')
import pandas as pd
pd.read_csv(COMPETITION/'test.csv',usecols=['StudyInstanceUID'],dtype=str).to_csv(
    '/kaggle/working/test_manifest.csv',index=False)
print(f'Wrote {len(submission)} live test predictions.')
display(submission.head())
''',True),cell('markdown','''## What was measured

AUC measures how well positive studies are ranked above negative ones. The
screening comparison held folds, images, model, initialization and training budget
fixed; only supplied training scores changed. Its report-derived reference is
noisy and favors its label source. One fold is not full cross-validation.

The runtime audit includes real train DICOM decoding, transfer-syntax checks and
comparison of inference precision. Gold58 are held out of all training and are
audited once after the candidate is frozen, not used for epoch selection.

## Rebuild and limits

The package contains training code, folds, config, hashes, credits and offline
wheels. Build the MRI cache in a private notebook after accepting competition
terms. MRI pixels and reports are not included. The compact cache is lossy;
the11GiB storage probe does not establish optimal geometry for every encoder.
See MODEL_CARD.md and NOTICE.md for sources and limitations.
''')]
    if skip_qa:
        cells[2]['source']=[line.replace('RUN_QA=True','RUN_QA=False') for line in cells[2]['source']]
        cells[4]['source'] += ['\nThe complete 90-study runtime audit passed in version 1, including exact pixel hashes\n',
            'for all 58 held-out studies. This scoring version skips that completed audit and\n',
            'reads only the live test data. Independent CPU FP32 versus T4 FP16 probability\n',
            'difference was at most 0.002205 across all 58 studies.\n']
    save_json(notebook/'compact_coatnet.ipynb',dict(nbformat=4,nbformat_minor=5,
        metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python3'}},cells=cells))
    save_json(notebook/'kernel-metadata.json',dict(id=KERNEL,title='Knee MRI: Compact CoAtNet',
        code_file='compact_coatnet.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',
        competition_sources=['rsna-knee-abnormality-detection'],dataset_sources=[SLUG],kernel_sources=[],model_sources=[]))
    print(json.dumps({'dataset':SLUG,'kernel':KERNEL,'files':len(files),'bytes':sum(p.stat().st_size for p in package.rglob('*') if p.is_file())}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['weights','package','notebook']:p.add_argument('--'+key,required=True)
    p.add_argument('--skip-qa',action='store_true',help='Scoring version after completed separate QA')
    a=p.parse_args();build(a.weights,a.package,a.notebook,a.skip_qa)
