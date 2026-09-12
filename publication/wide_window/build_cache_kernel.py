"""Build private CPU cache notebook using the already verified geometry."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'artifacts/wide_window/cache_kernel'

def build():
    DEST.mkdir(parents=True,exist_ok=True)
    original=(ROOT/'experiments/cache_budget/recheck/pipeline.py').read_text(encoding='utf-8')
    tree=ast.parse(original)
    materialize=next(ast.get_source_segment(original,n) for n in tree.body
                     if isinstance(n,ast.FunctionDef) and n.name=='materialize')
    geometry=(ROOT/'publication/consensus_coatnet/geometry.py').read_text(encoding='utf-8')
    prelude='''from pathlib import Path
import os, sys, subprocess, time, json, hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
PACKAGE=next(p for p in [Path('/kaggle/input/rsna-knee-compact-coatnet'),
    Path('/kaggle/input/datasets/dmitriigluzdov/rsna-knee-compact-coatnet'),
    Path('/kaggle/input/dmitriigluzdov/rsna-knee-compact-coatnet')] if (p/'MODEL_CONFIG.json').is_file())
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
    '--find-links',str(PACKAGE),'pydicom==3.0.2','pylibjpeg==2.1.0',
    'pylibjpeg-libjpeg==2.3.0','pylibjpeg-openjpeg==2.5.0','pyjpegls==1.5.1'],check=True)
THREADS=min(4,os.cpu_count() or 2)
START=time.time()
def log(message): print(f'[{time.time()-START:.0f}s] {message}',flush=True)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()
def save_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
'''
    end='''
root=next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')] if (p/'train_series.csv').is_file())
cfg=dict(img=224,n_slices=9,crop_mm=130,window=[.10,.90])
spec=materialize(root,Path('/kaggle/working/cache'),config=cfg)
save_json('/kaggle/working/BUILD_RECEIPT.json',dict(status='COMPLETE',
    spec_sha256=sha256('/kaggle/working/cache/SPEC.json'),elapsed_seconds=time.time()-START,
    geometry=cfg,n_studies=spec['n_studies'],pixel_gib=spec['pixel_gib']))
print('PRIVATE_WIDE_CACHE_COMPLETE',flush=True)
'''
    code=prelude+'\n'+geometry+'\n'+materialize+'\n'+end
    for name in ['NOTICE.md','LICENSE-APACHE-2.0.txt']:
        content=(ROOT/'publication/consensus_coatnet'/name).read_text(encoding='utf-8')
        code+=f"Path('/kaggle/working/cache/{name}').write_text({content!r},encoding='utf-8')\n"
    compile(code,'build_cache.py','exec')
    (DEST/'build_cache.py').write_text(code,encoding='utf-8')
    meta=dict(id='dmitriigluzdov/knee-mri-private-wide-cache',title='Knee MRI Private Wide Cache',
        code_file='build_cache.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=False,enable_tpu=False,enable_internet=False,
        competition_sources=['rsna-knee-abnormality-detection'],
        dataset_sources=['dmitriigluzdov/rsna-knee-compact-coatnet'],kernel_sources=[],model_sources=[])
    (DEST/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(dict(kernel=meta['id'],source_sha256=hashlib.sha256(code.encode()).hexdigest())))

if __name__=='__main__':build()
