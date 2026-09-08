import base64
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEST=HERE/'private_kernel'

def main():
    DEST.mkdir(exist_ok=True)
    folds=(ROOT/'ops/assets/FOLDS_V1/folds.csv').read_bytes()
    assert hashlib.sha256(folds).hexdigest()=='3086df3341333f44adb883292da386857c3230eaa2d501514ddf827a2da11b1a'
    payload=base64.b64encode(gzip.compress(folds)).decode()
    code=(HERE/'pipeline.py').read_text(encoding='utf-8')
    code+='\nimport base64, gzip\n'
    code+=f"Path('/kaggle/working/folds.csv').write_bytes(gzip.decompress(base64.b64decode({payload!r})))\n"
    code+="receipt,spec=run_all('/kaggle/working/folds.csv','/kaggle/working/evidence','/kaggle/working/cache')\n"
    for filename in ['NOTICE.md','LICENSE-APACHE-2.0.txt']:
        code+=f"Path('/kaggle/working/cache/{filename}').write_text({(HERE/filename).read_text(encoding='utf-8')!r},encoding='utf-8')\n"
    code+="print('PRIVATE_CACHE_COMPLETE',sha256('/kaggle/working/cache/SPEC.json'),flush=True)\n"
    (DEST/'build_cache.py').write_text(code,encoding='utf-8')
    meta=dict(id='dmitriigluzdov/rsna-knee-private-cache-verifier',title='RSNA Knee Private Cache Verifier',
        code_file='build_cache.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=False,enable_tpu=False,enable_internet=False,
        dataset_sources=['pilkwang/rsna-knee-llm-labels'],competition_sources=['rsna-knee-abnormality-detection'],
        kernel_sources=[],model_sources=[])
    (DEST/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Built private CPU kernel',hashlib.sha256(code.encode()).hexdigest())

if __name__=='__main__':main()
