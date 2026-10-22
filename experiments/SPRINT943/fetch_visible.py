"""Download selected outputs of the exact launched private notebook version."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

ROOT=Path(__file__).resolve().parent
FILES=['submission.csv','btkd_v559_complete.json','sprint943_run_manifest.json',
       '_coat_raptor_blend_receipt.json','_coat_arm_receipt.json','_coat_g96_receipt.json',
       '_coat_repairv1_receipt.json','d4_input/d4.receipt.json',
       'diagnostics/a5_metadata_applicability.json','diagnostics/a5_applicability.json',
       'raptor_raw.npz','residual_coat_runtime.log','global96_runtime.log','repairv1_runtime.log']


def main(version):
    launched=json.loads((ROOT/'launch_receipt.json').read_text())
    assert launched['status']=='launched' and launched['kernel_version']==version
    out=ROOT/'visible'
    out.mkdir(exist_ok=True)
    def fetch(name):
        api=KaggleApi();api.authenticate()
        q=ApiDownloadKernelOutputRequest()
        q.owner_slug='dmitriigluzdov';q.kernel_slug='rsna-knee-v34-documented-strict'
        q.version_number=version;q.file_path=name
        try:
            with api.build_kaggle_client() as client:
                redirect=client.kernels.kernels_api_client.download_kernel_output(q)
            match=re.search(r'/kf/(\d+)/',redirect.url)
            assert match,'No immutable output version identity'
            with requests.get(redirect.url,stream=True,timeout=45) as response:
                response.raise_for_status()
                body=bytearray()
                for chunk in response.iter_content(65536):
                    body.extend(chunk)
                    if len(body)>4*1024*1024:
                        raise RuntimeError('Unexpectedly large output')
            dest=out/name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(body)
            return {'file':name,'status':'OK','bytes':len(body),
                    'sha256':hashlib.sha256(body).hexdigest(),'kernel_version_id':int(match[1])}
        except Exception as exc:
            return {'file':name,'status':'ERROR','error_type':type(exc).__name__}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(fetch,FILES))
    ids={r['kernel_version_id'] for r in rows if r['status']=='OK'}
    assert len(ids)==1,'Mixed or absent output versions'
    receipt={'kernel_version':version,'kernel_version_id':next(iter(ids)),'files':rows}
    (out/'fetch_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))
    assert all(r['status']=='OK' for r in rows[:3]),'Missing mandatory release output'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',type=int,required=True)
    main(p.parse_args().version)
