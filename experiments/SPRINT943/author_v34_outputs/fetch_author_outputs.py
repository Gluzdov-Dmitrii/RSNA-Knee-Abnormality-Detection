"""Download small, public V34 prediction/receipt artifacts; never download weights."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest, ApiListKernelFilesRequest

ROOT = Path(__file__).resolve().parent
OWNER='mattiaangeli'; SLUG='bend-the-knee-to-speedy-raptors-the-original'; VERSION=34
REQUIRED=['submission.csv','btkd_v559_complete.json','_coat_raptor_blend_receipt.json',
          '_coat_arm_receipt.json','_coat_g96_receipt.json','_coat_repairv1_receipt.json',
          'd4_input/d4.receipt.json','diagnostics/a5_metadata_applicability.json',
          'diagnostics/a5_applicability.json','raptor_raw.npz']
LIMIT=2*1024*1024

def fetch(item):
    filename,size=item
    if size is not None and size>LIMIT:
        return {'name':filename,'status':'SKIPPED_TOO_LARGE','size':size}
    api=KaggleApi(); api.authenticate()
    req=ApiDownloadKernelOutputRequest(); req.owner_slug=OWNER; req.kernel_slug=SLUG
    req.version_number=VERSION; req.file_path=filename
    try:
        with api.build_kaggle_client() as client:
            redirect=client.kernels.kernels_api_client.download_kernel_output(req)
        match=re.search(r'/kf/(\d+)/',redirect.url)
        if not match or int(match[1])!=351665011:
            raise RuntimeError('Unexpected kernelVersionId in download redirect')
        response=requests.get(redirect.url,stream=True,timeout=45)
        if response.status_code != 200:
            return {'name':filename,'status':'HTTP_ERROR','http_status':response.status_code}
        body=bytearray()
        for chunk in response.iter_content(65536):
            body.extend(chunk)
            if len(body)>LIMIT:
                response.close()
                raise RuntimeError('Artifact exceeds small-output download cap')
        path=ROOT/filename; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(body)
        return {'name':filename,'status':'OK','bytes':len(body),
                'sha256':hashlib.sha256(body).hexdigest(),'kernel_version_id':int(match[1])}
    except Exception as exc:
        # Avoid printing transient signed URLs embedded in HTTP exception strings.
        return {'name':filename,'status':'ERROR','error_type':type(exc).__name__}

def main():
    api=KaggleApi(); api.authenticate(); listing=[]
    req=ApiListKernelFilesRequest(); req.user_name=OWNER; req.kernel_slug=SLUG
    req.version_label=str(VERSION); req.page_size=100
    try:
        with api.build_kaggle_client() as client:
            for _ in range(10):
                page=client.kernels.kernels_api_client.list_kernel_files(req)
                listing.extend({'name':f.name,'bytes':f.size} for f in page.files)
                if not page.next_page_token: break
                req.page_token=page.next_page_token
    except Exception as exc:
        print('Optional listing unavailable: '+type(exc).__name__,flush=True)
    (ROOT/'file_listing.json').write_text(json.dumps(listing,indent=2),encoding='utf-8')
    sizes={r['name']:r['bytes'] for r in listing}
    optional=[r['name'] for r in listing if r['bytes']<=LIMIT and
              (r['name'].startswith('d4_input/study_input_receipts/input_') and r['name'].endswith('.jsonl') or
               r['name'].endswith('.npz') and any(x in r['name'].lower() for x in ('prob','prediction')))]
    files=list(dict.fromkeys(REQUIRED+optional))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(fetch,[(p,sizes.get(p)) for p in files]))
    receipt={'source':f'{OWNER}/{SLUG}/{VERSION}', 'queried_at_utc':datetime.now(timezone.utc).isoformat(),
             'files':rows}
    (ROOT/'fetch_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
