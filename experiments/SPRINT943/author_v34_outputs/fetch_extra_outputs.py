"""Fetch optional tiny probability payloads and exact V34 input-grid receipts."""
from concurrent.futures import ThreadPoolExecutor
import json
from fetch_author_outputs import ROOT, OWNER, SLUG, fetch
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelFilesRequest

files=['coat_resgated_ep10_top3_predictions.npz','coatnet_global96_top3_predictions.npz',
       'coatnet_repairv1_top3_predictions.npz','d4_input/coatnet_d4_depthzone_swa3_predictions.npz']
api=KaggleApi();api.authenticate()
req=ApiListKernelFilesRequest();req.user_name=OWNER;req.kernel_slug=SLUG
req.version_label='351665011';req.page_size=100
listing=[]
try:
    with api.build_kaggle_client() as client:
        for _ in range(10):
            page=client.kernels.kernels_api_client.list_kernel_files(req)
            listing.extend({'name':f.name,'bytes':f.size} for f in page.files)
            if not page.next_page_token:break
            req.page_token=page.next_page_token
    (ROOT/'exact_id_file_listing.json').write_text(json.dumps(listing,indent=2))
    files.extend(r['name'] for r in listing if r['name'].startswith('d4_input/study_input_receipts/input_') and r['name'].endswith('.jsonl') and r['bytes']<2*1024*1024)
except Exception as exc:
    print('Exact-ID optional listing unavailable: '+type(exc).__name__)
with ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(fetch,[(p,None) for p in files]))
(ROOT/'extra_fetch_receipt.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
