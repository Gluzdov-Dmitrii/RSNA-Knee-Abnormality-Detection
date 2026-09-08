"""Read-only remote Dataset verification; never creates or updates a Dataset."""
import json
from pathlib import Path
from datetime import datetime,timezone
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/cache_budget_recheck'
REF='dmitriigluzdov/rsna-knee-uint8-224-9-c130'

def main():
    validation=json.loads((OUT/'cache_validation.json').read_text())
    assert validation['status']=='validated' and validation['all_shards_sha256_verified']
    api=KaggleApi();api.authenticate()
    status=api.dataset_status(REF)
    assert status=='ready',f'Dataset not ready: {status}'
    request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=REF.split('/')
    with api.build_kaggle_client() as client:
        info=client.datasets.dataset_api_client.get_dataset(request)
    # Record actual visibility; this read-only verifier never changes access.
    names={};token=None
    while True:
        page=api.dataset_list_files(REF,page_token=token,page_size=100)
        for f in page.files:names[f.name]=f.total_bytes
        token=page.next_page_token
        if not token:break
    local={p.name:p.stat().st_size for p in (OUT/'cache').iterdir() if p.name!='dataset-metadata.json'}
    assert names==local,f'Remote file names/sizes differ: {set(names)^set(local)}'
    receipt=dict(status='ready',url='https://www.kaggle.com/datasets/'+REF,
        is_private=info.is_private,dataset_id=info.id,version=info.current_version_number,
        verified_at_utc=datetime.now(timezone.utc).isoformat(),n_studies=4407,
        pixel_gib=validation['pixel_gib'],total_bytes=sum(names.values()),n_files=len(names),
        spec_sha256=validation['spec_sha256'],remote_files_and_bytes_verified=True)
    (OUT/'dataset_publication.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
