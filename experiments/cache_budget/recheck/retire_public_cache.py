"""Create/verify a private replacement before explicitly retiring the public cache."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/cache_budget_recheck'
COPY=OUT/'private_copy'
OLD='dmitriigluzdov/rsna-knee-uint8-224-9-c130'
NEW='dmitriigluzdov/rsna-knee-uint8-224-9-c130-personal'

def info(api,ref):
    request=ApiGetDatasetRequest()
    request.owner_slug,request.dataset_slug=ref.split('/')
    with api.build_kaggle_client() as client:
        return client.datasets.dataset_api_client.get_dataset(request)

def files(api,ref):
    result={};token=None
    while True:
        page=api.dataset_list_files(ref,page_token=token,page_size=100)
        result.update({f.name:f.total_bytes for f in page.files})
        token=page.next_page_token
        if not token:return result

def prepare():
    validation=json.loads((OUT/'cache_validation.json').read_text())
    assert validation['all_shards_sha256_verified'] and validation['n_studies']==4407
    assert hashlib.sha256((COPY/'SPEC.json').read_bytes()).hexdigest()==validation['spec_sha256']
    meta=json.loads((OUT/'visibility_2026_09_10/dataset-metadata.json').read_text())['info']
    description=meta['description'].split('## Source, licence and limitations')[0]
    description=description.replace('/kaggle/input/rsna-knee-uint8-224-9-c130','/kaggle/input/rsna-knee-uint8-224-9-c130-personal')
    description+='\n## Source, licence and limitations\n\n'+(COPY/'NOTICE.md').read_text()
    description+='\nThis is the owner-only replacement for the retired public cache. MRI files and their original SPEC.json are unchanged; any dataset URL inside that original manifest is historical provenance, not an active download. No collaborators are added.\n'
    result=dict(id=NEW,title='RSNA Knee uint8 224 x 9 crop 130 personal',
        subtitle='Personal derived MRI cache: 4,407 studies, 11.12 GiB, NumPy uint8',
        licenses=[{'name':'other'}],description=description,keywords=meta['keywords'])
    (COPY/'dataset-metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Prepared PRIVATE creation metadata for '+NEW)

def verify(api):
    d=info(api,NEW)
    assert d.is_private is True and api.dataset_status(NEW)=='ready'
    local={p.name:p.stat().st_size for p in COPY.iterdir() if p.name!='dataset-metadata.json'}
    assert len(local)==41 and files(api,NEW)==local
    check=OUT/'private_remote_manifest';check.mkdir(exist_ok=True)
    api.dataset_download_file(NEW,'SPEC.json',path=str(check),force=True,quiet=True)
    assert (check/'SPEC.json').read_bytes()==(COPY/'SPEC.json').read_bytes()
    receipt=dict(url='https://www.kaggle.com/datasets/'+NEW,is_private=True,
        dataset_id=d.id,version=d.current_version_number,n_files=len(local),
        total_bytes=sum(local.values()),local_shards_sha256_verified=True,
        remote_names_sizes_and_spec_verified=True,verified_at_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'private_replacement.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','delete-public'])
    action=parser.parse_args().action
    if action=='prepare':return prepare()
    api=KaggleApi();api.authenticate()
    verify(api)
    if action=='delete-public':
        old=info(api,OLD)
        assert old.id==11946093 and old.is_private is False
        owner,slug=OLD.split('/')
        result=api.dataset_delete(owner,slug,no_confirm=True)
        assert result is True,repr(result)
        receipt=dict(ref=OLD,dataset_id=old.id,delete_success=True,
            deleted_at_utc=datetime.now(timezone.utc).isoformat())
        (OUT/'public_cache_retirement.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
