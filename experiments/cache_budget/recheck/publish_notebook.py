"""Publish verified figures with Kaggle Quick Save, preserving runnable rebuild cells."""
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/cache_budget_recheck'
PUBLIC=ROOT/'experiments/cache_budget/public'

def main():
    api=KaggleApi();api.authenticate()
    status=api.kernels_status('dmitriigluzdov/rsna-knee-private-cache-verifier')
    assert str(status.status).endswith('COMPLETE'), f'Private source run not complete: {status.status}'
    meta=json.loads((PUBLIC/'kernel-metadata.json').read_text())
    notebook=(PUBLIC/meta['code_file']).read_text(encoding='utf-8')
    nb=json.loads(notebook)
    assert sum(len(c.get('attachments',{})) for c in nb['cells'])==2
    for c in nb['cells']:
        if c['cell_type']=='code':
            if c['id'] != 'settings':
                assert c['metadata']['jupyter']['source_hidden'] is True
            compile(c['source'],f"cell:{c['id']}",'exec')
    settings=next(c['source'] for c in nb['cells'] if c['id']=='settings')
    assert 'SAVE_CACHE_OUTPUT = False' in settings
    assert 'rsna-knee-uint8-224-9-c130' not in notebook
    assert all(not c.get('outputs') for c in nb['cells'])
    request=ApiSaveKernelRequest()
    request.id=133521917
    request.slug=meta['id'];request.new_title=meta['title'];request.text=notebook
    request.language='python';request.kernel_type='notebook'
    request.is_private=False;request.enable_gpu=False;request.enable_tpu=False;request.enable_internet=False
    request.dataset_data_sources=meta['dataset_sources']
    request.competition_data_sources=meta['competition_sources']
    request.kernel_data_sources=[];request.model_data_sources=[]
    request.kernel_execution_type=KernelExecutionType.QUICK_SAVE
    with api.build_kaggle_client() as client:
        response=client.kernels.kernels_api_client.save_kernel(request)
    assert not response.error,response.error
    assert not response.invalid_dataset_sources and not response.invalid_competition_sources
    receipt=dict(url=response.url,version=response.version_number,kernel_id=response.kernel_id,
        save_type='QUICK_SAVE',measurements_run='dmitriigluzdov/rsna-knee-private-cache-verifier/1',
        measurements_run_status='COMPLETE',source_sha256=hashlib.sha256(notebook.encode()).hexdigest(),
        displayed_figures_from_verified_run=True,cpu_only=True,internet=False,competition_submission=False)
    (OUT/'notebook_publication.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
