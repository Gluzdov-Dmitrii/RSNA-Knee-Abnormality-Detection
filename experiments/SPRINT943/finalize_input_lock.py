"""Build compact version lock from read-only Kaggle probe evidence."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
source = ROOT / 'source_v34/bend-the-knee-to-speedy-raptors-the-original.ipynb'
evidence=json.loads((ROOT/'input_version_evidence.json').read_text())
metadata=json.loads((ROOT/'input_metadata_evidence.json').read_text())
for kernel in evidence['kernels']:
    kernel['exact_metadata']={k:v for k,v in kernel.get('exact_metadata',{}).items() if k in ('kaggle','papermill')}
(ROOT/'input_version_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
datasets=[]
for row, remote in zip(metadata['datasets'],evidence['datasets']):
    assert row['slug']==remote['slug']
    match=re.search(r'/kaggle-data-sets/(\d+)/(\d+)/',remote['location_without_query'])
    assert remote['status']==302 and match and int(match[2])==row['dataset_version_id']
    datasets.append({'ref': row['slug'], 'version_number': remote['version'],
                     'dataset_version_id': row['dataset_version_id'],
                     'pinned_ref': row['slug']+'/'+str(remote['version']),
                     'verification': 'Kaggle download redirect embeds exact datasetVersionId; archive not downloaded'})
kernels=[]
for row in evidence['kernels']:
    verified=row['current_version_number']==1 and row['cell_source_match']
    kernels.append({'ref':row['slug'],'kernel_version_id':row['kernel_version_id'],
                    'version_number':1 if verified else None,
                    'pinned_ref':row['slug']+'/1' if verified else None,
                    'current_version_number_at_audit':row['current_version_number'],
                    'historical_source_sha256':row['exact_sha256'],
                    'current_cell_sources_match_historical':row['cell_source_match'],
                    'verification':'Only version 1 exists and its cell sources match exact historical source ID' if verified else 'Current version 2 cell sources match historical source ID, but output version mapping is unresolved'})
redirect_file=ROOT/'kernel_output_redirect_evidence.json'
if redirect_file.exists():
    redirects=json.loads(redirect_file.read_text())
    for row in kernels:
        for redirect in redirects:
            if redirect['kernel_version_id']==row['kernel_version_id']:
                row['pinned_ref']=redirect['ref']
                row['version_number']=int(redirect['ref'].rsplit('/',1)[1])
                row['verification']='SDK DownloadKernelOutput(version_number) returned redirect /kf/<exact kernelVersionId>/; output bytes not downloaded'
lock={'schema_version':1,'checked_at_utc':metadata['queried_at_utc'],
      'source':{'ref':'mattiaangeli/bend-the-knee-to-speedy-raptors-the-original','version_number':34,
                'kernel_version_id':351665011,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
      'datasets':datasets,'kernels':kernels,
      'models':[{'ref':'metaresearch/dinov2/PyTorch/small/1','model_instance_version_id':4533}],
      'competition':{'ref':'rsna-knee-abnormality-detection','source_id':154281},
      'metadata_overrides':{'dataset_sources':[r['pinned_ref'] for r in datasets],
                            'kernel_sources':[r['pinned_ref'] or r['ref'] for r in kernels],
                            'model_sources':['metaresearch/dinov2/PyTorch/small/1']},
      'cli_version_syntax_verified':True,
      'cli_evidence':{'module':'kaggle/api/kaggle_api_extended.py',
                       'validate_dataset_string_lines':'8764-8782: owner/slug/version-number accepted',
                       'validate_kernel_string_lines':'8955-8992: owner/slug/version accepted',
                       'kernels_push_lines':'6395-6401 validation; 6440-6443 arrays sent unchanged'},
      'limitations':[] if all(r['pinned_ref'] for r in kernels) else ['Some kernel output versions unresolved; inspect kernels verification fields.'],
      'evidence_files':['input_metadata_evidence.json','input_version_evidence.json','kernel_output_redirect_evidence.json']}
(ROOT/'input_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
print(json.dumps({'datasets_pinned':len(datasets),'kernels':kernels,'lock':'input_lock.json'},indent=2))
