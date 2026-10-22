"""CPU packaging checks: source identity, cells, stage parity and pinned inputs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    package=ROOT/'kaggle'
    meta=json.loads((package/'kernel-metadata.json').read_text(encoding='utf-8'))
    lock=json.loads((ROOT/'input_lock.json').read_text(encoding='utf-8'))
    nb=json.loads((package/meta['code_file']).read_text(encoding='utf-8'))
    assert nb['nbformat']==4 and nb['nbformat_minor']==5
    ids=[c['id'] for c in nb['cells']]
    assert len(ids)==len(set(ids))
    assert all(c['cell_type'] in ('code','markdown') and isinstance(c['source'],list) for c in nb['cells'])
    code=[c for c in nb['cells'] if c['cell_type']=='code']
    assert len(code)==25
    for c in code:
        assert c['execution_count'] is None and c['outputs']==[]
        compile(''.join(c['source']),c['id'],'exec')
    stages=sorted((ROOT/'runtime').glob('*.py'))
    assert len(stages)==23
    for i,path in enumerate(stages,1):
        codecell=next(c for c in code if c['id']==f'stage-{i:02d}')
        assert ''.join(codecell['source'])==path.read_text(encoding='utf-8')
        expected=nb['metadata']['sprint943']['runtime_sha256'][path.name]
        assert hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()==expected
    assert 'id_no' not in meta
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is False
    assert meta['machine_shape']=='NvidiaTeslaT4'
    assert nb['metadata']['kaggle']['isGpuEnabled'] is True
    for key,value in lock['metadata_overrides'].items():
        assert meta[key]==value,(key,'lock drift')
    assert len(meta['dataset_sources'])==14 and len(meta['kernel_sources'])==2
    assert all(ref.rsplit('/',1)[1].isdigit() for ref in meta['dataset_sources']+meta['kernel_sources'])
    source=ROOT/'source_v34/bend-the-knee-to-speedy-raptors-the-original.ipynb'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==lock['source']['sha256']
    text='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert 'sprint_release_gate' in text and "'repairv1_top3'" in text
    result={'status':'PASS','notebook_cells':len(nb['cells']),'code_cells':len(code),
            'editable_stages':len(stages),'pinned_datasets':14,'pinned_kernels':2,
            'notebook_sha256':hashlib.sha256((package/meta['code_file']).read_bytes()).hexdigest(),
            'gpu_execution_verified':False}
    (ROOT/'package_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
