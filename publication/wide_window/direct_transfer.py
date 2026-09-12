"""Fetch remaining immutable shards directly; signed URLs stay in process memory."""
import json
from pathlib import Path
import shlex
import subprocess
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
from manage import ART,PROJECT

REMOTE = r'''
import concurrent.futures,hashlib,json,sys,urllib.request
from pathlib import Path
payload=json.load(sys.stdin); root=Path(payload['root'])
root.mkdir(parents=True,exist_ok=True)
def one(item):
    name=item['name']; assert Path(name).name==name
    p=root/name; tmp=root/(name+'.direct-part')
    def digest(f):
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
        return h.hexdigest()
    if p.exists():
        assert digest(p)==item['sha256']; return name+' already verified'
    try:
        with urllib.request.urlopen(item['url'],timeout=60) as source, tmp.open('xb') as dest:
            while True:
                b=source.read(8<<20)
                if not b:break
                dest.write(b)
        assert digest(tmp)==item['sha256']
        if p.exists(): raise RuntimeError('Concurrent destination exists')
        tmp.rename(p)
        return name+' directly verified'
    except Exception as exc:
        # Error text can contain a signed URL; expose only type and filename.
        return name+' FAILED '+type(exc).__name__
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for message in pool.map(one,payload['files']):print(message,flush=True)
'''

if __name__=='__main__':
    api=KaggleApi();api.authenticate()
    request=ApiListKernelSessionOutputRequest()
    request.user_name='dmitriigluzdov';request.kernel_slug='knee-mri-private-wide-cache'
    api._set_paging(request,200,None)
    with api.build_kaggle_client() as client:
        response=client.kernels.kernels_api_client.list_kernel_session_output(request)
    assert not response.next_page_token
    spec=json.loads((ART/'cache_receipt/cache/SPEC.json').read_text())
    items=[]
    for item in response.files:
        name=Path(item.file_name).name
        if item.file_name.startswith('cache/pixels-') and int(name[7:10])>=8:
            items.append(dict(name=name,url=item.url,sha256=spec['sha256'][name]))
    assert len(items)==27
    payload=dict(root=PROJECT+'/data/rsna-knee-uint8-224-9-c130-w10-90',files=items)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','nsu-a100',
        shlex.join(['python3','-c',REMOTE])],input=json.dumps(payload),text=True,timeout=3600)
    raise SystemExit(result.returncode)
