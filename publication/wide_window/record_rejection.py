"""Persist a measured rejection in the owned run manifest; never train or submit."""
import json
import subprocess
from manage import ART,RUN

if __name__=='__main__':
    result=json.loads((ART/'comparison.json').read_text())
    assert result['status']=='REJECT'
    p=ART/'run_manifest.json';manifest=json.loads(p.read_text())
    manifest.update(state='REJECT',comparison=result,full_refit=False,lb_submitted=False,
                    owned_gpu_leases='smoke and pilot RELEASED after verified exit',
                    continuation='rsna-p04-gpu PAUSED')
    p.write_text(json.dumps(manifest,indent=2))
    for source,name in [(p,'run.json'),(ART/'comparison.json','comparison.json')]:
        subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=10',str(source),
                        'nsu-a100:'+RUN+'/'+name],check=True,timeout=60)
    print('REJECT recorded locally and in owned remote run; no submit')
