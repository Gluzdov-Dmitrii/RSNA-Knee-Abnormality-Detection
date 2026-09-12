"""Read only owned P04 training progress or export its small comparison files."""
import argparse
import json
import subprocess
from manage import ART,RUN,ssh

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','pilot','full'],required=True)
    p.add_argument('--export',action='store_true');a=p.parse_args()
    if a.export:
        dest=ART/a.mode/'pilkwang';dest.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','history.json','result.json','validation_predictions.csv',
                     'train_uids.csv','validation_uids.csv']:
            subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=10',
                'nsu-a100:'+RUN+'/'+a.mode+'/pilkwang/'+name,str(dest/name)],check=True,timeout=120)
        print('Comparison files exported: '+str(dest))
    else:
        code="""import json,sys;from pathlib import Path
p=Path(sys.argv[1]);h=p/'pilkwang/history.json';log=p/'pilkwang.log'
print(json.dumps(dict(history=json.loads(h.read_text()) if h.exists() else None,
    log_tail=log.read_text()[-3000:] if log.exists() else None)))"""
        print(json.dumps(ssh('nsu-a100',['python3','-c',code,RUN+'/'+a.mode]),indent=2))
