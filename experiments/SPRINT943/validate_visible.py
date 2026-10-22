"""Validate downloaded visible Kaggle output before a code submission."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
MEMBERS = ['resgated_top3','global96_top3','d4_swa3','repairv1_top3']
AUTHOR_VISIBLE_SHA = '8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a'


def validate(folder):
    folder=Path(folder)
    path=folder/'submission.csv'
    frame=pd.read_csv(path,dtype={'StudyInstanceUID':str})
    expected=pd.read_csv(Path(__file__).resolve().parents[2]/'tmp/schema/test.csv',dtype=str)
    assert frame.columns.tolist()==['StudyInstanceUID']+LABELS, 'Column contract failed'
    assert frame.StudyInstanceUID.is_unique, 'Duplicate UID'
    assert frame.StudyInstanceUID.tolist()==expected.StudyInstanceUID.tolist(), 'Visible UID/order mismatch'
    values=frame[LABELS].to_numpy(float)
    assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    complete=json.loads((folder/'btkd_v559_complete.json').read_text())
    assert complete['status']=='COMPLETE'
    assert complete['dino_members']==20 and complete['a5_folds']==5 and complete['raptor_views']==4
    release=complete['strict_release']
    assert release['status']=='VERIFIED' and set(release['coats'])==set(MEMBERS)
    assert release['family']['family_reduction']=='rank_of_member_probability_mean'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert complete['submission_sha256']==digest, 'Completion receipt SHA mismatch'
    run=json.loads((folder/'sprint943_run_manifest.json').read_text())
    assert run['source']['script_version_id']==351665011
    assert run['submission_sha256']==digest
    result={'status':'PASS','studies':len(frame),'submission_sha256':digest,
            'author_visible_sha256':AUTHOR_VISIBLE_SHA,
            'author_visible_byte_match':digest==AUTHOR_VISIBLE_SHA,
            'elapsed_seconds':run['elapsed_seconds'],'coats':MEMBERS,
            'public_score':None,'full_graph_oof_available':False}
    (folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder')
    validate(p.parse_args().folder)
