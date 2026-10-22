"""CPU regression: rebuild V34 final blend from its saved visible components.

This verifies ensemble arithmetic and alignment, not neural inference or OOF.
No coefficients are fitted and no Kaggle submission is made.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent/'author_v34_outputs'
from validate_visible import LABELS, AUTHOR_VISIBLE_SHA


def main():
    reference=pd.read_csv(ROOT/'submission.csv',dtype={'StudyInstanceUID':str})
    ids=reference.StudyInstanceUID.tolist()
    frames=[]
    files=['coat_resgated_ep10_top3_predictions.npz','coatnet_global96_top3_predictions.npz',
           'd4_input/coatnet_d4_depthzone_swa3_predictions.npz','coatnet_repairv1_top3_predictions.npz']
    for name in files:
        with np.load(ROOT/name,allow_pickle=False) as z:
            uids=z['study_uids'].astype(str).tolist()
            assert len(uids)==len(set(uids)) and set(uids)==set(ids)
            prob=z['probability_mean'].astype(float) if 'probability_mean' in z.files else z['raw_probabilities'].astype(float).mean(axis=0)
            if 'labels' in z.files:
                assert z['labels'].astype(str).tolist()==LABELS
            assert np.isfinite(prob).all() and prob.shape==(len(ids),len(LABELS))
            frames.append(pd.DataFrame(prob,index=uids,columns=LABELS).loc[ids].reset_index(drop=True))
    coat=(sum(frames)/len(frames)).rank(method='average',pct=True)
    raptor=pd.read_csv(ROOT/'_raptor_public.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,LABELS].reset_index(drop=True).rank(method='average',pct=True)
    hybrid=.6*raptor+.4*coat
    with np.load(ROOT/'btk_v32_rest_input.npz',allow_pickle=False) as z:
        assert z['labels'].astype(str).tolist()==LABELS
        parent=pd.DataFrame(z['values'],index=z['study_uids'].astype(str),columns=LABELS).loc[ids].reset_index(drop=True).rank(method='average',pct=True)
    weights=dict.fromkeys(LABELS,.60)
    weights.update({'ACL':.75,'Medial Meniscus':.80,'Lateral Meniscus':1.,'Lateral OA':.75,'Fracture':.75})
    final=parent.copy()
    for target,weight in weights.items():
        final[target]=(1-weight)*parent[target]+weight*hybrid[target]
    final=final.rank(method='average',pct=True)
    final.insert(0,'StudyInstanceUID',ids)
    out=ROOT/'submission_cpu_replay.csv'
    # Kaggle runs Linux; fix line endings for byte-level comparison on Windows.
    final.to_csv(out,index=False,lineterminator='\n')
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    error=float(np.max(np.abs(final[LABELS].to_numpy()-reference[LABELS].to_numpy())))
    assert error<1e-14,error
    assert digest==AUTHOR_VISIBLE_SHA,(digest,AUTHOR_VISIBLE_SHA)
    result={'status':'PASS','studies':len(ids),'max_abs_error_vs_parsed_csv':error,
            'byte_identical_to_author':True,'sha256':digest,'four_coat_members':True,
            'new_gpu_inference':False,'full_graph_oof':False,'fitted_parameters':0}
    (ROOT/'blend_replay_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
