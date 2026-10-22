"""Verify author probability artifacts against their signed-release receipts."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from harden_source import STRICT_HELPERS
ns={}; exec(compile(STRICT_HELPERS,'strict_helpers','exec'),ns)
submission=pd.read_csv(ROOT/'submission.csv',dtype={'StudyInstanceUID':str})
ids=submission['StudyInstanceUID'].tolist(); labels=submission.columns[1:].tolist()
rows=[]
for name in ['_coat_arm_receipt.json','_coat_g96_receipt.json','_coat_repairv1_receipt.json','d4_input/d4.receipt.json']:
    receipt=json.loads((ROOT/name).read_text())
    relative=receipt['predictions'].removeprefix('/kaggle/working/')
    path=ROOT/relative
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==receipt['predictions_sha256'],relative
    with np.load(path,allow_pickle=False) as z:
        raw=z['raw_probabilities']
        assert raw.shape==(receipt['models'],len(ids),len(labels))
        assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
        mean=z['probability_mean'] if 'probability_mean' in z else raw.astype(np.float64).mean(0).astype(np.float32)
        ns['sprint_validate_probability_payload'](z['study_uids'],mean,ids,labels)
        rows.append({'path':relative,'receipt_sha256_match':True,'raw_shape':list(raw.shape),'probability_payload_gate':'PASS'})
result={'status':'PASS','artifacts':rows,'limitations':['D4 per-process input_*.jsonl receipts were not fetched: versioned listing API returned 404 and receipt names contain unknown process IDs. D4 receipt reports input_coverage.studies=3, zero fallback_studies and zero shard preparation_warnings; JSONL grid checks were not exercised.', 'Author artifacts validate schema compatibility only; this is not our GPU smoke run or private-data validation.']}
(ROOT/'probability_artifact_validation.json').write_text(json.dumps(result,indent=2))
gate_path=ROOT/'release_gate_validation.json'; gate=json.loads(gate_path.read_text()); gate['limitations']=result['limitations'];gate_path.write_text(json.dumps(gate,indent=2))
print(json.dumps(result,indent=2))
