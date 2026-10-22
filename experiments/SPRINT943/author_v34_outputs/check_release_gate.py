"""Exercise current STRICT_HELPERS on immutable downloaded V34 artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from harden_source import STRICT_HELPERS

ns={}
exec(compile(STRICT_HELPERS,'strict_helpers','exec'),ns)
complete=json.loads((ROOT/'btkd_v559_complete.json').read_text())
submission=pd.read_csv(ROOT/'submission.csv',dtype={'StudyInstanceUID':str})
ids=submission['StudyInstanceUID'].tolist()
members=['resgated_top3','global96_top3','d4_swa3','repairv1_top3']
receipt={'checked_at_utc':datetime.now(timezone.utc).isoformat(),
         'source_kernel_version_id':351665011,
         'strict_helpers_sha256':hashlib.sha256(STRICT_HELPERS.encode()).hexdigest(),
         'studies':len(ids),'events':len(complete['events']),
         'd4_input_receipts_present':len(list((ROOT/'d4_input/study_input_receipts').glob('input_*.jsonl')))}
try:
    receipt['result']=ns['sprint_release_gate'](ROOT,ids,members,complete['events'])
    receipt['status']='PASS'
except Exception as exc:
    receipt['status']='FAIL'; receipt['error_type']=type(exc).__name__
    receipt['error']=str(exc); receipt['traceback']=traceback.format_exc()
(ROOT/'release_gate_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
