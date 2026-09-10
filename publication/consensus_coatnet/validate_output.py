"""Validate downloaded Kaggle output before a single authorized submission."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import UID,TARGETS,sha256,save_json

def validate(outputs,package,training_gold,fp32_reference=None,qa_outputs=None):
    outputs,package=Path(outputs),Path(package)
    pred=pd.read_csv(outputs/'submission.csv',dtype={UID:str})
    expected=pd.read_csv(outputs/'test_manifest.csv',dtype={UID:str})
    if pred.columns.tolist()!=[UID]+TARGETS:raise ValueError('Wrong submission columns/order')
    if not len(pred) or pred[UID].isna().any() or pred[UID].duplicated().any():raise ValueError('Invalid submissionUIDs')
    if pred[UID].tolist()!=expected[UID].tolist():raise ValueError('Live test count/IDs/order differ')
    values=pred[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Invalid predictions')
    if np.all(values==.5):raise ValueError('All0.5 fallback is not a model output')
    qa_dir=Path(qa_outputs) if qa_outputs else outputs
    qa=json.loads((qa_dir/'qa.json').read_text());audit=json.loads((outputs/'submission.audit.json').read_text())
    if qa['cache_pixel_parity']!='58/58 exact SHA256':raise ValueError('Pixel parity incomplete')
    if qa['fp16_fp32_max_error']>.01:raise ValueError('Precision mismatch')
    if audit['weights_sha256']!=sha256(package/'weights.pt'):raise ValueError('Unexpected modelweights')
    a=pd.read_csv(training_gold,dtype={UID:str}).set_index(UID)
    b=pd.read_csv(qa_dir/'gold_predictions.csv',dtype={UID:str}).set_index(UID)
    if set(a.index)!=set(b.index) or len(a)!=58:raise ValueError('Gold prediction UID mismatch')
    error=float(np.abs(a.loc[b.index,TARGETS].to_numpy()-b[TARGETS].to_numpy()).max())
    fp32_error=None
    if fp32_reference:
        ref=Path(fp32_reference)
        receipt=json.loads(ref.with_suffix('.json').read_text())
        if receipt['weights_sha256']!=audit['weights_sha256'] or receipt['dtype']!='CPU FP32':
            raise ValueError('FP32 reference contract differs')
        c=pd.read_csv(ref,dtype={UID:str}).set_index(UID)
        if set(c.index)!=set(b.index) or len(c)!=58:raise ValueError('FP32 UID mismatch')
        fp32_error=float(np.abs(c.loc[b.index,TARGETS].to_numpy()-b[TARGETS].to_numpy()).max())
        if not np.isfinite(fp32_error) or fp32_error>.01:raise ValueError(f'FP32 reference divergence: {fp32_error}')
    elif error>.015:raise ValueError(f'NSU BF16 / Kaggle FP16 divergence: {error}')
    report=dict(status='PASS',n_studies=len(pred),submission_sha256=sha256(outputs/'submission.csv'),
        weights_sha256=audit['weights_sha256'],nsu_bf16_kaggle_fp16_max_error=error,
        independent_cpu_fp32_kaggle_fp16_max_error=fp32_error,
        test_seconds=audit['elapsed_seconds'],qa_seconds=qa['elapsed_seconds'],
        gold_macro_auc=qa['gold']['macro_auc'],gold_scored_classes=qa['gold']['n_scored_classes'],
        gpu=qa['gpu'],peak_vram_gib=qa['peak_vram_gib'],versions=qa['versions'],
        runtime_limit_hours=9,evidence='Visible-run validation; hidden-runtime/score not yet known')
    save_json(outputs/'pre_submit_validation.json',report);print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['outputs','package','training-gold']:p.add_argument('--'+key,required=True)
    p.add_argument('--fp32-reference');p.add_argument('--qa-outputs')
    a=p.parse_args();validate(a.outputs,a.package,a.training_gold,a.fp32_reference,a.qa_outputs)
