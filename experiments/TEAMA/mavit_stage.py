"""MaViT-288 stage: hengck23/mavit-288-baseline-01-1-5sec-study-1gpu (public 0.935), inference loop unchanged.

Changes, all outside the model path: the competition root can be redirected for the rehearsal (RSNA_FAKE_ROOT);
the output goes to /kaggle/working/mavit_sub.csv; a study whose read or forward pass fails is recorded in
mavit_status.csv (the source writes all-zero probabilities for it, which the blend treats as a failure).
"""
import os
import sys
import time
import glob

DS = next(p for p in glob.glob('/kaggle/input/**/hengck23-rnsa-knee', recursive=False) + [
    '/kaggle/input/datasets/hengck23/hengck23-rnsa-knee', '/kaggle/input/hengck23-rnsa-knee'] if os.path.isfile(p + '/model.py'))
sys.path.append(DS)

import numpy as np
import torch
import pandas as pd

from model import *
from my_fastread_knee import *

LABEL = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
         'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
NUM_VIEW = 6
NUM_SLICE = 8
NUM_INJURY = 12
SIZE = 288
DEVICE = "cuda"
CHECKPOINT = DS + "/00000008.pth"
KAGGLE_DIR = os.environ.get('RSNA_FAKE_ROOT') or next(
    p for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection')
    if os.path.isfile(p + '/test.csv'))
DICOM_DIR = f"{KAGGLE_DIR}/test_series"
STUDY = pd.read_csv(f"{KAGGLE_DIR}/sample_submission.csv")
SERIES = pd.read_csv(f"{KAGGLE_DIR}/test_series.csv")
STUDY_ID = STUDY["StudyInstanceUID"].to_list()
print("MAVIT root", KAGGLE_DIR, "studies", len(STUDY_ID), flush=True)

model = KneeModel(pretrained=False)
saved = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
print("resume model:", model.load_state_dict(saved["model_state_dict"], strict=False), flush=True)
model.to(DEVICE)
model = model.eval()

fastreader = make_knee_dataset_reader(
    out_size=SIZE, num_slice=NUM_SLICE,
    percentile=(1.0, 99.0), crop_mm=130.0,
    io_threads=32,
    crop_only_if_nonsquare=True,
)

predict, ok = [], []
start_time = time.time()
for i, study_id in enumerate(STUDY_ID):
    series = SERIES[SERIES["StudyInstanceUID"] == study_id]
    try:
        r = fastreader(study_id, series, DICOM_DIR, "valid")
        volume = torch.from_numpy(r["volume"]).unsqueeze(0).byte().to(DEVICE)
        view_mask = torch.from_numpy(r["view_mask"]).unsqueeze(0).bool().to(DEVICE)
        slice = torch.from_numpy(r["slice"]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logit = model.infer(volume, view_mask, slice)
                prob = torch.sigmoid(logit)
        prob = prob.detach().float().cpu().numpy()
        ok.append(1)
    except Exception as e:
        print('MAVIT_STUDY_FAILED', study_id, repr(e)[:200], flush=True)
        prob = np.zeros((1, 12), np.float32)
        ok.append(0)
    predict.append(prob)
    if i < 5 or i % 100 == 0:
        print(f"{i} time: {(time.time() - start_time) / (i + 1): 5.3f} sec/study", flush=True)
predict = np.concatenate(predict, 0)

submit_df = pd.DataFrame(predict, columns=LABEL, index=STUDY_ID)
submit_df.index.name = "StudyInstanceUID"
submit_df.to_csv("/kaggle/working/mavit_sub.csv", index=True)
pd.DataFrame({"StudyInstanceUID": STUDY_ID, "ok": ok}).to_csv("/kaggle/working/mavit_status.csv", index=False)
print("MAVIT_DONE", len(STUDY_ID), "failed", len(ok) - sum(ok), "seconds", round(time.time() - start_time, 1), flush=True)
