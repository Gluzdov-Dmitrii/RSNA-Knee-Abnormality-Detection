# FOLDS_V1 origin

- Competition: `rsna-knee-abnormality-detection`
- Train universe: official `train.csv` StudyInstanceUID column (4407 unique rows). The Report column was read only as a schema check and is not stored here.
- Stratify source: `pilkwang/rsna-knee-llm-labels` `report_labels_v2.csv`, SHA-256 `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920`
- One train UID is absent from Pilkwang V1 and is listed in `SPEC.json`; Steven V6 and Lixin V1 both include that UID.
- Created: 2026-09-07T10:33:44Z on local CPU. No GPU lease.
