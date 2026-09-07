# FOLDS_V1

Locked study-level 5-fold split for week-1 label heads (S11+).

| Field | Value |
| --- | --- |
| Studies | 4407 official train `StudyInstanceUID`s |
| Splits | 5 |
| Seed | 2026 |
| Algorithm | iterative stratification (Sechidis et al. 2011) |
| Stratify labels | `LABEL_PILKWANG_V1` scores ≥ 0.5 |
| `folds.csv` SHA-256 | `3086df3341333f44adb883292da386857c3230eaa2d501514ddf827a2da11b1a` |
| Builder | `ops/tools/build_folds_v1.py` |

`folds.csv` contains only `StudyInstanceUID,fold`. It does not contain reports or teacher scores.

Rebuild only to repair a proven builder bug; do not change seed or algorithm without a new registry key.
