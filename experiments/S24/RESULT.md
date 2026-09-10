# S24 — r3d_18 9-slice 3D on PIXEL_CACHE_V1

- Status: `local_evaluated` (new pixel-path control)
- Device: NSU A100 80 GB (`ngpu01`), queue `rsna-s24-20260910T0545Z`
- Independent recompute from `oof.csv` + Pilkwang V1: **0.800641** (matches `metrics.json`)
- Fold best val AUC: 0.808 / 0.807 / 0.821 / 0.796 / 0.804
- vs S22 ResNet-18 2.5D: **+0.032**
- Weakest targets remain MCL 0.745 and Lateral Meniscus 0.742; Fracture rose to 0.779

Same cache/labels/folds as S22. Hidden test still needs a live-decode 3D notebook.
Promote to a standalone Kaggle submit after fp16 export and a visible runtime gate.
Do not mix into A0 at 0.40.
