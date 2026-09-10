# S24-r2plus1d v1 — r2plus1d_18 9-slice 3D

- Status: `local_evaluated`
- Device: NSU A100 80 GB (`ngpu01`), queue `rsna-s24-r2p-20260910T0745Z`
- Independent recompute: **0.810611**
- Fold best val AUC: 0.814 / 0.814 / 0.828 / 0.810 / 0.821
- vs S24 r3d_18 0.801: **+0.010**
- New single-model pixel-path control

Same cache/labels/folds/recipe as S24 r3d_18. Only the backbone changed.
Not mixed into A0. Hidden test still needs a live-decode 3D notebook.
