# S24 slice-attention v1 — ResNet-18 per slice, pool over T=9

- Status: `local_evaluated` (lost)
- Device: NSU A100 80 GB (`ngpu01`), queue `rsna-s24-attn-20260910T1108Z`
- Independent recompute: **0.801746**
- Fold best val AUC: 0.800 / 0.807 / 0.818 / 0.797 / 0.803
- vs r2plus1d 0.811: **−0.009**
- vs r3d_18 0.801: **+0.001** (tie)

Same vol9 cache as S24. ImageNet 2D + slice attention did not beat 3D r2plus1d.
Rank-mean R2P+ATTN 0.823; r3d+r2plus1d+attn 0.827. Control stays r2plus1d 0.811.
Not a Kaggle submit.
