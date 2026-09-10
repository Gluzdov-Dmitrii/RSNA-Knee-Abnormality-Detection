# S22-head v1 — MLP fusion on ResNet-18 2.5D

- Status: `local_evaluated`, decision `defer_submit`
- OOF macro AUC: **0.769** (Pilkwang weak-label proxy)
- Δ vs S22 baseline 0.768: **+0.0008** (point estimate only; fold-noise tie)
- Recipe: fusion 1536→512→12, dropout 0.25, weight_decay 3e-4, MixUp 0.2, backbone LR 0.3×
- NSU A100 `GPU-61c0078d-a4a6-37a2-3aba-0378e7794c46`, queue `rsna-s22-head-20260910T0425Z`

Not a Kaggle submit. Next local levers: S24 3D, or a second seed, not public-LB retuning.
