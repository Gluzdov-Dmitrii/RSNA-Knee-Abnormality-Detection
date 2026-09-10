# S22-reg v1 — regularized ResNet-18 2.5D

- Status: `local_evaluated`, decision `defer_submit`
- OOF macro AUC: **0.757** (Pilkwang weak-label proxy)
- Δ vs S22 baseline 0.768: **−0.012**
- Recipe: dropout 0.4, weight_decay 1e-3, MixUp 0.4, backbone LR 0.1×, 24 epochs, patience 8
- NSU A100 `GPU-61c0078d-a4a6-37a2-3aba-0378e7794c46`, queue `rsna-s22-reg-20260910T0335Z`, ~16.5 s/epoch
- Train AUC stayed near 0.90 (S22 went to ~0.999). Regularization worked; the locked OOF did not improve.

Pixel-path control remains original S22. Do not submit. No new S-ID.
