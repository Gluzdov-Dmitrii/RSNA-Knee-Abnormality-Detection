# S22-head v1 — MLP fusion on ResNet-18 2.5D

Not a new S-ID. Same PIXEL_CACHE_V1 / FOLDS_V1 / LABEL_PILKWANG_V1 / seed 2026.

S22-reg overshot (OOF 0.757 vs 0.768). This run is milder regularization plus one extra
fusion layer, selected on locked weak-label OOF only.

Intended deltas vs original S22:

- fusion Linear(1536,12) → Linear(1536,512) GELU Dropout Linear(512,12)
- dropout 0.25 (S22 0.2, S22-reg 0.4)
- weight_decay 3e-4 (S22 1e-4, S22-reg 1e-3)
- MixUp 0.2 (S22 0, S22-reg 0.4)
- backbone LR 0.3× (S22 1×, S22-reg 0.1×)
- epochs 20, patience 6

No Kaggle submit in this packet. Promote only if OOF beats 0.768.
