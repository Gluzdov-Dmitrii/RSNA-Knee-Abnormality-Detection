# S22 — ResNet-18 2.5D transformer mix on A0

- Status: `submitted` (public score pending hidden grading)
- Local OOF macro AUC: **0.768** (Pilkwang weak-label proxy)
- Kernel: `dmitriigluzdov/rsna-s22-a0-rank-blend` v2
- Submission ref: `56118771` at 2026-09-09T09:09:04Z
- Visible `submission.csv` SHA-256: `5a2b3fb5894b7be8ae01ea9bbb6a15e7ae376d2713ce2b3cfebc5674a222d471`
- Visible overlay: 7.9 s, 3/3 studies decoded, parent A0 SHA matched
- Recipe: 40% S22 rank into A0 transformer ranks, then raptor 0.60 / meniscus T30/R60/bag10
- Weights: private `dmitriigluzdov/rsna-s22-resnet18-25d-folds` (A100-verified fp16)
- Hidden estimate: ~450 min (A0 ~420 plus live 224×9 decode)

Do not resubmit this kernel version. Poll `kaggle competitions submissions` only.
