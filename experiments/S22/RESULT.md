# S22 — ResNet-18 2.5D transformer mix on A0

- Status: `scored`, decision **`reject`** for the 0.40 overlay
- Public score: **0.931** (ref `56118771`, kernel v2 COMPLETE). Δ vs A0 **−0.006**
- Team best remains S01 **0.937**. Kernel stays **private** — 0.931 is not a writeup score
- Local OOF macro AUC: **0.768** (Pilkwang weak-label proxy). This own-model result is kept
- Recipe: 40% S22 rank into A0 transformer ranks, then raptor 0.60 / meniscus T30/R60/bag10
- Visible `submission.csv` SHA-256: `5a2b3fb5894b7be8ae01ea9bbb6a15e7ae376d2713ce2b3cfebc5674a222d471`

Do not resubmit this overlay or retune blend weights from public LB.
Next local iterations: `reg_v1/` OOF **0.757** (lost); `head_v1/` OOF **0.769** (tie / +0.0008). Selection stays on locked OOF, not public LB.

