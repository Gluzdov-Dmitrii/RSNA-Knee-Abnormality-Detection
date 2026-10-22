# CF01 — CoAtNet P03 + r2p/Swin rank-mean

- Status: `scored` (independent CoAtNet graph + scored 3D bag, no A0 mix)
- Members: compact CoAtNet public **0.875** and r2p+Swin public **0.871**
- No locked pair OOF (CoAtNet is a 4,349-study refit, not FOLDS_V1)
- Kernel: `dmitriigluzdov/rsna-cf01-coatnet-3d-rankmean` **v1**, private, internet off, T4
- Visible: 3/3 studies, 73.1 s CUDA
- Hidden estimate ~7.27 h from split 3D/CoAtNet timings; engineering gate 8.5 h
- Submit once: ref **56220447** at 2026-09-14T03:35:47Z
- Public LB: **0.886** (`COMPLETE`)
- Result: improves the own-model branch over compact CoAtNet `0.875` and r2p+Swin `0.871`, but remains below A0/public routes (`0.937+`).
- Do not retry this kernel version. Do not mix into A0.
