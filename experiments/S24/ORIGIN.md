# S24 — r3d_18 9-slice 3D on PIXEL_CACHE_V1

Local pixel-path control after this run. Not mixed into A0.

- torchvision `r3d_18` Kinetics-400, three plane heads
- Input (3, 3, 9, 224, 224): fluid / struct / mean, T=9 from PIXEL_CACHE_V1
- Same FOLDS_V1 seed 2026, LABEL_PILKWANG_V1
- MixUp 0.2, dropout 0.25, backbone LR 0.3×, batch 8, patience 5

Cache has 9 slices (not the original 16-slice S24 sketch). Spec geometry otherwise matches PIXEL_CACHE_V1.
