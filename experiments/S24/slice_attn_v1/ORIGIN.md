# S24 slice-attention v1 — ResNet-18 per slice, pool over T=9

Not a new S-ID. Same PIXEL_CACHE_V1 / FOLDS_V1 / LABEL_PILKWANG_V1 / seed 2026
and the S24 vol9 layout (fluid / struct / mean). Only the temporal mixer changes:
3D `r3d_18` → ImageNet ResNet-18 on each of 9 slices + learned attention pool.

Hypothesis: 2D ImageNet features plus all 9 slices can beat 3D Kinetics r3d_18
on locked weak-label OOF and move the pixel path toward 0.85.

Promote over S24 only if OOF clearly beats 0.801.
