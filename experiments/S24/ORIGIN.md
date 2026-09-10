# S24 — r3d_18 9-slice 3D on PIXEL_CACHE_V1

Not a Kaggle submit until locked OOF beats S22 0.768 by a clear margin.

- torchvision `r3d_18` Kinetics, three plane heads
- Input (3, 3, 9, 224, 224): fluid / struct / mean, T=9 from PIXEL_CACHE_V1
- Same FOLDS_V1 seed 2026, LABEL_PILKWANG_V1
- Mild MixUp 0.2, dropout 0.25, backbone LR 0.3×
