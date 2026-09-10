# S22-reg v1 — same ResNet-18 2.5D, stronger regularization

Not a new S-ID. Architecture remains torchvision ResNet-18 2.5D on PIXEL_CACHE_V1
(6ch slices 1/4/7, FOLDS_V1 seed 2026, LABEL_PILKWANG_V1).

S22 baseline overfit: train AUC ~0.999 vs val ~0.77; some folds picked epoch 2.
The 0.40 A0 overlay scored public 0.931 and is rejected. This run only tries to
raise the locked weak-label OOF of the own CNN.

Intended deltas vs `runs/20260908T1635Z-s22-resnet18-25d`:

- dropout 0.4 (was 0.2)
- AdamW weight_decay 1e-3 (was 1e-4)
- MixUp α=0.4 on train batches (labels and plane masks combined with max labeled/mask)
- backbone LR 0.1× head LR (`lr=1e-4`)
- epochs 24, patience 8 (was 16 / 5)

Locked unchanged: backbone, 6-channel plane heads, fusion Linear(3*512, 12),
batch 16, seed 2026, AMP, grad clip 1.0. No 90° rotate, no vertical flip.

No Kaggle submit in this packet.
