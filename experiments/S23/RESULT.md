# S23 — ConvNeXt-Tiny 2.5D on PIXEL_CACHE_V1

- Status: `local_evaluated` (no Kaggle submit)
- Device: NSU A100 80 GB (`ngpu01`), queue `rsna-s23-cv-20260909T0635Z`
- Data: 11.12 GiB uint8 cache, FOLDS_V1, Pilkwang V1
- OOF macro AUC: **0.713** (weak-label proxy)
- Fold val AUC: 0.734 / 0.666 / 0.669 / 0.731 / 0.770
- vs S22 ResNet-18: **−0.055** (S22 remains the pixel-path control)

Same slice sampler and training recipe as S22. Hidden test still needs a scoring notebook that decodes test DICOM live.
