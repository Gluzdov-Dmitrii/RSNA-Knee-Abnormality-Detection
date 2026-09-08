# S22 — ResNet-18 2.5D on PIXEL_CACHE_V1

- Status: `local_evaluated` (no Kaggle submit)
- Device: NSU Quadro RTX 6000 (`prepost`)
- Data: 11.12 GiB uint8 cache, FOLDS_V1, Pilkwang V1
- OOF macro AUC: **0.768** (weak-label proxy)
- Fold val AUC: 0.773 / 0.776 / 0.783 / 0.769 / 0.760
- Stronger targets: Synovitis 0.818, Effusion 0.810, Medial OA 0.806, ACL 0.801
- Weaker: MCL 0.716, Lateral Meniscus 0.716, Fracture 0.720

This is local evidence only. Hidden test still needs a scoring notebook that decodes test DICOM live.
