# ORIGIN — S22 A0 transformer-mix overlay

Pinned parent graph (same as S01):

- owner/slug: `renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid`
- URL: https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid
- kernel ID: `133076816`
- pinned strategy version: V1 / scriptVersionId `347142162`
- visible A0 `submission.csv` SHA-256: `11bb66f2bc7ca21d4282de3696f5ea5531f90c6c2958164b6ad6931811af2213`

S22 local training:

- NSU Quadro RTX 6000 run `20260908T1635Z-s22-resnet18-25d`
- PIXEL_CACHE_V1 224×9 crop 130 mm, FOLDS_V1 seed 2026, LABEL_PILKWANG_V1
- OOF macro AUC 0.768 (weak-label proxy, not hidden-test AUC)
- Export/verify: A100 `GPU-61c0078d-a4a6-37a2-3aba-0378e7794c46`, queue `rsna-s22-export-20260909T0735Z`, fp16 fold state_dicts

Inference recipe (locked S21–S25 `transformer_component_replacement_share` = 0.40):

- Live PIXEL_CACHE_V1 decoder (`build_cache.py` geometry: physical order, groups of 3, crop 130 mm, percentile 1–99, scipy zoom order=1)
- 5-fold ResNet-18 bag, mean sigmoid
- `tr_mix = 0.60 * A0_transformer_rank + 0.40 * rank(S22)`
- default outer `0.40 * tr_mix + 0.60 * raptor_rank`, then average-tie percentile rerank
- Medial Meniscus: `0.30 * tr_mix + 0.60 * raptor + 0.10 * bag` then the same rerank

Kernel: `dmitriigluzdov/rsna-s22-a0-rank-blend` (new slot; does not mutate `rsna-week1-a0-s01-s05`).
Internet off. `machine_shape` NvidiaTeslaT4. Submit `-f submission.csv` from this kernel version.

Attached inputs: A0 Renta datasets/model plus private `dmitriigluzdov/rsna-s22-resnet18-25d-folds`.
S23 lost the pixel-path control (OOF 0.713). S24/S25 are not in this kernel.
