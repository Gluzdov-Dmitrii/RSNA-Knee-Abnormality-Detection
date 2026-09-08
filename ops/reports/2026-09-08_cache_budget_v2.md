# Cache-budget v2 — denser curve

Private CPU kernel [`dmitriigluzdov/rsna-knee-on-a-storage-budget`](https://www.kaggle.com/code/dmitriigluzdov/rsna-knee-on-a-storage-budget) **v2 COMPLETE** (~27 min). Internet off, no GPU, **not submitted**. 200 studies (40 / `FOLDS_V1` fold, seed 2026). 977/1200 public slots decoded (81.4%). Publication notebook is the same source; v1 of this slug was a `main()` order bug.

This is **not** public LB and not DINOv2. Error bars are 800 study-level bootstraps of the OOF GBDT.

## Resolution at 9 slices, crop 130 mm

Downsampled from one 336² decode, so slices and crop are fixed.

| id | GiB | AUC | 95% CI | SSIM vs 336 |
| --- | ---: | ---: | --- | ---: |
| res_128x9 | 3.63 | 0.628 | 0.598–0.660 | 0.933 |
| res_160x9 | 5.67 | 0.633 | 0.602–0.663 | 0.977 |
| res_192x9 | 8.17 | 0.628 | 0.598–0.658 | 1.000 |
| res_224x9 | 11.12 | 0.623 | 0.591–0.654 | 0.991 |
| res_256x9 | 14.53 | 0.622 | 0.592–0.656 | 0.989 |
| res_288x9 | 18.38 | 0.621 | 0.590–0.653 | 0.995 |
| res_336x9 | 25.02 | 0.619 | 0.591–0.646 | 1.000 |

The resolution curve is **flat**. 128²–336² sit inside one another's CIs. 336 is wasted GiB for this probe.

## Slice count at 224², crop 130 mm

| slices | GiB | AUC | 95% CI |
| ---: | ---: | ---: | --- |
| 3 | 3.71 | 0.589 | 0.557–0.623 |
| 6 | 7.41 | 0.625 | 0.593–0.658 |
| 9 | 11.12 | 0.623 | 0.591–0.654 |
| 12 | 14.83 | 0.637 | 0.608–0.669 |
| 15 | 18.54 | 0.619 | 0.590–0.649 |

The cliff is **3 slices**. 6–12 is the plateau. 15 does not pay.

Tiny 128×3 / 160×3 (~1.2–1.9 GiB) match the 3-slice 224 point (~0.59), not the 9-slice 128 point (~0.63). Coverage beats extra pixels.

## Crop at 11.12 GiB (224² × 9)

| crop mm | AUC | 95% CI | SSIM vs 130 mm 336 |
| ---: | ---: | --- | ---: |
| 110 | 0.611 | 0.575–0.648 | 0.186 |
| 130 | 0.623 | 0.591–0.654 | 0.991 |
| 160 | 0.637 | 0.606–0.666 | 0.200 |

Crop is not a size knob. GBDT still likes 160 mm (larger FOV in the intensity stats). CIs overlap 130 vs 160. For a DINO token, pitch is `14 × crop_mm / img`; keep **130 mm** for encoder caches until an encoder probe says otherwise.

## What to materialise

**224² × 9, crop 130 mm, ≈ 11.1 GiB** remains the default: it sits on both plateaus, matches Steven's published design, and does not spend 25 GiB. Cheaper 160² × 9 (~5.7 GiB) is indistinguishable here; use it only if disk is the constraint. Do not publish a Dataset of derived MRI.

Plots: `ops/assets/CACHE_BUDGET_V1/cache_budget_v2_curve.png`, `cache_budget_v2_ssim.png`.
