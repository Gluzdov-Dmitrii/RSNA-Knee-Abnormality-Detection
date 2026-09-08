# How we measure “quality vs compression”

570 GB → 11 GiB is **not** lossless. It is series selection, physical crop, slice window, and 224² resize. Quality is therefore “how much diagnostic signal a **fixed** simple model still sees”, not PSNR to raw DICOM (that would punish every resize by construction).

## Primary metric

Locked-fold weak-label OOF **macro ROC-AUC** of one probe family, identical across cache variants:

- Targets: Pilkwang V1 scores binarized at 0.5 (same as `FOLDS_V1` stratify)
- Split: `FOLDS_V1` fold ids, restricted to a **fixed** study subset (seed 2026)
- Model A (fast, CPU): `HistGradientBoostingClassifier` on slot summary features (mean, std, p10/p90, center-third mean, adjacent-slice MAD, missing flag)
- Model B (simple DL): tiny 2.5D CNN on 6 slots × 3 central slices downsampled to 64², 8 epochs, no ImageNet pretrain

A cache that is smaller but within ~noise of Steven’s 11 GiB point on **both** probes is the one we later materialise on Kaggle CPU for all 4407 studies.

## Probe kernel

Private CPU notebook `dmitriigluzdov/rsna-cache-budget-probe` (internet off, no GPU, no submit):

| id | img | slices | crop_mm | window | full-corpus GiB |
| --- | --- | --- | --- | --- | --- |
| `tiny_160x3` | 160 | 3 | 130 | 0.40,0.60 | 2.11 |
| `steven_224x9_c130` | 224 | 9 | 130 | 0.35,0.65 | 11.13 |
| `steven_224x9_c160` | 224 | 9 | 160 | 0.35,0.65 | 11.13 |
| `wide_224x15` | 224 | 15 | 130 | 0.10,0.90 | 18.54 |
| `hi_336x9_c130` | 336 | 9 | 130 | 0.35,0.65 | 25.03 |

Subset is 32 studies per `FOLDS_V1` fold (160 total), seed 2026. Slot scheme is public `train_series` plane × `Fluid_Sensitive` (not Steven’s recovered FS/T1 mapping). The probe does **not** write a full-corpus cache.

### v1 measured (kernel COMPLETE, 2026-09-08)

| id | GiB | GBDT | tiny CNN | SSIM vs 336 |
| --- | ---: | ---: | ---: | ---: |
| tiny_160x3 | 1.891 | 0.572 | 0.469 | 0.504 |
| steven_224x9_c130 | 11.121 | 0.604 | 0.493 | 0.994 |
| steven_224x9_c160 | 11.121 | 0.639 | 0.486 | 0.199 |
| wide_224x15 | 18.535 | 0.620 | 0.493 | 0.421 |
| hi_336x9_c130 | 25.022 | 0.607 | 0.492 | 1.000 |

Tiny CNN did not train (≈0.49). Decision from GBDT + SSIM: materialise **224² × 9**, not 336 and not 160×3. Crop 130 vs 160 stays open for a DINO probe; default crop for encoder work remains 130 mm. Details: `ops/reports/2026-09-08_cache_budget.md`.

## Secondary metric

Fidelity to the **densest variant in this sweep** (336² × 9 slices, crop 130 mm), after resampling spatial and slice axes: mean SSIM. This is information loss vs a richer cache, not vs 570 GB.

## What we will not claim

- Public LB from this subset
- That 11 GiB is optimal before the curve is measured
- That GBDT AUC equals a DINOv2 head (S11 still needs the frozen A0 tap)

## Publication

Do not publish a Dataset of derived MRI without a separate permission check (MIRA + competition data security). A **private** CPU kernel for this probe is allowed; a public notebook is a later decision.
