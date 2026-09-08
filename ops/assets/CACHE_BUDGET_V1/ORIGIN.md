# CACHE_BUDGET_V1 origin

Public source (Apache 2.0), not our 11 GiB cache:

- owner/slug: `stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache`
- URL: https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache
- kernel id: `130645920`, version 1, CPU, internet off, competition data only
- Runtime reported by the author: **64.2 min**, **11.13 GiB** written (`4407 × 6 × 9 × 224 × 224` uint8, 4 shards)

Geometry knobs we treat as the published 11 GiB design:

- 6 slots (default recovered plane × fluid/FS; public `train_series` flags are a coarser 3-bit alternative)
- `group=3` adjacent slices per 2.5D triplet, `n_group=3` → 9 slices/slot
- `RSNA_IMG=224`, `RSNA_CROP_MM=130`, `RSNA_WINDOW=0.35,0.65`
- Physical slice order (ImagePositionPatient × orientation normal), not filename
- Per-series 1st–99th percentile, then uint8

Credit inside that notebook: Pilkwang slot scheme, Karnakbayev 2.5D, Will physical millimetres.

This asset does **not** vendor his report lexicon. Labels for the quality probe are `LABEL_PILKWANG_V1`. Folds are `FOLDS_V1`.

Private probe kernel (CPU, internet off, no submit): `dmitriigluzdov/rsna-cache-budget-probe` **v1 COMPLETE** (~17 min compute). Receipt: `RECEIPT.json`. Curve: `cache_budget_curve.png`.

Denser methods notebook: `dmitriigluzdov/rsna-knee-on-a-storage-budget` **v2 COMPLETE** (private measurement, 15 points, 200 studies) and **v3 COMPLETE public** (same protocol, SSIM families split). Receipt: `RECEIPT_V2.json`. Curves: `cache_budget_v2_curve.png`, `cache_budget_v2_ssim.png`.
