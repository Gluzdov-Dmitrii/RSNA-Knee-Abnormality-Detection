# Cache-budget probe — Kaggle v1 COMPLETE

Private CPU kernel [`dmitriigluzdov/rsna-cache-budget-probe`](https://www.kaggle.com/code/dmitriigluzdov/rsna-cache-budget-probe) v1. Internet off, no GPU, **not submitted**. Wall ~17 min after a ~7 min import lag. 160 studies (32/fold, `FOLDS_V1` seed 2026), Pilkwang V1 ≥ 0.5. Public `train_series` slots: 784/960 decoded (81.7%).

This is **not** public LB and not a DINOv2 ranking.

## Curve (GBDT slot stats)

| id | GiB | GBDT macro AUC | tiny CNN | SSIM vs 336²×9 c130 |
| --- | ---: | ---: | ---: | ---: |
| tiny_160x3 | 1.891 | 0.572 | 0.469 | 0.504 |
| steven_224x9_c130 | 11.121 | 0.604 | 0.493 | 0.994 |
| steven_224x9_c160 | 11.121 | **0.639** | 0.486 | 0.199 |
| wide_224x15 | 18.535 | 0.620 | 0.493 | 0.421 |
| hi_336x9_c130 | 25.022 | 0.607 | 0.492 | 1.000 |

Plot: `ops/assets/CACHE_BUDGET_V1/cache_budget_curve.png`.

## What we will materialise

- **224² × 9 slices**, full-corpus ≈ **11.1 GiB**. 336² (+13.9 GiB) is a wash vs 224 at crop 130 (0.607 vs 0.604, SSIM 0.994). Tiny 160×3 loses ~0.03 AUC.
- **Do not** spend 25 GiB on 336 for this cache.
- Crop **130 vs 160** is **not** settled here. GBDT intensity stats like 160; Steven’s DINO mm/token argument prefers 130, and SSIM vs the 130 mm reference collapses at 160 (0.20). Keep 130 for any DINO/A0-tap cache until an encoder probe says otherwise.
- Extra slices (15 vs 9) help GBDT a little (+0.016 at crop 130) and cost 7.4 GiB. Not enough to default to 18.5 GiB.

Tiny CNN stayed ~0.49 with 6 epochs on 160 studies. Ignore it for selection.

## Next

Kaggle CPU conversion of the full 4407 studies at the chosen 224×9 geometry, still private, still not a Dataset publish, still not a submit.
