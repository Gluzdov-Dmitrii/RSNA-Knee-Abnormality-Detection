# Cache recheck, frozen before execution

Date: 2026-09-08. Competition: rsna-knee-abnormality-detection.
No submission, test inference, public leaderboard selection, GPU, or notebook internet.
Inputs: official competition and pilkwang/rsna-knee-llm-labels only.

Preserve FOLDS_V1 (study-level iterative stratification, 5 folds, seed 2026,
SHA256 3086df3341333f44adb883292da386857c3230eaa2d501514ddf827a2da11b1a).
Select the identical 40 studies per fold by sequential NumPy default_rng(2026)
shuffles of each fold's original CSV order. Freeze and hash the selected UIDs.
Labels: report_labels_v2.csv, numeric scores >=0.5, 12 original targets.
Missing labels must fail rather than silently become negatives.

Independent verifier: spatial oriented-gradient encoder plus ridge regression
on binary targets (ranking scores, not calibrated probabilities). Compute edges
at each cache's actual resolution, pool eight unsigned orientation channels onto
an 8x8 grid, normalize histograms, retain three ordered depth bins. Add an 8x8
intensity grid per depth bin and six missing-slot flags. This is a fixed image
descriptor, not a pretrained foundation model or the old eight slot statistics.
Train-only StandardScaler; Ridge(alpha=1000, solver='cholesky'); identical settings
on every variant. No hyperparameter or epoch search.

Evaluate the original 15 configurations, with native decodes for every point.
This deliberately fixes the previous resolution family's two-stage resizing so
the 224 point measures the exact downloadable cache. Everything else retains
the old geometry, including percentiles across sampled cropped slices of each
series and scalar row PixelSpacing (anisotropy is counted and disclosed).
Cache selection stays at 224x9 crop130. Do not publish additional variants.

Report pooled out-of-fold macro AUC (equal mean of 12 target ranking AUCs),
per-target and per-fold AUC, 800 paired study bootstrap draws, difference CIs
against 224x9 crop130, and Holm-adjusted paired p-values across 14 comparisons.
The bootstrap does not include model-refitting or split uncertainty. Overlapping
intervals are not proof of equivalence. AUC within 0.01 is the prespecified
practical equivalence margin; assert equivalence only if the whole paired 95%
interval lies inside +/-0.01. A different cache is considered only with a
paired interval beyond +/-0.01 and Holm p<0.05; inspect geometry implications.
Run one label-permutation sanity control on the default, without model selection.

Success: completed verifier evidence, two separate human-readable figures,
rewritten public notebook with hidden runnable implementation, one validated
uint8 4407-study cache with hashes and a real Kaggle Dataset link. Use private
Dataset visibility because competition rule 2.4.b.1 restricts non-participant
access. No reports, report lexicon, labels, or train.csv in the cache Dataset.
Public notebook rebuilds pixels under /kaggle/temp, avoiding public saved pixels;
private materializer persists them under /kaggle/working/cache.
Validate synthetic geometry/failure/shard tests locally and all remote receipts.
