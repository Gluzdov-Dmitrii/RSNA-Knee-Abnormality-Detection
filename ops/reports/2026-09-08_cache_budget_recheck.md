# Spatial image probe and full cache

Private source run: [rsna-knee-private-cache-verifier v1](https://www.kaggle.com/code/dmitriigluzdov/rsna-knee-private-cache-verifier).
COMPLETE. CPU, internet off, no GPU, no competition submission. Total code runtime
5,151 seconds. Source/config hashes are recorded in
`experiments/cache_budget/recheck/RUN.json`.

## Locked protocol

Same 200 studies as the previous probe, 40 per FOLDS_V1 fold, seed 2026. Folds and
Pilkwang report_labels_v2.csv match their original SHA-256 values. Threshold >=0.5.
Subset CSV SHA-256: `207359e564979a8ee4279cd9aae57458815968a8df5f27a6c74b993cb8b3baeb`.
977 of 1,200 slots available, matching the prior probe.

Original verifier: spatial oriented-gradient histograms and image grids, with
three ordered depth bins; 10,374 features; train-fold-only scaling and ridge
regression (alpha 1000) on binary targets. No hyperparameter search. It is a fixed
image descriptor, not a learned image encoder or foundation model.
Label-permutation control macro AUC: 0.485663.

Each of the 15 cache settings now uses native decoding, correcting the prior
resolution family's second resize of an already quantized 336² parent. The actual
download matches the measured native 224² setting. This is the only intended
geometry-comparison correction; the subset, folds, threshold, windows, public
slot scheme, sampled-series percentile scope, and crop policy remain fixed.

## Results

Pooled out-of-fold macro AUC. Differences use 800 paired study bootstrap draws;
these do not include retraining or alternative-fold uncertainty. Holm correction
covers the 14 comparisons against 224² × 9 crop 130. The prespecified practical
equivalence margin is ±0.01 AUC.

| Setting | GiB | AUC | Paired difference vs primary (95% interval) |
|---|---:|---:|---:|
| 160² × 9, crop 130 | 5.674 | 0.647944 | −0.012326 [−0.019075, −0.005048] |
| 192² × 9, crop 130 | 8.170 | 0.657821 | −0.002449 [−0.006407, +0.002104] |
| **224² × 9, crop 130** | **11.121** | **0.660270** | reference |
| 336² × 9, crop 130 | 25.022 | 0.665026 | +0.004756 [−0.002976, +0.013093] |
| 224² × 3, crop 130 | 3.707 | 0.645240 | −0.015030 [−0.030486, +0.000547] |
| 224² × 12, crop 130 | 14.828 | 0.660486 | +0.000216 [−0.005103, +0.005262] |
| 224² × 9, crop 110 | 11.121 | 0.669855 | +0.009585 [−0.007915, +0.025968] |
| 224² × 9, crop 160 | 11.121 | 0.642062 | −0.018208 [−0.037540, +0.001544] |

Primary AUC 95% interval: [0.633447, 0.688602].

**Decision:** retain 224² × 9, crop 130. Extra resolution does not show a clear
gain worth another 13.90 GiB in this probe. This is a practical storage choice,
not proof of universal optimality or strict equivalence to 336².

**Corrections to earlier claims:** 160² is lower here (Holm-adjusted p=0.034956),
so no "small but equivalent" extra is published. Three slices score lower, but
their paired interval crosses zero; the new model does not establish the earlier
"only real quality cliff" assertion. Crop 160 is nominally worse, with 617/977
series skipping the physical crop; there is no clear reason to switch from 130.

## Materialized cache

All 4,407 official training studies, 35 dense uint8 NumPy shards, shape per study
(6,9,224,224). Pixel payload: 11.120721817 GiB. No 336² or three-slice cache is
shipped. Study index, slot mask, small SPEC.json, audit and licence notices only;
no reports, report lexicon, train.csv or labels in the Dataset.

21,334 selected series used physical geometry ordering; no fallback ordering was
needed. 5,108 absent public slots are zero-filled and explicitly masked. Cropping
applied to 20,976 series and skipped for short FOV in 358. No missing or anisotropic
spacing was recorded. Every remote shard passed shape, dtype and hash checks.

SPEC SHA-256: `043485b6c0ea543a5f4061ce15b6fef868bb28aebc2af6d124adfd940c9aaf50`.
Data remains subject to competition rules and MIRA. Geometry attribution and the
Apache-2.0 licence are included; that code licence does not grant MRI redistribution.

Full-download validation passed: all shard SHA-256 values, dtype/shape, official
study UID coverage, masks and present-slot signal checked. All 15 AUC values were
independently recomputed from saved held-out predictions; original subset, folds
and thresholded labels match exactly.

[Private Dataset version 1](https://www.kaggle.com/datasets/dmitriigluzdov/rsna-knee-uint8-224-9-c130)
is READY, ID 11946093, verified private with exactly 41 remote files and matching
byte sizes (11,941,204,980 bytes total). No reports or label files are included.

[Public notebook version 5](https://www.kaggle.com/code/dmitriigluzdov/knee-mri-in-11-gib)
retains kernel ID 133521917. Kaggle changed the slug when the title changed.
The completed private CPU run supplies the actual figures, embedded as Markdown
attachments because Quick Save's public renderer omitted supplied code outputs.
Kaggle-specific `_kg_hide-input` supplements standard Jupyter source-hiding
metadata; the live page shows four collapsed code cells, both figures, and the
private Dataset link. Public output has zero files, so no MRI shards are exposed.
The source retains precisely the original two inputs, CPU and internet off.

Hidden Save & Run code rebuilds both verifier and full cache, placing rebuilt MRI
in /kaggle/temp to avoid public saved pixels. The public Quick Save is not another
full execution; the underlying private full execution completed in 5,151 seconds.

## Reader clarification, version 6

Clarified that 4,407 counts training studies, not unique people or train plus test.
Added a short comparison of slice selection, bilinear resizing and physical crop;
uint8 conversion is fixed across the sweep. Crop numbers now explicitly mean the
side length retained, with the crop-skipping exception stated.

Explained the context/detail tradeoff at fixed 224 pixels: a 10 mm structure is
about 20 pixels wide for an applied 110 mm crop and 14 for 160 mm. This is a scale
illustration, not a proven explanation for the measured AUC. The highest 110 mm
mean differs from 130 mm by +0.009585 with paired CI [-0.007915, +0.025968], so it
is not a confirmed winner. The prespecified cache stays unchanged.

Replaced the plateau claim with diminishing returns and explicit uncertainty:
224 to 336 costs 13.90 GiB for observed +0.004756 AUC, paired CI crossing zero.
Both figures were relabelled and regenerated from the same saved measurements.
AST comparison confirmed only the plotting function changed in pipeline.py;
no model, data, folds, cache conversion or metrics changed. No new run or points.
