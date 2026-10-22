# TEAMA — variant A: five public models, fixed family weights

Goal (owner, 16.10): a reliable private-LB solution for team Kneerds; no public-LB tuning.
Baseline: team 079 = 0.957 (sub 57228091), byte-identical on visible rows to
`heliosli/this-is-best-public-dataset-new-version-0-9576`, so on private it ties with every copy.

## Members (all public, run unchanged in their own stages)
| Member | Source | Notes |
|---|---|---|
| Raptor CoAtNet-384 | `nartaa/rsna-knee-publication-swa-weights-20261007` | anatomical mirror TTA; script byte-identical to 079 |
| Raptor CoAtNet-224 | same dataset, 0.8 centre crop | plain + mirror, as in laimerbotrolenk V1 (0.957) |
| R7 compact ConvNeXt | `heliosli/rsna-knee-compact-r7cnxfsss` (bundle 8c7b666e) | no flip TTA (as published) |
| R6 compact ConvNeXt | `heliosli/rsna-knee-r6-frozen-fe055406-w1` (bundle fe055406) | flip TTA, as in 079 |
| S6 all-data ConvNeXt | `nartaa/rsna-knee-s6fall-weights` | embedded script unchanged |
Code/wheels: `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`.

## Weight rule (fixed 16.10 before any run)
Raptor family 0.5 (384 = 0.65, 224 = 0.35) + compact family 0.5 (R6, R7, S6 = 1/3 each); per-finding
percentile-rank blend. A failed member is dropped per study inside its family; > 3 % failures of any member
stops the notebook. R6 and S6 must agree with R7 (mean Spearman ≥ 0.6, min ≥ 0.3), the source's S6 gate.

## Files
- `source/` — pinned base notebooks (SHA in `build_notebook.py`).
- `build_notebook.py` — builds `kaggle/scored/` (REHEARSAL_N = 0) and `kaggle/rehearsal/` (N = 64 seeded train
  studies through a fake competition root; only timing and sanity, never an AUC claim).
- `test_fusion.py` — CPU tests of the blend cell (weights, per-study drop, gates).

## Steps
1. Rehearsal on Kaggle (T4x2): every stage completes; per-stage seconds → upper-bound hours for ~1300 studies.
   Gate: estimated total ≤ ~7 h, else reduce (first candidate: R6 without flip TTA) and re-rehearse.
2. Scored build: visible run on the 3 test studies, outputs checked, then one submission (daily free one).
3. Final pair (21.10): variant A + 079 (0.957).
