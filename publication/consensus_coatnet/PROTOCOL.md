# P01/P02: reproducible CoAtNet with consensus labels

Owner: Codex, authorized 2026-09-10. This lane is independent of Grok S22–S25.
Only `publication/consensus_coatnet/` is writable source for this lane. Shared
folds, cache, labels and installed environments are read-only. No changes to
Grok processes, launchers, registry or reservation. A Codex helper audits source;
no interface to launch an external Grok assistant was available.

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
Official metric: mean ROC AUC over twelve findings. Local reference: Pilkwang
report_labels_v2 scores >= 0.5, finite entries only, unchanged FOLDS_V1, seed2026.
Study is the independent unit, not series/slice. Gold labels, if evaluated later,
must be excluded from training of their predicting fold; never tune on gold58.
No submissions authorized in this work. Check current deadline/quota and schema
before requesting permission for a concrete, verified submission.

Hypothesis: replacing one source of soft targets with the median of three raw
sources improves an otherwise identical model. P01=Pilkwang; P02=median of
supplied Pilkwang + Steven v2 + Lixin scores (finite sources; no confidence weighting).
Numeric0.5 is retained as uncertainty, not converted to missing. Evaluation
always uses the same original Pilkwang scores, never each arm's training labels.
Its teacher bias is a declared limitation. Steven v2 has a Synovitis repair and
0.5 means unmentioned; Pilkwang UNK may be0.28. This is explicitly a median of
supplied scores, not three independent annotators or the unknown-excluding S12.
Both arms use the same4,406 eligible UIDs minus the globally held-out gold58.
No raw reports or label tables ship. Gold is not read for pilot model selection.

Initial geometry is the existing private PIXEL_CACHE_V1: six slots, three groups
of three adjacent slices per slot, uint8,224²,crop130mm,window.35–.65.18 windows
stay within series/group boundaries. Missing windows are masked out of attention.
This is a compact adaptation of Raptor, NOT reproduction of its wider384 recipe
or its0.924 score. Full train and live-DICOM inference share geometry and norm.

Backbone: timm coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k, adapted to224 with generic
ImageNet initialization; no knee-trained initialization. Per-finding attention
follows the Apache2.0 Raptor approach. This exact pretrained CoAtNet uses mean/std
(.5,.5,.5)/(.5,.5,.5), shared by train/inference and asserted against timm config.
Do not substitute conventional ResNet ImageNet normalization.
No augmentation and no MixUp in first A/B: this avoids changing missing-target
handling or adding another variable. All18 windows used in train and validation.

Screening: fold0, fixed four epochs per arm, same seed/data order/init, final
epoch primary metric (not best-of-validation). Pilot first verifies forward,
backward, loss/mask, checkpoint reload and speed; pilot is not quality evidence.
Keep latest resumable checkpoint + final inference weights. Promotion requires
finite valid predictions, no leakage, positive macrodelta without broad class
regression. Screening is directional, not proof. Only then five folds and a
second seed, paired study-bootstrap, fold/class deltas, runtime on Kaggle T4.
No LB blend tuning. Stop if checks fail or measured cost exceeds the declared
bounded allocation; checkpoint and reassess instead of silently expanding.

Resource plan: one reserved GPU,8CPU threads,48GiB host RAM,<=6GiB new storage,
<=120min per bounded stage; prefer compatible smaller GPU unless occupied or
large backbone activations justify A100. Existing11GiB cache is reused, not copied.
Each remote run has its own code snapshot, output directory and source/input
hashes. No modifications to existing environment. Queue on canonical nsu-quadro.

Release goal: one model dataset (weights/config/credits/rebuild), one offline
live-test inference notebook, one isolated improvement and a measured score/cost.
Public release and LB claims wait for verified results. No MRI cache publishing.
