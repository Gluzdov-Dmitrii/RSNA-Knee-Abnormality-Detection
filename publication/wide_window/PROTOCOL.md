# P04: wider slice coverage, fixed storage and model

Authorized continuation, 2026-09-12: inspect current results and make the next
attempt. At most one new LB submission after a passed experiment and runtime
checks. Never retry a submitted/ambiguous request. Private development assets.

Competition: rsna-knee-abnormality-detection. Metric: mean twelve ROC AUCs.
Unit/group: StudyInstanceUID. FOLDS_V1 seed 2026; fold 0 screening, 875 validation
studies, 3,474 training studies, gold58 globally excluded. Same Pilkwang v2
reference binarized at 0.5; same supplied soft training scores. No gold tuning.
Gold58 were already inspected for P03; do not call them an untouched final test
for this later experiment. They remain excluded and only serve a frozen audit.
Official pages refreshed 2026-09-12. Deadline 2026-10-22 23:59 UTC. Latest read
shows one submission today; recheck limits immediately before any submit.

Hypothesis: the narrow central slice window discards useful peripheral anatomy.
Change only anchor window 0.35-0.65 to 0.10-0.90. Each of three anchors still
provides three physically adjacent slices: not nine evenly separated slices.
Six slots, 224 squared, nine slices/slot, crop 130 mm, uint8 remain fixed.
The existing sampled-volume percentile rule remains fixed; changing sampled
slices can change its numerical limits. This is a coverage-recipe comparison,
not a claim that all intensity values on overlapping slices remain identical.

Control: P01 fixed-final-epoch Pilkwang fold0 AUC 0.837165070275057. Reuse its
verified predictions, exact generic ImageNet initialization, seed, row ordering,
model, optimizer, batch4/accum2 and four epochs. New folds are never initialized
from knee-trained P03 or public knee weights. New cache cannot be derived from
the central-only cache: read official DICOM in a private CPU Kaggle notebook.

Predeclared promotion: macro delta >=0.005 and paired study-bootstrap 95% lower
bound >0 on the identical 875 rows; no more than one target may drop >0.02 AUC.
This is conditional single-fold screening, not full OOF proof. If it fails,
record rejection and do not consume LB quota merely to retry a weak hypothesis.
If it passes, refit the same fixed recipe on 4,349 non-gold studies, audit frozen
FP32/FP16 predictions and live DICOM, run the final private notebook, then make
one authorized standalone LB evaluation. No grid or LB-derived blend tuning.

Cache: about 11.12 GiB, all 4,407 studies, explicit masks, shards and SHA256 spec.
No reports, train.csv Report or raw DICOM exported; no public MRI dataset.
CPU conversion <=9h, up to4 workers. Prefer NSU existing immutable environment
and a single reserved A100 for <=2h chunks after inputs/storage are ready;
Kaggle T4 is a fallback subject to account-wide quota/concurrency. Do not reserve
a GPU while waiting for cache construction. Peak new working data budget <=16
GiB per selected staging pool, with archive duplication avoided by shard transfer.

Done: measured promotion/rejection with source/input hashes; if promoted, one
terminal Kaggle submit result. Never modify Grok source, processes or ledger.
