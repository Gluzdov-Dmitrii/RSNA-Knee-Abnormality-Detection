# One authorized standalone LB evaluation

User instruction2026-09-10: wait for the result and independently complete through
an LB submission. Exactly ONE competition submit attempt is authorized for this
new CoAtNet candidate. Kernel Save&Run and private weight upload are preparation.
Do not submit again after an error/ambiguous response without new authorization.

Completed fixed-epoch fold0 comparison: Pilkwang0.8371650703, median0.8266990467,
delta-0.0104660236. The median hypothesis is rejected for this screening protocol;
it is not evidence that every consensus-label method is inferior. Evaluation is
biased toward its Pilkwang reference. No LB feedback used to select this arm.

Select P01, fixed4epochs, then refit from the same generic initialization on all
4,349 non-gold training studies. This is P03 refit, not a new CV result. No best
epoch selection, blend optimization or tuning on gold58. Record gold predictions
only at the end and evaluate once as a small held-out audit. Do not relabel the
prior0.937 ensemble score as this model's result.

Freeze model/config/code/hashes, package weights and offline dependencies in a
private Kaggle dataset, and run a new private CPU-decode/GPU-inference notebook.
Verify predictions on current test DICOM, arbitrary studycount/order, schema,
runtime, compressed DICOM support and FP16/BF16 agreement. Submit the concrete
completed notebook version once, then query submissions read-only until terminal.
If score is weak, report it honestly; do not publish a high-score claim.
