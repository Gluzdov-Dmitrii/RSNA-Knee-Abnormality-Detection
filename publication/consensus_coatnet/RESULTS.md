# Compact CoAtNet experiment

The supplied-score median did not improve this controlled comparison. On the
same 875 fold-0 studies, final-epoch AUC was 0.837165 for Pilkwang and 0.826699
for the median. The paired study-bootstrap 95% interval for the difference was
[-0.016471, -0.003872]. This interval is conditional on one fitted pair and does
not capture training-seed or fold variation. The reference labels are Pilkwang,
so source-label bias is a limitation.

The control's epoch AUC sequence was 0.787747, 0.827389, 0.839044, 0.837165.
The four-epoch endpoint was fixed before comparison; we did not switch to the
best observed epoch. The decreasing training loss alone does not establish that
longer training would improve generalization.

P03 refits that control from generic ImageNet initialization on 4,349 studies,
with gold58 excluded. Four epochs took 20.43 minutes on one reserved A100, with
6.047 GiB peak allocated VRAM. Final weights are 292,820,514 bytes; the private
package including offline dependencies is about 314 MB.

Real Kaggle T4 verification decoded 90 training studies in 196.25 seconds.
Pixels and masks matched the training cache exactly on all gold58. The frozen
model's gold58 macro AUC was 0.862483; this is a small held-out audit, not LB.
Independent CPU FP32 and T4 FP16 probabilities differed by at most 0.002205
on all 58 studies. A100 BF16 had a larger 0.017295 difference from T4 FP16;
that initial failed precision check and its resolution are retained.

Version 2 performs only live test inference, with no train-data QA in scoring.
Its visible predictions match version 1 byte for byte. One authorized submission
was accepted as 56142857 at 2026-09-10 11:04:11.587 UTC. It completed with
**Public LB 0.875**, observed at 11:44:51.720 UTC. That roughly 41-minute interval
includes scheduling/scoring overhead and is not a measured inference duration.
Notebook version 2 / scriptVersionId 348764510 was scored. No second submit.

The score is above the separate S22 ResNet-18 submission's 0.825, but below the
existing 0.937 ensemble. It does not support publishing this candidate as a
new high-score solution. Weights and notebook remain private development assets.

This experiment does not measure a new resolution/slice/crop curve, reproduce
the wider Raptor recipe, establish an improvement over the existing 0.937
ensemble, or justify a public high-score claim. It supplies a runnable compact
baseline and a negative result for one simple label-consensus hypothesis.
