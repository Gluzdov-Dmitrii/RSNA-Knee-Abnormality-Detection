# Wider slice coverage did not improve this model

P04 is rejected. On the same 875 validation studies, widening the slice-anchor
window from 35–65% to 10–90% lowered the final-epoch mean AUC by 0.02271.

| Fixed four-epoch comparison | Mean AUC |
|---|---:|
| P01 central window, 35–65% | 0.837165 |
| P04 wider window, 10–90% | 0.814450 |
| P04 minus P01 | -0.022715 |

Paired study bootstrap, 1,000 replicates, seed 2026: 95% interval for the
difference **[-0.030628, -0.014682]**. Six of twelve targets dropped by more
than 0.02 AUC. All predeclared promotion conditions failed. There was no full
refit or LB submission, and the continuation automation was paused.

The largest declines were medial meniscus (-0.06046), contusion (-0.04491),
and lateral meniscus (-0.04364). ACL (+0.00664) and fracture (+0.01149) improved,
but these post hoc class results do not justify a selected class-wise blend.

Both recipes use 224x224 uint8, six slots, nine slices per slot (three anchors
with three adjacent slices each), and crop 130 mm. Storage remains 11.12072 GiB
for all 4,407 studies. Same FOLDS_V1 fold 0, exact generic ImageNet initialization,
ordered training/validation IDs, labels, model source, optimizer and four-epoch
budget. The train set contains 3,474 studies; gold58 remain excluded. Both
comparisons run on A100 with the same torch/timm versions.

The changed window also changes the sampled-volume percentile limits. This
tests the entire window policy; it does not isolate anatomical coverage from
intensity scaling. The interval is conditional on one fold, seed and trained
pair, not a multi-seed training uncertainty estimate or LB prediction.

Keep the existing 35–65% default for this compact recipe. This experiment does
not establish the best window for other encoders, denser sampling or more epochs.
It supplies no evidence that more peripheral slices at a fixed nine-slice budget
improve the model. Do not ship or promote the wider-window cache as an upgrade.

Execution: smoke passed, checkpoint reload error 0. Pilot supervisor completed
in 1,044.61 seconds, peak training VRAM 6.042 GiB. Both owned GPU leases were
released after verified exit. The cache and experiment artifacts remain private.

Evidence files (workspace-relative):

- `artifacts/wide_window/comparison.json`: exact metrics, class deltas and bootstrap.
- `artifacts/wide_window/pilot/pilkwang/`: configuration, predictions, history and IDs.
- `artifacts/wide_window/REMOTE_READY.json`: complete input/environment preflight.
- `PROTOCOL.md`: gate frozen before training.

Candidate weights SHA256:
`671f5b9de8e8c88a1acb46f51050b00fbdbb25de4f2535fe0b1c214683d554ba`.
Candidate predictions SHA256:
`c8291a735cd9e91af3c05e45b4d65a5054129b470beeb793cf6b0fa9b8e88e3c`.
Cache SPEC SHA256:
`afce0b82784a444d5fc73bac80493c0e4ab2379a0ea4f765af4edb17eae37b46`.
