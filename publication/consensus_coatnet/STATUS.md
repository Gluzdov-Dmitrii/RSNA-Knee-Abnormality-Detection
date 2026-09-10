# Execution handoff — 2026-09-10

## Latest: COMPLETE, Public LB 0.875

The earlier run description below is retained as execution history. Both arms
completed. Pilkwang AUC 0.837165; median 0.826699; delta -0.010466 on 875 fold-0
studies. Paired bootstrap 95% interval [-0.016471, -0.003872], conditional on this
single fitted pair. No full OOF claim. Median hypothesis rejected for this screen.

P03 refit uses the selected Pilkwang recipe, four fixed epochs from generic
initialization, 4,349 non-gold studies. Training completed in 1,225.69 seconds,
6.047 GiB peak VRAM. Its own GPU lease was released; Grok was untouched.
Weights SHA256: f658b89db67997c3c0975a79298f1cbb962c30483ba9a7dec9bbffc4926c6213.
Full outputs: artifacts/consensus_coatnet/full/pilkwang/.

Seventeen local tests pass, including stale sample / hidden test.csv remount.
Private weights dataset: dmitriigluzdov/rsna-knee-compact-coatnet.
Private kernel: dmitriigluzdov/knee-mri-compact-coatnet, version 2 COMPLETE.
Version 1 passed 90-study real DICOM QA and 58/58 exact pixel hashes. A100 BF16
versus T4 FP16 initially failed the 0.015 tolerance (0.017295); an independent
CPU FP32 reference on all 58 studies established T4 max error only 0.002205.
Version 2 skips the completed train-data QA; test predictions match v1 exactly.

ONE authorized submit occurred at 2026-09-10 11:04:11.587 UTC, ref 56142857.
Description: P03 compact CoAtNet Pilkwang full4349 ep4 f658b89d.
Immediate read-only query confirmed PENDING. Terminal COMPLETE / Public LB 0.875
was observed at 2026-09-10 11:44:51.720 UTC; the read-only monitor exited normally.
No second submission occurred. Notebook and weights remain private. This beats
the separate S22 ResNet-18 score 0.825 but not the existing ensemble score 0.937.
Do not present it as an improvement over that ensemble. See RESULTS.md and
SUBMISSION_READINESS.md. Artifact receipt: artifacts/consensus_coatnet/submission_status.json.

## Earlier execution history

Lane: Codex publication/consensus_coatnet, isolated from Grok's S22–S25.
Remote project: /home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
Run: runs/20260910T0913Z-codex-consensus/verified_pair
Frozen code: code/consensus-coatnet-v3 (never modify while running).
Lease ID: rsna-codex-consensus-verified_pair-20260910.
Device: A100 index1, GPU-04efb7bd-1f45-38cd-4a13-c79b6aeaa002.
SupervisorPID67828, Linux process start identity425204457.
Grok's GPU0 lease/process were not modified.

## Passed before training

- All35 cache shard hashes, index/mask, locked folds and label source hashes.
- Real pretrained CoAtNet CPU forward at224²;73,136,096 parameters.
-16 CPU tests: labels, missing values, per-slot triplets, masked attention/loss,
  live synthetic DICOM for arbitrary study counts/order, and exact geometry
  equivalence to the existing cache pipeline. Model mocked in those unit tests.
- Correct-normalization GPU smoke:8steps,4.51s measured stage,5.43GiB peak,
  checkpoint reload max error0. This was a wiring check, not quality evidence.
- Exact timm pretrained config checked: mean/std(.5,.5,.5)/(.5,.5,.5).
  Correct-normalization initialization SHA256:
  a7d32ac8c3f006d2b60d1837a90a27c1cf6ed7ad80bdef018c8de52decc4a8c3.

Both arms use4,349 eligible non-gold studies across folds. The one study absent
from Pilkwang is also among gold58, so exclusions overlap. Fold0 validation875,
training3,474. Both arms have exactly the same rows, seed and initial weights.

## Current bounded run

verified_pair first repeats8 smoke steps with the correct normalization, then
trains Pilkwang and supplied-score median sequentially for4epochs each, fold0.
Batch4, accumulation2, no augmentation/MixUp, workers0. Final epoch is primary.
The supervisor stops on failure and has a110min cap. No real gold labels are
read for model selection. Expectation from the earlier wiring run is roughly
4–5min per epoch including validation, not a guaranteed completion time.

Local monitor_existing.py watches only this run, refreshes its reservation,
fetches reports/predictions after completion, and releases its verified-ended
lease. It never launches follow-ups or submits. Live state and outputs:
artifacts/consensus_coatnet/monitor_verified_pair/status.json and
artifacts/consensus_coatnet/verified_pair/. Token files are ignored/private.

## Recorded failures, not quality experiments

smoke: DataLoader IPC failed because the canonical TMPDIR exceeded the Unix
socket path limit. Own trainPID67061 and its4 workers were stopped, supervisor
exited, lease released. Fixed by workers0 on this memory-mapped cache.

pair/v2: stopped during the first epoch after discovering that conventional
ResNet mean/std differed from this CoAtNet's pretrained config. No A/B conclusion
was drawn. Its own trainPID67394 stopped; monitor released its lease. v3 uses
fresh shared initialization with the correct buffers. Earlier logs/source hashes
remain as diagnostics. Grok's processes were never stopped.

Cleanup: after checking the active command and resolved paths, removed only
prepared/initialization.pt (292,864,790bytes) and smoke2/pilkwang/smoke.pt
(292,790,538bytes). Both were obsolete own checkpoints; input hashes, source and
smoke results were preserved locally. Reclaimed585,655,328bytes. The correct
initialization, generic timm cache and running/final checkpoints remain pinned
to this comparison and its forthcoming validation; review after comparison.

## Next gates

Read comparison.json after BOTH final-epoch prediction files pass UID/schema
checks. One-fold screening is not full OOF and not LB. If promising, repeat on
all locked folds/another seed before making a quality claim. Compare source-label
bias against a separately untouched gold evaluation, without tuning on gold58.
The compact224×9 recipe does not reproduce Raptor's wider384 setup or0.924 score.

inference_draft.ipynb is a local development draft with an explicit unset model
package path. Do not publish it as ready-to-run yet. Final Kaggle package still
needs selected weights, offline dependencies, actual DICOM runtime/remount QA,
and actual authorized leaderboard evaluation. No submissions or dataset/notebook
publication were performed in this lane.
