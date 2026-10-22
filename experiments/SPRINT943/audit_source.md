# Speedy Raptors latest: independent source audit

Status: **complete static source audit; no GPU/Kaggle execution**. 2026-09-22.
Update during implementation: root obtained the exact **V34 / scriptVersionId 351665011** export,
SHA256 `a025fcd5aa25373ac380f60746ba5ca17e70e97f714fdc17e01ce414ec2163b5`, at
`experiments/SPRINT943/source_v34/bend-the-knee-to-speedy-raptors-the-original.ipynb`.
Root verified the source page displays Public Score **0.943** for V34. This identifies
the source, but is not yet our independent end-to-end reproduction.

Compared with the latest source audited below, only code cell 22 differs: V34 adds
Repair-v1 as a **fourth** CoAt family (epochs [12,7,11]) and uses **0.60** as default
outer CoAt/Raptor weight instead of 0.65. Other target overrides are unchanged.
The Repair-v1 files reside under `repairv1/` in the existing D4 dataset, according
to root's dataset-file inspection; no extra dataset slug is required. V34 author
outputs record all four families. Therefore V34 hardening must require exactly
four families and retain its successful four-way probability mean and weights.

`harden_source.py` now provides `harden_notebook(nb) -> deepcopy`, with exact
source-anchor assertions, failure gates, truthful release receipts and no changes
to successful prediction arithmetic. `test_hardening.py` runs the CPU gate tests.
The audit details below describe the earlier latest source unless explicitly stated.
Notebook: `tmp/audit_2026-09-22/speedy_raptors/bend-the-knee-to-speedy-raptors-the-original.ipynb`.
SHA256 verified: `60bbe1af1ddca2a304c954d08806826de4e66b285b015d970925d005f4831db4`.
Cell numbers below are zero-based JSON indices (0 is the opening markdown).
Line numbers refer to adjacent `source.txt`, read only for this audit.

## Decision for a one-hour engineering sprint

Use this complete, pinned public graph as the most ambitious available candidate, with minimal failure-handling corrections, useful section documentation, receipts, and one offline T4x2 smoke. Do not present it as a reproduced 0.943: the reviewed notebook is latest, not confirmed scored V34. Preserve the independently reproduced Rice 0.941 artifact as the result-backed fallback. Do not add weak local models or tune more target weights on the public leaderboard.

The source's metadata explicitly describes `v559c` as “data-driven aborts relaxed to flagged degradations” and has `t4_end_to_end_executed_here: false`. There are **no saved outputs**. The static comments “complete graph” and “no partial fallback” are contradicted by several actual code paths.

## Findings requiring changes before release

| Priority | Evidence | Consequence / bounded correction |
|---|---|---|
| P0 | Cell 22 `_coat_substitute`, lines 4256–4336; output acceptance 4357–4387 | Every ResGated, Global96, or D4 child/output failure is caught and that branch is omitted. All three can fail and return 0; cell 23 still publishes. Re-raise child/output failures and require the exact member set `resgated_top3, global96_top3, d4_swa3`. |
| P0 | Cell 22 `_ke_run_raptor_arms`, lines 4044–4083, 4102–4111 | A study failure leaves NaN predictions that are replaced by cohort means or 0.5. Even all-study model failure can produce a valid-looking submission. Re-raise errors and require finite raw predictions from every arm; preserve the existing FP32 retry before rejecting. |
| P0 | Cell 13 `infer_from_package.bank`, lines 2081–2088; cell 19 `_speed_a5_micro`, 2842–2845; cell 21 `_rad_validate`, 3354–3357 | Nonfinite DINO/A5 predictions become 0.5; invalid Rad outputs are repaired. Finite final ranks cannot detect the loss. Reject persistent nonfinite/out-of-range predictions after the already implemented precision retry. |
| P0 | Cell 19 A5 decode loop, lines 2964–2970 | Exceptions from `build_study` become “A5-absent”; eligibility is based on decoded masks, contradicting the receipt's `model_failure_fallback_allowed: false`. Reject acquired-data preparation failures. Actual absence of matching acquisition slots may retain documented parent-only handling. |
| P1 | Cell 1 `_asset_write_coat_runtime` lines 224–230; `_asset_patch_d4` lines 281–284 | Dense sampling errors silently switch back to another preprocessing recipe. D4 input receipts record `slots:[{fallback:...}]`, but no gate rejects this. Reject dense-preparation exceptions or explicitly gate zero fallback records; do not count reference-grid predictions as the intended dense graph. |
| P1 | Cell 22 lines 4253–4254, 4309–4310 | ResGated/Global96 `fallback_studies` are only logged. Require key presence and zero; check child shard/preparation failure fields where supplied. D4 already checks these more strictly. |
| P1 | Cell 22 `_coat_member_probability` and family reduction, lines 4393–4434 | Missing/bad NPZ silently switches from rank(mean probabilities) to mean ranks, or a single member. Require all three probability payloads, unique exact UIDs, 12 target columns, expected checkpoint count and finite probabilities. Require one explicit reduction contract. |
| P1 | Cell 23 lines 4501–4505 | Incomplete Raptor preparation identity set merely logs an event. Final `COMPLETE` is static and claims CoAt branches without checking them. `family_mix:[.5,.5]` is stale for three members, Global96 omitted. Raise on mismatches; derive final receipt from verified component receipts and actual three-member weights. |
| P1 | Cells 5/7/9 `probe`, `read_slot`, `build_cache`; cell 17 A5 reader; cell 22 Raptor reader | Selected acquired files can fail or disappear and be treated as absent. Gate serious events (`slot_all_decode_failed`, `slot_read_error`, `a5_slot_no_files`, `a5_study_failed`, `raptor_acquisition_empty`) distinctly from metadata-proven absence. Predefined partial-slice recovery can be logged, but must not hide full component failure. |

Do not reject all empty slot masks indiscriminately: this graph has specialized layouts, so a study genuinely missing a particular acquisition can be legitimate. Verify matching metadata existed before deciding preparation failed.

## Actual inference graph

1. **DINOv2-S:** 20 competition members, strict checkpoint loading and per-member numerical fingerprints, first six frozen blocks shared only after SHA check and path-equivalence test. Per-target window pooling: max for fracture, contusion, both menisci and Baker's; top2 for ACL/MCL; mean elsewhere. `_combine(public_frontier_members)` averages each member's percentile ranks. The declared legacy soft-pool functions and `soft_pred` calculations are **not consumed by this final combination**; do not document them as another active ensemble. Pixel geometry comes from the weight manifest via `adopt_config_globals`, not the top-level demonstration defaults.
2. **A5:** five `m_f0.pt`…`m_f4.pt` models with matched cfg. Six plane×fat-suppression slots, 16 slices, 336 px, 130 mm, central 12–88%, per-slice intensity. Five fold rankings are averaged and reranked. Available rows mix 0.48 DINO + 0.52 A5; truly unavailable A5 rows retain the parent. Duplicate model replicas on T4x2 preserve microbatches of eight.
3. **RadImageNet ResNet50:** five E10 reference heads on 3 FS planes/full frame; five E13 heads on 4 E13 slots/130 mm; the same E13 heads on E11 layout/130 mm. Eight slices, 224 px. Reference+E13 ranks mix 0.5/0.5, then enter parent with alpha .55 except Baker's/Fracture. Another E13/E11 ranking contributes .20. The explicit cross-layout pass is intentional input to the saved calibrator and should not be independently “fixed.”
4. **Saved Rad calibrator:** decoded inline payload contains a 12×88 coefficient matrix (209 nonzero coefficients), intercept, scaling, 4 pathology groups, and 12 acquisition-count columns. Its .4 correction gate covers ACL, medial/lateral/PF OA, effusion, Baker's, contusion. Payload lacks fit-cohort/protocol metadata; private generalization is unproven.
5. **Public Raptor:** three trained CoAtNet checkpoints, four views: MaxSpan v5 forward .60, v5 reverse .10, native384dense v10 .10, native384 v8 .20. Probability-average then rank. The current runtime overrides the old 42/62 windows with capacity-aware up-to-96 unique acquired slices / 94 adjacent windows. Forward/reverse share preparation. This is changed geometry, not merely faster scheduling.
6. **CoAt family:** ResGated epochs [4,6,8], Global96 epochs [16,23,18], D4 parent SWA epochs [15,16]+adapter SWA [11,9,5]. Intended current reduction: each family's checkpoint probability mean, then equal probability mean across the three families, then rank. Inner mixture = .60 Raptor rank + .40 CoAt family rank, with no inner rerank.
7. **Outer mixture:** CoAt/Raptor weight .65 generally, .75 ACL/lateral OA/fracture, .80 medial meniscus, **1.0 lateral meniscus**; remaining weight from DINO/A5/Rad calibrated parent. Final per-target percentile rank.

## Private-data risks and what not to change during the sprint

- The target-specific outer coefficients, target-specific DINO pooling, calibrator gate, and cross-layout Rad recipe have no full-graph OOF evidence here. The lateral meniscus weight discards all parent diversity. This is a material public-LB selection risk, not proof of leakage.
- The calibrator includes acquisition counts, which can represent site/scanner/protocol shortcuts. A shift of private hospital/protocol composition may change its utility. Preserve it for the source-faithful candidate; test removal/shrinkage only against independently held-out evidence in later development.
- Same-family public forks are not independent signals. New training and 5-fold validation cannot credibly be completed by editing this inference notebook in one hour.
- SMART FAT/Dixon water detection now changes DINO/Rad routing; DINO/Rad/A5/Raptor intentionally use different normalization, crop, orientation, and slice rules. Harmonizing them without retraining is unsafe. Raptor performs modality LUT and MONOCHROME1 inversion; DINO uses rescale slope/intercept but not that inversion; A5 uses raw pixel intensities. Preserve checkpoint-specific preprocessing for this candidate.
- All three CoAt branches are related models. Equal probability averaging is not equivalent to equal rank averaging and is sensitive to calibration; it is the latest source contract, not a validated superiority claim.

## Inputs and runtime

Required competition attachment: `rsna-knee-abnormality-detection`; `test.csv`, `test_series.csv`, `test_series/`, `sample_submission.csv` and currently `train.csv` (read only to size a cache, not for training). Model attachment: `metaresearch/dinov2/PyTorch/small/1`.

External `kernel-metadata.json` lists these 14 datasets:

- `dreaddevelopment/raptor-knee-maxspan`
- `dreaddevelopment/raptor-knee-native384`
- `dreaddevelopment/raptor-knee-native384dense`
- `mattiaangeli/knee-mri-fold-weights`
- `mattiaangeli/opencv-python-headless-4120088-x86`
- `marwanmath/resnet-50-radimagenet-marwan`
- `mattiaangeli/rsna-knee-coat-resgated-ep10-top3`
- `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2`
- `mattiaangeli/rsna-knee-coatnet-global96-top3`
- `antoinegg1/rsna-knee-e11-diverse-heads-v20`
- `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`
- `pilkwang/rsna-knee-llm-labels`
- `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`
- `pilkwang/rsna-knee-weights`

Kernel attachments: `sofiaanjenje/rsna-knee-e11-train`, `sofiaanjenje/rsna-knee-e13-train`. Some listed assets are source-era alternatives/unused by the final graph; keep all during first reproduction and prune only after proving the used-file inventory.

Metadata mismatch: embedded notebook metadata lists only **13** datasetVersion sources and `isGpuEnabled:false`, while external metadata lists 14 datasets and `enable_gpu:true`. Set the new package consistently to GPU T4x2 and all 14 dataset handles. External metadata pins a docker SHA; embedded metadata records image version 31430. Preserve provenance rather than assuming latest Kaggle defaults are identical.

Core dependencies: torch, torchvision, transformers, timm, pydicom with required DICOM codecs, numpy/pandas, cv2. Offline children pin cv2 4.12.0 via a SHA-checked wheel; Global96/D4 pin timm 1.0.22 via SHA-checked wheel. Parent A5/Raptor timm and transformers rely on the image and should have versions recorded. Main CUDA probe requires exactly two actual T4 GPUs. A5 explicitly requests BF16 on T4; test the actual image/device path (availability/emulation/performance vary) before changing precision. Existing retry only handles nonfinite output, not autocast/runtime exceptions.

Correction after re-reading the entire child function: `TIME_BUDGET` is 8h and
`_run_required_child` already computes the remaining budget and uses
`process.wait(timeout=remaining)`, followed by SIGTERM/SIGKILL process-group cleanup.
No timeout patch is needed. The initial version of this audit incorrectly said
the child lacked a deadline. A 3-visible-study smoke validates assets/contracts
but cannot establish hidden-cohort runtime. Dynamic up-to-94-window readers dominate
runtime; preserve microbatching, resident replicas and shared prefixes. Report
elapsed time by phase, cohort size, GPU/RAM peaks, and counts of diagnostic events.

## Validation performed

Read required project review, all 24 notebook-cell summaries, relevant functions and full output/blend/release paths. Parsed every code cell with Python AST; decoded the calibrator payload; verified notebook SHA; inspected both metadata records. No notebook or runtime code modified, no network/Kaggle calls, no GPU runs. The acceptance claim is static audit only.
