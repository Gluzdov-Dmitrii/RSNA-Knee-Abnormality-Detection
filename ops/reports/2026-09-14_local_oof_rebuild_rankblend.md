# 2026-09-14 Local OOF rebuild + rank blend

Goal: resume after Grok and maximize local OOF using available local/cluster
compute without touching Kaggle submissions.

## Compute status

- Linux aliases `nsu-quadro` / `nsu-a100`: DNS resolves only outside the
  sandbox, but SSH timed out on port 22 during this session.
- `nsu-pc`: reachable, RTX 3080 idle, but it currently has only `FOLDS_V1`
  and Pilkwang labels, not the 11 GiB `PIXEL_CACHE_V1`.
- Local RTX 2060 CUDA path was usable and completed all checkpoint rebuilds.

## Inputs recovered/verified

- Baseline cache:
  `artifacts/cache_budget_recheck/private_copy`
  - `SPEC.json` sha256: `043485b6c0ea543a5f4061ce15b6fef868bb28aebc2af6d124adfd940c9aaf50`
  - verifier smoke passed on 4407 studies / 35 shards.
- Pilkwang weak labels:
  `data/labels/pilkwang/report_labels_v2.csv`
  - sha256: `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920`

## New utilities

- `training/export_oof_from_checkpoints.py`
  - Rebuilds fold-out OOF predictions from exported fold checkpoints.
  - Supports S22 ResNet18 2.5D and S24 R3D/R2P/Swin3D checkpoints.
- `training/oof_rank_blend_search.py`
  - Rank-transforms OOF members and searches simplex weights.
  - Supports full global grid, target-wise upper-bound search, and local
    refinement around a center weight vector.

## Rebuilt full OOF results

All four members were rebuilt from checkpoint roots against the same cache:

| Member | OOF CSV | Macro AUC |
|---|---:|---:|
| S22 ResNet18 2.5D | `tmp/rebuilt_oof/s22_full/oof.csv` | `0.7683275282` |
| S24 R3D18 | `tmp/rebuilt_oof/r3d_full/oof.csv` | `0.8006560195` |
| S24 R2Plus1D18 | `tmp/rebuilt_oof/r2p_full/oof.csv` | `0.8106031301` |
| S24 Swin3D-T | `tmp/rebuilt_oof/swin_full/oof.csv` | `0.8106725403` |

The rebuilt Swin metric matches the previous recovery OOF
(`output/s24_swin_20260910T1644Z_recovery/oof.csv`) within expected floating
noise.

## Best local rank blend

Final reproducible global blend:

- Report:
  `experiments/LOCAL_OOF_RANKBLEND_REBUILT_global_refine00125/rankblend_report.json`
- OOF CSV:
  `experiments/LOCAL_OOF_RANKBLEND_REBUILT_global_refine00125/best_global_rankblend_oof.csv`
- Macro AUC: `0.8328745897`
- Weights:
  - `S22`: `0.075`
  - `R3D`: `0.2125`
  - `R2P`: `0.3500`
  - `SWIN`: `0.3625`
- Alignment: 4406 labelled rows; excluded the single known unlabeled UID
  `1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220`.

Reference searches:

- Full global grid step `0.05`:
  `0.8328031176`, weights `S22=0.10, R3D=0.20, R2P=0.35, SWIN=0.35`.
- Full global grid step `0.025`:
  `0.8328607747`, weights `S22=0.075, R3D=0.225, R2P=0.35, SWIN=0.35`.
- Local refinement step `0.0125`, radius `0.05`:
  `0.8328745897`, weights listed above.

Target-wise search at step `0.05` reached `0.8351425458`, but those per-target
weights are optimized on the same OOF labels and should be treated as an
optimistic upper-bound / search signal, not an unbiased promotion metric.

## CoAtNet note

Local CoAtNet artifacts are not a locked FOLDS_V1 5-fold OOF. The available
`artifacts/consensus_coatnet/verified_pair/pilkwang/validation_predictions.csv`
is one 875-study validation screen (`macro_auc=0.8371650703`), and the full
CoAtNet package is a refit. Do not mix it into the honest 4406-row OOF blend
without a separately defined protocol.

## Next work

1. Add a faster pre-ranked AUC backend to `oof_rank_blend_search.py` before
   trying full step `0.0125` or target-wise refinements.
2. If `nsu-pc` is to be used, transfer or mount `PIXEL_CACHE_V1` plus the
   rebuilt scripts/checkpoints; otherwise local RTX 2060 is sufficient for
   inference-only OOF rebuilds.
3. DINO/S11 remains blocked until DINO weights/cache are recovered. Do not fake
   S11 with another backbone.
4. Treat `0.8328745897` as the current best honest local full-OOF rank blend
   from available own-model checkpoints.
