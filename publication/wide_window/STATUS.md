# P04 execution handoff

User requested fresh status and next attempt, September 12 (local date moved
to September 13 during work). Scope: one window-policy experiment, then at most
one new LB submit if the predeclared gate and runtime validation pass.

Live Kaggle snapshot:

| Submission | Ref | Public LB |
|---|---:|---:|
| Existing A0/S01 ensemble | 56034297 | 0.937 |
| Own compact CoAtNet P03 | 56142857 | 0.875 |
| Grok R3D + R(2+1)D rank mean | 56176352 | 0.861 |
| Grok R(2+1)D | 56159319 | 0.859 |
| Grok R3D | 56158999 | 0.852 |
| Grok ResNet-18 | 56138090 | 0.825 |

All listed submissions COMPLETE. Grok's S25 markdown still said PENDING; the live
API establishes completion. Its source/ledger were not edited. Swin3D OOF
0.810674 was a tie; attention OOF 0.801746 lost. Do not repeat those submissions.

Active private CPU kernel:
https://www.kaggle.com/code/dmitriigluzdov/knee-mri-private-wide-cache
Version 1, COMPLETE at 2026-09-12 18:01 UTC. Actual slug is title-derived; do not use the
initial requested rsna-knee-private-wide-cache slug. Input official competition
plus own compact model package (used only for offline DICOM codec wheels).
Build source SHA256 aacf4fb39b5db4302b64e19a7309c713e94544f32a3d7a8c5915c8f376cc8370.
All 4,407 studies completed in 3,533.81 seconds, 35 verified shards, 11.12072 GiB.
Receipt/spec SHA256 verified: afce0b82784a444d5fc73bac80493c0e4ab2379a0ea4f765af4edb17eae37b46.
Local download: artifacts/wide_window/cache_output/cache. Stage --wait-local
pipelines verified shards to NSU. direct_transfer.py fetches shards 8-34 directly
to the same private server using temporary URLs in memory (no credential copy).
Both paths verify immutable SHA256; remote preflight still required before GPU.
The installed CLI supports `kaggle kernels logs OWNER/SLUG -f` for live progress;
ordinary kernels output/logs without follow may stay empty while running.
Output cache/ has all 4,407 uint8 studies, 224x9, crop130, window .10-.90, once
complete. No MRI publication. Use --file-pattern and --page-size 200 to download
receipt/spec first, then shards. Old cache remains immutable.

Prepared local source generator: prepare_source.py -> artifacts/wide_window/source.
Common window and train initialization guard differ; model and geometric
algorithms are unchanged. Exact original generic initialization reused;
initialization/window metadata explicitly separated. Seventeen established tests
PASS against P04 source via run_tests.py. Comparison checks both ordered UID
tables; they were retrieved for the old control. See PROTOCOL.md for fixed gate.

Remote target (source and complete cache staged and hashed; GPU work not yet started):
- code/coatnet-wide-window-v2 (explicit LF; v1 was never launched)
- data/rsna-knee-uint8-224-9-c130-w10-90
- runs/20260912-codex-wide-window/{smoke,pilot,full}
within /home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection.
Prelaunch audit found Windows line-ending translation changed raw model source
hash despite identical code. Version 2 explicitly writes LF so its model.py hash
matches P01 exactly; version 1 remains a small unlaunched diagnostic snapshot.
Existing immutable ngpu01 py3.11-torch-cu124-v1 environment. Ready-input receipt
required by manage.py before it allows any GPU reservation. REMOTE_READY.json
now records READY, all 38 cache hashes, identical row order/masks and original
initialization. Smoke lease rsna-codex-p04-smoke-20260912 is WAITING_RESOURCE,
not dispatched. Both A100s currently run Biohub jobs (PIDs 164524 and 166866).
Our waiting request is first; recheck physical state and common queue before
dispatch. No GPU held while cache built. Kaggle GPU quota snapshot: 27.91/30h remaining.

Cleanup script included inactive-process and queue checks for explicit old
optimizer/smoke duplicates, but auto-review rejected execution as expensive
destructive work without explicit permission. No files deleted. Do not bypass
that rejection. Old artifacts (~4.98GB) remain; this does not block the <=16GiB
new-task budget. Final models and generic initialization were never targets.

Next: await resource -> smoke, fixed fold0 pilot -> compare saved P01. No new
submit has occurred. Transfers complete and fully verified. New *.direct-part
temporary download duplicates were removed by finish_transfer.py only after
both temporary and final file hashes matched SPEC. Old checkpoints remain.
