# S24 Swin lease recovery — 2026-09-11

## Outcome

At 02:37:30 UTC the canonical `nsu-quadro` resource_queue.py released
`rsna-s24-swin-20260910T1644Z` using its existing owner token and
`--verified-stopped`. Response: `RELEASED`, `CURRENT`, host `prepost`.
The coordinating task was notified. No training or Kaggle submission was launched.

## Evidence before release

- Remote identity: `prepost`, `gluz_d_s`.
- Registered PID 187205 / start identity 427776461 absent.
- No owned process matched the run command, cwd, open files or parent PID.
  Two transient unreadable /proc entries were subsequently absent; the fresh
  full owned-process list contained only session/system processes, no trainers
  or data-loader workers.
- GPU compute-process list empty; RTX 6000 reported 5 MiB and 0% utilization.
- Metrics record completion at 2026-09-10T20:09:17Z, all five folds finished.
- Reported local OOF macro AUC: 0.81067355041603 (not leaderboard evidence;
  metric was read, not independently recomputed during lease recovery).
- OOF checked: 4407 rows, unique study IDs, folds 0–4, 12 prediction columns,
  all finite and in [0,1]. All five best checkpoints exist and are nonempty.

## Preserved artifacts and cleanup receipt

Remote run:
`/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection/runs/20260910T1644Z-s24-swin3d-t`

Local evidence copy: `output/s24_swin_20260910T1644Z_recovery/` contains
`metrics.json`, `oof.csv`, `train.log`. SHA-256 matched source and destination.
These generated artifacts are not staged in Git.

No files were deleted; reclaimed bytes: 0. Five best and five last checkpoints
(378750435 bytes each; 3787504350 bytes total), validation predictions, metrics
and log remain at the remote run. These are the retained best/latest artifacts,
not unverified disposable copies. Retention owner: rsna-agent. Reason: preserve
the completed Swin candidate for the next OOF comparison/export decision;
review on 2026-09-14 before pruning. No independent weight backup was made.
The legacy `/tmp/rsna-s24-swin` directory was empty and left untouched, as it is
outside the registered project root. Future launch cleanup/path correction is
separate work; this recovery did not change launch scripts.

## SHA-256 manifest

Paths below are relative to the remote run.

```text
7c5cc1e4df59a0df456c3ffe149b1a389c74df3ee6b770f0481a619bc37c8966  checkpoints/metrics.json
80106e9c852bea3bea274418a29044b5cc46f816feba8f44225ca75b4bbc149d  checkpoints/oof.csv
9050b86ad4f3e14e740a3af206f48841eaaaeb4d53b4e8eb29a600d593203aae  logs/train.log
27735b0e26151ac0e803bd7004bf5eb07ac31b0335e50b4a88f9d34dc683e84e  checkpoints/fold0/best.pt
926f2d9398f26eb380b8f17289cae4e4553567bf9862122eb108dddb10faeba4  checkpoints/fold1/best.pt
130305432078d70423e4104938051d814fe14d7d2ce01b9ea217b7826e84a1d3  checkpoints/fold2/best.pt
c2ed2d1ea59b4ab4abce1c8c9dc49939320f9430111efb3588c254f5fa3a027b  checkpoints/fold3/best.pt
7da106fc30a2a87ca6ee023af294b6318faeb35c6796e0aecb12c914c67b03ed  checkpoints/fold4/best.pt
9fc2db2d7023ede3defb1f79845bd3c28473056a4493aa712b74cc4b1becbea3  checkpoints/fold0/last.pt
a49608e8e80f2d35e108006aeb9c35248acde0107519c2a62e01fb752e508d83  checkpoints/fold1/last.pt
d1c4245beacbb45ffa88e5f52c581f80cb2ed1d6ba513d4422e82d535445144b  checkpoints/fold2/last.pt
8581c6c2d970452c75f313734209f5f7d53d510ff0a7ab20081dd95acb92389d  checkpoints/fold3/last.pt
de05e7afdd2595efcbfdd9481e41f4427c7b69026a148b7ead0eda3f7b7b6d23  checkpoints/fold4/last.pt
```
