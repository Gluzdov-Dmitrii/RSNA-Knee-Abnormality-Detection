# Reproducible knee CoAtNet — work in progress

This isolated lane implements an end-to-end, Raptor-inspired model on the existing
private11GiB cache. **No quality improvement or LB score is claimed yet.**
The first comparison changes only supplied training scores: Pilkwang versus the
median of Pilkwang, Steven v2 and Lixin. See PROTOCOL.md and NOTICE.md.

Grok's source, jobs, folds, cache and Python environment are not modified. The
Codex helper worked only on an independent audit and test_contract.py. This is
not an external Grok assistant and it does not run GPU work.

## Local correctness checks

```sh
python -m unittest discover -s publication/consensus_coatnet -p test_contract.py -v
```

These tests use a tiny mocked encoder; a real CPU pretrained forward and a
reserved-GPU backward/reload pilot are separate required checks.

## Rebuild

1. Accept competition terms. In your private copy, build the224×9,crop130,
   window.35–.65 cache with the geometry used here. Never publish derived MRI.
2. Obtain the three label sources listed in NOTICE.md. Preserve their exact
   versions. Use the supplied FOLDS_V1 hash from PROTOCOL/common.py.
3. Run export_gold_uids.py on official train.csv; it exports only58 study UIDs.
4. Run prepare.py with --cache, --folds, --labels, --gold-uids and --out.
   This checks input hashes and downloads generic ImageNet initialization on CPU.
5. Train the two arms in separate empty output directories, sharing initialization:

```sh
python train.py --cache CACHE --folds FOLDS.csv --labels LABEL_ROOT \
  --gold-uids gold_uids.csv --initialization prepared/initialization.pt \
  --arm pilkwang --fold 0 --epochs 4 --out p01
python train.py --cache CACHE --folds FOLDS.csv --labels LABEL_ROOT \
  --gold-uids gold_uids.csv --initialization prepared/initialization.pt \
  --arm median --fold 0 --epochs 4 --out p02
```

Use --smoke-steps8 in a distinct disposable output directory before full training.
It checks real forward/backward and checkpoint reload, not quality. --resume
resumes the last completed epoch; any interrupted partial epoch is replayed.
Do not modify source/config/dependencies between arms or during resume.

## Apply a trained checkpoint

```sh
python infer.py --competition /kaggle/input/rsna-knee-abnormality-detection \
  --weights weights.pt --out /kaggle/working/submission.csv
```

Inference builds pixels from the current test mount, preserves live test.csv UID order,
uses an explicit missing-slot mask, and runs without internet. The package needs
common.py,model.py,geometry.py,infer.py plus pinned dependencies and weights.
Actual Kaggle remount/runtime validation is still required before release.

Dependencies verified remotely are recorded in the preparation input manifest;
the current immutable project environment is reused read-only. No installer
changes an environment used by Grok. Final publication includes the dependency
lock, model config, hashes, credits, one short A/B table and measured runtime.
