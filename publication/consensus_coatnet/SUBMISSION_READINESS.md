# P03 standalone submission readiness

Candidate: dmitriigluzdov/knee-mri-compact-coatnet, version 2, COMPLETE.
Dataset: dmitriigluzdov/rsna-knee-compact-coatnet, private version 1.
Weights SHA256: f658b89db67997c3c0975a79298f1cbb962c30483ba9a7dec9bbffc4926c6213.

- PASS: official competition rules/evaluation/data pages read through Kaggle SDK;
  internet off, GPU runtime limit 9 hours, submission.csv, external generic
  ImageNet initialization with source credits. No MRI or reports in package.
- PASS: user explicitly authorized completion through one LB submit. History
  checked after interruption: no P03 submit exists; 4 daily slots remained.
- N/A: second attempt is not authorized or planned. Exactly one submit call.
- PASS: FOLDS_V1 study split, seed 2026, fixed final epoch 4; 875 validation
  studies. Gold 58 globally excluded, no gold-based epoch/recipe selection.
- N/A: full OOF/nested selection not performed. This is a single-fold screening
  followed by a standalone baseline evaluation, not a proven improvement.
- PASS: macro/per-class AUC and paired conditional bootstrap recorded. Median
  loses 0.010466 AUC, 95% interval [-0.016471,-0.003872]. No LB tuning/blending.
- N/A: fold/domain tail statistics unavailable from one fold; per-class results
  are retained. No claim of robustness across sites/folds.
- PASS: full refit 4,349 non-gold studies, 1,225.69 seconds, 6.047 GiB peak A100.
- PASS: actual clean Kaggle T4 v1 execution COMPLETE, offline wheel installation,
  90 real DICOM studies in 196.25 seconds; 1.310 GiB peak inference allocation.
- PASS: cache/live pixel and mask SHA256 identical for all 58 gold studies.
- PASS: JPEG2000/JPEG-LS lossless roundtrips; JPEG Lossless decoder available.
- PASS: independent CPU FP32 versus T4 FP16 max error 0.002205 on all 58 studies.
  Initial A100 BF16 versus T4 FP16 check failed (0.017295 > 0.015); retained as
  diagnostic evidence. CPU FP32 establishes the T4 path is numerically accurate;
  the original BF16 threshold was not silently widened. No weights changed.
- PASS: v1 live test predictions, intended weights hash, submission.csv schema,
  3 current test UIDs/order, numeric finite [0,1] probabilities, no fallback.
- PASS: 17 local tests including arbitrary test count, missing slots and stale
  sample UIDs; inference follows test.csv, sample only provides column schema.
- PASS: v2 disables already completed training-data QA, avoiding train-file
  dependency and repeated QA overhead during hidden scoring. Same model/code.
- N/A: hidden duration cannot be guaranteed before scoring. 90-study p95 is
  4.01 seconds/study; linear 1,300-study extrapolation is about 87 minutes at
  that rate, below 9 hours. This is an estimate, not a hidden-runtime claim.
- PASS: v2 terminal COMPLETE; downloaded output validation PASS. Its prediction
  CSV is byte-identical to v1 (SHA256 ba1a551282699305df88b1d10cb4882498f3885f0a77ea6f866ad202763c661f).
  Submitted notebook source SHA256: 566596021f6100b2c2ff2232c2cc46dd5470b91cccddad36657fb71e0011ea35.
  Package manifest SHA256: a4c7f89dbeccd46949ca3ad13e914b8f8eed4eccf8d0bd5fc86e0f95674a5c49.
- PASS: exactly one submit, immediately followed only by read-only submissions
  query. Ref 56142857, 2026-09-10 11:04:11.587 UTC, PENDING. Message:
  P03 compact CoAtNet Pilkwang full4349 ep4 f658b89d. No retries.
- PASS: terminal COMPLETE, Public LB 0.875, observed 2026-09-10 11:44:51.720 UTC.
  Scored scriptVersionId 348764510 (notebook version 2). Read-only monitor exited.

Detailed local evidence: artifacts/consensus_coatnet/{verified_pair,full,
kaggle_output,kaggle_output_v2,kaggle_verified_source_v2}/.
