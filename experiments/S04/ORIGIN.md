# ORIGIN — S01–S05 A0 batch

Pinned parent graph:

- owner/slug: `renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid`
- URL: https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid
- kernel ID: `133076816`
- pinned strategy version: V1 / scriptVersionId `347142162`
- pulled locally: 2026-09-05 (latest public source; kernel ID matches the pin)
- last public visible run: 2026-09-04 02:57 UTC, T4×2, ~159 s, internet off
- visible A0 `submission.csv` SHA-256: `11bb66f2bc7ca21d4282de3696f5ea5531f90c6c2958164b6ad6931811af2213`

V6 overlay formula (S02–S05 only):

- owner/slug: `prvsiyan/the-bee-s-knees-final-rsna-push`
- URL: https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push
- kernel ID: `133138832`
- pinned strategy version: V6 / scriptVersionId `347412699`
- exact public version pull returned HTTP 403; latest public notebook still contains
  cell `BEE_OUTER_WEIGHT_OVERLAY_V6` with the locked weights
  ACL `.80`, Lateral OA/PF OA/Synovitis `.35`, Baker's `.375`.

Attached inputs (copied from the Renta metadata that produced the visible A0 receipt):

- datasets: `dreaddevelopment/raptor-knee-maxspan`, `dreaddevelopment/raptor-knee-native384`,
  `dreaddevelopment/raptor-knee-native384dense`, `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`,
  `renta0426/rsna-knee-public0033-meniscus-bag-v1`
- model: `metaresearch/dinov2/PyTorch/small/1`
- competition: `rsna-knee-abnormality-detection`
- docker: `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

Licenses follow the public notebooks (Apache 2.0 on the Head-and-shoulders/Renta lineage).
The local additions are the S01–S05 named-file cell and a final
`SELECTED_SUBMISSION_SID` copy into `submission.csv`. Inference stays internet-off.

Kaggle `CreateCodeSubmission` returns HTTP 400 for `-f submission_Sxx.csv`.
Each S-ID was therefore submitted as `-f submission.csv` from a distinct version:

| S-ID | kernel version | submission ref | submitted_at_utc |
|---|---:|---:|---|
| S01 | 1 | 56034297 | 2026-09-05T13:57:58Z |
| S02 | 2 | 56034425 | 2026-09-05T14:05:35Z |
| S03 | 3 | 56034555 | 2026-09-05T14:12:43Z |
| S04 | 4 | 56034661 | 2026-09-05T14:20:11Z |
| S05 | 5 | 56034764 | 2026-09-05T14:26:04Z |

Static audit of the pulled Renta source: no `requests`/`urllib`/`subprocess`/`os.system`.
`os.environ` is used for offline HF flags and work-dir overrides. The meniscus bag loads
hash-pinned `public0033_runtime.py` from the attached dataset via `importlib`.
