# Knee MRI without the 500 GB footprint

A compact **derived pixel cache of all 4,407 training studies** from RSNA Knee Abnormality Detection. Each study is a `uint8` array of shape **(6, 9, 224, 224)**: six scan slots, nine selected slices per slot, and 224 × 224 pixels per slice. Pixel payload: **11.12 GiB** (about 11.94 decimal GB including metadata).

This is a lossy training convenience, not lossless DICOM or a substitute for the official train zip. Studies are not necessarily unique people. No test data, radiology reports, report lexicon, `train.csv`, or label scores are included.

## Start here

1. Join the [competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) and accept its rules and the [RSNA MIRA terms](http://rsna.org/mira-license). This derived competition MRI is intended for participants, not non-participants.
2. Add this Dataset to your Kaggle notebook, or download its files locally. NumPy and pandas are sufficient to read it; no DICOM reader is needed.
3. Use `studies.csv` to locate a study. Match labels from your authorized source by `StudyInstanceUID`, never by row position.

```python
from pathlib import Path
import numpy as np
import pandas as pd

# Use the actual mount path shown in your notebook's Input panel.
cache = Path('/kaggle/input/rsna-knee-uint8-224-9-c130')
index = pd.read_csv(cache / 'studies.csv')
study_i = 0
r = index.iloc[study_i]
shard = np.load(cache / r['shard'], mmap_mode='r', allow_pickle=False)
study = shard[int(r['row'])]  # view of ONE study, not the whole corpus
mask = np.load(cache / 'slot_mask.npy', allow_pickle=False)[study_i]
print(study.shape, study.dtype, mask)  # (6, 9, 224, 224), uint8

# Example: first slot, middle retained slice.
slot = 0
if mask[slot]:
    image = study[slot, 4]
    # Optional model input; choose further normalization for your own model.
    x = study.astype(np.float32) / 255.0
```

Memory mapping reads file portions on demand. One uint8 study is about **2.58 MiB**; a 128-study shard is about **330.75 MiB**. Converting the whole corpus to float32 would require about 44.48 GiB; convert batches instead.

## Files and axes

| File | Contents |
| --- | --- |
| `pixels-000.npy` … `pixels-034.npy` | 35 dense NumPy shards: first 34 hold 128 studies each; last holds 55. Axes: study within shard, slot, retained slice, height, width. |
| `studies.csv` | 4,407 unique study IDs with `shard` filename and zero-based `row` inside that shard. CSV row order also indexes the mask. |
| `slot_mask.npy` | Shape `(4407, 6)`, uint8. 1 = selected series exists, 0 = absent public slot. Absent slots are zero-filled; zeros do not mean a healthy knee. |
| `SPEC.json` | Exact geometry, dtype, storage, shard sizes and SHA-256 checksums. |
| `AUDIT.json` | Counts of physical ordering, decoded slices, missing slots and crop application. |
| `NOTICE.md` | Original build attribution and data-use restrictions. |
| `LICENSE-APACHE-2.0.txt` | Licence for the credited geometry code, not a licence to the MRI data. |

| Slot index | Scan plane | Public `Fluid_Sensitive` flag |
| --- | --- | --- |
| 0 | Sagittal | 1 |
| 1 | Coronal | 1 |
| 2 | Axial | 1 |
| 3 | Sagittal | 0 |
| 4 | Coronal | 0 |
| 5 | Axial | 0 |

The flag comes from official `train_series.csv`; it is not a recovered report label or a substitute for the separate fat-suppression flag. Where several series match a slot, the series with the most DICOM files is selected; CSV order breaks ties.

## How the cache was made

Physical slice order uses image position projected onto the orientation normal, then `SliceLocation`, then `InstanceNumber`. Decoding/order errors fail the build; missing series are explicitly masked.

Nine slices are three neighboring-slice groups, anchored within positions **0.35–0.65 along the ordered stack**. This is a slice sampling window, not an intensity window. Short stacks can repeat slices.

The requested center crop keeps approximately **130 × 130 mm**, using row pixel spacing, before bilinear resizing to 224 × 224. It is the size retained, not the amount removed. Cropping is skipped if the source field is too small or spacing is missing. Intensities are scaled per sampled, cropped series using its 1st–99th percentiles, clipped and rounded to uint8 (0–255).

All 4,407 study IDs and all 35 shard hashes were validated. The full build recorded 21,334 present slots and 5,108 absent slots; crop applied to 20,976 selected series and was skipped for short fields of view in 358.

## Why this size?

The [companion notebook: Knee MRI in 11 GiB](https://www.kaggle.com/code/dmitriigluzdov/knee-mri-in-11-gib) compares 15 settings on the same fixed 200 studies and five study-level folds. A spatial image descriptor plus regularized linear model scored **0.660 macro AUC**, with 95% interval 0.633–0.689, on this recipe using report-derived weak labels.

224² × 9 with crop 130 mm is a practical storage compromise. More resolution had a small, uncertain gain; a flat plateau is not proven. This cache is not a validated substitute for full-resolution imaging, a pretrained encoder, or a leaderboard result. No labels or folds are shipped here; use the notebook for the evaluation protocol and rebuild code.

## Source, licence and limitations

**Data licence: competition rules + RSNA MIRA.** Public hosting does not grant unrestricted rights or waive [competition access restrictions, section 4.b](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules). Intended for participants who accepted those terms; not for non-participants. No independent organizer endorsement or redistribution permission is claimed. The original version-1 manifest/notices describe the initial private build; the Dataset page shows current visibility, and the underlying data-use restrictions remain unchanged.

**Geometry credit:** [Steven Lee's CPU pixel cache](https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache), Apache-2.0, reimplemented and modified. No report or lexicon code was copied. The full attribution is in `NOTICE.md`. Pilkwang's [labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) were used only for the separate quality probe.

This is an offline cache for research workflows. It discards slices, peripheral anatomy, scanner metadata and intensity precision, and it does not correct orientation across studies. Fixed geometry may miss findings that another crop, slice selection or model would use. It is not intended for clinical decisions. This is a fixed release with no scheduled updates.
