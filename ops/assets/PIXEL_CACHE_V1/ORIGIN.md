# PIXEL_CACHE_V1 origin

Private derived uint8 cache, **not** raw DICOM and not a DINO embedding cache.

- Dataset: `dmitriigluzdov/rsna-knee-uint8-224-9-c130`
- URL: https://www.kaggle.com/datasets/dmitriigluzdov/rsna-knee-uint8-224-9-c130
- Visibility: **private** (participants who accepted competition + MIRA terms)
- Companion notebook: https://www.kaggle.com/code/dmitriigluzdov/knee-mri-in-11-gib
- Geometry credit: Steven Lee `stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache` (Apache 2.0), reimplemented; no report lexicon

Shape: `4407 × 6 × 9 × 224 × 224` uint8 ≈ **11.12 GiB** pixels (`pixels-000.npy` … `pixels-034.npy`).

Local path (gitignored): `data/rsna-knee-uint8-224-9-c130/`
NSU path (shared NFS, once): `/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection/data/rsna-knee-uint8-224-9-c130/`

Reader: `ops/tools/pixel_cache_v1.py`. Dataset join: `ops/tools/pixel_dataset.py`. Pixel-model skeleton: `training/s22_resnet18_25d.py`.

This unblocks **local/NSU training on pixels**. It does **not** replace `DINO_CACHE_V1` (frozen encoder features). Hidden test is still decoded live in the Kaggle scoring notebook. Do not download the ~500 GB official train zip as the default training corpus.
