# S22 fold weights

Private inference weights for `dmitriigluzdov/rsna-s22-a0-rank-blend`.

These files are derived from the RSNA Knee Abnormality Detection training MRI
under the competition rules and RSNA MIRA licence. They are for accepted
competition participants only. Do not redistribute outside the competition.

Architecture: torchvision ResNet-18, 6-channel 2.5D plane heads, Linear(3*512, 12).
Geometry: PIXEL_CACHE_V1 224×9 crop 130 mm, window 0.35–0.65.
Folds: FOLDS_V1 seed 2026. Labels: LABEL_PILKWANG_V1. dtype float16 state_dicts.
