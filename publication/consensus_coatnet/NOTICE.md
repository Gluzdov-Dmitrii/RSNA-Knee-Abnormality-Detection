# Attribution and boundaries

Model approach: Johnathan Wagner / dreaddevelopment, per-finding attention over
2.5D windows, from the Apache2.0 training notebook:
https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model
Reference inference:
https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model

This is a new compact implementation with explicit missing-window masking,
within-slot triplets, shared normalization, fixed folds and final-epoch selection.
It is not compatible with Raptor knee weights and does not inherit their score.
Generic ImageNet CoAtNet initialization uses timm (Apache2.0), Ross Wightman et al.:
https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k

Geometry: Steven Lee, Apache2.0, and the existing local cache pipeline:
https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache
The geometry implementation retains physical ordering, grouped slice sampling,
physical central crop, sampled-series percentiles and uint8 conversion. No report
parser or lexicon is copied. Apache2.0 covers code, not redistribution of MRI.
Competition MRI remains under competition rules and RSNA MIRA.

Label inputs are privately read from their original datasets; weights packages
contain neither label tables nor reports. Finite supplied scores are retained,
including uncertainty sentinels. These sources are not claimed independent:
- https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels
- https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels
- https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56

Exact input hashes and dependency versions are recorded in the preparation manifest.

Retain this notice and LICENSE-APACHE-2.0.txt when distributing the adapted code.
