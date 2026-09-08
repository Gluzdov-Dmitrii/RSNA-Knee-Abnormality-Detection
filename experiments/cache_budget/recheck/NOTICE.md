# Source and use

MRI pixels are derived from the official RSNA Knee Abnormality Detection training
data. They remain subject to the [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
and [RSNA MIRA licence](http://rsna.org/mira-license). This private cache is for
competition participants who have accepted those terms. Do not redistribute to
non-participants. It is not a substitute for the official train zip or lossless DICOM.

Geometry credit: Steven Lee,
[RSNA Knee: 500GB to 11GiB CPU Pixel Cache](https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache),
Apache License 2.0. His notebook credits Pilkwang's slot scheme, Karnakbayev's
three-slice grouping, and Will's physical crop. The new pipeline reimplements
physical ordering, slice grouping, millimetre center cropping, percentile scaling,
and uint8 caching. Modifications: public train_series flags; native SciPy bilinear
resize; explicit absent-slot mask; fail on decoding/order errors; sharded NumPy
files; spatial gradient verifier with study-level validation. No report parsing or
report lexicon code is included or copied. Apache-2.0 covers geometry code, not
permission to redistribute MRI. See LICENSE-APACHE-2.0.txt.

No radiology reports, train.csv, report lexicon, or report-derived labels are
included in the cache Dataset. Label scores are used only to evaluate the probe.
