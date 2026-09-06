# ORIGIN — S06–S10 A0 outer-weight packet

Pinned parent graph (same as S01–S05):

- owner/slug: `renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid`
- URL: https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid
- kernel ID: `133076816`
- pinned strategy version: V1 / scriptVersionId `347142162`
- visible A0 `submission.csv` SHA-256: `11bb66f2bc7ca21d4282de3696f5ea5531f90c6c2958164b6ad6931811af2213`

Overlay formula (Bee V6 / S02–S05): for changed targets,
`(1-w)*transformer_rank + w*raptor_rank`, then average-percentile rerank.
Untouched targets stay byte-identical to the A0 parent CSV. Medial Meniscus
stays on public0033 T30/R60/bag10. S07 must not touch Medial Meniscus.

- Bee formula source: `prvsiyan/the-bee-s-knees-final-rsna-push` kernel `133138832` V6 / script `347412699`

Kernel slot: `dmitriigluzdov/rsna-week1-a0-s01-s05` (extended batch cell; not a public fork).
Internet off. `machine_shape` NvidiaTeslaT4. Named `-f submission_Sxx.csv` is HTTP 400;
each S-ID is submitted as `-f submission.csv` from a distinct version.

Attached inputs match the Renta metadata that produced the visible A0 receipt.
Static audit of the parent graph was already completed for A0 / Day 1.
