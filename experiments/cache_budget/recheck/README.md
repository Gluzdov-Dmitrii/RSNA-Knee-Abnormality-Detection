# Reproduce the storage recheck

Read PROTOCOL.md for the frozen experimental plan and NOTICE.md for attribution
and MRI restrictions. The only remote inputs are the competition and Pilkwang's
labels. Neither kernel enables internet, GPU, or competition submission.

1. Run `python experiments/cache_budget/recheck/test_pipeline.py` locally with
   NumPy, pandas, SciPy, scikit-learn and pydicom installed. The script also checks
   `tmp/cache_budget_deps` for the local DICOM reader used during development.
2. `python experiments/cache_budget/recheck/build_private.py` embeds FOLDS_V1
   and creates the private CPU materializer in `private_kernel/`.
3. Push that directory with the Kaggle CLI. Monitor that same run to completion.
4. Download its output under `artifacts/cache_budget_recheck/`. Validate all
   pixels with `python experiments/cache_budget/recheck/validate_dataset.py
   artifacts/cache_budget_recheck/cache`.
5. Change into the validated `artifacts/cache_budget_recheck/cache/` directory,
   then run `kaggle datasets create -p . -t`. This also avoids a Windows CLI bug
   in resumable-upload paths containing forward slashes. The CLI default is
   private. Do not pass `--public`.
6. Inspect Dataset readiness, privacy and complete file listing; save the confirmed
   publication receipt as `artifacts/cache_budget_recheck/dataset_publication.json`.
7. `python experiments/cache_budget/public/build_notebook.py` requires those
   actual evidence/publication receipts. It generates the public notebook with
   two measured figures and hidden source cells. `publish_notebook.py` uses the
   SDK Quick Save type with Markdown image attachments for the verified figures.
   Kaggle-specific `_kg_hide-input` metadata collapses all implementation cells.
   It requires
   the private execution to be COMPLETE. Check the public rendering, hidden code,
   inputs, private Dataset link, and absence of saved public MRI pixel shards.
   Save & Run remains a complete rebuild, using the same pipeline proven by the
   private execution. A Quick Save is not represented as a second execution.

The public run rebuilds its cache under `/kaggle/temp/rsna-knee-cache`; the private
run persists it under `/kaggle/working/cache`. Only the private output is uploaded
as a Dataset. No raw DICOM train archive needs to be downloaded locally.

`pipeline.py` implements an original spatial gradient descriptor and a
train-fold-only ridge model. It is not a learned image encoder. It keeps the exact
200-study subset while correcting the old two-stage resolution resize. The
15-point evaluation is not a hyperparameter search: classifier settings were
frozen before execution. Every prediction is held out at study level.

The notebook generator derives narrative claims from paired confidence intervals.
Failure to find a difference does not establish equivalence. Its prespecified
equivalence margin is ±0.01 AUC, with Holm correction used when considering a
clearly better setting. Cache selection stays at the user's primary recipe
unless contrary evidence warrants a separately documented decision.

Published and verified: [notebook version 8](https://www.kaggle.com/code/dmitriigluzdov/knee-mri-in-11-gib)
and [cache version 1](https://www.kaggle.com/datasets/dmitriigluzdov/rsna-knee-uint8-224-9-c130).
The notebook retains kernel ID 133521917; Kaggle changed its slug with the title.
All 35 shards passed full local hash/shape/content validation. All 41 remote
Dataset files and byte sizes match, with privacy and readiness confirmed.

`DATA_CARD.md` and `build_dataset_metadata.py` reproduce the improved Dataset
documentation and cover. The builder preserves recorded visibility. Kaggle's
metadata API saved the card/tags/provenance/cover, but ignored file and column
descriptions: all 41 file descriptions were saved through the browser editor.
Column definitions are included in the index description and data card.
Usability is 6.88. Public access was requested but blocked by automatic approval
review pending explicit confirmation after the non-participant access risk;
do not treat documentation completion as permission to bypass that rejection.

The user subsequently changed Dataset visibility manually; a read-only check
confirmed it is accessible as listed, with all 41 files unchanged. Notebook
version 8 removes private/public wording from reader-facing text while retaining
competition and MIRA terms. The assistant did not change Dataset visibility.
