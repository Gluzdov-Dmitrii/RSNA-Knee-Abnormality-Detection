# 2026-09-15 Public941 submit

Objective: submit the best available public-code LB solution and compare its
OOF evidence against our best local OOF.

## Submitted

- Public source: `ricecakar/rsna-knee-public-0-941-control`
- Private kernel: `dmitriigluzdov/rsna-knee-public941-control` v2
- Submission ref: `56243894`
- Status verified 2026-09-16: `COMPLETE`, public score **0.941**.
- Message: `PUBLIC941 control probe22 ff848c73`
- Monitor: `monitor-rsna-public941-submission`, every 15 minutes, quiet while
  pending
- Visible `submission.csv` SHA-256:
  `ff848c73ba6f31e487d175161304d26df307e6e777189c01dfb508a76532ddfc`

## Packaging notes

- First `kernels push` attempts failed with HTTP 403 until the pulled public
  metadata's foreign `id_no` was removed.
- Kernel v1 then failed because `kernel_sources` had been removed; the notebook
  requires `sofiaanjenje/rsna-knee-e13-train` for the intended E13 head SHA.
- Kernel v2 restored the kernel sources and completed visible validation.
- Ordinary local file submit returned HTTP 400. Correct code-competition route:
  `kaggle competitions submit rsna-knee-abnormality-detection -k dmitriigluzdov/rsna-knee-public941-control -v 2 -f submission.csv -m "..."`

## OOF comparison

- Our current best honest local full OOF:
  `0.832874589670475`
  from `experiments/LOCAL_OOF_RANKBLEND_REBUILT_global_refine00125`.
- Public941 final-solution OOF: not available in the pulled public notebook.
  Programmatic search found no `OOF` / `oof` / `validation_predictions` artifact
  for the final graph. The notebook is inference-only.
- Numeric OOF delta: not computable.

Component note: some attached RadImageNet datasets contain component OOF CSVs,
but they are not the final public941 graph OOF and should not be compared as the
OOF of the public top solution.

## Current LB context

- Current leaderboard top at polling time: `0.957`.
- Our previous best submitted score: `0.937`.
- CF01 completed at `0.886`, improving the own-model branch but below A0/public
  top-code routes.
- PUBLIC941 ref `56243894` completed at **0.941**: +0.004 over the previous
  account best and +0.055 over CF01. Full-solution OOF remains unavailable.
