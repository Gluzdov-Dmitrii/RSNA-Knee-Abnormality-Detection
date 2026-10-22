# PUB941 — public 0.941 control / probe22

- Status: `scored`, COMPLETE; public LB **0.941**, new account best (+0.004 vs 0.937).
- Public source: `ricecakar/rsna-knee-public-0-941-control`
- Local/private kernel: `dmitriigluzdov/rsna-knee-public941-control` v2
- Submit ref: `56243894` at `2026-09-15 03:09:09.290000`
- Submit route: code competition submit from kernel v2 output,
  `-k dmitriigluzdov/rsna-knee-public941-control -v 2 -f submission.csv`
- Monitor: `monitor-rsna-public941-submission` deleted after completion.
- Visible `submission.csv` SHA-256:
  `ff848c73ba6f31e487d175161304d26df307e6e777189c01dfb508a76532ddfc`
- Notebook SHA-256:
  `bb34d22e97d29f6cb0638bcf5effd1de9237778e420baba3d9b4258485855b91`
- Metadata SHA-256:
  `1463157d3001e2c8f49075fea5f81f34e2bef68fd76b977fabb3378633455021`

Validation:

- Visible output shape: `3 x 13`
- Schema and UID order matched `tmp/schema/sample_submission.csv`
- Prediction values finite, in `[0, 1]`, non-constant per visible target
- `probe22_outer_routing_receipt.json` status: `passed`
- Receipt output SHA matched `submission.csv`

Notes:

- v1 failed because `kernel_sources` were removed from pulled metadata; the
  notebook required `sofiaanjenje/rsna-knee-e13-train` for the intended E13
  `v52_e11_heads.pt` SHA.
- v2 restored kernel sources after removing the original foreign `id_no`, then
  completed visible run successfully.
- This is an inference-only public graph. The pulled notebook contains no full
  final-solution OOF artifact, so numeric OOF delta is not computable.
