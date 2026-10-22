# PUB939A: global outer weight ablation

Purpose: measure the combined public-LB contribution of target-specific outer
weights in PUB941 (verified LB 0.941). All model arms and inner weights match
PUB941 kernel v2. The global-weight parent uses outer Raptor/CoAt weight 0.60
for every target; the parent source describes a 0.939 score, not yet verified
on this account. This is a controlled ablation, not an expected new best.

Kernel: `dmitriigluzdov/rsna-pub939-global060-ablation`, version 1.
Primary output: `submission.csv` copied from `submission_0939_parent_exact.csv`.
Probe22 output is preserved as `submission_probe22_reference.csv`.

Original visible parent SHA256:
`7d3b8bd4e76b309171e2c44a58301e71da26104d1049cc9f8c445c17eee3c770`.
Original visible output validated: schema, UID order, finite values in [0,1],
shape 3 x 13, receipt hash match. This does not validate hidden-test performance.

Direct submission of the auxiliary CSV returned HTTP 400. A dedicated private
kernel packages the same parent as the required primary submission.csv.

Status: submitted, ref `56275828`, `PENDING`.
Submitted at UTC `2026-09-16 09:58:04.273000`.
Completed kernel output passed schema/UID/range validation and exactly matched
the original parent SHA256. Four submissions remained today after acceptance.
Our local OOF: 0.832874589670475. Final public-graph OOF unavailable.
