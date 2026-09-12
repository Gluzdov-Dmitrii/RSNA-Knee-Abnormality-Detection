# S25 — rank-mean of the two best pixel 3D branches

Locked architecture-batch ensemble: two best S21–S24 members that already have
exported weights and independent public scores.

Members:

- S24 r2plus1d_18 9-slice OOF 0.811 / public 0.859
- S24 r3d_18 9-slice OOF 0.801 / public 0.852

S23 ConvNeXt-Tiny lost. S21 was not an independent pixel OOF. Same-seed (2026)
S22+r3d rank-mean OOF 0.806 lost to the two 3D members (0.822) and is not submitted.
Swin3D-T tied r2plus1d on OOF and is not in this kernel.

Combine: per-target rank-percentile mean of the two 5-fold bags. No A0 mix.
Seed 2027 retrains are not in this file.
