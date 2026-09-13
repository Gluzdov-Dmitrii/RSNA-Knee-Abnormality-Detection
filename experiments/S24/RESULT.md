# S24 — r3d_18 9-slice 3D on PIXEL_CACHE_V1

- Status: `scored` (standalone 5-fold bag, no A0 mix)
- Local OOF: **0.800641** vs S22 0.768
- Public LB: **0.852** (ref 56158999 COMPLETE) vs S22 0.825, CoAtNet 0.875, A0 0.937
- Kernel: `dmitriigluzdov/rsna-s24-r3d18-9slice` v1, private, internet off, T4
- Visible: 3/3 studies, 25.9 s
- Do not retry this kernel version. Do not mix into A0.

Single-model public control is `r2plus1d_v1/` **0.859**. S25 is public **0.861**. r2p+Swin rank-mean is ref 56204398.
