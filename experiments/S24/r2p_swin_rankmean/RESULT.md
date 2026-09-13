# S24 r2plus1d + Swin3D-T rank-mean

- Status: `submitted` (shared decode, two complementary 5-fold 3D bags, no A0 mix)
- Local OOF: **0.8287** vs r2plus1d/Swin solo 0.811 and S25 r3d+r2p 0.822
- Kernel: `dmitriigluzdov/rsna-s24-r2p-swin-rankmean` **v2**, private, internet off, T4
- Visible: 3/3 studies, 53.2 s CUDA
- Hidden estimate ~6.5 h from 1322/3 scaling; engineering gate 8.5 h
- Submit once: ref **56204398** at 2026-09-13T09:51:58Z
- Do not retry this kernel version. Do not mix into A0.

v1 visible ERROR was a 3-row MCL rank tie, not a failed decode.
