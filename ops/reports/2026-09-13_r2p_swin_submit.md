# Daily report: 2026-09-13 — r2plus1d+Swin rank-mean submit

## Before

- UTC timestamp: 2026-09-13T09:51:58Z
- Remaining submissions: 5/5 before this packet; after submit 4/5
- Weekly GPU: 3.11 h used before this visible run, reset 2026-09-19T00:00:00Z
- Plan: R2 submit mode. S11–S15 still blocked on DINO_CACHE_V1.
- Team best: S01 A0 0.937. Own-model LB: S25 0.861, r2plus1d 0.859, r3d 0.852, CoAtNet P03 0.875.

## Planned batch and readiness

| ID | Hypothesis | Artifact | Ready/blocker |
|---|---|---|---|
| S24 r2p+Swin rank-mean | complementary 3D bags despite solo tie | Swin export + visible 3/3 53.2 s | submitted v2 |
| S25 r3d+r2p | two best CNN 3D bags | already scored 0.861 | do not retry |
| S24 Swin standalone | beat r2plus1d 0.811 | OOF tie 0.810674 | defer_submit |
| S11–S15 | DINO heads | missing DINO cache | blocked |
| A0 overlay | mix own CNN into A0 | public 0.931 | do-not-retry |

## Results

| ID | Kernel/version | Ref | Visible | Public | Decision |
|---|---|---|---:|---|---|
| S25 r3d+r2p | `dmitriigluzdov/rsna-s25-r3d-r2p-rankmean` v1 | 56176352 | 48.9 s | **0.861** | +0.002 vs r2plus1d 0.859 |
| S24 r2p+Swin | `dmitriigluzdov/rsna-s24-r2p-swin-rankmean` v2 | 56204398 | 53.2 s | PENDING | poll only |

Local OOF: r2p+Swin 0.8287 vs S25 0.822. Hidden estimate ~6.5 h from visible 53.2 s × 1322/3. Kernel private, internet off, T4. Submitted once. v1 visible ERROR was a 3-row MCL rank tie.

## Interpretation

S11–S15 still not ready. Next independent hypothesis after S25 is Swin complementarity with the current 3D LB leader. No A0 mix. Standalone Swin remains a tie and was not submitted.

## Prepared for next package

- Poll 56204398 read-only until scored. Do not retry 56204398 / 56176352.
- Next local work: DINO cache for S11, or three-member r3d+r2p+Swin only if this pair beats S25 and hidden time still fits.
- Blocker for S11–S15: DINO_CACHE_V1.
