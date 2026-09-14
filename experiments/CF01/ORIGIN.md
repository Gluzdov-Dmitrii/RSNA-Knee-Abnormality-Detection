# CF01 — cross-family rank-mean of compact CoAtNet and r2p+Swin

S01–S10 and A0 overlay stay frozen. Own 3D bags moved public LB from S22 0.825
to r2plus1d 0.859, S25 0.861, then r2p+Swin 0.871. That last step is real, but
it still sits 0.004 below compact CoAtNet P03 (0.875) and 0.066 below A0 (0.937).
Another 3D-only rank-mean is the same family, not a new mechanism.

CF01 keeps both scored own graphs independent and rank-means them:

- Compact CoAtNet P03 (`dmitriigluzdov/rsna-knee-compact-coatnet`), public 0.875
- S24 r2plus1d_18 + Swin3D-T rank-mean bag, public 0.871

The 3D member is decoded first with the existing PIXEL_CACHE_V1 vol9 path so
pydicom stays on the docker that produced 0.871. CoAtNet then installs its
offline wheels and runs `infer.py` unchanged. Equal per-target rank-percentile
mean; no weight search on public LB; no A0/Raptor mix.

There is no locked 5-fold OOF for this pair: CoAtNet was a 4,349-study refit,
not FOLDS_V1 OOF on 4,406. The hypothesis is complementary families, not a
third 3D backbone.
