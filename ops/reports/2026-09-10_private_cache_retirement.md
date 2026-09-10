# Private cache replacement and notebook v9

User authorization: create a private copy of the existing MRI cache, delete the
public Dataset after confirming the copy, and retain the revised notebook publicly.
No competition submission or support message is authorized by this operation.

## Notebook

Published version 9 of `dmitriigluzdov/knee-mri-in-11-gib`, kernel 133521917,
using QUICK_SAVE. Remote source was pulled back and checked: public, CPU only,
internet/GPU/TPU disabled; exactly the official competition and Pilkwang inputs;
no old cache slug anywhere in source; no code outputs; two embedded figures;
four hidden implementation cells and one visible settings cell near the top.

Replaced the cache download with rule 2.4.b.1 and private generation instructions.
The settings change cache geometry only, preserving the locked curve protocol.
Defaults remain 224 pixels, nine slices, 130 mm, window .35-.65. The independent
probe can be skipped with RUN_VERIFIER=False. SAVE_CACHE_OUTPUT=False preserves
the temporary MRI destination for public rebuilds. Readers must make their own
notebook private before retaining MRI outputs and creating a private Dataset.

No full MRI rebuild was run for this editorial/configuration change. Existing
measured figures are retained. Focused tests passed for default and custom
160-pixel/six-slice/110-mm DICOM materialization, shape, uint8, masks, hashes,
invalid slice counts and refusal to mix recipes in an existing output folder.
Existing geometry, spatial-feature and paired-bootstrap tests also passed.

## Cache copy validation

The local source cache had been cleaned up before this task. Downloaded the
public Dataset and validated all 35 shards against their recorded SHA-256,
shape, dtype and masks, plus the exact set of 4,407 official training study IDs.
41 files; 11,941,204,980 total bytes; pixel payload 11.120721817 GiB.
No reports or labels included.

SPEC.json SHA-256:
`043485b6c0ea543a5f4061ce15b6fef868bb28aebc2af6d124adfd940c9aaf50`.

The new slug is `dmitriigluzdov/rsna-knee-uint8-224-9-c130-personal`.
The original archive contents are preserved, including the old URL as historical
provenance in SPEC.json; the new private card explains this. No public link to
either cache is included in notebook v9.

Runtime receipts (ignored artifacts): `cache_validation.json`,
`notebook_publication.json`, `private_replacement.json` and
`public_cache_retirement.json` under `artifacts/cache_budget_recheck/`.

## Completed remote operations

Private replacement created: Dataset ID 11968817, version 1, ready,
`is_private=True`. All 41 remote names and byte sizes match the validated local
copy, and downloaded SPEC.json is byte-identical. Verified 2026-09-10 08:32 UTC.

Original Dataset ID 11946093 was deleted successfully at 08:32:21 UTC after
rechecking the replacement. An anonymous check then returned HTTP 404 for the
old Dataset page, 403 for its SPEC download, and 403 for the new private SPEC
download. Previously downloaded third-party copies cannot be revoked by deletion.
The new private copy has not been published or shared with collaborators.

No support message was sent. Browser inspection was unavailable in this session;
the notebook's saved source, metadata, figures and outputs were checked through
the Kaggle API instead. The completed private source remains available locally
under `artifacts/cache_budget_recheck/private_copy`.
