# META / label-origin findings — 2026-10-16

Inputs: `out/series_meta.csv` (kernel dmitriigluzdov/rsna-dicom-meta-scan v1; one DICOM header per series, train + 3
visible test studies), `data/official/train.csv` (58 gold rows), LLM label sets in `tmp/labels/`.

## Facts
- PatientID is a per-study hash: 4410 studies → 4410 IDs; no study shares a patient. Patient linkage is impossible.
- Allowlisted tags present: Manufacturer, ManufacturerModelName, SoftwareVersions, MagneticFieldStrength, ReceiveCoilName,
  SeriesDescription (777 values), PatientSex (95 %), Laterality (49 % of series). No age, dates, institution or station.
- 108 scanner fingerprints (manufacturer|model|software) in train; largest 8 %.
- Label AUC vs the 58 gold rows (label sets built from report text only): steven v4 blend mean 0.893, lixin binary 0.835.
  Weakest: Synovitis 0.79, Fracture 0.79, Lateral OA 0.83, Contusion 0.86, Effusion 0.88 (binary 0.70).
- lixin vs gold disagreements: 108 of 696 cells. Synovitis: 14 gold-positive cells the LLM marks negative.
  Effusion: 14 gold-negative cells the LLM marks positive (e.g. a report that explicitly describes a moderate effusion has gold 0).
  Two synovitis FN reports do not mention synovitis at all.
- Scanner fingerprint alone (5-fold target encoding) predicts the report-derived labels with mean AUC 0.647
  (Synovitis 0.746, ACL 0.680, Lateral OA 0.679); PatientSex alone 0.546.
- Gold prevalence is high (Effusion 0.60, Synovitis 0.47, ACL 0.41, Fracture 0.31): an enriched sample.

## Hypotheses (not proven)
1. Gold (and probably test) labels are an expert image read with its own per-finding definitions, not report extraction.
   Evidence: report-explicit findings labelled 0, gold positives absent from the report. 58 rows only.
2. Part of every report-derived label is the clinic's reporting habit (scanner fingerprint predicts labels at 0.65);
   models trained on these labels learn that habit, which is noise for an expert-read target.
3. AUC ignores where the expert threshold lies, so ordinal severity targets from reports should transfer better than
   binary or LLM-confidence targets. Testable at label level on the 58 gold rows before any training.

## Gold58 as a selector for the second final pick (2026-10-17)
- Provenance: no strong public member is clean of the 58. R6 manifest: gold_in_training 0, but run_config.gold_studies 58
  (evaluated each run) and its Effusion fluid / Baker's state stackers come from `rsna_align_probe/runs/r6_meta_cnx`;
  S6: train_studies 4263, gold_studies 58 (evaluated); Raptor: selected on the 58 (079 notes); MaViT: ships
  train_golden58.csv for its local evaluation.
- Paired bootstrap (1000) on the 58, rank AUC (experiments/TEAMA/rehearsal_b_out): A 0.9372 (95 % CI 0.909–0.959);
  B 0.9384, 079-like 0.9391, compact-only 0.9407, equal-5 0.9380 — all CIs of the difference vs A include 0
  (widths ±0.003–0.008). Only Raptor-only (0.924) and MaViT-alone (0.920) are reliably worse than the blends.
- Conclusion: the 58 cannot rank our candidates; using them would select noise plus the members aligned on them.
