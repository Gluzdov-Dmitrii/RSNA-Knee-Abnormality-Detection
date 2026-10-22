# 2026-10-18 15:33 UTC: C submitted 57359803 (PENDING). Arseniy's 093 (D4 addition to 079) failed hidden rerun.
# Arseniy's handover package: RSNA_experience_2026-10-16/ (99 experiments; confirms R6 v1/v2 trap; own-model additions ≈0 or negative).

# 2026-10-17 10:00 UTC: B 57294542 COMPLETE public 0.956 (A 0.957); submit→score 5.85 h. B is not a final pick.

# 2026-10-17 09:45 UTC: C visible run PASS (263 s, preset kneerds, all members). Owner: submit C on 18.10 (free slot):
#   kaggle competitions submit rsna-knee-abnormality-detection -k dmitriigluzdov/rsna-team-c-goldmedal-noprobe -v 1 -f submission.csv
# Final plan: A + C unless C scores clearly below 0.956. B 57294542 pending.

# RSNA status — 2026-10-17 09:30 UTC
- Owner 17.10: final = A + one more candidate (079 doubtful). Search for a different-lineage second pick.
- Found lavinwins/rsna-gold-medal: public 0.958 with NO nartaa/heliosli components (Mattia CoAt family D4/Global96/resgated/repairv1,
  dreaddevelopment Raptor maxspan/native384/dense, DINOv2-20, A5, RadImageNet heads). Its 5 per-label routes are documented
  LB probes. Fork experiments/TEAMC (preset 'kneerds' = v6a + route_shrink 0) visible run RUNNING. Submit needs 18.10 free slot or owner OK.
- Public scores now via anonymous internal ListKernels (tmp/team_2026-10-16/top/scores.json); browser pane navigation was refused.

# RSNA status — 2026-10-17 04:15 UTC
- B submitted: 57294542 (kernel dmitriigluzdov/rsna-team-b-ensemble v1), PENDING; visible run 562 s, 0 failures. 17.10 free slot used.
- Gold58 cannot select the 2nd final pick (see experiments/META/FINDINGS.md). Final pair stays A + 079 unless the owner changes it.

# Owner decision 2026-10-16: FINAL PAIR = A (57267685, 0.957) + 079 (57228091, 0.957). B goes 17.10 as the free submission (info/backup only).

# RSNA status — 2026-10-16 17:45 UTC

- Variant A sub 57267685 COMPLETE: public 0.957 (same as 079); submit→score ≈ 6.1 h incl. queue, so the 4-member graph fits.
- Next: 17.10 free submission = variant B (kaggle/scored_b; push visible run first). Final pair decided after B's score.

# RSNA status — 2026-10-16 17:30 UTC

- B rehearsal (58 gold + 6 random) PASSED, 0 failures. Marginal: R6 plain 11.4 s/study/worker; MaViT ~1.1 s/study;
  est. hidden B ≈ 5.2 h (A ≈ 6.8 h).
- MaViT is NOT independent: mean Spearman 0.945 with Raptor-384 (0.87 with R6). Gold58 (contaminated) blend AUC:
  w_mavit 0 → 0.9371, 0.15 → 0.9376. B ≈ A with more runtime margin. Owner: B goes as the 17.10 free submission.
- A (57267685) still PENDING; background poll running.

# RSNA status — 2026-10-16 16:50 UTC

- SUBMITTED variant A v1: sub 57267685 (kernel dmitriigluzdov/rsna-team-a-ensemble v1), PENDING; visible run OK 567 s.
  Receipt experiments/TEAMA/submission_receipt_v1.json. Today's free submission is used.
- Variant B built (experiments/TEAMA/kaggle/scored_b): A with R6 plain (no flip TTA) + hengck23 MaViT-288 (public 0.935,
  independent family, ~1.5 s/study) at 0.15; rule RULE_B in build_notebook.py. Rehearsal rsna-team-b-ensemble-rehearsal v1
  RUNNING on 58 gold + 6 random train studies (timing + contaminated gold58 diagnostics).
- Untouched candidates from ListKernels (no crowd dataset): hengck23 MaViT 0.935; tonylica DINO-RadImageNet 0.940 (41 ckpts,
  heavy, includes DINOv2 0.899 branch); handudu teacher-probs 0.942 (private dataset, unusable).
- B submission needs the owner's OK or waits for 17.10 free slot.

# RSNA status — 2026-10-16 16:30 UTC

- Five-member rehearsal (64 train studies, v1) PASSED all stages, 0 failures; R6-R7 Spearman 0.968 (near duplicates).
  Marginal s/study/worker: R6-TTA 23, R7 12; Raptor384 ~1.5 s/study, Raptor224 ~1, S6 ~3.9 → five members ≈ 9.0 h for 1300: too long.
- Variant A now = Raptor384 + Raptor224 + R6-TTA + S6 (R7 dropped, documented in RULE), est. ≈ 6.8 h.
- Scored kernel dmitriigluzdov/rsna-team-a-ensemble v1 pushed (visible run). Rehearsal of this build waits for a free GPU slot
  (Kaggle limit 2 batch GPU sessions). Then: check visible outputs → submit once (today's free submission).
- Label-origin findings: experiments/META/FINDINGS.md (PatientID per-study; gold looks like expert image re-read; site predicts report labels 0.65).

# RSNA status — 2026-10-16 15:50 UTC: variant A in rehearsal

- Owner: goal = reliable private solution, not higher public. Claude builds variant A; 1 submission/day free, more by approval.
- Variant A = five public models (Raptor 384/224, R7, R6-TTA, S6), fixed family weights; see experiments/TEAMA/README.md.
- Rehearsal kernel dmitriigluzdov/rsna-team-a-five-model-rehearsal v1 pushed (64 seeded train studies) — RUNNING.
- Next: read a5_rehearsal_report.json; if est. total <= ~7 h push kaggle/scored, check visible outputs, submit once.

# RSNA status — 2026-10-16 (team Kneerds = arsgorynich + dmitriigluzdov)

- Team best 0.957: sub 57228091 (arsgorynich, 15.10), kernel arsgorynich/rsna-079-r6tta-three-arm-pinned v358648366.
  Components, all public: nartaa Raptor CoAtNet-384 SWA + 224-crop SWA, Heliosli R6 ConvNeXt-small (4 folds, flip TTA);
  per-class rank blend 0.5 R6 / 0.5 Raptor (384 .65 / 224 .35). Visible run 470 s on T4x2. No own model inside.
- Visible submission.csv identical (maxdiff 0 on 3 studies) to heliosli/this-is-best-public-dataset-new-version-0-9576.
- LB 2026-10-16 (6073 teams): gold r22 0.963, silver r303 0.958, bronze r607 0.957; we are rank 598 inside a 367-team tie at 0.957.
  Crowd 0.956-0.958 ≈ 1300 teams, all permutations of R6/R7/S6/Raptor public checkpoints. Test ≈1300 studies.
- Own team models: DINOv2-S distill 0.916 best; 4 hidden-rerun errors on 11.10.
- Files: tmp/team_2026-10-16/{k079,k075,out079,pubout,lb}.

# RSNA status update — 2026-09-30

- SPRINT943 (56464896) FAILED on hidden rerun: "unhandled error while rerunning"; no score.
  Best confirmed remains PUB941 0.941 (~rank 1036/4608).
- Live LB 2026-09-30 04:05 UTC (`tmp/lb_2026-09-30/`): 4608 teams; gold rank 19 = 0.956,
  silver rank 230 = 0.944, bronze rank 460 = 0.943 (588-team tie at 0.943, ranks 258-845).
- Public notebooks at 0.945 exist (pjmathematician/rsna-knee-d4-lite, aastikrajan15/knee-s75-w50)
  but list private dataset inputs; forkability not verified.
- Decision context: user's priority is LoRA / other competitions; RSNA = low-effort lottery only.

# RSNA current state — 2026-09-22, one-hour sprint submitted

- Current objective (user): strongest feasible LB solution within one hour, with
  private-generalization awareness and an understandable, extensible notebook.
- Window: 11:56–12:56 UTC. Chosen exact Speedy Raptors **V34 / 351665011**, author
  LB 0.943. Source export obtained and SHA pinned; it has **four** CoAt branches,
  including Repair-v1 bundled inside D4 v2; usual outer weight .60.
- Artifact: `experiments/SPRINT943/kaggle/rsna-knee-v34-documented-strict.ipynb`;
  23 documented editable stages in `runtime/`, 50 notebook cells, preview.html,
  README.md, source_v34/, input_lock.json and build/package receipts alongside.
- Validation: 10 hardening CPU tests passed; 25 code cells compile; stage parity
  and all 14 dataset + 2 kernel version pins verified. Full visible GPU run PASS
  on T4x2 in 346.995s; all families complete; CSV byte-identical to author V34.
- Kaggle V1 launched 12:45:40 UTC and is COMPLETE; kernel
  `dmitriigluzdov/rsna-knee-v34-documented-strict`, scriptVersionId 351861109.
  User explicitly chose WAIT, not cancel Biohub jobs; none were stopped.
  Both bounded launcher processes ended. No automations/callbacks/background waiters.
  Durable launch proof: `experiments/SPRINT943/launch_receipt.json` and launch_last.log.
- Submitted once: **56464896**, 12:53:35 UTC, V1, submission.csv; verified
  **PENDING** 12:54 UTC. No own SPRINT943 score yet.
- Next: read submission 56464896 result and update RESULT.json. Compare with
  PUB941 0.941. If hidden inference fails, inspect its error before retrying.
  No automatic monitors. Outputs: `experiments/SPRINT943/visible/`; submission
  receipt alongside RESULT.json. Report: `ops/reports/2026-09-22_v34_sprint.md`.
- Confirmed fallback: PUB941 **0.941**, submission 56243894. No new training.
- Development evaluator: `experiments/SPRINT943/evaluate_oof.py`, VALIDATION.md.
  Existing R2P legacy .8106031301 / known-only .8389554743, coverage reported;
  this is NOT OOF of the new public graph. Full-graph OOF remains unavailable.
- Extra evidence: actual author V34 receipts and four CoAt NPZ pass release checks;
  CPU replay of the final blend is byte-identical to the author visible CSV. This
  is arithmetic/schema regression, not a new GPU inference. D4 per-study JSONL
  artifacts unavailable from source listing; that additional check not replayed.

## Earlier analysis context

- Objective: compare our self-trained cluster solution with current public bases;
  provide an evidence-based improvement plan. Review complete; no training launched.
- Decision: public scored base is the main competition track; bounded own training
  on a reproducible public architecture is the research track.
- Live Kaggle verified: CF01 own-model best 0.886 (56220447); PUB941 reproduced public
  best 0.941 (56243894); PUB939A 0.939 COMPLETE (56275828, old experiment JSON stale).
- Current external candidate: Mattia Speedy Raptors V34 0.943,
  scriptVersionId 351665011; Jiwei V8 and Maverick V7 both 0.942.
  Latest downloaded notebook is not certified as scored V34. Leaderboard top 0.958.
- Latest own controlled new train P04: rejected; no refit/submit. CF01 is an ensemble
  of previous checkpoints. Rebuilt four-model OOF blend 0.83287459 is weight-selected
  on the same OOF; not unbiased evaluation of the selected blend and not CF01 OOF.
- Weak labels: UNK=0.28 in 27.32% of 52,872 cells; current AUC thresholds UNK as
  negative. Own fold-out CSVs were independently recalculated, UID/folds checked.
- Plan/evidence: `ops/reports/2026-09-22_solution_review_and_plan.md`.
- Source snapshot: `tmp/audit_2026-09-22/speedy_raptors/` (inference code only).
- Rejected approaches: repeat P04, old CNN blend weight sweeps, treating public
  competition checkpoints evaluated on train as OOF, assuming fork independence.
- Running jobs started by this analysis: none. Live cluster capacity not checked.
- Next action: obtain exact scored public version + pin inputs, validate runtime;
  define validation/UNK policy and pilot a trainable DINO recipe before large runs.
