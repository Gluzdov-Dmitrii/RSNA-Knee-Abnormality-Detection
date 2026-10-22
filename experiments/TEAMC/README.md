# TEAMC — fork of lavinwins/rsna-gold-medal without leaderboard-probed routes

Source: public notebook `lavinwins/rsna-gold-medal` (Mattia Angeli CoAtNet family D4/Global96/resgated/repairv1,
dreaddevelopment Raptor maxspan/native384/dense, DINOv2-20, A5, RadImageNet heads), not republished here.
Only change: a new default preset `kneerds` = the source's `v6a` with `route_shrink = 0.0`, so the five per-label
CoAt/Raptor routes that the source documents as public-LB probes (ACL 0.75, Medial Meniscus 0.80, Lateral Meniscus 1.00,
Lateral OA 0.75, Fracture 0.75) fall back to `base_coat_w` 0.62.

Result: submission 57359803, public **0.941** (18.10). The listed 0.958 of the source was its best score over all
versions; the current source code is the 0.943-lineage "v6", so the fork lands near 0.943 minus the removed probes.
Lesson: check the score of the exact version you fork, not the kernel's best score.
