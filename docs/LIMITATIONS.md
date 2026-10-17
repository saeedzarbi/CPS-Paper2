# Limitations

1. **bb joint coverage (env_chunks `_fs`).** With chunk-aligned calibration and rule-selected primary
   (`maxscore_a0.2_safe_fs`), test joint coverage on bb is **0.781** vs nominal **0.80** (α=0.20); with `a0.15_safe_fs` on the
   same test it is **0.834**. Raw-tick `_fs` (previous adaptive test) gave **0.646**. bb QoE gain vs RobustMPC remains small
   (+0.9 cluster CI) and rebuffer is unchanged; outage traces dominate tails (p90 rebuffer ~109.5 s for all methods).
2. **Zero rebuffer is conditional.** The lemma guarantees no rebuffer only for a committed plan, under joint coverage, on
   non-fallback steps. The startup chunk and fallback steps are not covered; observed rebuffer is not zero.
3. **5g_harsh QoE.** QoE difference vs RobustMPC is not significant (−0.1 cluster); fixed-ρ / fixed-quantile MPC have higher QoE
   (~88) but ~3× more rebuffering than the primary tube arm.
4. **Per-family α.** α is selected per family on val (`configs/tube_v3_alpha_rule.yaml`); a single global α lowers harsh QoE
   (see `alpha_selection_envchunks.json`).
5. **Prior work.** A weighted max-score conformal construction for multi-step prediction already appears in
   Cleaveland et al., AAAI 2024; we make no novelty claim for the max-score tube itself, only for its use in ABR MPC with a
   pre-refill safety constraint, chunk-aligned `_fs` calibration, and the evaluation protocol.
6. Small trace counts per family (e.g. 12 traces on 5g_markov) → wide cluster CIs.

Headline test report (env_chunks): `docs/reports/tube_v3_adaptive_test_envchunks.md`.
