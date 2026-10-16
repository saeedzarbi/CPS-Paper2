# Limitations

1. **bb coverage failure.** Joint coverage of the primary arm on bb is 0.646 vs nominal 0.85. `_fs` calibrates on raw trace
   samples, which do not match chunk-level throughput on bb, so exchangeability is empirically violated. The coverage theorem does
   not apply there; bb QoE gain (+2.7) is small and rebuffer is unchanged. bb test also contains outage traces (p90 rebuffer 109.5 s
   for every method).
2. **Zero rebuffer is conditional.** The lemma guarantees no rebuffer only for a committed plan, under joint coverage, on
   non-fallback steps. The startup chunk and fallback steps are not covered; observed rebuffer is not zero.
3. **5g_harsh QoE.** QoE difference vs RobustMPC is not significant (−1.4); fixed-ρ / fixed-quantile MPC have higher QoE (88–89)
   but ~3× more rebuffering.
4. **Per-family α.** α is selected per family on val; a single global α=0.1 lowers harsh QoE (65.1).
5. **Prior work.** A weighted max-score conformal construction for multi-step prediction already appears in
   Cleaveland et al., AAAI 2024; we make no novelty claim for the max-score tube itself, only for its use in ABR MPC with a
   pre-refill safety constraint and the evaluation protocol.
6. Small trace counts per family (e.g. 12 traces on 5g_markov) → wide cluster CIs.
