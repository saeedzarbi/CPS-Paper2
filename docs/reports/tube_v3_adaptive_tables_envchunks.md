
## Key table (primary = rule-selected max-score `_safe_fs`, env_chunks calibration) vs RobustMPC, test

| set | primary arm | QoE | rebuf (s) | joint cov | ΔQoE vs RobustMPC (cluster) | Δrebuf (cluster) |
|---|---|---|---|---|---|---|
| 5g_markov | `tube_v2_maxscore_a0.15_safe_fs` | 11.2 | 1.37 | 0.869 | +40.1 [+24.1,+56.7] | -11.96 [-15.38,-8.67] |
| 5g_harsh | `tube_v2_maxscore_a0.2_safe_fs` | 81.0 | 0.34 | 0.846 | -0.1 [-6.2,+6.4] | -2.14 [-3.15,-1.27] |
| bb | `tube_v2_maxscore_a0.2_safe_fs` | -32.5 | 16.27 | 0.781 | +0.9 [-1.0,+3.0] | -0.01 [-0.31,+0.28] |
| 5g_irish | `tube_v2_maxscore_a0.1_safe_fs` | -22.6 | 5.96 | 0.949 | +123.0 [+39.8,+215.7] | -36.13 [-59.60,-14.58] |

Full tables: `docs/reports/tube_v3_adaptive_test_envchunks.md`. Figures: `docs/figures/tube_v3_adaptive_envchunks/` (on server).
