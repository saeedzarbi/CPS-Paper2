# Tube-v2 smoke (Phase 3 baselines)

Episodes per arm: 5

| trace | host | arm | QoE mean | rebuffer | joint_cov |
|-------|------|-----|----------|----------|-----------|
| 5g_markov | bba | `raw` | 31.77 | 13.424 | — |
| 5g_markov | bba | `static_conformal` | 58.79 | 5.374 | — |
| 5g_markov | bba | `tube_conformal` | 61.06 | 5.194 | — |
| 5g_markov | bba | `robust_mpc_logu` | 85.44 | 1.444 | — |
| 5g_markov | bba | `mpc_fixed_quantile` | 66.37 | 0.002 | — |
| 5g_markov | bba | `mpc_fixed_rho_0.6` | 81.80 | 3.484 | — |
| 5g_markov | bba | `mpc_fixed_rho_0.7` | 80.48 | 4.614 | — |
| 5g_markov | bba | `mpc_fixed_rho_0.8` | 61.59 | 9.344 | — |
| 5g_markov | bba | `mpc_fixed_rho_0.9` | 51.70 | 12.727 | — |
| 5g_markov | bba | `tube_v2_step_static_nocap` | 67.75 | 1.632 | 0.667 |
| bb | bba | `raw` | -271.90 | 78.648 | — |
| bb | bba | `static_conformal` | 20.21 | 1.640 | — |
| bb | bba | `tube_conformal` | 33.99 | 1.420 | — |
| bb | bba | `robust_mpc_logu` | 35.15 | 1.240 | — |
| bb | bba | `mpc_fixed_quantile` | 1.00 | 1.240 | — |
| bb | bba | `mpc_fixed_rho_0.6` | 26.59 | 1.240 | — |
| bb | bba | `mpc_fixed_rho_0.7` | 22.48 | 3.850 | — |
| bb | bba | `mpc_fixed_rho_0.8` | 31.87 | 2.320 | — |
| bb | bba | `mpc_fixed_rho_0.9` | 38.26 | 1.550 | — |
| bb | bba | `tube_v2_step_static_nocap` | 21.52 | 4.470 | 0.695 |
