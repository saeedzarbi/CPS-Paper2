# Clean ABR — extended validation (`clean_v3_extended`)

Headline benchmark unchanged: [`clean_v3_full_metrics.md`](clean_v3_full_metrics.md).
Extended artifacts: `results/clean_v3_extended/`.

## Task 2 — OOD & mobility generalization

Headline arms on extended trace sets. **Bold** = best in row (rebuffer & smoothness: minimum; QoE: maximum).

#### Mean rebuffer (s/ep)

| Trace | Policy | raw | static | tube | p(tube vs static) |
|---|---|---|---|---|---|
| `5g_v18` | mpc | 1.18 | **0.00** | 0.00 | 1.00e+00 |
| `5g_v18` | greedy | 2.83 | **0.00** | 0.00 | 1.00e+00 |
| `5g_v18` | bba | **0.00** | 0.00 | 0.00 | 1.00e+00 |
| `5g_v18` | bola | **0.00** | 0.00 | 0.00 | 1.00e+00 |
| `5g_harsh` | mpc | 6.13 | 3.84 | **3.12** | 1.01e-04 |
| `5g_harsh` | greedy | 68.79 | 7.14 | **4.86** | 1.09e-14 |
| `5g_harsh` | bba | 6.91 | 3.74 | **3.28** | 4.89e-03 |
| `5g_harsh` | bola | 6.90 | 3.42 | **2.71** | 2.23e-04 |
| `norway_vehicular` | mpc | 5.13 | 0.79 | **0.65** | 1.30e-03 |
| `norway_vehicular` | greedy | 923.76 | 1.40 | **0.62** | 4.79e-26 |
| `norway_vehicular` | bba | 70.60 | 1.22 | **0.61** | 3.83e-22 |
| `norway_vehicular` | bola | 19.18 | 0.97 | **0.60** | 9.24e-10 |
| `fcc_fixed` | mpc | 9.69 | 2.60 | **1.95** | 3.09e-02 |
| `fcc_fixed` | greedy | 976.74 | 3.60 | **3.57** | 7.68e-06 |
| `fcc_fixed` | bba | 98.84 | **2.59** | 2.70 | 1.47e-02 |
| `fcc_fixed` | bola | 18.10 | **1.96** | 2.13 | 3.39e-03 |

#### Mean smoothness (|Δlog u|)

| Trace | Policy | raw | static | tube | p(tube vs static) |
|---|---|---|---|---|---|
| `5g_v18` | mpc | **0.06** | 0.12 | 0.12 | 3.14e-01 |
| `5g_v18` | greedy | **0.00** | 0.09 | 0.10 | 1.10e-08 |
| `5g_v18` | bba | 0.21 | 0.22 | **0.20** | 5.59e-10 |
| `5g_v18` | bola | 0.26 | 0.26 | **0.24** | 4.04e-05 |
| `5g_harsh` | mpc | **0.17** | 0.27 | 0.23 | 1.99e-15 |
| `5g_harsh` | greedy | **0.00** | 0.37 | 0.24 | 9.70e-35 |
| `5g_harsh` | bba | 0.40 | 0.41 | **0.27** | 1.24e-34 |
| `5g_harsh` | bola | 0.68 | 0.53 | **0.35** | 8.38e-35 |
| `norway_vehicular` | mpc | **0.09** | 0.15 | 0.12 | 9.86e-16 |
| `norway_vehicular` | greedy | **0.00** | 0.59 | 0.13 | 3.17e-35 |
| `norway_vehicular` | bba | 0.16 | 0.54 | **0.13** | 3.17e-35 |
| `norway_vehicular` | bola | 0.78 | 0.57 | **0.14** | 3.17e-35 |
| `fcc_fixed` | mpc | **0.09** | 0.18 | 0.11 | 1.05e-33 |
| `fcc_fixed` | greedy | **0.00** | 0.57 | 0.13 | 3.17e-35 |
| `fcc_fixed` | bba | 0.17 | 0.54 | **0.13** | 3.17e-35 |
| `fcc_fixed` | bola | 0.76 | 0.56 | **0.15** | 3.17e-35 |

#### Mean episodic QoE

| Trace | Policy | raw | static | tube | p(tube vs static) |
|---|---|---|---|---|---|
| `5g_v18` | mpc | **131.55** | 131.19 | 129.39 | 1.10e-06 |
| `5g_v18` | greedy | 131.63 | **135.86** | 133.45 | 3.16e-19 |
| `5g_v18` | bba | **124.06** | 123.22 | 122.88 | 2.21e-01 |
| `5g_v18` | bola | 119.49 | **119.71** | 119.01 | 1.38e-01 |
| `5g_harsh` | mpc | 76.49 | **78.58** | 75.68 | 6.18e-11 |
| `5g_harsh` | greedy | -152.01 | 70.48 | **74.42** | 1.52e-03 |
| `5g_harsh` | bba | 68.12 | **77.38** | 75.90 | 8.20e-02 |
| `5g_harsh` | bola | 48.18 | 69.32 | **69.62** | 3.03e-01 |
| `norway_vehicular` | mpc | 29.71 | **42.90** | 40.58 | 6.43e-12 |
| `norway_vehicular` | greedy | -3828.36 | 27.46 | **43.45** | 3.62e-35 |
| `norway_vehicular` | bba | -236.83 | 29.90 | **43.25** | 7.56e-35 |
| `norway_vehicular` | bola | -64.29 | 27.70 | **41.51** | 5.80e-35 |
| `fcc_fixed` | mpc | 12.09 | 35.53 | **37.23** | 1.13e-02 |
| `fcc_fixed` | greedy | -4056.17 | 18.51 | **31.63** | 2.12e-22 |
| `fcc_fixed` | bba | -355.84 | 23.16 | **34.77** | 2.17e-20 |
| `fcc_fixed` | bola | -58.71 | 22.90 | **35.00** | 4.58e-24 |

## Task 3 — Content complexity (VBR / SI-TI strata)

Videos grouped by `siti_summary.csv` motion/complexity labels and VBR chunk-size CV (top tertile → high).

| Stratum | Videos | Δ rebuffer (tube−static) s/ep | Δ smoothness (tube−static) | n episodes |
|---|---|---|---|---|
| `low_medium` | ducks, into_tree, rush_hour, sintel | -0.918 [-1.185, -0.650] | -0.2416 | 512 |
| `high_vbr_motion` | bigbuckbunny, cosmos, elephants_dream, kristen_and_sara, old_town_cross, parkjoy, sunflower, tearsofsteel | -0.988 [-1.179, -0.797] | -0.2587 | 1120 |

## Task 4 — Component ablation (`test_traces_5g_markov`, bba & bola)

`sym. tube`: symmetric ±1-tier rate limits (no static emergency cap). `fixed-α tube`: fixed α₀ bounds without online ACI updates.

#### Mean rebuffer (s/ep)

| Policy | raw | heur. 0.85 | static H=1 | sym. tube | fixed-α tube | tube (proposed) |
|---|---|---|---|---|---|---|
| bba | 11.19 | 7.88 | 4.54 | 7.17 | **2.94** | 3.41 |
| bola | 10.33 | 7.30 | 4.31 | 6.58 | 3.31 | **2.93** |

#### Mean smoothness (|Δlog u|)

| Policy | raw | heur. 0.85 | static H=1 | sym. tube | fixed-α tube | tube (proposed) |
|---|---|---|---|---|---|---|
| bba | 0.49 | 0.48 | 0.50 | **0.22** | 0.27 | 0.35 |
| bola | 0.85 | 0.78 | 0.66 | **0.22** | 0.38 | 0.44 |

#### Mean episodic QoE

| Policy | raw | heur. 0.85 | static H=1 | sym. tube | fixed-α tube | tube (proposed) |
|---|---|---|---|---|---|---|
| bba | 41.30 | 55.36 | 63.08 | 62.08 | **67.82** | 64.76 |
| bola | 18.45 | 35.84 | 50.57 | **63.38** | 56.50 | 57.16 |

### Horizon sweep (α=0.10)

H=1: exact 1-step `static_conformal`. H∈{2,3,5}: `tube_conformal` with reserve ramping to full H=5.

#### Mean rebuffer (s/ep)

| Policy | H=1 | H=2 | H=3 | H=5 |
|---|---|---|---|---|
| bba | 4.49 | **0.34** | 0.92 | 3.54 |
| bola | 4.28 | **0.05** | 0.76 | 3.18 |

#### Mean smoothness (|Δlog u|)

| Policy | H=1 | H=2 | H=3 | H=5 |
|---|---|---|---|---|
| bba | 0.50 | **0.04** | 0.15 | 0.34 |
| bola | 0.66 | **0.01** | 0.18 | 0.44 |

#### Mean episodic QoE

| Policy | H=1 | H=2 | H=3 | H=5 |
|---|---|---|---|---|
| bba | 63.09 | 36.98 | **72.47** | 64.75 |
| bola | 50.97 | 14.14 | **71.24** | 56.88 |

### α sweep (tube, H=5)

| Policy | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|---|---|---|---|---|
| bba | 2.88 | 3.18 | 3.63 | 3.75 |
| bola | 2.62 | 3.01 | 3.27 | 3.66 |

### Extended statistics on headline `clean_v3`

## Task 4 — Extended statistics (`results/clean_v3/`)

Paired bootstrap 10k resamples for tube − static; CVaR₉₅ on episode rebuffer; zero-stall rate; decision latency (`perf_counter_ns`, mean μs per chunk).

| Job | Δ rebuffer [95% CI] | Δ smoothness [95% CI] | CVaR₉₅ tube | CVaR₉₅ static | Zero-stall % tube | Latency μs (tube) [static] |
|---|---|---|---|---|---|---|
| `mpc_5g_markov` | -0.795 [-1.167, -0.448] | -0.0776 [-0.0864, -0.0692] | 14.18 | 16.13 | 33.3% | 48827.6 [1162.1] |
| `mpc_bb` | -0.174 [-0.333, -0.029] | -0.0394 [-0.0452, -0.0339] | 115.85 | 115.85 | 26.0% | 45519.9 [1152.9] |
| `greedy_5g_markov` | -2.700 [-3.401, -2.025] | -0.2202 [-0.2320, -0.2085] | 17.50 | 22.40 | 11.3% | 45865.1 [47.6] |
| `greedy_bb` | -0.471 [-0.699, -0.245] | -0.4235 [-0.4429, -0.4029] | 116.39 | 116.39 | 27.5% | 42964.9 [52.5] |
| `bba_5g_markov` | -1.735 [-2.322, -1.192] | -0.2234 [-0.2340, -0.2130] | 13.48 | 18.52 | 28.9% | 46863.1 [61.1] |
| `bba_bb` | -0.469 [-0.783, -0.175] | -0.3947 [-0.4151, -0.3735] | 110.73 | 110.73 | 24.0% | 44718.1 [62.8] |
| `bola_5g_markov` | -1.230 [-1.790, -0.680] | -0.2788 [-0.2936, -0.2641] | 14.25 | 18.20 | 27.5% | 49106.3 [76.2] |
| `bola_bb` | -0.158 [-0.412, 0.089] | -0.3691 [-0.3859, -0.3514] | 116.54 | 116.54 | 22.1% | 45092.2 [78.3] |
