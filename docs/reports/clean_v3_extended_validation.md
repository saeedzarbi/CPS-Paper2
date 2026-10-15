# Clean ABR — extended validation (`clean_v3_extended`)

Headline benchmark unchanged: [`clean_v3_full_metrics.md`](clean_v3_full_metrics.md).
Extended artifacts: `results/clean_v3_extended/`.

## Task 2 — OOD & mobility generalization

| Trace | Policy | raw | static | tube | p(tube vs static) |
|---|---|---|---|---|---|
| `5g_mmwave` | mpc | 0.05 | 0.00 | **0.00** | 1.00e+00 |
| `5g_mmwave` | greedy | 0.05 | 0.00 | **0.00** | 1.00e+00 |
| `5g_mmwave` | bba | 0.00 | 0.00 | **0.00** | 1.00e+00 |
| `5g_mmwave` | bola | 0.00 | 0.00 | **0.00** | 1.00e+00 |
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
| `fcc_fixed` | bba | 98.84 | 2.59 | **2.70** | 1.47e-02 |
| `fcc_fixed` | bola | 18.10 | 1.96 | **2.13** | 3.39e-03 |

## Task 3 — Content complexity (VBR / SI-TI strata)

Videos grouped by `siti_summary.csv` motion/complexity labels and VBR chunk-size CV (top tertile → high).

| Stratum | Videos | Δ rebuffer (tube−static) s/ep | Δ smoothness (tube−static) | n episodes |
|---|---|---|---|---|
| `low_medium` | into_tree, kristen_and_sara, old_town_cross, sintel, sunflower | -0.883 [-1.121, -0.652] | -0.2504 | 704 |
| `high_vbr_motion` | bigbuckbunny, cosmos, ducks, elephants_dream, parkjoy, rush_hour, tearsofsteel | -1.030 [-1.238, -0.827] | -0.2556 | 928 |

## Task 4 — Component ablation (`test_traces_5g_markov`, bba & bola)

| Policy | raw | heur. 0.85 | static H=1 | sym. tube | fixed-α tube | **tube (proposed)** |
|---|---|---|---|---|---|---|
| bba | 10.06 | 7.89 | 4.35 | 2.61 | 3.11 | **3.25** |
| bola | 9.42 | 6.84 | 4.11 | 2.30 | 3.06 | **2.95** |

### Horizon sweep (tube, α=0.10)

| Policy | H=1 | H=2 | H=3 | H=5 |
|---|---|---|---|---|
| bba | 0.27 | 0.27 | 0.95 | 3.25 |
| bola | 0.12 | 0.12 | 0.81 | 3.01 |

### α sweep (tube, H=5)

| Policy | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|---|---|---|---|---|
| bba | 2.88 | 3.18 | 3.63 | 3.75 |
| bola | 2.62 | 3.01 | 3.27 | 3.66 |

### Extended statistics on headline `clean_v3`

## Task 4 — Extended statistics (`results/clean_v3/`)

Paired bootstrap 10k resamples for tube − static; CVaR₉₅ on episode rebuffer; zero-stall rate; decision latency (profiling pass).

| Job | Δ rebuffer [95% CI] | Δ smoothness [95% CI] | CVaR₉₅ tube | CVaR₉₅ static | Zero-stall % tube | Latency μs (tube) |
|---|---|---|---|---|---|---|
| `mpc_5g_markov` | -0.795 [-1.167, -0.448] | -0.0776 [-0.0864, -0.0692] | 14.18 | 16.13 | 33.3% | — |
| `mpc_bb` | -0.174 [-0.333, -0.029] | -0.0394 [-0.0452, -0.0339] | 115.85 | 115.85 | 26.0% | — |
| `greedy_5g_markov` | -2.700 [-3.401, -2.025] | -0.2202 [-0.2320, -0.2085] | 17.50 | 22.40 | 11.3% | — |
| `greedy_bb` | -0.471 [-0.699, -0.245] | -0.4235 [-0.4429, -0.4029] | 116.39 | 116.39 | 27.5% | — |
| `bba_5g_markov` | -1.735 [-2.322, -1.192] | -0.2234 [-0.2340, -0.2130] | 13.48 | 18.52 | 28.9% | — |
| `bba_bb` | -0.469 [-0.783, -0.175] | -0.3947 [-0.4151, -0.3735] | 110.73 | 110.73 | 24.0% | — |
| `bola_5g_markov` | -1.230 [-1.790, -0.680] | -0.2788 [-0.2936, -0.2641] | 14.25 | 18.20 | 27.5% | — |
| `bola_bb` | -0.158 [-0.412, 0.089] | -0.3691 [-0.3859, -0.3514] | 116.54 | 116.54 | 22.1% | — |
