# Clean ABR `clean_v3` — full metrics

Headline arms: `raw`, `static_conformal`, `tube_conformal` (asymmetric multi-horizon tube + static cap).
204 paired episodes; traces: `test_traces_5g_markov`, `test_traces_v18`.

Source: `results\clean_v3/**/summary.json`

### Mean rebuffer (s/ep)

*Mean rebuffer (s/ep)* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 7.96 | 4.06 | **3.37** | 2.34e-07 | 3.79e-22 |
| greedy | 90.63 | 8.55 | **5.68** | 2.44e-17 | 3.17e-35 |
| bba | 10.60 | 4.64 | **3.14** | 1.16e-10 | 1.21e-31 |
| bola | 11.01 | 4.55 | **3.34** | 1.03e-07 | 3.34e-28 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 13.96 | 9.87 | **9.82** | 9.38e-02 | 5.71e-30 |
| greedy | 1113.31 | 10.10 | **9.54** | 1.36e-08 | 3.17e-35 |
| bba | 162.28 | 11.01 | **10.61** | 8.67e-08 | 3.17e-35 |
| bola | 27.07 | 10.00 | **9.58** | 7.20e-03 | 4.23e-35 |

### p95 chunk rebuffer (s)

*p95 chunk rebuffer (s)* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 0.61 | 0.07 | **0.05** | 6.28e-01 | 6.26e-17 |
| greedy | 8.72 | 0.53 | **0.18** | 5.62e-15 | 3.17e-35 |
| bba | 0.92 | 0.09 | **0.04** | 2.58e-02 | 1.50e-26 |
| bola | 0.75 | 0.10 | **0.03** | 4.07e-02 | 1.93e-20 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 1.02 | 0.65 | **0.64** | 5.87e-01 | 2.86e-10 |
| greedy | 43.55 | 0.69 | **0.65** | 5.60e-04 | 2.80e-35 |
| bba | 9.49 | 0.71 | **0.68** | 4.40e-02 | 3.17e-35 |
| bola | 2.88 | 0.66 | **0.63** | 4.26e-01 | 2.89e-34 |

### Mean smoothness (|Δlog u|)

*Mean smoothness (|Δlog u|)* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 0.22 | **0.33** | 0.24 | 3.40e-33 | 2.22e-09 |
| greedy | 0.00 | **0.46** | 0.24 | 3.47e-35 | 3.17e-35 |
| bba | **0.50** | 0.50 | 0.28 | 3.17e-35 | 3.22e-35 |
| bola | **0.85** | 0.66 | 0.37 | 3.22e-35 | 3.17e-35 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 0.08 | **0.14** | 0.10 | 1.15e-28 | 6.65e-12 |
| greedy | 0.00 | **0.51** | 0.11 | 7.61e-35 | 7.54e-35 |
| bba | 0.11 | **0.50** | 0.11 | 7.61e-35 | 1.09e-01 |
| bola | **0.70** | 0.51 | 0.12 | 7.61e-35 | 3.17e-35 |

### Mean log-bitrate utility

*Mean log-bitrate utility* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | **2.19** | 2.08 | 1.91 | 1.41e-34 | 1.99e-34 |
| greedy | **3.00** | 2.34 | 2.05 | 3.17e-35 | 3.17e-35 |
| bba | **2.32** | 2.18 | 1.91 | 4.02e-35 | 3.17e-35 |
| bola | **2.16** | 2.10 | 1.86 | 1.03e-34 | 4.94e-35 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | **1.01** | 0.95 | 0.87 | 3.78e-34 | 1.22e-34 |
| greedy | **3.00** | 1.12 | 0.95 | 7.61e-35 | 3.15e-35 |
| bba | **1.50** | 1.06 | 0.88 | 7.61e-35 | 3.16e-35 |
| bola | 1.00 | **1.05** | 0.89 | 1.23e-34 | 2.71e-15 |

### Mean episodic QoE

*Mean episodic QoE* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 60.33 | **66.56** | 65.69 | 2.88e-02 | 9.80e-02 |
| greedy | -245.93 | 53.45 | **62.74** | 2.81e-08 | 3.17e-35 |
| bba | 41.96 | 60.62 | **64.70** | 2.66e-02 | 1.02e-20 |
| bola | 15.59 | 49.51 | **56.97** | 2.04e-08 | 5.56e-30 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | -15.48 | **-3.66** | -5.26 | 1.94e-07 | 6.15e-19 |
| greedy | -4643.44 | -14.04 | **-1.07** | 3.75e-31 | 3.17e-35 |
| bba | -631.36 | -20.67 | **-8.52** | 8.91e-29 | 3.17e-35 |
| bola | -101.83 | -16.76 | **-4.23** | 2.90e-29 | 3.22e-35 |

## Rebuffer win count (tube vs static & raw)

- Tube ≤ static on rebuffer: **8/8** benchmarks
- Tube ≤ raw on rebuffer: **8/8** benchmarks
