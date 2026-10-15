# Clean ABR `clean_v3` — full metrics

Headline arms: `raw`, `static_conformal`, `tube_conformal` (asymmetric multi-horizon tube + static cap).
204 paired episodes; traces: `test_traces_5g_markov`, `test_traces_v18`.

Source: `results/clean_v3/**/summary.json`

### Mean rebuffer (s/ep)

*Mean rebuffer (s/ep)* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 7.77 | 3.93 | **3.14** | 2.15e-06 | 1.07e-23 |
| greedy | 92.74 | 8.67 | **5.97** | 3.14e-14 | 3.17e-35 |
| bba | 10.14 | 4.92 | **3.19** | 3.10e-08 | 2.55e-28 |
| bola | 10.27 | 4.61 | **3.38** | 2.73e-06 | 1.68e-30 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 14.97 | 9.42 | **9.25** | 2.84e-02 | 4.42e-34 |
| greedy | 1119.52 | 9.90 | **9.43** | 2.35e-09 | 3.17e-35 |
| bba | 149.09 | 9.76 | **9.29** | 6.21e-06 | 3.17e-35 |
| bola | 27.56 | 9.76 | **9.60** | 4.25e-02 | 5.89e-35 |

### p95 chunk rebuffer (s)

*p95 chunk rebuffer (s)* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 0.56 | 0.11 | **0.05** | 1.40e-02 | 4.29e-17 |
| greedy | 9.10 | 0.60 | **0.19** | 2.71e-12 | 3.17e-35 |
| bba | 0.72 | 0.16 | **0.02** | 8.10e-04 | 9.20e-23 |
| bola | 0.61 | 0.06 | **0.02** | 8.08e-02 | 1.68e-22 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 1.26 | 0.63 | **0.62** | 2.34e-01 | 5.75e-21 |
| greedy | 43.23 | 0.68 | **0.60** | 7.30e-06 | 2.54e-35 |
| bba | 8.86 | 0.71 | **0.61** | 9.48e-05 | 3.20e-35 |
| bola | 2.89 | 0.65 | **0.61** | 6.70e-02 | 1.55e-34 |

### Mean smoothness (|Δlog u|)

*Mean smoothness (|Δlog u|)* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 0.22 | **0.32** | 0.25 | 6.48e-32 | 2.12e-06 |
| greedy | 0.00 | **0.46** | 0.24 | 3.32e-35 | 3.17e-35 |
| bba | 0.49 | **0.50** | 0.28 | 3.17e-35 | 3.27e-35 |
| bola | **0.85** | 0.65 | 0.37 | 3.17e-35 | 3.17e-35 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 0.08 | **0.14** | 0.10 | 1.33e-26 | 3.57e-10 |
| greedy | 0.00 | **0.54** | 0.11 | 7.61e-35 | 7.56e-35 |
| bba | 0.13 | **0.51** | 0.11 | 7.61e-35 | 5.21e-01 |
| bola | **0.69** | 0.49 | 0.12 | 7.61e-35 | 3.17e-35 |

### Mean log-bitrate utility

*Mean log-bitrate utility* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | **2.17** | 2.06 | 1.90 | 1.34e-32 | 7.56e-35 |
| greedy | **3.00** | 2.34 | 2.04 | 3.17e-35 | 3.17e-35 |
| bba | **2.35** | 2.22 | 1.96 | 3.47e-35 | 3.17e-35 |
| bola | **2.17** | 2.11 | 1.86 | 9.26e-34 | 4.45e-35 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | **0.99** | 0.93 | 0.84 | 1.71e-33 | 7.61e-35 |
| greedy | **3.00** | 1.11 | 0.93 | 7.61e-35 | 3.15e-35 |
| bba | **1.50** | 1.10 | 0.93 | 7.61e-35 | 3.60e-35 |
| bola | 1.01 | **1.05** | 0.90 | 2.35e-34 | 1.70e-15 |

### Mean episodic QoE

*Mean episodic QoE* — **bold** = best (rebuffer: lowest; utility/QoE: highest).

#### 5g_markov

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | 60.02 | **66.44** | 65.93 | 8.99e-03 | 3.82e-02 |
| greedy | -255.00 | 52.89 | **60.73** | 1.15e-05 | 3.17e-35 |
| bba | 45.54 | 61.52 | **67.25** | 4.78e-03 | 1.67e-17 |
| bola | 19.01 | 50.14 | **56.84** | 6.53e-06 | 3.11e-31 |

#### bb

| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |
|---|---|---|---|---|---|
| mpc | -20.62 | **-2.57** | -4.28 | 7.09e-11 | 2.89e-25 |
| greedy | -4670.15 | -15.11 | **-1.32** | 8.33e-29 | 3.17e-35 |
| bba | -575.11 | -13.44 | **-0.63** | 2.08e-28 | 3.17e-35 |
| bola | -103.07 | -15.12 | **-3.87** | 1.52e-27 | 3.17e-35 |

## Rebuffer win count (tube vs static & raw)

- Tube ≤ static on rebuffer: **8/8** benchmarks
- Tube ≤ raw on rebuffer: **8/8** benchmarks
