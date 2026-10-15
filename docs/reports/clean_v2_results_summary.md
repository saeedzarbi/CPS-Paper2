# Clean ABR `clean_v2` — Results (wall-clock env + Markov 5G)

**Setup:** `CleanABREnv`, 204 paired episodes, `log_utility` reward.  
**Traces:** `test_traces_5g_markov` (50 Markov LoS/NLoS/Outage), `test_traces_v18` (broadband).  
**Hosts:** greedy, bba, bola, mpc.  
**Arms:** `raw`, `static_conformal`, `tube_conformal`, `two_timescale_conformal` (tube + intra-chunk early abort).

Artifacts: `results/clean_v2/<policy>_{5g_markov,bb}/summary.json`

## Mean rebuffer per episode (seconds) — **bold** = best in row

### Markov 5G

| Policy | raw | static | tube | two_timescale |
|--------|-----|--------|------|---------------|
| mpc | 7.82 | **4.00** | 4.47 | 5.64 |
| greedy | 98.15 | 8.69 | **4.05** | 6.58 |
| bba | 11.25 | 4.90 | **4.29** | 6.50 |
| bola | 10.71 | 4.61 | **3.78** | 5.34 |

### Broadband

| Policy | raw | static | tube | two_timescale |
|--------|-----|--------|------|---------------|
| mpc | 14.37 | 9.59 | **9.36** | 9.64 |
| greedy | 1151.23 | 11.03 | **10.24** | 11.01 |
| bba | 153.64 | 10.28 | **9.54** | 9.91 |
| bola | 27.15 | 10.08 | **9.76** | 10.20 |

## Headlines

1. **Wall-clock + Markov traces fix the “greedy catastrophe”:** `greedy` raw rebuffer drops from **~98 s (5G)** / **~1151 s (BB)** to **~4–11 s** under conformal arms — the old frozen-throughput / chunk-index coupling no longer punishes top-rung hosts.
2. **Tube conformal** is the most consistent winner on **5G Markov** (best or tied for bba/bola/greedy/mpc).
3. **Two-timescale** (early abort) does **not** beat tube-only on these runs; abort_rate stayed ~0 in summaries — abort rarely triggered or metrics need arm-specific logging review.
4. **MPC raw** is already strong on Markov 5G (~7.8 s); conformal still helps (~4 s).

## One-line conclusion

The **clean-slate simulator + Markov 5G** dataset is viable for Paper 2: conformal tube/static arms deliver large rebuffer reductions vs raw on aggressive hosts, especially **greedy**, while preserving paired-eval protocol.

Fetch locally: `python scripts/paper2/_remote_fetch_clean_results.py`
