# Tube-MPC `tube_mpc_r1` — Results Only

**Setup:** 204 paired episodes; arms `raw_mpc` (RobustMPC v18), `tube_static` (split-conformal tube, no ACI), `tube_aci` (online ACI tube); horizon 5, α=0.10 (coverage target ~0.90).  
**Traces:** 5G (`test_traces_5g_v18`), broadband (`test_traces_v18`). **Metrics:** rebuffer (s), mean log-bitrate utility (no VMAF).

## Mean rebuffer per episode (seconds)

| Trace | raw_mpc | tube_static | tube_aci |
|-------|---------|-------------|----------|
| **5g** | 66.77 | **37.60** | 75.00 |
| **bb** | 13.39 | 9.94 | **9.33** |

**Bold** = lowest rebuffer in that row.

## Mean log utility (higher = more aggressive bitrate)

| Trace | raw_mpc | tube_static | tube_aci |
|-------|---------|-------------|----------|
| 5g | **1.817** | 1.508 | 1.541 |
| bb | **0.787** | 0.697 | 0.670 |

## Tube throughput coverage (arms with tube only)

| Trace | tube_static | tube_aci |
|-------|-------------|----------|
| 5g | 0.919 | 0.905 |
| bb | 0.890 | 0.916 |

## Paired comparisons (Wilcoxon)

**5G**

| Comparison | Δ rebuffer (s) | p (reb) | Δ log utility | p (logU) |
|------------|----------------|---------|---------------|----------|
| tube_aci vs raw_mpc | +8.22 | ≪0.001 | −0.276 | ≪0.001 |
| tube_aci vs tube_static | +37.39 | ≪0.001 | +0.033 | 0.015 |

**Broadband**

| Comparison | Δ rebuffer (s) | p (reb) | Δ log utility | p (logU) |
|------------|----------------|---------|---------------|----------|
| tube_aci vs raw_mpc | −4.07 | ≪0.001 | −0.117 | ≪0.001 |
| tube_aci vs tube_static | −0.62 | 0.002 | −0.027 | ≪0.001 |

(`tube_static` vs `raw_mpc` on 5G: ≈−29.2 s rebuffer by means — not in `summary.json` comparisons; add in next eval revision.)

## One-line conclusion

- **5G:** **Static conformal tube-MPC** cuts rebuffer sharply vs raw MPC (~44% relative); **online ACI tube is worse than raw MPC** (same qualitative pattern as OACS: fixed/static bound works, naive online ACI over-shields on volatile 5G).  
- **Broadband:** Both tubes beat raw MPC on rebuffer; **tube_aci** edges **tube_static** slightly on rebuffer at the cost of lower log utility.  
- **Coverage** is near the 0.90 target for both tube arms; the 5G failure mode is **policy conservatism / bound tightness**, not missing coverage.

Artifacts: `results/tube_mpc/campaigns/tube_mpc_r1/{5g,bb}/summary.json` (fetched from server).
