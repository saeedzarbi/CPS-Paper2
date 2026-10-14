# OACS `oacs-r1_mm` — Results Only

**Setup:** 204 paired episodes; arms `raw`, `static_cps` (CPS v18), `aci_feasibility`, `aci_oacs`; ε=1.0, α=0.10 (coverage target 0.90).  
**Policies:** greedy, bba, bola, mpc. **Traces:** 5G (`test_traces_5g_v18`), broadband (`test_traces_v18`).

## Mean rebuffer per episode (seconds)

| Job | raw | static_cps | aci_feasibility | aci_oacs |
|-----|-----|------------|-----------------|----------|
| greedy_5g | 87.47 | **39.84** | 81.84 | 72.20 |
| greedy_bb | 1216.25 | 15.81 | 15.00 | **14.76** |
| bba_5g | 78.14 | **36.60** | 78.06 | 69.28 |
| bba_bb | 175.31 | 14.67 | 13.76 | **13.51** |
| bola_5g | 78.09 | **37.98** | 78.00 | 69.72 |
| bola_bb | 39.74 | 15.13 | 13.54 | **13.45** |
| mpc_5g | 66.25 | **33.46** | 64.48 | 56.70 |
| mpc_bb | 12.84 | 9.78 | 9.36 | **9.34** |

**Bold** = lowest rebuffer in that row.

## Headline (`greedy_5g`)

| Metric | static_cps | aci_oacs |
|--------|------------|----------|
| Mean rebuffer (s) | **39.84** | 72.20 |
| Mean VMAF | 79.29 | 77.48 |
| Conformal coverage | 0.919 | 0.905 |

**aci_oacs vs static_cps:** +32.4 s rebuffer (+81%, paired Wilcoxon significant); TOST VMAF ±1.0 **not** equivalent.

**aci_oacs vs aci_feasibility:** −9.6 s rebuffer (−12%, significant); TOST VMAF equivalent.

**aci_oacs vs raw:** −15.3 s rebuffer (−17%).

## One-line conclusion

- **5G:** CPS v18 (`static_cps`) wins on rebuffer; OACS beats raw and ACI-only but **does not beat v18**.  
- **Broadband:** OACS slightly beats v18 (small margins).  
- **aci_feasibility on 5G** ≈ **raw** → current ACI throughput bound is too weak for aggressive shielding on 5G.

Artifacts: `results/oacs/campaigns/oacs-r1_mm/*/summary.json`
