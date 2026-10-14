# Paper 2 — Work completed & experimental results (2026-10)

Single summary of the **Tube-MPC** line after the OACS pivot: what was built, what was run on the server, and what the numbers say.  
Detailed OACS write-up: [`oacs_r1_campaign_report.md`](oacs_r1_campaign_report.md). Spec: [`../decision_rule_tube_mpc.md`](../decision_rule_tube_mpc.md).

---

## 1. Strategic direction

| Phase | Topic | Outcome |
|-------|--------|---------|
| **OACS (`oacs-r1_mm`)** | Online ACI + per-chunk VMAF + CPS-style shield | Completed; **did not beat CPS v18 on 5G** rebuffer; archived spec [`../decision_rule.md`](../decision_rule.md) |
| **Pivot** | Multi-horizon **Conformal Tube-MPC**, log-utility, **no VMAF** | Active Paper 2 direction [`../pivot_note.md`](../pivot_note.md) |
| **Data** | Synthetic 5G (Paper 1) + **real Irish 5G** (MMSys 2020) + broadband v18 | Real 5G pipeline: [`../data_5g_real.md`](../data_5g_real.md) |

---

## 2. Software delivered (Tube-MPC)

| Component | Path | Role |
|-----------|------|------|
| Conformal throughput tube | `src/training/conformal_tube.py` | Static split-conformal or online ACI lower bounds over horizon \(H\) |
| Tube-MPC policy | `src/training/tube_mpc_policy.py` | Log-utility MPC + tube-implied download times + action mask |
| Campaign evaluator | `src/evaluation/eval_tube_mpc_campaign.py` | Arms `raw_mpc`, `tube_static`, `tube_aci`; paired seeds `1000+ep` |
| Smoke | `experiments/paper2/run_tube_smoke.sh` | 20-step sanity on 5G traces |
| Campaign (synthetic + BB) | `experiments/paper2/run_tube_campaign.sh` | Mattermost + `results/tube_mpc/campaigns/` |
| Campaign (real 5G) | `experiments/paper2/run_tube_campaign_irish.sh` | Irish trace dir only |
| Real trace prep | `scripts/paper2/prepare_traces_5g_irish_mmsys20.py` | CSV → `{"throughput_kbps": [...]}` @ 1 Hz |
| Server deploy | `_remote_run_tube_campaign.py`, `_remote_run_tube_irish_campaign.py` | Tarball sync, venv smoke, `screen` jobs |
| Fetch results | `_remote_fetch_tube_results.py` (`--irish` for real 5G campaign) | Pull `summary.json` / `episodes.csv` |

**Baselines:** `raw_mpc` = `RobustMPCPolicy` (v18). Same env physics as Paper 1 (`paper2_abr_env` → `abr_multi_env_v18`). **204 episodes** unless noted.

**Fixes during rollout:** `run_tube_campaign.sh` bash `for pair in ...` syntax; remote commands use `bash -lc` + `source .venv`; optional `_tube_campaign_runner.sh` for long `screen` commands.

---

## 3. Campaigns executed (server)

| Campaign ID | Traces | Status | Output |
|-------------|--------|--------|--------|
| `tube_mpc_r1` | `test_traces_5g_v18` (synthetic), `test_traces_v18` (broadband) | **Done** | `results/tube_mpc/campaigns/tube_mpc_r1/{5g,bb}/` |
| `tube_mpc_r1_irish5g` | `test_traces_5g_irish_mmsys20` (78 real sessions) | **Done** | `results/tube_mpc/campaigns/tube_mpc_r1_irish5g/5g_irish/` |

Notifications: Mattermost on start/finish (Irish run: some notify timeouts; eval completed).

---

## 4. Results — mean rebuffer per episode (seconds)

Lower is better. **Bold** = best in row.

### 4.1 Synthetic 5G (`test_traces_5g_v18`)

| Arm | Rebuffer (s) | Mean log utility | Tube coverage |
|-----|--------------|------------------|---------------|
| raw_mpc | 66.77 | **1.817** | — |
| **tube_static** | **37.60** | 1.508 | 0.919 |
| tube_aci | 75.00 | 1.541 | 0.905 |

- `tube_static` vs `raw_mpc`: **−29.2 s** (~44% relative).
- `tube_aci` **worse than** `raw_mpc` (+8.2 s, paired Wilcoxon significant).

### 4.2 Broadband (`test_traces_v18`)

| Arm | Rebuffer (s) | Mean log utility | Tube coverage |
|-----|--------------|------------------|---------------|
| raw_mpc | 13.39 | **0.787** | — |
| tube_static | 9.94 | 0.697 | 0.890 |
| **tube_aci** | **9.33** | 0.670 | 0.916 |

- Both tubes beat raw; **tube_aci** slightly best on rebuffer (−0.6 s vs static).

### 4.3 Real 5G — Irish operator (MMSys 2020)

Source: [uccmisl/5Gdataset](https://github.com/uccmisl/5Gdataset); citation in [`../data_5g_real.md`](../data_5g_real.md).

| Arm | Rebuffer (s) | Mean log utility | Tube coverage |
|-----|--------------|------------------|---------------|
| raw_mpc | 249.04 | **0.551** | — |
| **tube_static** | **232.62** | 0.517 | 0.869 |
| tube_aci | 233.63 | 0.522 | 0.855 |

- **Modest gain:** `tube_static` −16.4 s vs raw (~7% relative); `tube_aci` −15.4 s vs raw.
- `tube_static` vs `tube_aci`: essentially tied (~1 s; static slightly better).
- Absolute rebuffer is high on all arms (hard real traces vs high-capacity synthetic 5G).

---

## 5. Cross-trace comparison (headline)

| Trace type | Best arm | vs raw_mpc (rebuffer) | ACI online helpful? |
|------------|----------|------------------------|---------------------|
| 5G synthetic | **tube_static** | Large win | **No** (worse than raw) |
| Broadband real | **tube_aci** | Clear win | Yes (slight edge over static) |
| 5G Irish real | **tube_static** | Small win | Marginal (≈ static; both beat raw) |

**Log utility:** `raw_mpc` always highest → tubes trade bitrate aggressiveness for fewer stalls.

**Coverage:** Tube arms target ~0.90; Irish run ~0.86–0.87 (still reasonable; disclose in paper).

---

## 6. Relation to OACS / Paper 1 (one paragraph)

OACS showed **static CPS (Paper 1)** winning on **synthetic 5G**, while **online ACI** without full CPS banking tracked **raw** ([`oacs_r1_results_summary.md`](oacs_r1_results_summary.md)). Tube-MPC **without VMAF** repeats the pattern on synthetic 5G: **static conformal tube wins; online ACI fails**. On **easier** (broadband) or **real but capacity-limited** 5G traces, tubes still help, but **margins shrink** on Irish data. Paper 1 numbers are **not** directly in the Tube campaigns (no `static_cps` arm yet)—next comparison if needed.

---

## 7. Scientific takeaway (draft for paper)

1. **Contributions supported by experiments:** Conformal **multi-horizon tube** inside MPC reduces rebuffer vs RobustMPC under paired evaluation; **static** conformal is the reliable default; evaluation spans **synthetic cellular stress**, **real 5G field traces**, and **wireline** traces with **log-utility** metrics (no VMAF in the loop).

2. **Honest limits:** Online **ACI tube** is not uniformly better than static; on synthetic high-throughput 5G it **hurts**. Real Irish traces yield **smaller** rebuffer gains—likely trace capacity / preprocessing (G-NetTrack `DL_bitrate` @ 1 Hz) vs simulator ladder mismatch.

3. **Do not claim:** “ACI tube-MPC dominates on all 5G” or “replaces CPS v18” without adding a **CPS/shield baseline** arm and tuning ACI on 5G.

---

## 8. Suggested next steps

- [ ] Add eval comparisons: `tube_static` vs `raw_mpc` in `summary.json` (Wilcoxon); optional `static_cps` arm.
- [ ] Paper table: three trace columns (synthetic 5G, Irish 5G, broadband).
- [ ] Trace diagnostics: distribution of mean `throughput_kbps` per Irish file vs synthetic.
- [ ] ACI calibration pass on 5G (or hybrid: static tube width + ACI only on broadband).
- [ ] Git commit: Tube-MPC code, scripts, docs (exclude `server.txt`, webhook, large `data/`).
- [ ] Zenodo bundle: Irish zip + generated JSON manifest (`trace_stats_5g_irish_mmsys20.json`).

---

## 9. Artifact index

```
results/tube_mpc/campaigns/tube_mpc_r1/5g/summary.json
results/tube_mpc/campaigns/tube_mpc_r1/bb/summary.json
results/tube_mpc/campaigns/tube_mpc_r1_irish5g/5g_irish/summary.json
data/standardized/test_traces_5g_irish_mmsys20/   (local; 78 traces with --all-test)
docs/reports/tube_mpc_r1_results_summary.md        (synthetic+bb short)
docs/data_5g_real.md
```

---

*Generated for handoff / Academic Lead review. Update after new campaigns or baseline arms.*
