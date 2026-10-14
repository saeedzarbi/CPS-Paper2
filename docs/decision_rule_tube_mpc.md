# Project Specification: Multi-Horizon Conformal Tube-MPC for Safe ABR (Paper 2)

**Status:** Active direction (2026-10).  
**Supersedes for new work:** perceptual OACS path (`docs/decision_rule.md` — archived for implementation).

---

## 1. Motivation & positioning

Paper 1 (CPS v18, `paper_v1/`) coupled **VMAF-knee banking** with **split-conformal throughput bounds** and post-policy shielding. That line is strong for perceptual streaming papers but:

- Depends on heavy VMAF assets (encode + libvmaf pipelines).
- Frames quality as an **application-layer** perceptual metric.
- Our first OACS prototype (`oacs-r1_mm`) showed that **online ACI + per-chunk VMAF** did not beat CPS v18 on 5G rebuffer without further calibration (see `docs/reports/oacs_r1_results_summary.md`).

**Paper 2 (new)** targets **networking & control** venues (e.g., Computer Networks, ToN-style framing): **robust, safe ABR under non-stationary throughput** using only **bytes, buffer, and throughput**—no VMAF in the control loop or evaluation utility.

---

## 2. Problem statement

At each chunk \(k\), choose bitrate rung \(a_k \in \{0,\ldots,L-1\}\) given buffer \(B_k\), chunk sizes, and uncertain future throughput. Classical RobustMPC plans over a horizon with a **point** throughput forecast. Under 5G LOS/NLOS and broadband drift, point forecasts mis-estimate tail risk → rebuffer spikes and unstable bitrate trajectories.

**Goal:** A **multi-step safe MPC** that plans inside a **conformal throughput tube** (sequence of lower bounds over horizon \(H\)), adapts miscoverage online (ACI), and minimizes **log-utility** plus rebuffer/smoothness penalties—without perceptual ladders.

---

## 3. Core technical stack

### 3.1. Quality model (no VMAF)

Use standard **logarithmic rate utility** (RobustMPC / BOLA family):

\[
u(r_k) = \log(r_k / r_0), \quad r_k = \text{BITRATE\_LEVELS}[a_k].
\]

Reward in simulation may remain env-native; **paper claims** use log-utility and rebuffer/smoothness, not VMAF.

### 3.2. Conformal throughput tube (multi-horizon)

- **Point predictor:** harmonic mean of last \(k\) observed throughputs (as in v18).
- **Online ACI** (reuse ideas from `src/training/aci_perceptual_shield.py` **estimator only**): adaptive \(\alpha_k\), weighted residual quantiles, prior-seeded cold start.
- **Tube:** for horizon \(h = 1,\ldots,H\), use throughput lower bounds \(\underline{C}_{k+h}\) (same conformal scale or horizon-discounted variant) to simulate buffer trajectories.
- **Safety:** only rungs whose implied download times keep buffer above margin \(m\) over the tube simulation are **feasible**.

No VMAF knee, no `vmaf_perchunk.csv` in the controller.

### 3.3. Tube-MPC policy (host algorithm)

Extend `RobustMPCPolicy` (`eval_certified_shield_v18.py`):

1. Observe past throughputs; update ACI estimator.
2. Build tube \(\{\underline{C}_{k+1},\ldots,\underline{C}_{k+H}\}\).
3. Enumerate or search rung sequences (length \(H\), small \(L\)) **feasible** under tube-induced buffer dynamics.
4. Maximize \(\sum_h \big( u(r_{k+h}) - \lambda_{\mathrm{reb}} \cdot \mathrm{reb}_h - \lambda_{\mathrm{sm}} \cdot |\Delta u|_h \big)\).
5. Execute first action \(a_k\).

Optional: **action mask** for RL (MaskablePPO) = feasible rungs under tube at \(h=1\) (network safety only, no perceptual mask).

### 3.4. Evaluation arms (paired seeds)

| Arm | Description |
|-----|-------------|
| `raw_mpc` | Baseline RobustMPC (point forecast, no tube) |
| `cps_v18` | Paper 1 shield wrapping host (optional baseline) |
| `tube_mpc` | Tube-MPC with static conformal tube |
| `tube_mpc_aci` | Tube-MPC + online ACI tube widening/tightening |

Same traces: `test_traces_5g_v18`, `test_traces_v18`; seeds `1000 + ep`; **no VMAF violation metrics**.

**Primary metrics:** mean/p95 rebuffer, stall-free rate, mean log-utility, bitrate smoothness, empirical tube coverage, intervention rate (if shielded baselines included).

---

## 4. What we reuse vs retire

| Reuse | Retire / archive |
|-------|------------------|
| `abr_multi_env_v18` / `paper2_abr_env` dynamics | Per-chunk VMAF lookup in control |
| Online ACI estimator (refactor to `src/training/conformal_tube.py`) | `aci_perceptual_shield` banking + knee |
| Trace splits, 204 episodes, paired stats | OACS campaign as paper headline |
| Server workflow, Mattermost, `results/oacs/` layout → rename campaign family | `eval_oacs_campaign` 4-arm VMAF story |

`paper_v1/` remains frozen reference for **cps_v18** comparisons only.

---

## 5. Repository invariants

- Do not modify `paper_v1/`.
- `configs/videos.py`: `CPS_EPISODES = 204`, `MAX_CHUNKS = 48` unless an RFC is opened.
- Paired seeding across arms unchanged.
- **VMAF data optional** for Paper 2 code paths (env may still load VMAF for legacy reward; Tube-MPC controller ignores it).

---

## 6. Implementation roadmap (high level)

1. **`conformal_tube.py`** — ACI + horizon lower bounds; unit tests vs v18 conformal on single trace.
2. **`tube_mpc_policy.py`** — feasible tube-MPC; smoke on one episode.
3. **`eval_tube_mpc_campaign.py`** — arms above; outputs under `results/tube_mpc/campaigns/`.
4. **Drift / cold-start scripts** — reuse coverage logic without VMAF.
5. **Paper assets** — tables: tube vs raw MPC vs cps_v18 on 5G/broadband.

---

## 7. Success criteria (scientific)

- **5G:** `tube_mpc_aci` matches or beats `raw_mpc` on rebuffer with **non-inferior** log-utility (TOST on paired episodes).
- **Coverage:** tube lower bounds meet \(1-\alpha\) on rolling windows under mid-episode shock.
- **Narrative:** clear networking contribution (safe MPC under conformal uncertainty), no dependency on VMAF pipeline for reproducibility.

---

## 8. Source note

Strategic pivot documented in `docs/help2.txt` (Persian). This file is the **English canonical spec** for Cursor agents and Academic Lead.
