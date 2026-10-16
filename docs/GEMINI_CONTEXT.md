# CPS-Paper2 / Tube-v3 — self-contained context (branch `tube-v3-val`)

## 1. Problem
Chunk-level ABR video streaming (BBA base policy, 6 bitrate rungs, chunk duration p, buffer B_max). We plan with finite-horizon MPC (H=5) using a **conformal lower bound tube on future throughput**, so that bitrate plans are feasible (buffer stays above a margin) under calibrated throughput uncertainty. Paper 1 (CPS) = 1-step conformal shield; Paper 2 = multi-step tube.

## 2. Method (equations)
**Point forecast.** At decision j (made after chunk j−1 is observed) the forecast is the harmonic mean of the last k=5 throughputs, ŷ_j = k / Σ_{i=1..k} 1/y_{j−i}. The same ŷ_j is used for all steps h=1..H.

**Alignment (fixed in `dd6765a`).** Step h of decision j refers to chunk **j+h−1**: h=1 is the chunk downloaded right now.

**Residual score** (relative, one-sided): s_{j,h} = max(0, (ŷ_j − y_{j+h−1}) / ŷ_j). (A log score is also available.)

**Per-step split/online conformal.** For each h, a window W=200 of past scores S_h holds. With the finite-sample rank r = ⌈(n+1)(1−α_h)⌉, q_h = S_h(r) and L_{j,h} = ŷ_j (1 − q_h). If n < ⌈1/α_h⌉−1, the cold start uses L = 0.8 ŷ. The score windows are warm-started offline from calibration traces only (train or calib split, never test).
- `per_step`: α_h = α, which gives marginal per-step coverage ≈ 1−α.
- `bonferroni`: α_h = α/H, which gives joint coverage ≥ 1−α by the union bound.

**Max-score (simultaneous) tube** (`MaxScoreConformalTube`). Scales σ_{j,h} = max(q_h^{(1−α)}, 0.02) are frozen at decision time. The joint score is S_j = max_h s_{j,h}/σ_{j,h}. With one quantile Q = Quant_{1−α}{S_{j'}: completed j'}, the bound is L_{j,h} = ŷ_j (1 − Q σ_{j,h}). The joint event "all H steps covered" ⇔ S_j ≤ Q. Bounds are stored **before** any update. A decision's score enters the window only after chunk j+H−1 is realized.

**ACI** (Gibbs–Candès) per step: α_{t+1,h} = clip(α_{t,h} + γ(α_h − err_{t,h}), α_h/10, min(1, 3α_h)), with γ ∈ {0.002,…,0.02}. The cold-start gate uses the nominal α_h.

**Tube-MPC** (`TubeV2MPCPolicy.plan_action`). Enumerate monotone bitrate sequences a_{1..H}, each step moving at most one rung up or down. Download time is d_h = size(a_h, chunk j+h−1) / L_{j,h}. Buffer recursion: b_{h+1} = min(max(0, b_h − d_h) + p, B_max). The objective is Σ_h [log-utility(a_h) − 100·max(0, d_h − b_h) − |Δ log-utility|], subject to the constraint b_h ≥ margin ∀h. Execute a_1, then **cap** it with the static 1-step conformal projector: a = min(a_tube, a_static).

**Episode QoE** = Σ log-utility − W_reb · rebuffer_s − W_smooth · Σ|Δu| (`clean_abr_eval.py:123`).

## 3. Code map
| path | key items |
|---|---|
| `src/training/conformal_tube_v2.py` | `ConformalTubeV2Config` (L26; `legacy_step_offset` alignment flag), `ConformalThroughputTubeV2` (L46): `_score` L109, `_finite_sample_quantile` L117, `_quantile_alpha` (ACI) L136, `horizon_lower_bounds_kbps` L155 (stores pre-update bounds), `_register_score` (coverage + ACI) L178, `_register_marginal_scores` L195, `_maybe_complete_joint_decision` L210, `ingest_calibration_sample` L229, `observe` L239 |
| `src/training/conformal_tube_v3.py` | `MaxScoreConformalTube` (L27): `_sigmas` L55, `horizon_lower_bounds_kbps` L67, `_maybe_complete_joint_decision` L100 (joint score after pre-update check), `ingest_calibration_sample` L126 |
| `src/training/tube_mpc_policy_v2.py` | `TubeV2MPCConfig` L18, `TubeV2MPCPolicy.plan_action` L123 (buffer recursion L155–163; `split_objective` L146 — rejected variant) |
| `src/control/two_timescale_conformal.py` | `ControlArmConfig` (alpha, margin, `effective_safety_margin` L147), `parse_tube_v3_arm` L100, `StaticConformalProjector` L159 (1-step cap), `TubeV2ConformalPolicy` L457, `_warm_start_tube_v2` L568, `make_controlled_policy` L577 |
| `src/training/robust_mpc_logu_policy.py` | RobustMPC baseline (harmonic / (1 + max recent rel. error)) |
| `src/training/fixed_quantile_mpc.py` | fixed-quantile (offline q on calib) and fixed-ρ MPC baselines |
| `src/training/tube_v2_calib.py` | offline warm-start replay |
| `configs/trace_sets_v2.py` | splits: train−val / val / test; harsh family 40/20/40 calib/val/test; `resolve_trace_dir` L309; `calibration_dirs_for_eval` L380 |
| `src/evaluation/trace_provenance.py` | sha256 manifest verification (`verify_or_exit` L132) |
| `src/evaluation/run_tube_v3_val.py`, `run_tube_v3_final.py` | VAL campaign; frozen TEST runner (reads `configs/tube_v3_final.yaml`, checks manifest sha) |
| `scripts/paper2/shift_coverage_v3.py` | offline markov→harsh shift coverage |
| `scripts/paper2/analyze_tube_v3_final.py` | cluster/paired bootstrap tables + figures |
| `tests/test_tube_v3.py`, `tests/test_conformal_tube_v2.py` | alignment pin test, max-score iid joint coverage, ACI sign guards, shift transient |

## 4. Protocol
- Trace sets: `5g_markov` (synthetic Markov, 12 test traces), `5g_harsh` (synthetic harsh family, 20 test), `bb` (broadband, 22 test), `5g_irish` (real MMSys'20 Irish 5G, 11 test).
- Splits are disjoint by filename (asserted). Calibration uses train−val (or harsh-calib); α, γ, and arms were selected on **val** only.
- Every reported run verifies the sha256 manifest. Config frozen in commit `938bf27` **before** the single test run.
- 100 episodes per arm, with deterministic per-episode seeds (paired across arms).
- Primary CIs: **trace-cluster bootstrap**. Secondary: paired-episode bootstrap and W/T/L.

## 5. Final TEST results (frozen config)
Primary arm: per-step tube α=0.15. Δ vs RobustMPC with cluster 95% CI:

| set | ΔQoE | Δrebuf (s) | per-step cov h1..h5 |
|---|---|---|---|
| 5g_markov | +26.0 [+7.7,+43.7] | −5.4 | 0.845–0.851 |
| 5g_harsh | +3.5 [−1.6,+9.5] | −0.9 [−1.9,−0.1] | 0.84–0.85 |
| bb | +1.8 [+0.2,+3.5] | +0.1 | 0.85 |
| 5g_irish | +3.9 (ns) | +0.5 | 0.84–0.85 |

Other arms:
- α=0.1: per-step coverage 0.89–0.90 everywhere. markov ΔQoE +40.3 vs RobustMPC; irish +24.9 [−1.4,+57.7].
- Max-score α=0.1: joint coverage 0.89–0.90 in all sets. Irish +92 QoE and −26 s rebuffering vs RobustMPC.
- Bonferroni: joint coverage 0.92–0.95. Irish +116 QoE.
- vs fixed-quantile MPC (same calib data), α=0.15: irish +19.0 [+6.5,+32.7]; markov −9.6; harsh −3.5; bb tie.
- 5g_harsh best QoE: fixed ρ=0.5 (89.2) and fixed-quantile (88.0) vs tube 84.5.

Full tables: `docs/reports/tube_v3_final_test.md`. Figures: `docs/figures/tube_v3_final/*.png`.

## 6. Known weaknesses
1. The best α depends on the network (0.15 on harsh/bb, 0.1 or a joint tube on markov/irish). A single fixed α is not uniformly best, and we pick it by hand on val.
2. On 5g_harsh, simple fixed-ρ / fixed-quantile MPC beats the tube on QoE.
3. ACI gives no measurable gain: val, test, and an offline markov→harsh shift that turned out *easier*, not harder.
4. The theory is currently informal. Scores are temporally dependent (overlapping horizons, a sliding window, and online updates), so exchangeability does not strictly hold.
5. Novelty vs conformal-MPC (e.g., conformal prediction regions for stochastic MPC and safe planning; Lindemann et al.; Dixit et al.; Stamouli et al.) and vs Conformal-ABR-type work is not yet sharply framed.
6. The 1-step static cap means the final action = min(tube, static). Any theory must cover the composed policy.

## 7. Exact questions
A. A data-driven **per-network α selection** with a validity argument and a leakage-free evaluation design.
B. A **joint coverage theorem** for the max-score tube with the j+h−1 alignment, with honest assumptions under temporal dependence, plus a feasibility lemma (coverage ⇒ no rebuffer for the executed plan).
C. Related-work positioning and novelty claims.
D. A file-level implementation plan with tests.
