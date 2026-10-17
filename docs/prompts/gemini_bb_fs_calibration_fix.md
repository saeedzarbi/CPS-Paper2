# Gemini handoff: fix broadband (`bb`) joint coverage for Tube-v3 `_fs` calibration

**Repo:** [CPS-Paper2](https://github.com/saeedzarbi/CPS-Paper2), branch **`tube-v3-val`** (commit ~`199cf9a` or later).  
**Your role:** Propose a **leakage-free** fix, implementation plan, tests, and an updated evaluation protocol—not ad-hoc tuning on the test split.

---

## 1. Scientific context (one paragraph)

We run chunk-level adaptive bitrate streaming with **Tube-MPC**: a finite-horizon planner (H=5) uses **conformal lower bounds** on future throughput. The headline method is the **max-score simultaneous tube** (`MaxScoreConformalTube` in `src/training/conformal_tube_v3.py`) with arms suffixed **`_safe_fs`**:

- **`_safe`:** pre-refill buffer safety constraint in `TubeV2MPCPolicy` (plan only sequences with no planned rebuffer on the tube lower bound).
- **`_fs`:** **frozen scales** σ and joint quantile Q fitted offline from calibration traces (even/odd trace split; stride H for joint scores)—so the theorem’s “fixed σ” assumption (A1) is approximated in code.

**α** is chosen **per trace family** on **validation only** via a pre-registered rule (`configs/tube_v3_alpha_rule.yaml`, applied by `src/evaluation/select_alpha_rule.py`). Config is frozen before a **single** test run (`configs/tube_v3_adaptive_final.yaml`).

---

## 2. The problem we need you to solve

### 2.1 Symptom (frozen TEST, primary arm on `bb`)

For **`tube_v2_maxscore_a0.15_safe_fs`** on the **broadband (`bb`)** test set:

| Metric | Value | Nominal target |
|--------|-------|----------------|
| **Joint simultaneous coverage** | **0.646** | **1 − α = 0.85** (α=0.15 chosen on val) |
| Per-step marginal coverage (h1..h5) | ~0.84–0.86 | ~0.85 (looks acceptable) |
| ΔQoE vs RobustMPC (cluster 95% CI) | +2.7 [+0.9, +4.5] | small gain |
| Δrebuffer vs RobustMPC | −0.06 s (CI includes 0) | not meaningful |

On **5G families** (markov, harsh, irish), the **same** method reaches joint coverage **at or above** 1−α (e.g. irish 0.936 at α=0.1). The failure is **specific to `bb`** in our current pipeline.

Full numbers: `docs/reports/tube_v3_adaptive_test.md`, honest summary in `docs/LIMITATIONS.md`.

### 2.2 Root-cause hypothesis (to validate or refine)

**Offline `_fs` calibration** replays **raw trace samples** from JSON:

- `src/training/tube_v2_calib.py` → `replay_throughputs_on_tube()` reads `throughput_kbps[]` and calls `tube.ingest_calibration_sample(actual)` for each sample.
- `finalize_frozen_calibration()` builds σ and Q from those samples using the same harmonic-mean forecaster and h-step alignment as online.

**Online evaluation** feeds the tube **chunk effective throughput** from the simulator:

- `clean_abr_env.py` → after each chunk, `last_raw_throughput = effective_tp` where  
  `effective_tp ≈ bits_downloaded / download_time` integrated over wall-clock trace samples during the download (plus early-abort / lowest-bitrate retry logic).
- `tube_mpc_policy_v2.py` → `tube.observe(last_raw_throughput, realized_chunk=…)`.

So calibration scores are computed on **one stochastic process** (dense/raw trace indices), while test scores use **another** (chunk-aggregated, action-dependent effective kbps). On 5G this mismatch may be mild; on **`bb`** it appears to break **joint** coverage badly (marginals can still look fine if errors correlate across horizons within a decision).

The LaTeX remark in `docs/theory/tube_mpc_theory.tex` already notes that **y_k depends on the executed action**; offline replay ignores MPC and uses raw samples only—another layer of mismatch.

### 2.3 What “success” means

1. **Primary:** On **`bb` test**, joint coverage of the rule-selected primary arm within **reasonable tolerance** of 1−α (e.g. ≥0.80–0.85 for α=0.15), without destroying gains on 5G families.
2. **Protocol:** Any change to calibration must follow: **code + pre-registered note → VAL (diagnostics + optional re-run α rule) → FREEZE config → single TEST**. No peeking at test to tune.
3. **Theory alignment:** Prefer a calibration distribution that matches what the tube sees at runtime, while keeping **`_fs`** “σ and Q frozen from disjoint calibration data” story defensible.
4. **Engineering:** Minimal diff; reuse `configs/trace_sets_v2.py` calib dirs; keep legacy arms reproducible.

---

## 3. Code map (read these files in the bundle)

| File | Why |
|------|-----|
| `src/training/conformal_tube_v3.py` | Max-score tube, `_fs` finalize (even/odd split, stride H) |
| `src/training/conformal_tube_v2.py` | Scores, `ingest_calibration_sample`, `_h_step_scores`, per-step `_fs` |
| `src/training/tube_v2_calib.py` | **Current** offline replay (raw JSON samples) |
| `src/control/two_timescale_conformal.py` | `_warm_start_tube_v2()` calls replay + `finalize_frozen_calibration()` |
| `src/environment/clean_abr_env.py` | `_simulate_download`, `effective_tp` definition |
| `src/training/tube_mpc_policy_v2.py` | What gets passed to `tube.observe()` |
| `configs/tube_v3_alpha_rule.yaml` | α selection (tail rebuffer vs RobustMPC on val) |
| `configs/tube_v3_adaptive_final.yaml` | Frozen test arms (bb primary: `maxscore_a0.15_safe_fs`) |
| `docs/theory/tube_mpc_theory.tex` | Theorem (A1–A3), lemma, remarks on `_fs` and dependence |
| `docs/reports/tube_v3_adaptive_test.md` | Full TEST tables |
| `docs/LIMITATIONS.md` | Current limitation text |

Trace JSONs are **not** in the git bundle (large; on eval host under paths in `configs/trace_sets_v2.py`, manifest `configs/trace_manifests/canonical_traces.json`). Assume each trace file has `throughput_kbps: [float, …]` sampled in time.

---

## 4. Concrete questions for you

**A. Calibration design**

1. Propose **one or two** calibration pipelines that produce chunk-level (or decision-aligned) throughputs consistent with `observe()` at runtime. Options to consider (you may combine or reject):
   - Replay traces through `clean_abr_env` with a **fixed calibration policy** (e.g. always lowest bitrate, or cyclic bitrates) and log `effective_tp` per chunk.
   - Aggregate raw trace to **non-overlapping windows of length `CHUNK_DURATION`** at a fixed start offset (does this match the simulator’s integral?).
   - Separate calibration env flag `calibration_mode=True` that disables abort/variation for reproducible y_k.
2. How should **`_fs` even/odd trace split** and **stride H** be applied on chunk-indexed series?
3. Does action-dependent y_k require **calibrating per bitrate** or is a single conservative calibration sufficient for the paper?

**B. Diagnostics (VAL only, before any new TEST)**

1. What plots/statistics compare **raw-sample scores vs chunk-effective scores** on `bb` val (e.g. joint score distribution, P(S>Q), autocorrelation)?
2. How to detect “fix worked” on val without using test?

**C. α rule and freeze**

1. If calibration changes, must we **re-run** `select_alpha_rule.py` on val and commit a **new** `tube_v3_adaptive_final.yaml`?
2. Could bb’s chosen α shift (currently 0.15 on val for maxscore_safe_fs)?

**D. Implementation plan**

1. File-level changes (new module vs extend `tube_v2_calib.py`).
2. Unit tests (synthetic trace where raw ≠ chunk-mean; joint coverage on synthetic iid data).
3. Runtime cost of calibration replay (bb calib set size).

**E. Paper wording**

1. One paragraph for **Limitations** if we ship the fix.
2. One sentence if we **cannot** fully match action-dependent y_k but improve bb joint coverage.

---

## 5. Constraints (do not violate)

- Do **not** use the **test** split to pick hyperparameters or to iterate until bb joint ≥ 0.85.
- Keep **`tube_v3_final`** legacy experiment and CSVs as historical audit trail.
- Preserve manifest SHA verification for reported runs.
- The executed action is still **`min(tube_mpc, static_1step_cap)`**—any feasibility claim must mention the cap if relevant.

---

## 6. Deliverables we want back

1. **Root-cause analysis** (confirm or replace the raw-vs-chunk hypothesis using val-side reasoning).
2. **Recommended fix** (single primary approach + fallback).
3. **Step-by-step protocol** (commits order, config files, commands).
4. **Pseudocode** for chunk-aligned `_fs` warm-start replacing or augmenting `replay_throughputs_on_tube`.
5. **Risk table:** what could regress on 5g_irish (+125 QoE vs RobustMPC) or harsh (rebuffer −2 s).

---

## 7. Reference numbers (primary vs RobustMPC, TEST, cluster CI)

| Set | Primary arm | ΔQoE | Δrebuf (s) | Joint cov |
|-----|-------------|------|------------|-----------|
| 5g_markov | `maxscore_a0.1_safe_fs` | +36.5 | −12.05 | 0.924 |
| 5g_harsh | `maxscore_a0.2_safe_fs` | −1.4 (ns) | −2.05 | 0.818 |
| bb | `maxscore_a0.15_safe_fs` | +2.7 | −0.06 (ns) | **0.646** |
| 5g_irish | `maxscore_a0.1_safe_fs` | +124.8 | −36.83 | 0.936 |

Legacy `tube_v2_maxscore_a0.1` (no `_fs`) on bb test had joint **0.896** but worse QoE than the primary `_fs` arm—so the fix must address **`_fs` + bb**, not “drop `_fs`”.

---

*End of handoff prompt.*
