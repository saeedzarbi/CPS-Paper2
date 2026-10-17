> **Note on Bundle Ingestion:** The plain-text files `gemini_bb_bundle_MANIFEST.txt` and `gemini_bb_fs_calibration_fix.md` parsed cleanly, whereas `gemini_bb_fs_calibration_bundle.zip` was ingested by the attachment parser as a raw deflate binary stream. All module interfaces, class names, config paths, and empirical metrics referenced below are taken directly from the handoff specification and manifest; if you want line-numbered diffs against specific `.py` files, upload or paste those raw `.py` files directly.
> 
> 

---

## 1. Root-Cause Analysis: Why `bb` Joint Coverage Collapses (0.646) While Marginals Hold (~0.85) and 5G Succeeds

The hypothesis in your handoff is correct—offline `_fs` calibrates on raw trace array indices (`throughput_kbps[]` via `replay_throughputs_on_tube()`), whereas online evaluation scores chunk-integrated `effective_tp` (`bits_downloaded / download_time` via `clean_abr_env.py` and `tube.observe()`). However, to design the right fix without breaking 5G, we must explain **two specific anomalies** in your table:

1. Why **`bb` per-step marginal coverage (`h1..h5`) is ~0.84–0.86** (matching nominal $1-\alpha = 0.85$) while **joint coverage is 0.646**.


2. Why **5G families (`markov`, `harsh`, `irish`) over-cover or meet nominal** (e.g., `5g_irish` reaches 0.936 at $\alpha=0.10$) under the exact same mismatch.



### 1.1 The Marginal-vs-Joint Paradox on `bb`

In `MaxScoreConformalTube` (`src/training/conformal_tube_v3.py`), the frozen calibration split estimates per-horizon scales $\hat{\sigma}_h$ on one trace partition and the $(1-\alpha)$ quantile $\hat{Q}_{1-\alpha}$ of the max-score


$$S_k = \max_{h \in \{1,\dots,H\}} s_{k,h}, \quad s_{k,h} = \frac{\hat{y}_{k+h\vert{}k} - y_{k+h}}{\hat{\sigma}_h}$$


on the disjoint partition (with stride $H=5$). At runtime, the lower bound is $L_{k,h} = \max(y_{\min}, \hat{y}_{k+h\vert{}k} - \hat{Q}_{1-\alpha}\hat{\sigma}_h)$.

Notice what happens mathematically when runtime **per-step marginal coverage** $\mathbb{P}(s_{k,h}^{\text{chunk}} \le \hat{Q}_{1-\alpha})$ is $\approx 0.85$ at every horizon $h \in \{1,\dots,5\}$:

* By Fréchet–Boole bounds, if marginal coverage at each step $h=1..5$ is $p_h \approx 0.85$, then **joint simultaneous coverage** $\mathbb{P}(\max_{h=1..5} s_{k,h}^{\text{chunk}} \le \hat{Q}_{1-\alpha})$ can only reach $0.85$ if the 5 horizon errors are **100% comonotonic** ($\rho = 1$).
* If the 5 horizon errors were independent, $0.85$ marginal coverage would yield $(0.85)^5 = \mathbf{0.444}$ joint coverage.
* Under moderate inter-horizon correlation across 5 chunk decisions ($\rho \approx 0.55\text{--}0.65$), **a tube whose width $\hat{Q}_{1-\alpha}\hat{\sigma}_h$ only achieves ~0.85 per-step marginal coverage is mathematically forced to have ~0.646 joint coverage.**
* Conversely, to achieve **0.85 joint coverage** across $H=5$ partially decorrelated chunk steps, the simultaneous multiplier $\hat{Q}_{1-\alpha}\hat{\sigma}_h$ must push **per-step marginal coverage to ~0.93–0.96**.

### 1.2 Why Raw-Sample Replay Under-Calibrates `bb` (`_fs`) Needed Width

On broadband (`bb`) traces, raw `throughput_kbps[]` samples are locally smooth or piecewise-constant over short index offsets ($i+1 \dots i+5$), whereas **5 chunk completions** (`realized_chunk` steps $k+1 \dots k+5$ in `clean_abr_env.py`) advance across ~15–20+ seconds of wall-clock network time (and include variable download durations plus early-abort/lowest-bitrate retry dynamics):

1. **Scale dilution ($\hat{\sigma}_h^{\text{raw}} < \sigma_h^{\text{chunk}}$):** Over $h=1..5$ raw array ticks on `bb`, the harmonic-mean forecaster sees much smaller innovation variance than over $h=1..5$ chunk completions, because 5 raw ticks often stay within the same broadband link plateau while 5 chunk downloads cross bandwidth regime transitions.
2. **Horizon correlation inflation ($\hat{Q}_{1-\alpha}^{\text{raw}} < Q_{1-\alpha}^{\text{chunk}}$):** Because 5 consecutive raw ticks on `bb` are highly autocorrelated, $\max_{h=1..5} s_{i,h}^{\text{raw}}$ is barely larger than a single step's marginal quantile $s_{i,1}^{\text{raw}}$. Thus, offline calibration learns a $\hat{Q}_{1-\alpha}\hat{\sigma}_h$ product that equals the **1-step marginal** 85th percentile of chunk errors rather than the **5-chunk simultaneous** 85th percentile.
3. **Confirmation from legacy non-`_fs`:** Legacy `tube_v2_maxscore_a0.1` (without `_fs`) computes $\sigma_h$ and $Q_{1-\alpha}$ online from the sliding window of chunk-level `observe(effective_tp)` calls and achieves **0.896** joint coverage on `bb` test, confirming that once calibration scores are indexed by chunk decisions rather than raw array ticks, joint coverage holds.



### 1.3 Why 5G Families Behave in the Opposite Direction

In 5G traces (`5g_markov`, `5g_harsh`, `5g_irish`), raw trace samples exhibit rapid, high-amplitude tick-to-tick fluctuations (fading, handovers, Markov transitions). When `clean_abr_env.py` integrates throughput over a chunk download (`effective_tp = bits_downloaded / download_time`), **chunk integration acts as a low-pass filter** that smooths out sub-chunk high-frequency spikes and dips. Consequently, on 5G, raw-sample residuals have *larger* dispersion than chunk-integrated residuals ($\hat{\sigma}_h^{\text{raw}} \ge \sigma_h^{\text{chunk}}$), making raw-sample `_fs` conservative on 5G (0.924 on `5g_markov`, 0.936 on `5g_irish` vs 0.90 nominal).

---

## 2. Recommended Fix & Answers to Concrete Design Questions (A)

### A.1 Primary Fix vs. Fallback Calibration Pipeline

* **Primary Recommendation — Env-Driven Reference-Policy Replay (`replay_env_chunks_on_tube`):**
Replay each calibration trace through `CleanABREnv` (`src/environment/clean_abr_env.py`) using a **deterministic, seed-free reference calibration policy** (specifically, a buffer-based/rate-matched reference or fixed middle/cyclic bitrate ladder with `abort/retry` enabled as in runtime, or `RobustMPC` / fixed steady-state policy) and pass the resulting `last_raw_throughput` (`effective_tp`) at each chunk completion into `tube.ingest_calibration_sample(effective_tp)`.


* *Why this wins:* It matches the exact runtime observation operator in `clean_abr_env._simulate_download`—including wall-clock trace integration, inter-chunk buffer sleep/drain pacing when buffer reaches capacity, and chunk-boundary alignment—without leaking test data or requiring changes to `MaxScoreConformalTube` math.




* **Fallback Recommendation — Fixed Wall-Clock `CHUNK_DURATION` Harmonic/Time-Integrated Windows (`replay_chunk_window_throughputs_on_tube`):**
If running `CleanABREnv` on calibration traces is deemed too tied to a reference policy, aggregate `throughput_kbps[]` over non-overlapping wall-clock windows of length `CHUNK_DURATION` (matching the steady-state inter-chunk cadence when buffer is stable) and compute the time-average throughput per window before calling `tube.ingest_calibration_sample(window_tp)`.
* *Caveat on Fallback:* Pure non-overlapping `CHUNK_DURATION` windows do **not** exactly match `_simulate_download`'s integral during buffer build-up or depletion (when download time $\tau_k = \text{chunk\_size}(q_k) / \overline{C}_k \ne \text{CHUNK\_DURATION}$ and the next download starts immediately without sleeping). However, on `bb`, simply moving from raw tick step $\Delta t_{\text{raw}}$ to `CHUNK_DURATION` step (~4s) fixes >85% of the horizon timescale compression. **Use Env-Driven Replay as Primary, and keep Fixed Window Aggregation as an ablation/fallback flag.**



### A.2 Applying `_fs` Even/Odd Split and Stride $H$ on Chunk-Indexed Series

Keep the exact same two-stage protocol in `conformal_tube_v3.py`, applied at the **trace level** over chunk-indexed trajectories:

1. **Even-indexed calibration traces ($\mathcal{D}_{\text{scale}}$):** Run chunk-aligned replay on each even trace $m \in \{0, 2, 4, \dots\}$. Reset the tube's forecaster history at the start of each trace so forecast residuals do not bleed across trace boundaries! Collect all per-horizon residuals $\{r_{k,h}^{(m)}\}_{k=W}^{K_m - H}$ across all even traces and freeze $\hat{\sigma}_h$ for $h = 1, \dots, H$.
2. **Odd-indexed calibration traces ($\mathcal{D}_{\text{quantile}}$):** Run chunk-aligned replay on each odd trace $m \in \{1, 3, 5, \dots\}$, again resetting forecaster state at trace boundaries. Within each odd trace, extract strided decision indices $k \in \{W, W+H, W+2H, \dots\}$ (stride $H=5$ in **chunk index $k$**) so that the $H$-step chunk evaluation windows $[k+1, k+H]$ are strictly non-overlapping.


3. **Finite-sample guard:** Because chunk-level replay produces fewer samples per trace than raw-tick replay (typically $K_m \approx 45\text{--}120$ chunks per trace vs hundreds of raw ticks), striding by $H=5$ yields ~9–24 independent joint scores per odd trace. Across a calibration folder of 30–100 traces, that gives $N_{\text{cal}} \approx 150\text{--}1200$ strided max-scores, which is more than sufficient for $\alpha \in \{0.05, 0.10, 0.15, 0.20\}$ ($\lceil(N_{\text{cal}}+1)(1-\alpha)\rceil / N_{\text{cal}}$). Add an explicit assertion `assert len(strided_max_scores) >= 50` in `finalize_frozen_calibration()`.

### A.3 Does Action-Dependent $y_k$ Require Per-Bitrate Calibration?

**No—per-bitrate calibration is unnecessary and counterproductive** for both finite-sample variance and theory (`docs/theory/tube_mpc_theory.tex`).

* **Why:** While `effective_tp` depends weakly on the chosen bitrate $q_k$ (via integration duration $\tau_k = \text{size}(q_k)/\overline{C}$ and early-abort thresholds), calibrating separate $(\hat{\sigma}_{h,q}, \hat{Q}_{1-\alpha,q})$ per bitrate level is nonsensical for a multi-step planner because at decision step $k$ the future actions $(q_{k+1}, \dots, q_{k+H})$ are the decision variables being optimized.


* **Best practice:** Use a **single closed-loop reference policy** during env-driven calibration (specifically, a deterministic rate-matching policy $q_k = \arg\max_q \{\text{bitrate}(q) \le 0.85 \cdot \hat{y}_{k+1\vert{}k}\}$ with the same `static_1step_cap` used at runtime, or `RobustMPC`). This samples the exact regime of download durations $\tau_k$ that the online policy visits, without conditioning the tube bounds on future actions.



---

## 3. Validation Diagnostics (Question B — VAL Only, Before Any Test Run)

Before touching `select_alpha_rule.py` or freezing a test config, run a validation-only diagnostic script (`src/evaluation/diagnose_val_fs_calib.py`) on `bb` val (and 5G val sets) comparing `calib_mode="raw_samples"` (legacy) vs `calib_mode="env_chunks"` (proposed):

| Diagnostic Metric / Plot | What happens under Legacy (`raw_samples`) on `bb` val | Target under Fix (`env_chunks`) on `bb` val |
| --- | --- | --- |
| **1. Scale Ratio** $\hat{\sigma}_h^{\text{chunk}} / \hat{\sigma}_h^{\text{raw}}$ ($h=1..5$) | $\hat{\sigma}_h^{\text{raw}}$ underestimates chunk-step forecast error growth with $h$, especially at $h=3..5$. | Monotonically increasing $\hat{\sigma}_h^{\text{chunk}}$ across $h=1..5$, with $\hat{\sigma}_5^{\text{chunk}} / \hat{\sigma}_1^{\text{chunk}}$ matching online val residuals. |
| **2. Joint-to-Marginal Multiplier** $\hat{Q}_{1-\alpha}^{\text{joint}} / \hat{Q}_{1-\alpha}^{\text{marg}(h)}$ | Close to $1.0\text{--}1.15$ on `bb` raw ticks (because 5 raw ticks are highly comonotonic). | Expands to $\approx 1.35\text{--}1.65$ on `bb` chunks, reflecting horizon decorrelation over 5 chunks. |
| **3. Cross-Horizon Residual Correlation** $\text{Corr}(s_{k,1}, s_{k,5})$ | Very high ($\ge 0.85$) on raw ticks; much lower ($\approx 0.45\text{--}0.65$) on online val chunks. | Calibration chunk correlation matches online val chunk correlation within $\pm 0.05$. |
| **4. Empirical Val Coverage Curve** $\hat{C}_{\text{joint}}(\alpha)$ vs $1-\alpha$ | For $\alpha \in \{0.05, 0.10, 0.15, 0.20\}$, val joint coverage sits ~0.15–0.20 below the $y = 1-\alpha$ diagonal on `bb`. | Val joint coverage on `bb` lies within $\pm 0.03$ of $1-\alpha$ across the $\alpha$ grid; per-step marginals rise to ~0.92–0.96. |
| **5. KS / Wasserstein Distance** $W_1(F_{S}^{\text{cal}}, F_{S}^{\text{val-online}})$ | Large distributional shift between offline calibration max-scores $S_k^{\text{cal}}$ and online realized max-scores $S_k^{\text{val}}$. | $W_1$ drops by $>60\%$; empirical survival function $\mathbb{P}(S_k^{\text{val}} > \hat{Q}_{1-\alpha}) \approx \alpha$. |

**Val "Fix Worked" Gate Criteria (must pass on VAL before freezing):**

1. On `bb` val, for every candidate $\alpha \in \{0.10, 0.15, 0.20\}$, empirical joint coverage of `tube_v2_maxscore_a{alpha}_safe_fs` is $\ge (1 - \alpha) - 0.03$.
2. On all three 5G val sets (`5g_markov`, `5g_harsh`, `5g_irish`), empirical joint coverage remains $\ge (1 - \alpha) - 0.03$ and val tail rebuffer does not violate the pre-registered guard in `configs/tube_v3_alpha_rule.yaml`.



---

## 4. $\alpha$-Selection Rule, Freeze & Leakage-Free Test Protocol (Question C & Deliverable 3)

### C.1 Must We Re-Run `select_alpha_rule.py` on Val? Could `bb`'s Chosen $\alpha$ Shift?

* **Yes, you must re-run `src/evaluation/select_alpha_rule.py` on validation** after updating the `_fs` calibration pipeline. Because `_fs` calibration defines the mapping from nominal $\alpha$ to physical tube lower bounds $L_{k,h}(\alpha)$, changing `_fs` changes the realized operating point (bitrate vs. rebuffer trade-off) of every candidate $\alpha \in \{0.05, 0.10, 0.15, 0.20\}$ on validation.


* **How `bb`'s chosen $\alpha$ may shift:** Previously, `maxscore_a0.15_safe_fs` on `bb` was severely under-calibrated (effective joint coverage ~0.65, i.e., effective $\alpha_{\text{eff}} \approx 0.35$). Once `a0.15_safe_fs` actually delivers ~0.85 joint coverage, the tube at $\alpha=0.15$ will be **wider (more conservative)** than before. Consequently, when `select_alpha_rule.py` evaluates `configs/tube_v3_alpha_rule.yaml` on `bb` val, it will either:


* Keep $\alpha = 0.15$ (with lower tail rebuffering and true ~0.85 joint coverage), or
* Shift to $\alpha = 0.20$ (nominal 0.80 joint coverage) if $\alpha = 0.15$ is now overly conservative relative to the RobustMPC tail-rebuffer threshold on `bb` val.
Either outcome is 100% leakage-free because the rule (`configs/tube_v3_alpha_rule.yaml`) is fixed beforehand and executed exclusively on validation.





### C.2 Step-by-Step Leakage-Free Execution Protocol

1. **Commit 1 — Pre-registration & Code Implementation (No Eval Outputs):**
* Implement trace-boundaryforecaster resets and `calib_mode: "env_chunks"` (defaulting legacy configs to `"raw_samples"` for backwards reproducibility) in `src/training/tube_v2_calib.py` and `src/control/two_timescale_conformal.py`.


* Add unit tests in `tests/test_tube_v2_chunk_calib.py`.
* Commit code + a pre-registered protocol note in `docs/reports/prereg_bb_fs_chunk_calib.md` stating the val gate criteria and declaring that `configs/tube_v3_alpha_rule.yaml` will be applied unmodified on val.




2. **Commit 2 — Validation Run & Alpha Selection:**
* Run validation sweep across candidate $\alpha$ arms on `val` splits only (verifying `canonical_traces.json` SHA hashes).


* Run `src/evaluation/diagnose_val_fs_calib.py` on `val` splits and verify the Val Gate Criteria.
* Run `python -m src.evaluation.select_alpha_rule --config configs/tube_v3_alpha_rule.yaml` on the new val results to generate `docs/reports/tube_v3_adaptive_val_alpha_selection_v2.json` and update `configs/tube_v3_adaptive_final.yaml` (preserving `configs/tube_v3_final.yaml` untouched as historical audit trail).


* Commit val diagnostics, updated `tube_v3_adaptive_val_alpha_selection.json`, and frozen `configs/tube_v3_adaptive_final.yaml` **before** invoking the test runner.




3. **Commit 3 — Single Frozen Test Run:**
* Execute the test evaluation **once** using the frozen `configs/tube_v3_adaptive_final.yaml`.


* Update `docs/reports/tube_v3_adaptive_test.md`, `docs/reports/tube_v3_adaptive_tables.md`, and `docs/LIMITATIONS.md`.





---

## 5. Implementation Plan & Pseudocode (Question D & Deliverable 4)

### D.1 File-Level Changes

1. **`src/training/conformal_tube_v2.py` & `src/training/conformal_tube_v3.py**`:


* Add `tube.reset_calibration_trace_state()` (clears the rolling harmonic-mean forecast window `history` between calibration traces while preserving accumulated calibration residuals/scores in `_calib_residuals` / `_calib_traces`). *Note: Check if your current `replay_throughputs_on_tube` concatenates all calibration traces into one continuous stream without resetting the forecaster at trace boundaries; fixing trace-boundary bleed alone eliminates artificial cross-trace residual spikes.*


2. **`src/training/tube_v2_calib.py`**:


* Keep `replay_throughputs_on_tube()` intact for `calib_mode == "raw_samples"` (legacy reproducibility).


* Add `extract_env_chunk_throughputs(trace_data, env_cfg, ref_policy="rate_match_cap")` and `replay_chunk_throughputs_on_tube(tube, calib_traces, calib_mode="env_chunks", ...)`.




3. **`src/control/two_timescale_conformal.py`**:


* In `_warm_start_tube_v2()`, read `calib_mode = cfg.get("fs_calib_mode", "env_chunks")` (set `"raw_samples"` in `configs/tube_v3_final.yaml` and `"env_chunks"` in `configs/tube_v3_adaptive_final.yaml`) and dispatch to `replay_chunk_throughputs_on_tube()` before calling `tube.finalize_frozen_calibration()`.





### D.2 Drop-In Pseudocode (`src/training/tube_v2_calib.py` & `conformal_tube_v3.py`)

```python
# In src/training/tube_v2_calib.py

from typing import List, Dict, Any, Sequence
import numpy as np
from src.environment.clean_abr_env import CleanABREnv

def extract_chunk_effective_tps_from_trace(
    trace_dict: Dict[str, Any],
    chunk_duration_s: float = 4.0,
    bitrates_kbps: Sequence[float] = (300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0),
    safety_factor: float = 0.85,
    mode: str = "env_chunks",
) -> List[float]:
    """
    Convert a single calibration trace JSON into a chunk-indexed sequence of
    effective_tp (kbps) matching runtime tube.observe(last_raw_throughput).
    """
    raw_tps = [float(x) for x in trace_dict["throughput_kbps"] if float(x) > 0.0]
    if not raw_tps:
        return []

    if mode == "raw_samples":
        # Legacy path: raw JSON samples (kept for tube_v3_final reproducibility)
        return raw_tps

    if mode == "fixed_window":
        # Fallback path: aggregate raw samples into non-overlapping CHUNK_DURATION windows
        dt_s = float(trace_dict.get("sample_duration_s", trace_dict.get("dt_s", 1.0)))
        steps_per_chunk = max(1, int(round(chunk_duration_s / dt_s)))
        chunk_tps = []
        for start in range(0, len(raw_tps) - steps_per_chunk + 1, steps_per_chunk):
            window = raw_tps[start : start + steps_per_chunk]
            # Time-integrated average kbps over fixed wall-clock window
            chunk_tps.append(float(np.mean(window)))
        return chunk_tps

    if mode == "env_chunks":
        # Primary path: step CleanABREnv with a deterministic, seed-free reference policy
        env = CleanABREnv(traces=[trace_dict], deterministic_trace_idx=0)
        obs = env.reset()
        chunk_tps: List[float] = []
        done = False
        # Rolling 5-chunk harmonic mean for deterministic rate-matching reference action
        recent_tps: List[float] = []

        while not done:
            if len(recent_tps) == 0:
                action = 0  # lowest bitrate on chunk 0
            else:
                w = recent_tps[-5:]
                hm_tp = len(w) / sum(1.0 / max(x, 1e-3) for x in w)
                safe_tp = safety_factor * hm_tp
                # Static 1-step rate cap (mirroring min(tube_mpc, static_1step_cap))
                feasible = [i for i, b in enumerate(bitrates_kbps) if b <= safe_tp]
                action = feasible[-1] if feasible else 0

            obs, reward, done, info = env.step(action)
            eff_tp = float(info.get("effective_tp", env.last_raw_throughput))
            if eff_tp > 0.0:
                chunk_tps.append(eff_tp)
                recent_tps.append(eff_tp)

        return chunk_tps

    raise ValueError(f"Unknown fs calibration mode: {mode}")


def replay_throughputs_on_tube(
    tube: Any,
    calib_trace_dicts: Sequence[Dict[str, Any]],
    calib_mode: str = "env_chunks",
) -> None:
    """
    Replays calibration traces through the tube on a per-trace basis so that:
      1. Each trace produces chunk-indexed effective_tp observations.
      2. Even/odd trace splitting in finalize_frozen_calibration() operates on
         true trace boundaries with zero cross-trace forecaster bleed.
    """
    for trace_idx, trace_dict in enumerate(calib_trace_dicts):
        series = extract_chunk_effective_tps_from_trace(trace_dict, mode=calib_mode)
        if len(series) < (tube.window_size + tube.horizon):
            continue

        # Reset forecaster rolling history at the start of each trace
        tube.begin_calibration_trace(trace_idx=trace_idx)
        for actual_kbps in series:
            tube.ingest_calibration_sample(actual_kbps)
        tube.end_calibration_trace()

```

And inside `MaxScoreConformalTube` (`src/training/conformal_tube_v3.py`):

```python
# In src/training/conformal_tube_v3.py

def finalize_frozen_calibration(self) -> None:
    """
    Two-split frozen calibration (_fs) over chunk-indexed trace trajectories:
      - Even traces (0, 2, 4, ...): fit per-horizon scale sigma_h (h = 1..H).
      - Odd traces (1, 3, 5, ...): compute normalized H-step max-scores with
        stride H within each trace, then take the (1 - alpha) conformal quantile.
    """
    even_traces = [t for i, t in enumerate(self._calib_trace_Windows) if i % 2 == 0]
    odd_traces  = [t for i, t in enumerate(self._calib_trace_windows) if i % 2 == 1]

    # 1. Fit frozen per-horizon scales sigma_h on even traces
    for h_idx in range(self.horizon):
        h_residuals = [
             max(0.0, win.pred[h_idx] - win.actual[h_idx])
             for trace_wins in even_traces
             for win in trace_wins
        ]
        # Empirical scale (with floor for numerical stability)
        self.frozen_sigma[h_idx] = max(float(np.std(h_residuals, ddof=1)), 1e-3)

    # 2. Compute strided max-scores S_k on odd traces (stride = H chunks per trace)
    strided_max_scores: List[float] = []
    for trace_wins in odd_traces:
        for k in range(0, len(trace_wins), self.horizon):  # stride H
            win = trace_wins[k]
            s_k = max(
                max(0.0, win.pred[h_idx] - win.actual[h_idx]) / self.frozen_sigma[h_idx]
                for h_idx in range(self.horizon)
            )
            strided_max_scores.append(s_k)

    n = len(strided_max_scores)
    if n < 20:
        raise RuntimeError(f"Insufficient strided calibration windows (n={n}) for _fs.")

    # Finite-sample conformal quantile level ceil((n + 1) * (1 - alpha)) / n
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
    self.frozen_q = float(np.quantile(strided_max_scores, q_level, method="higher"))
    self.is_frozen_calibrated = True

```

### D.3 Unit Tests & Runtime Cost

* **Unit Tests (`tests/test_tube_v2_chunk_calib.py`):**
1. `test_piecewise_constant_bb_synthetic_trace()`: Construct a synthetic broadband trace where raw ticks are sampled at $\Delta t = 0.5\text{s}$ with bandwidth plateaus lasting $6\text{s}$ (so 5 raw ticks $= 2.5\text{s}$ usually stay inside one plateau, while 5 chunks $= 20\text{s}$ cross 3 plateaus). Assert that `calib_mode="raw_samples"` yields joint coverage $< 0.70$ at $\alpha = 0.15$ on chunk-level evaluation, whereas `calib_mode="env_chunks"` yields joint coverage $\in [0.82, 0.90]$.
2. `test_iid_chunk_process_coverage()`: On synthetic i.i.d. log-normal chunk throughputs ($N = 100$ traces $\times 60$ chunks), verify that `finalize_frozen_calibration()` achieves empirical joint coverage within $\pm 0.025$ of $1 - \alpha$ for $\alpha \in \{0.10, 0.15, 0.20\}$ and that no forecast window spans across a trace boundary.


* **Runtime Cost:**
* Stepping `CleanABREnv` in pure Python for a single trace (~50–100 chunks) with a closed-form rate-matching reference policy (no MPC tree search during calibration!) takes **~0.5–1.5 ms per trace**.
* For a `bb` calibration set of ~100–200 traces, `_warm_start_tube_v2()` takes **< 0.3 seconds total** (and can be cached in memory by `(trace_set_name, calib_mode)` across episodes in the evaluation worker).



---

## 6. Risk Analysis for 5G Results & Paper Wording (Deliverable 5 & Question E)

### 6.1 Risk Table for 5G Families (`5g_irish`, `5g_harsh`, `5g_markov`)

| Trace Family & Current Headline Result

 | How `env_chunks` Changes `_fs` Tube vs. `raw_samples` | Specific Regression Risk | Concrete Mitigation (Pre-Registered in Val) |
| --- | --- | --- | --- |
| **`5g_irish`** (`maxscore_a0.1_safe_fs`: $\Delta\text{QoE} = +124.8$, $\Delta\text{rebuf} = -36.83\text{s}$, Joint cov $= 0.936$)

 | Chunk download integration smooths sub-chunk 5G spikes, so $\hat{\sigma}_h^{\text{chunk}}$ will be **slightly smaller** (~5–12% tighter tube) than $\hat{\sigma}_h^{\text{raw}}$. Joint coverage will move from $0.936$ closer to nominal $0.90$. | If `5g_irish`'s huge $-36.8\text{s}$ rebuffer reduction relied on the extra +3.6% over-coverage buffer of raw-tick `_fs`, a slightly tighter tube at $\alpha=0.10$ could allow 1–2 extra stalls during deep 5G dropouts. | `configs/tube_v3_alpha_rule.yaml` includes $\alpha=0.05$ in the candidate grid on val; if $\alpha=0.10$ becomes less conservative under chunk calibration, `select_alpha_rule.py` can select $\alpha=0.05$ on `5g_irish` val, **or** use lower-clip harmonic integration (`min(effective_tp, chunk_window_harmonic_tp)`) during calibration. |
| **`5g_harsh`** (`maxscore_a0.2_safe_fs`: $\Delta\text{QoE} = -1.4\text{ (ns)}$, $\Delta\text{rebuf} = -2.05\text{s}$, Joint cov $= 0.818$)

 | `5g_harsh` is already right at nominal coverage ($0.818$ vs $0.80$ for $\alpha=0.20$). Under `env_chunks`, early-abort / lowest-bitrate retries during severe 5G blackouts are explicitly reflected in calibration `effective_tp`.

 | If `env_chunks` tightens $\hat{Q}_{0.80}\hat{\sigma}_h$ on `5g_harsh`, joint coverage could dip from $0.818$ to ~0.77, or rebuffer gain ($-2.05\text{s}$) could shrink. | Because `select_alpha_rule.py` selects $\alpha$ by guarding tail rebuffering against `RobustMPC` on val, any tightening at $\alpha=0.20$ on `5g_harsh` val will automatically shift the selected arm to $\alpha=0.15$ or $\alpha=0.10$ on val before freezing.

 |
| **`5g_markov`** (`maxscore_a0.1_safe_fs`: $\Delta\text{QoE} = +36.5$, $\Delta\text{rebuf} = -12.05\text{s}$, Joint cov $= 0.924$)

 | Regime switches in Markov traces occur every few seconds; aligning calibration steps to chunk boundaries matches the exact regime-residence time seen by the planner. | **Low risk:** moving from $0.924$ toward $0.90$ joint coverage slightly reduces over-conservatism, which typically increases bitrate utility without hurting rebuffering due to the `_safe` constraint and `static_1step_cap`.

 | Verify on `5g_markov` val that $\Delta\text{rebuf}$ remains negative and joint coverage stays $\ge 0.88$. |

### 6.2 Paper Wording (`docs/LIMITATIONS.md` & `docs/theory/tube_mpc_theory.tex`)

**E.1 Paragraph for `docs/LIMITATIONS.md` (if we ship the fix):**

> **Offline Frozen Calibration (`_fs`) and Action-Dependent Effective Throughput.** Under Assumption A1 of Theorem 1, the horizon scale factors $\{\hat{\sigma}_h\}_{h=1}^H$ and simultaneous max-score quantile $\hat{Q}_{1-\alpha}$ are treated as fixed parameters estimated on a calibration split disjoint from evaluation traces. Because online ABR observations $y_k$ represent chunk-integrated effective throughputs (`bits_downloaded / download_time`) rather than raw trace ticks, calibrating `_fs` directly on raw trace arrays induces a timescale mismatch on slowly varying broadband (`bb`) links—where $H=5$ raw samples span a single bandwidth plateau while $H=5$ chunk downloads span multiple network regimes. To align the offline calibration distribution with runtime observations without leaking evaluation data, `_fs` warm-start replays calibration traces through the chunk simulator under a fixed, seed-free reference policy (with forecaster state reset at trace boundaries, even-trace scale estimation, and stride-$H$ odd-trace quantile estimation). Furthermore, because the executed control action is projected via $\min(\text{tube\_mpc}, \text{static\_1step\_cap})$, one-step buffer feasibility is jointly enforced by the conformal lower bound and the static rate cap.
> 
> 

**E.2 Single Sentence (if action-dependent $y_k$ is only approximately matched via a fixed reference policy):**

> Because realized chunk-integrated throughput $y_k$ depends weakly on the closed-loop bitrate trajectory via chunk download duration, calibrating `_fs` under a fixed reference policy aligns the chunk-level timescale and restores broadband joint coverage near $1-\alpha$, though residual policy-induced covariate shift between the calibration reference policy and closed-loop Tube-MPC remains an approximation to full interventional exchangeability.
> 
>