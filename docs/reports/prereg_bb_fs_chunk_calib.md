# Pre-registration: `_fs` chunk-aligned calibration (`env_chunks`)

**Branch:** `tube-v3-val` (after chunk-calib code lands).  
**Motivation:** `bb` test joint coverage 0.646 vs nominal 0.85 for `maxscore_*_safe_fs` when `_fs` uses raw `throughput_kbps[]` ticks; runtime `tube.observe()` uses chunk `effective_tp` from `CleanABREnv`.

## Frozen before any new TEST run

1. `configs/tube_v3_alpha_rule.yaml` — **unchanged** (apply on val only).
2. `fs_calib_mode: env_chunks` in the new adaptive freeze yaml (not `tube_v3_adaptive_final.yaml` commit `818629c` until re-frozen).

## Val gate (must pass on `bb` val before freeze)

For each $\alpha \in \{0.10, 0.15, 0.20\}$ and arm family `maxscore_safe_fs`, empirical **joint** coverage on **bb val** (offline replay of val traces through the tube + env, or full val campaign) satisfies:

$$\hat{C}_{\text{joint}} \ge (1 - \alpha) - 0.03.$$

On `5g_markov`, `5g_harsh`, `5g_irish` val: same tolerance, and tail-rebuffer rule in `select_alpha_rule.py` still admissible for the chosen $\alpha$.

Diagnostic helper: `python -m src.evaluation.diagnose_val_fs_calib --trace-set bb --split val`.

## Protocol order

1. Code + this prereg + tests (`tests/test_tube_v2_chunk_calib.py`).
2. `run_tube_v3_adaptive_val.py --fs-calib-mode env_chunks` → `select_alpha_rule.py` → new freeze yaml.
3. Single TEST run → update `tube_v3_adaptive_test.md` / `LIMITATIONS.md`.

`configs/tube_v3_final.yaml` stays `fs_calib_mode: raw_samples` for historical reproducibility.
