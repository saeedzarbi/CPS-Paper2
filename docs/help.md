
> `@docs/decision_rule.md @docs/development_roadmap.md Implement Phase X step-by-step and run the verification command.`

---

```markdown
# Engineering & Development Roadmap: CPS-Paper2 (V20)

This document defines the step-by-step implementation plan, target file paths, reference templates, and verification commands for developing **Paper 2 (Online ACI & Action-Masked Perceptual Shield)** inside the `CPS-Paper2` repository.

**Golden Rule for AI Coding Assistants (Cursor):**
- **NEVER** modify any file inside `paper_v1/` (frozen baseline archive).
- **ALWAYS** preserve episode seeding (`random.seed(1000 + ep)` and `env.reset(seed=1000 + ep)`) so paired statistical tests across arms remain 100% aligned.
- **ALWAYS** keep `configs/videos.py` (`CPS_EPISODES = 204`, 12 videos, `MAX_CHUNKS = 48`) and `src/environment/abr_multi_env_v18.py` as the shared environment foundation.

---

## Phase 1: Dependencies, Paths & Data Verification

### 1.1. Update `requirements.txt`
- **Action:** Edit `requirements.txt` in the repository root.
- **Details:** Add `sb3-contrib>=2.2,<3.0` alongside `stable-baselines3>=2.2,<3.0` to enable `MaskablePPO` and `QRDQN`.

### 1.2. Verify `configs/paths.py` and Data Symlinks
- **Action:** Ensure `data/` is accessible at the repository root (via symlink or directory copy as described in `docs/data_setup.md`).
- **Required Data Assets:**
  - `data/standardized/train_traces_v18/` and `data/standardized/test_traces_v18/`
  - `data/standardized/train_traces_5g_v18/` and `data/standardized/test_traces_5g_v18/`
  - `data/vmaf_scores/vmaf_summary.csv` and `data/vmaf_scores/vmaf_perchunk.csv`
  - `data/content_features/siti_summary.csv` and per-video `*_siti.json`
- **Verification Command:**
  ```bash
  python configs/paths.py
  python configs/videos.py

```

---

## Phase 2: Core Shield & Online ACI Module (`src/training/`)

### 2.1. Create `src/training/aci_perceptual_shield.py`

* **Reference Template:** `@src/training/certified_perceptual_shield.py` and `@paper_v1/src/evaluation/ablation_perchunk_ladder.py`.
* **Components to Implement:**
1. **`ACIConfig` (dataclass):**
* `alpha: float = 0.10` (target miscoverage)
* `gamma_aci: float = 0.04` (online step size for $\alpha_t$ adaptation)
* `decay: float = 0.90` (exponential forgetting factor for non-stationary residuals)
* `window: int = 200`, `k_predict: int = 5`
* `prior_scale: float = 0.78`, `prior_weight: int = 3` (pseudo-prior residuals to eliminate the 20-chunk `cold_start` window).


2. **`OnlineACIEstimator` (class):**
* Harmonic mean point predictor `predict()` over last `k_predict` realizations.
* Exponentially weighted quantile `_weighted_quantile()` over rolling residual ratios $r_i = C_i^{\text{act}} / \hat{C}_i$ evaluated at effective level $\alpha_t$.
* Online update `update(actual_kbps)`: computes indicator error $\text{err}_t = \mathbb{I}\{C_t^{\text{act}} < \underline{C}_t\}$ and updates $\alpha_{t+1} = \text{clip}(\alpha_t + \gamma_{\text{aci}}(\alpha - \text{err}_t), 0.005, 0.45)$.


3. **`PerChunkVMAFLookup` (class):**
* Loads `data/vmaf_scores/vmaf_perchunk.csv` into a fast dictionary keyed by `(video_slug, chunk_idx, bitrate_kbps)`.
* Provides `get(env, rung_idx) -> float` with automatic fallback to `env.current_vmaf_scores` if a specific chunk is not in the CSV.


4. **`ACIShieldConfig` & `aci_certified_action(...)`:**
* Computes per-chunk VMAF knee $j_k^{\text{knee}}(a) = \min \{ j \le a : V_k(a) - V_k(j) \le \epsilon_{\text{eff}} \}$.
* Projects onto buffer-feasible set under $\underline{C}_k$: $d_k(j) \le \max(B_k - m, 0.1)$.
* Logs diagnostics (`banked_bits`, `vmaf_given_up`, `tp_lb_kbps`, `alpha_t`, `bound_held`, `cold_start = 0`, `perchunk_vmaf_used`).


5. **`ACIPerceptualShieldWrapper(gym.Wrapper)`:**
* Wraps `ABREnv` for both heuristic and RL hosts.
* Implements `action_masks(self) -> np.ndarray` (boolean array of length $L=6$) for `sb3_contrib.MaskablePPO`: sets `True` only for rungs that are both buffer-feasible under $\underline{C}_k$ and $\le j_k^{\text{knee}}(a_{\text{max\_feasible}})$. Always keeps rung `0` `True` if all higher rungs fail.




* **Verification Command (Smoke Test):**
```bash
python -c "from src.environment.abr_multi_env_v18 import ABREnv; from src.training.aci_perceptual_shield import ACIPerceptualShieldWrapper; env = ACIPerceptualShieldWrapper(ABREnv()); obs, _ = env.reset(seed=1000); print('Mask:', env.action_masks()); obs, r, d, t, info = env.step(5); print('Step OK, alpha_t:', info['alpha_t'], 'tp_lb:', info['tp_lb_kbps'])"

```



---

## Phase 3: Evaluation & Diagnostic Scripts (`src/evaluation/` & `experiments/`)

### 3.1. Create `src/evaluation/eval_aci_shield_v20.py`

* **Reference Template:** `@src/evaluation/eval_certified_shield_v18.py`.
* **Specifications:**
* Support policies: `greedy`, `bba`, `bola`, `mpc`, `ppo` (standard SB3), and `maskable_ppo` (`sb3_contrib.MaskablePPO`).
* Support 4 paired evaluation arms on identical episode seeds (`1000 + ep`):
1. `raw`: unshielded base policy.
2. `cps_v18`: Paper 1 baseline (`CertifiedPerceptualShieldWrapper` with static split-conformal, `min_calib=20`, session-mean VMAF).
3. `aci_safety`: Paper 2 online ACI feasibility only (`enable_banking=False`).
4. `aci_certified`: Paper 2 full shield (Online ACI + exponential weighting + per-chunk VMAF-knee banking).


* Compute paired comparisons (`aci_certified_vs_cps_v18`, `aci_certified_vs_aci_safety`, `aci_certified_vs_raw`) with Wilcoxon signed-rank $p$-values, TOST equivalence within $\pm\epsilon$, and true per-chunk VMAF budget violation rates.
* Save `summary.json` and `episodes.csv` under `--out`.



### 3.2. Create `src/evaluation/stall_cause_breakdown_v20.py`

* **Reference Template:** `@src/evaluation/stall_cause_breakdown_v18.py`.
* **Specifications:**
* Run `cps_v18` and `aci_certified` side-by-side over the 204 paired 5G episodes.
* Attribute every second of rebuffering to `emergency`, `cold_start`, `bound_miss`, or `bound_held`.
* Output comparative JSON/CSV proving the elimination of the 91.3% `cold_start` stall bucket.



### 3.3. Create `src/evaluation/coverage_under_drift_v20.py`

* **Reference Template:** `@src/evaluation/coverage_under_drift.py`.
* **Specifications:**
* Replay the 204 paired 5G episodes with the mid-episode throughput shock (`shock_start=28`, `shock_len=10`, `shock_scale=0.30`).
* Record and plot both `cps_v18` (static window) and `aci_certified` (online ACI + decay) rolling coverage curves and $\alpha_t$ trajectory on the same figure (`paper/figures/fig_v20_coverage_drift.pdf`).



### 3.4. Create Master Evaluation Runner `experiments/run_v20_eval.sh`

* **Specifications:**
* Bash script that runs `eval_aci_shield_v20.py` for `greedy`, `bba`, `bola`, and `mpc` on both `test_traces_5g_v18` and `test_traces_v18` (broadband), followed by `stall_cause_breakdown_v20.py` and `coverage_under_drift_v20.py`.


* **Verification Command:**
```bash
bash experiments/run_v20_eval.sh

```



---

## Phase 4: Action-Masked Multi-Seed RL Training (`src/training/`)

### 4.1. Port Lagrangian Wrappers to `src/training/`

* **Action:** Copy `@paper_v1/src/training/constrained_abr.py` and `@paper_v1/src/training/constrained_abr_v14.py` into `src/training/`.

### 4.2. Create `src/training/train_maskable_v20.py`

* **Reference Template:** `@paper_v1/src/training/train_all_models_v18.py`.
* **Specifications:**
* Add model key `maskable_aci`: wraps the training and eval environments in `ACIPerceptualShieldWrapper` and trains `sb3_contrib.MaskablePPO("MlpPolicy", ...)` so policy gradients only update over valid, non-saturated rungs via `action_masks()`.
* Support multi-seed training (`--seeds 0,1,2`) and save checkpoints under `results/models/v20_5g/maskable_aci/seed_<S>/`.


* **Verification Command (Quick Smoke Test):**
```bash
python src/training/train_maskable_v20.py --models maskable_aci --seeds 0 --timesteps-scale 0.01 --num-envs 2

```



---

## Phase 5: Automated Paper Tables, Figures & Macros (`src/paper/`)

### 5.1. Create `src/paper/make_paper2_assets.py`

* **Reference Template:** `@paper_v1/src/paper/make_cps_paper_assets.py`.
* **Specifications:**
* Read `results/v20_aci/*/summary.json` and `episodes.csv`.
* Generate LaTeX macros in `paper/tables/macros_v20.tex`.
* Generate main comparison tables in `paper/tables/`:
* `table_v20_main_5g.tex` (4 arms across all hosts on 5G)
* `table_v20_broadband.tex` (Broadband coverage and stall reduction: `0.890` in v18 vs. target `0.900` in v20)
* `table_v20_stall_breakdown.tex` (Cold-start vs. bound-miss attribution)
* `table_v20_multiseed_rl.tex` (Multi-seed `MaskablePPO` vs. post-shielded `PPO`)


* Generate publication-ready PDF figures in `paper/figures/`:
* `fig_v20_overview.pdf` (Head-to-head bar chart: CPS-v18 vs. ACI-v20 rebuffering & bitrate savings)
* `fig_v20_coverage_drift.pdf` (Rolling coverage + adaptive $\alpha_t$ under 0.3x shock)
* `fig_v20_cdf_rebuffer.pdf` (Episode rebuffering CDF across arms)




* **Verification Command:**
```bash
python src/paper/make_paper2_assets.py

```



```

---

### راهنمای شروع سریع در Cursor
1. فایل اول (از پیام قبلی) را در `docs/decision_rule.md` و فایل بالا را در `docs/development_roadmap.md` ذخیره کنید[cite: 36].
2. در پنجره Composer در Cursor بنویسید:
   > `@docs/decision_rule.md @docs/development_roadmap.md Let's execute Phase 1 and Phase 2. Create src/training/aci_perceptual_shield.py and run the smoke test.`
3. بلافاصله پس از پاس شدن تست فاز ۲، دستور اجرای **فاز ۳** را بدهید تا در کمتر از چند دقیقه کل نتایج مقایسه‌ای مقاله دوم روی کنترل‌کننده‌های `greedy`، `bba`، `bola` و `mpc` در پوشه `results/` ساخته شود[cite: 34, 36].

```