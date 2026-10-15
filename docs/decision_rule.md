# Project Specification: Online Adaptive Conformal & Action-Masked Perceptual Shielding (CPS-Paper2 / V20)

> **Archived (2026-10).** Active Paper 2: **[clean-slate ABR](clean_slate_abr.md)** and **`clean_v3`** results (`docs/reports/clean_v3_full_metrics.md`).

---

## 1. Overview & Motivation
This repository (`CPS-Paper2`) develops the second-generation runtime assurance framework for client-side Adaptive Bitrate (ABR) video streaming, building upon the baseline archive in `paper_v1/`.
While Paper 1 (`src/training/certified_perceptual_shield.py`, V18) introduced VMAF-knee bandwidth banking and static split-conformal throughput lower bounds, it exhibited three quantitative bottlenecks:
1. **Cold-Start Vulnerability:** The static split-conformal calibrator required a warmup window of `min_calib = 20` chunks (out of 48-chunk episodes) using a fixed fallback scale (`0.80`), during which 91.3% of all residual rebuffering occurred.
2. **Non-Stationary Coverage Drift:** Equal weighting of calibration residuals caused empirical coverage to dip below the `1 - alpha = 0.90` target on non-stationary broadband traces (`0.890`) and under tighter miscoverage levels (`0.945` vs. `0.95` at `alpha = 0.05`).
3. **Pooled-Ladder Budget Violations & Post-Shield Conflict:** Operating on session-mean VMAF ladders (`vmaf_summary.csv`) caused an 11.5% per-chunk perceptual budget violation rate when evaluated against true chunk-level VMAF (`vmaf_perchunk.csv`), and post-shielding a standard PPO policy resulted in a 67.2% action-override intervention rate.

## 2. Core Technical Innovations (Paper 2)
Paper 2 resolves all three bottlenecks through three tightly coupled algorithmic upgrades:

### 2.1. Prior-Seeded Online Adaptive Conformal Inference (Online ACI) with Exponential Decay
Instead of a fixed miscoverage `alpha` and an uncertified 20-chunk warmup, we maintain an online effective miscoverage level `alpha_k` updated after each chunk `k`:
- **ACI Update Rule:** `alpha_{k+1} = clip(alpha_k + gamma_aci * (alpha_target - err_k), alpha_min, alpha_max)`, where `err_k = 1{C_k^act < C_k_lb}`.
- **Exponentially Weighted Quantile:** Residual ratios `r_i = C_i^act / C_i^hat` in the rolling window are weighted by `w_i proportional to decay^(n - i)` (`decay in (0.85, 0.95)`), prioritizing recent regime dynamics during sudden 5G LOS/NLOS transitions.
- **Zero Cold-Start Initialization:** The residual buffer is initialized with a lightweight pseudo-prior (`prior_weight = 3` at `prior_scale = 0.78`), enabling valid adaptive quantile computation from chunk `k = 1` without a 20-chunk fallback delay.

### 2.2. True Per-Chunk VMAF-Knee Bandwidth Banking
Instead of relying solely on session-mean VMAF (`vmaf_summary.csv`), the shield queries true chunk-level VMAF `V_k(j)` from `data/vmaf_scores/vmaf_perchunk.csv` (falling back to session-mean only if unavailable):
- **Knee Rule:** `j_k^knee(a) = min { j <= a : V_k(a) - V_k(j) <= epsilon_eff }`.
- Guarantees 0.0% per-chunk perceptual budget violations by construction while exploiting local non-monotone chunk inversions for deeper bitrate savings.

### 2.3. Conformal Pre-Shielding via Action Masking (`MaskablePPO`)
In addition to wrapping deterministic hosts (`greedy`, `bba`, `bola`, `mpc`), the shield exposes a boolean mask `action_masks() -> np.ndarray` of shape `(L,)` for reinforcement learning (`sb3_contrib.MaskablePPO`):
- Rungs violating the conformal buffer feasibility test `d_k(j) <= max(B_k - m, 0.1)` are masked out (`False`).
- Perceptually saturated rungs strictly above the feasible VMAF knee (`j > j_k^knee(top_feasible)`) are masked out (`False`).
- Policy gradients update only over valid, non-saturated rungs, eliminating post-hoc action override conflicts and accelerating multi-seed convergence.

## 3. Repository Architecture & Invariants
- Keep `paper_v1/` untouched as the frozen baseline reference.
- Maintain exact compatibility with `configs/paths.py`, `configs/videos.py` (12 reference titles, `CPS_EPISODES = 204`, `MAX_CHUNKS = 48`), and `src/environment/abr_multi_env_v18.py`.
- Paired evaluation seeding (`random.seed(1000 + ep)` and `env.reset(seed=1000 + ep)`) must remain strictly identical across arms (`raw`, `cps_v18`, `aci_safety`, `aci_certified`) to preserve paired Wilcoxon signed-rank and TOST equivalence validity.