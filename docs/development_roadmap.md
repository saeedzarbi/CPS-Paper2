# Engineering & Development Roadmap: CPS-Paper2 (V20)

This document defines the step-by-step implementation plan, target file paths, reference templates, and verification commands for developing **Paper 2 (Online ACI & Action-Masked Perceptual Shield)** inside the `CPS-Paper2` repository.

**Paper 2 versioning:** use **OACS** / release `oacs-r1` and `results/oacs/` — not `v20`. Mapping from this file's `*_v20` names: [`docs/paper2_naming.md`](paper2_naming.md). Server runs: [`docs/server_runs.md`](server_runs.md).

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