# OACS Release `oacs-r1`: Work Log & First Campaign Results

**Repository:** CPS-Paper2 (Paper 2 — Online ACI + Action-Masked Perceptual Shield)  
**Report date:** 2026-10-14  
**Campaign ID:** `oacs-r1_mm`  
**Release tag:** `oacs-r1` (see `configs/paper2_release.py`)

---

## 1. Purpose

This document records engineering work completed on CPS-Paper2, the first end-to-end **heuristic evaluation campaign** run on a remote Linux server, and an honest read of the numbers relative to Paper 1’s **Certified Perceptual Shield (CPS v18)** baseline.

Paper 2 targets three V18 bottlenecks (see `docs/decision_rule.md`):

1. Cold-start rebuffering under static split-conformal warmup (`min_calib = 20`).
2. Coverage drift under non-stationary traces.
3. Pooled VMAF ladders and post-hoc RL shielding conflicts.

The **OACS** codename (Online ACI Conformal Shield) denotes Paper 2 artifacts; they are intentionally **not** named `v20` to avoid confusion with Paper 1’s `v18` environment lineage.

---

## 2. Engineering & Infrastructure Completed

### 2.1 Versioning & layout (distinct from Paper 1)

| Paper 1 convention | Paper 2 (OACS) |
|--------------------|----------------|
| `results/v18_certified/` | `results/oacs/campaigns/<name>/` |
| `abr_multi_env_v18.ABREnv` in new code | `paper2_abr_env.Paper2StreamingEnv` (same dynamics) |
| Arms `raw`, `safety`, `certified` | `raw`, `static_cps`, `aci_feasibility`, `aci_oacs` |

Documentation: `docs/paper2_naming.md`, `docs/server_runs.md`.

### 2.2 Core software delivered

| Component | Path | Role |
|-----------|------|------|
| Online ACI shield + masks | `src/training/aci_perceptual_shield.py` | `OnlineACIEstimator`, `PerChunkVMAFLookup`, `ACIPerceptualShieldWrapper`, `action_masks()` |
| Paper 2 env entry | `src/environment/paper2_abr_env.py` | Thin alias over v18 physics |
| Four-arm evaluator | `src/evaluation/eval_oacs_campaign.py` | Paired 204-episode rolls; Wilcoxon/TOST in `summary.json` |
| Campaign runner | `experiments/paper2/run_campaign.sh` | greedy, bba, bola, mpc × 5G + broadband |
| Smoke / screen helpers | `experiments/paper2/run_smoke.sh`, `start_oacs_screen.sh` | Server preflight |
| Remote deploy | `scripts/paper2/_remote_bootstrap.py`, `_remote_run_campaign.py`, `_remote_fetch_results.py` | Tarball sync + `screen` jobs |
| Mattermost logging | `src/utils/mattermost_notify.py` | ~10 messages per campaign (start, 8 jobs, end) |
| Dependencies | `requirements.txt`, `requirements-smoke.txt` | Full stack vs lightweight eval |

**Invariant preserved:** paired seeds `random.seed(1000 + ep)` and `env.reset(seed=1000 + ep)` across arms. **`paper_v1/`** left untouched.

### 2.3 Data & server

- Dataset layout under `data/` (standardized `*_v18` traces, `vmaf_scores`, including `vmaf_perchunk.csv`).
- Server clone at `/root/CPS-Paper2`; sync via `git` + tarball for unpublished files.
- `CPS_PAPER2_DATA_ROOT` supported in `configs/paths.py`.
- Git conflict on server resolved with `git reset --hard origin/main` + `git clean -fd` (`scripts/paper2/server_git_sync.sh`).

### 2.4 Agent workflow (optional process)

Dual-agent docs under `docs/agents/` (Academic Lead + Execution Arm). Not required to reproduce numbers.

---

## 3. First Full Heuristic Campaign (`oacs-r1_mm`)

### 3.1 Protocol

| Setting | Value |
|---------|--------|
| Policies | `greedy`, `bba`, `bola`, `mpc` |
| Traces | `data/standardized/test_traces_5g_v18`, `test_traces_v18` |
| Episodes per arm | 204 (`CPS_EPISODES`) |
| Arms | `raw`, `static_cps` (CPS v18), `aci_feasibility` (ACI, banking off), `aci_oacs` (full OACS) |
| Perceptual budget ε | 1.0 VMAF |
| Miscoverage α | 0.10 (coverage target 0.90) |
| Output | `results/oacs/campaigns/oacs-r1_mm/<policy>_{5g,bb}/` |

Each job writes `summary.json` and `episodes.csv`.

### 3.2 Completion status

All **8** jobs finished successfully (`summary.json` present for each).

Local copy fetched to: `results/oacs/campaigns/oacs-r1_mm/`.

---

## 4. Results Summary

### 4.1 Mean total rebuffering per episode (seconds)

| Job | raw | static_cps | aci_feasibility | aci_oacs | aci_oacs − static_cps |
|-----|-----|------------|-----------------|----------|------------------------|
| greedy_5g | 87.47 | **39.84** | 81.84 | 72.20 | +32.36 |
| greedy_bb | 1216.25 | 15.81 | 15.00 | **14.76** | −1.06 |
| bba_5g | 78.14 | **36.60** | 78.06 | 69.28 | +32.68 |
| bba_bb | 175.31 | 14.67 | 13.76 | **13.51** | −1.16 |
| bola_5g | 78.09 | **37.98** | 78.00 | 69.72 | +31.74 |
| bola_bb | 39.74 | 15.13 | 13.54 | **13.45** | −1.67 |
| mpc_5g | 66.25 | **33.46** | 64.48 | 56.70 | +23.24 |
| mpc_bb | 12.84 | 9.78 | 9.36 | **9.34** | −0.44 |

**Reading:**

- **5G:** Paper 1 `static_cps` yields the **lowest** mean rebuffering for all four hosts. OACS (`aci_oacs`) improves over `raw` and over `aci_feasibility` but remains **substantially worse** than CPS v18 (~23–33 s higher mean rebuffer per episode).
- **Broadband:** OACS is **slightly better** than `static_cps` (sub-second to ~1.7 s improvements). Differences are small relative to 5G gaps.
- **`aci_feasibility` ≈ `raw` on 5G** suggests the dominant issue is the **online throughput lower bound**, not VMAF-knee banking alone.

### 4.2 Headline job: `greedy_5g` (paired statistics)

From `results/oacs/campaigns/oacs-r1_mm/greedy_5g/summary.json`:

| Arm | Mean rebuffer (s) | Mean VMAF | Mean bitrate (kbps) | Conformal coverage |
|-----|-------------------|-----------|---------------------|--------------------|
| raw | 87.47 | 89.08 | 6000 | — |
| static_cps | **39.84** | 79.29 | 2956 | **0.919** (meets 0.90) |
| aci_feasibility | 81.84 | 77.50 | 3348 | 0.905 |
| aci_oacs | 72.20 | 77.48 | 3128 | 0.905 |

**`aci_oacs` vs `static_cps`:**

- Rebuffer: **+32.4 s** (+81%), Wilcoxon *p* ≈ 5.5×10⁻³⁵.
- VMAF: −1.80 points; **TOST equivalence within ±1.0 rejected** (*p*<sub>equiv</sub> ≈ 1).
- Bitrate: OACS ~5.8% lower (banking effect).

**`aci_oacs` vs `aci_feasibility`:**

- Rebuffer: **−9.6 s** (−12%), Wilcoxon *p* ≈ 2.9×10⁻⁵; TOST VMAF **accepted** → banking + per-chunk knee helps **relative to ACI-only**, but not enough to beat v18.

**`aci_oacs` vs `raw`:**

- Rebuffer: **−15.3 s** (−17%) — shielding helps vs unshielded greedy.

**Stall-free rate:** 0% for all arms on this job (no episode with zero rebuffer in the aggregate mean).

### 4.3 Anomaly note

`greedy_bb` **raw** mean rebuffer (~1216 s) is extreme compared to other arms (~15 s). Worth inspecting per-episode outliers in `episodes.csv` before using broadband raw as a baseline figure in prose.

---

## 5. Interpretation (Preliminary)

### 5.1 What this run **does** support

1. **Evaluation pipeline works:** four arms, paired seeds, eight policies/traces, reproducible JSON/CSV outputs.
2. **OACS reports ~90% empirical coverage** on shielded arms (similar to target).
3. **Banking matters:** `aci_oacs` consistently beats `aci_feasibility` on 5G rebuffer.
4. **Broadband:** small gains vs CPS v18; may support a “non-stationary / coverage” subplot if confirmed with drift scripts.

### 5.2 What this run **does not** support (yet)

1. Claim that **OACS replaces CPS v18 on 5G** for heuristic hosts under the current `aci_perceptual_shield.py` calibration.
2. Paper 2’s **cold-start elimination** narrative at the quantitative level (stall attribution not yet run; 5G regression dominates).
3. **MaskablePPO** or post-shield vs pre-shield RL comparisons (Phase 4 not executed).
4. Publication tables/figures (`make_oacs_assets.py`, stall breakdown, coverage-under-drift plots — Phase 3.2–3.3, Phase 5).

### 5.3 Likely technical cause (hypothesis for next iteration)

The implementation uses **prior-seeded, exponentially weighted quantiles** for `tp_lb`, whereas CPS v18 uses **finite-sample split-conformal** `(n+1)` quantiles with a 20-chunk fallback. On high-throughput 5G traces, the current ACI bound appears **less aggressive in downgrading** than v18 (`aci_feasibility` near `raw`), while still meeting nominal coverage. **Calibration alignment** (not necessarily abandoning online ACI) is the highest-priority engineering follow-up.

---

## 6. Work Not Yet Done (Roadmap Gap)

| Item | Roadmap ref | Status |
|------|-------------|--------|
| `stall_attribution_oacs.py` | Phase 3.2 | Not implemented |
| `coverage_drift_oacs.py` | Phase 3.3 | Not implemented |
| PPO / MaskablePPO training & eval | Phase 4 | Not started |
| `make_oacs_assets.py` | Phase 5 | Not started |
| ACI calibration fix + re-campaign | Scientific | Not done |
| Official “paper-quality” campaign tag after fix | — | Pending |

The **`oacs-r1_mm` campaign is the first complete heuristic sweep**, not the final paper submission bundle.

---

## 7. Recommended Next Steps

1. **Diagnostics:** Per-chunk time series of `tp_lb_kbps`, `alpha_t`, `shielded_action` for `greedy_5g` — compare OACS vs `static_cps` on identical seeds.
2. **Calibration:** Align weighted quantile / prior with v18 coverage–tightness tradeoff; short rerun (`OACS_EPISODES=20`) before another full 204-episode campaign.
3. **Attribution & drift:** Implement Phase 3.2–3.3 scripts to test cold-start and shock narratives.
4. **RL arm:** Phase 4 `maskable_aci` once throughput bound is competitive on 5G.
5. **Version control:** Push local OACS code (eval, notify, reports) to `origin/main`; keep `server.txt` and `mattermost_webhook.url` out of git.

---

## 8. Reproduction Commands

**Local smoke:**

```bash
export PYTHONPATH=$PWD
python configs/paths.py
bash experiments/paper2/run_smoke.sh
```

**Server campaign:**

```bash
cd /root/CPS-Paper2
bash scripts/paper2/server_git_sync.sh   # after git push
bash experiments/paper2/start_oacs_screen.sh
# or: bash experiments/paper2/run_campaign.sh
```

**Fetch summaries:**

```bash
python scripts/paper2/_remote_fetch_results.py
```

---

## 9. References in Repo

- Specification: `docs/decision_rule.md`
- Naming: `docs/paper2_naming.md`
- Server ops: `docs/server_runs.md`
- Roadmap: `docs/help.md` (full phases), `docs/development_roadmap.md` (partial)

---

*End of report.*
