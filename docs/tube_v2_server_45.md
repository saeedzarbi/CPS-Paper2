# Tube-v2 execution (dedicated VPS)

Dedicated host for **Tube-v2** campaigns (`cursor_prompt_paper2.md`). Shared **tools-new** stays for legacy `clean_v3_extended` only.

## Access

Credentials: gitignored `server.txt`. Typical remote tree: `/root/CPS-Paper2`.

## CHECKPOINT 0 — done on host (2026-10-16)

- Repo: `main` @ `dc07b3e` (pulled after stash of old local edits).
- Venv: `.venv-tube-v2` + `requirements-smoke.txt`.
- Data restored from `stash@{0}^3`: `prepare_markov_5g_traces.py`, Markov train/test (80/50), `eval_fcc_fixed` (14). **Irish MMSys'20 not on disk yet** (Phase 0.4).
- Audit JSON: `results/tube_v2/checkpoint0_audit.json` (via `scripts/paper2/phase0_tube_v2_audit.py`).

| ACI on tube estimator | |
|--|--|
| `symmetric_tube` | yes |
| `tube_conformal` | yes |
| `fixed_alpha_tube` | no (static split, `static_adapt=False`) |

Pilot (6 arms × 5 ep, `bba`, Markov, `--workers=1`): **~10 s** → extrap. headline 2×3×204 ≈ **7 min**, OOD 4×6×204 ≈ **28 min** (single core; tube arms dominate).

Tube decision latency (μs mean): symmetric ~2.6 ms; capped tube ~18 ms; hosts bba ~7 μs, mpc ~0.5 ms.

## Phase 0.4 — Irish MMSys'20 (2026-10-16)

Default group split (not `--all-test`): **11 test**, **67 train** under `data/standardized/test_traces_5g_irish_mmsys20/`.

## CHECKPOINT 1 — Phase 1 (2026-10-16)

- Code: `conformal_tube_v2.py`, `tube_mpc_policy_v2.py`, `docs/tube_v2_theory.md`, `tube_coverage_report.py`, `tests/test_conformal_tube_v2.py`.
- **pytest:** 25 passed on server 45 (~66 s, `--workers` N/A).
- Coverage artifacts: `results/tube_v2/coverage/coverage_summary.csv`, `coverage_<trace>_<mode>_aci*.png/pdf`.
- Example mean covered_rate @ h=1, Markov, no ACI: `per_step` ~0.90, `bonferroni` ~0.96 (real traces ≠ i.i.d.).
- `TubeV2MPCPolicy` plan latency (bb, H=5, μs): `none` ~1.2k, `monotone` ~5.7k, `beam` ~9.3k (beam sorts full grid — tune later).

## CHECKPOINT 0 — resources (2026-10-16)

| Item | Value |
|------|--------|
| Host | `srv1020091255` |
| CPUs | **1** (`nproc`) |
| RAM | **~1.9 GiB** (≈1.1 GiB available) |
| Swap | 1 GiB |
| Disk `/` | 40G, ~31% used |
| GPU | none |
| Load | low (~0.04) |
| Default `--workers` | **1** (leave headroom on single core; set `OMP_NUM_THREADS=1`) |

**Implication:** full 204-episode × many-arm grids will be **slow**; use `--pilot` / `--smoke` first, `tmux` session `tube_v2`, and `--resume` incremental CSVs per the master prompt.

## Repo on server (before sync)

- Branch: `main` @ `a373097` (behind laptop `dc07b3e`).
- Action: `git pull` or `rsync` from laptop after `tube-v2` branch is pushed **with user approval**, or deploy tarball.

## Environment (Phase 0)

```bash
cd /root/CPS-Paper2
python3 -m venv .venv-tube-v2
source .venv-tube-v2/bin/activate
pip install -r requirements.txt
pip install pytest
pip freeze > results/tube_v2/env_freeze.txt
```

Create `logs/` (gitignored) and `results/tube_v2/` before long runs.

## Long jobs

```bash
tmux new -s tube_v2
source .venv-tube-v2/bin/activate
export PYTHONPATH=/root/CPS-Paper2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# scripts/paper2/run_all_tube_v2.sh --pilot ...  (once implemented)
```

Monitor: `tmux attach -t tube_v2`, `tail -f logs/<run>.log`.

## Do not

- Use this box for unrelated users' processes (single-tenant VPS).
- `git push` without user OK (project rule).
