# Running OACS campaigns on a server

Paper 2 jobs are designed for **Linux** batch nodes (no emoji in logs; UTF-8 safe `paths.py`).

## 1. Clone and environment

```bash
git clone <repo-url> CPS-Paper2
cd CPS-Paper2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="${PWD}"
export PYTHONIOENCODING=utf-8
```

## 2. Data (do not commit large `data/`)

Point the repo at a shared dataset mount:

```bash
export CPS_PAPER2_DATA_ROOT=/datasets/cps-paper2/data
# or: export CPS_DATA_ROOT=...
python configs/paths.py
```

Expected under that root (same layout as Paper 1):

- `standardized/test_traces_5g_v18/`, `test_traces_v18/`
- `vmaf_scores/vmaf_perchunk.csv`, `vmaf_summary.csv`
- `content_features/` (if scripts need SITI)

See also `docs/data_setup.md`.

## 3. Deploy from laptop (optional)

```bash
python scripts/paper2/_remote_bootstrap.py   # git clone + tarball sync + venv
python scripts/paper2/_remote_start_screen.py # smoke in screen session `oacs`
```

Log files persist under `results/oacs/logs/` even after screen exits.

## 4. Smoke test (after deploy)

```bash
bash experiments/paper2/run_smoke.sh
```

## 5. Full campaign (when eval scripts land)

```bash
export OACS_CAMPAIGN=oacs-r1_main          # optional; default in paper2_release
bash experiments/paper2/run_campaign.sh
```

Outputs:

```text
results/oacs/campaigns/<OACS_CAMPAIGN>/
results/models/oacs/
```

## 6. tmux / nohup pattern

```bash
tmux new -s oacs
source .venv/bin/activate
export CPS_PAPER2_DATA_ROOT=/datasets/cps-paper2/data
export PYTHONPATH=$PWD
bash experiments/paper2/run_campaign.sh 2>&1 | tee results/oacs/campaigns/logs/run_$(date +%Y%m%d_%H%M).log
```

## 7. Sync results back (optional)

```bash
rsync -avz results/oacs/ user@laptop:~/CPS-Paper2/results/oacs/
```

## 8. What stays on laptop vs server

| Task | Where |
|------|--------|
| Editing shield / eval code | Laptop or server |
| 204-episode × multi-policy sweeps | **Server** |
| LaTeX asset build (`make_oacs_assets.py`) | Either (needs `results/oacs/`) |

Paper 1 artifacts remain under `paper_v1/` and `results/v18_*` on the old tree; Paper 2 never writes into those paths.
