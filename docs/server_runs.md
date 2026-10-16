# Running `clean_v3` on a server

Paper 2 headline eval runs on **Linux** (204 paired episodes per policy×trace job).

## 1. Clone and environment

```bash
git clone <repo-url> CPS-Paper2
cd CPS-Paper2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-smoke.txt   # or requirements.txt
export PYTHONPATH="${PWD}"
export PYTHONIOENCODING=utf-8
```

## 2. Data

```bash
export CPS_PAPER2_DATA_ROOT=/datasets/cps-paper2/data   # optional
python scripts/paper2/verify_data.py
python data/prepare_markov_5g_traces.py   # if test_traces_5g_markov missing
```

See `docs/data_setup.md`.

## 3. Smoke

```bash
python experiments/run_clean_campaign.py --policy mpc --episodes 2 --trace 5g_markov --out /tmp/clean_smoke
```

## 4. Full `clean_v3` sweep

```bash
export CLEAN_CAMPAIGN=clean_v3
mkdir -p results/${CLEAN_CAMPAIGN}/logs
for p in mpc greedy bba bola; do
  python experiments/run_clean_campaign.py --policy "$p" --episodes 204 --out "results/${CLEAN_CAMPAIGN}"
done 2>&1 | tee "results/${CLEAN_CAMPAIGN}/logs/run_$(date +%Y%m%d_%H%M).log"
```

Outputs: `results/clean_v3/<policy>_{5g_markov,bb}/summary.json` and `episodes.csv`.

## 5. tmux / screen

```bash
tmux new -s clean_v3
# run section 4 inside the session
```

## 6. Extended validation (`clean_v3_extended`)

```bash
python data/prepare_eval_trace_subsets.py
bash experiments/paper2/run_extended_validation.sh
```

Deploy/restart on **tools-new** (SSH host `tools-new`): `python scripts/paper2/deploy_extended_tools_new.py`  
Status: `python scripts/paper2/status_extended_server.py`

Pull artifacts: `rsync -avz tools-new:/root/CPS-Paper2/results/clean_v3_extended/ results/clean_v3_extended/`  
Regenerate report locally (needs `results/clean_v3/`):  
`python src/evaluation/run_extended_validation.py --report-only --latency-episodes 17`

Report: `docs/reports/clean_v3_extended_validation.md`

## 7. Copy results off the server (optional)

```bash
rsync -avz results/clean_v3/ user@laptop:~/CPS-Paper2/results/clean_v3/
```

## Mattermost (optional)

Gitignored `mattermost_webhook.url` at repo root, or `MATTERMOST_WEBHOOK_URL`.

```bash
python scripts/paper2/mattermost_notify.py "clean_v3 smoke OK"
```

## 8. Tube-v2 (dedicated VPS, server 45)

Headline Tube-v2 campaigns run on **`45.195.250.136`** (credentials in gitignored `server.txt`), not tools-new. See [`docs/tube_v2_server_45.md`](tube_v2_server_45.md).

```bash
cd /root/CPS-Paper2 && git pull origin main
python3 -m venv .venv-tube-v2 && source .venv-tube-v2/bin/activate
pip install -r requirements-smoke.txt pytest
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/paper2/phase0_tube_v2_audit.py   # CHECKPOINT 0 inventory
tmux new -s tube_v2   # long jobs (once run_all_tube_v2.sh exists)
```
