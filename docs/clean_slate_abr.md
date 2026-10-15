# Clean-slate wall-clock ABR (h3)

Active simulator spec: [`h3.txt`](h3.txt).

## Implemented

| Phase | Artifact |
|-------|----------|
| 1 Markov 5G traces | `data/prepare_markov_5g_traces.py` → `test_traces_5g_markov/` |
| 2 Wall-clock env | `src/environment/clean_abr_env.py` (`CleanABREnv`) |
| 3 Control arms | `src/control/two_timescale_conformal.py` |
| 4 Campaign | `experiments/run_clean_campaign.py` |

Legacy `abr_multi_env_v5..v18` removed from `src/environment/` (frozen copy remains under `paper_v1/`).

## Verify

```bash
export PYTHONPATH=$PWD
python data/prepare_markov_5g_traces.py
bash experiments/paper2/run_clean_smoke.sh
python experiments/run_clean_campaign.py --policy mpc --episodes 204 --reward-mode log_utility
python experiments/run_clean_campaign.py --policy greedy --episodes 204 --reward-mode log_utility
```

Paper 2 entry alias: `Paper2StreamingEnv` = `CleanABREnv` in `src/environment/paper2_abr_env.py`.
