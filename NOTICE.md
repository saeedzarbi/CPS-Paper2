# Provenance

Core CPS evaluation code in this repository was adapted from:

- https://github.com/saeedzarbi/ABR-Awareness  
  (local working copy: `ABR-Awareness-1`)

Copied files are the minimal runtime slice for Paper 2 pilots:

- `src/training/certified_perceptual_shield.py`
- `src/environment/abr_multi_env_v{5,6,9,10,11,12,14,15,18}.py`
- `src/evaluation/eval_certified_shield_v18.py`
- `src/evaluation/stall_cause_breakdown_v18.py`
- `src/evaluation/ablation_eps_alpha.py`
- `src/evaluation/coverage_under_drift.py`
- `configs/paths.py`, `configs/videos.py`

Paper 1 manuscript, training scripts, and raw media are **not** vendored here.
Large `data/` assets are linked locally (see `docs/data_setup.md`), not committed.
