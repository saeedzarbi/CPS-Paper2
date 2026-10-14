# Data setup

This repo expects a `data/` directory at the project root with the same layout as
Paper 1 (`new/data`), including:

- `data/standardized/test_traces_5g_v18/` (synthetic 5G-like; Paper 1)
- `data/standardized/test_traces_5g_irish_mmsys20/` (optional **real** 5G — see [`docs/data_5g_real.md`](data_5g_real.md))
- `data/standardized/test_traces_v18/` (broadband)
- `data/vmaf_scores/` (session / per-chunk VMAF logs used by the env)

## Local development (Windows)

A directory junction was created once:

```text
CPS-Paper2\data  ->  ABR-Awareness-1\new\data
```

Re-create if needed:

```bat
mklink /J "%USERPROFILE%\OneDrive\Desktop\CPS-Paper2\data" "%USERPROFILE%\OneDrive\Desktop\ABR-Awareness-1\new\data"
```

## Independent machine / publication

Copy or regenerate traces and VMAF scores into `data/` inside this repo.
Do **not** commit raw videos or large CSVs; keep them local or on Zenodo later.

## Linux server (Paper 2 / OACS)

Mount the dataset once and point the repo at it:

```bash
export CPS_PAPER2_DATA_ROOT=/path/to/data
python scripts/paper2/verify_data.py
```

Full runbook: [`docs/server_runs.md`](server_runs.md).  
Naming (`oacs-r1`, `results/oacs/`) is **not** the Paper 1 `v18` convention — see [`docs/paper2_naming.md`](paper2_naming.md).
