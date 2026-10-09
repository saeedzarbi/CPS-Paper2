# Data setup

This repo expects a `data/` directory at the project root with the same layout as
Paper 1 (`new/data`), including:

- `data/standardized/test_traces_5g_v18/`
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
