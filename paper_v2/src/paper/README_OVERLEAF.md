# Paper 2 (Tube-MPC) — Overleaf upload

## What to upload

Zip and upload the contents of **`paper_v2/src/paper/`** (this folder):

- `main.ltx` — root file (set as Main document in Overleaf)
- `sections/*.tex`
- `tables/macros_tube_v3.tex`
- `figures/*.png`
- `references.bib`

## Compiler

- **Class:** `elsarticle` (included on Overleaf; or upload from CTAN)
- **Engine:** pdfLaTeX
- **Bibliography:** BibTeX with `elsarticle-num.bst`

Build order: pdfLaTeX → BibTeX → pdfLaTeX × 2.

## Syncing numbers

Update `tables/macros_tube_v3.tex` from `docs/reports/tube_v3_adaptive_tables_envchunks.md` when test results change.

## Figures

Copied from `docs/figures/tube_v3_adaptive_envchunks/`. Re-copy after regenerating analysis on the server.

## Relation to Paper 1

Paper 1 manuscript: `paper_v1/src/paper/main.ltx` (CPS / one-step shield). This draft is **Paper 2** only.
