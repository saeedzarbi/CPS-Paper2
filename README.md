# CPS Paper 2

Independent repository for the second paper on conformal ABR shielding.

This repo is **not** a fork of [ABR-Awareness](https://github.com/saeedzarbi/ABR-Awareness).
Paper 1 stays in that project. Here we only keep experiments, analysis, and manuscript
assets for Paper 2.

## Candidate directions (decide after the two-day pilot)

1. **Buffer-conditioned risk** — \(\alpha_k = f(B_k)\), joint with \(\varepsilon\), vs best fixed \((\varepsilon,\alpha)\).
2. **Cold-start / calibration recipe** — stall-budget accounting, warm-start, weighted window, ACI; fair \(\rho_{\mathrm{fb}}\) sweep.

See `docs/decision_rule.md` for the go/no-go criteria.

## Layout

```
src/           # calibrators, stall accounting, evaluation harness
experiments/   # launch scripts and configs
results/       # summaries and tables (tracked selectively)
paper/         # manuscript / Overleaf export
docs/          # design notes and decision rules
```

## Relation to Paper 1

Reuse ideas and, if needed, **copy only** the minimal shield + simulator pieces under a clear
license note. Do not keep a live dependency on the Paper 1 working tree.
