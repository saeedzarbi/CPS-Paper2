# CPS Paper 2

Independent repository for the second paper on conformal ABR shielding.

This is **not** a fork of [ABR-Awareness](https://github.com/saeedzarbi/ABR-Awareness).
A minimal CPS eval slice was copied here so Paper 2 experiments can run without
editing the Paper 1 tree. See `NOTICE.md`.

## Candidate directions

1. **Buffer-conditioned risk** — \(\alpha_k = f(B_k)\), joint with \(\varepsilon\).
2. **Cold-start / calibration recipe** — stall-budget accounting + fair \(\rho_{\mathrm{fb}}\) sweep.

Go/no-go criteria: `docs/decision_rule.md`.

## Layout

```
configs/       # paths + video roster
src/
  training/    # Certified Perceptual Shield
  environment/ # ABREnv chain through v18
  evaluation/  # paired arms, stall breakdown, eps/alpha grid, drift
experiments/   # Paper 2 launch scripts (to add)
results/       # run outputs
paper/         # manuscript assets
docs/          # design notes
data/          # local junction / copy of traces+VMAF (not in git)
```

## Quick start

```bash
cd CPS-Paper2
python -m pip install -r requirements.txt

# smoke: greedy certified arms on synthetic 5G (needs data/)
python src/evaluation/eval_certified_shield_v18.py ^
  --policy greedy ^
  --trace-dir data/standardized/test_traces_5g_v18 ^
  --episodes 8 --epsilon 1.0 --alpha 0.10 ^
  --arms raw,safety,certified ^
  --out results/smoke/greedy_5g
```

Stall accounting (Paper 2 Direction B):

```bash
python src/evaluation/stall_cause_breakdown_v18.py --help
```

## Data

See `docs/data_setup.md`. On this machine `data/` is a junction to Paper 1’s `new/data`.
