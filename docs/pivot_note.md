# Pivot note: OACS → Tube-MPC (Paper 2)

## Decision

- **Set aside:** Online ACI + per-chunk VMAF-knee + MaskablePPO perceptual shield (spec: `docs/decision_rule.md`).
- **Adopt:** Multi-Horizon Conformal **Tube-MPC** for safe ABR, log-utility, no VMAF in the loop (spec: `docs/decision_rule_tube_mpc.md`).

Rationale summary: lighter data stack, stronger fit for networking/control venues, and `oacs-r1_mm` did not outperform CPS v18 on 5G rebuffer.

## What stays valuable from `oacs-r1` work

| Asset | Use in Tube-MPC |
|-------|-----------------|
| Server deploy, screen, Mattermost | Same ops |
| `OnlineACIEstimator` ideas | Tube width / \(\alpha_k\) |
| `eval_*` paired-stats pattern | New evaluator |
| `oacs-r1_mm` results | Negative result / motivation appendix |

## Agent prompts

Point Academic Lead at **`docs/decision_rule_tube_mpc.md`**, not the old OACS mandate.
