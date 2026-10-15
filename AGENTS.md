# Agents — CPS-Paper2

This repo uses a **two-agent** workflow documented in [`docs/agents/README.md`](docs/agents/README.md).

| Agent | Prompt / rule | Suggested model (budget) |
|-------|----------------|---------------------------|
| Academic Lead | [`docs/agents/academic_lead_system_prompt.md`](docs/agents/academic_lead_system_prompt.md) | Composer 2.5 / Sonnet / GPT-5 mini |
| Execution Arm | [`.cursor/rules/cps-paper2-execution-arm.mdc`](.cursor/rules/cps-paper2-execution-arm.mdc) | Composer 2.5 Fast (Agent) |

Handoff templates: `docs/agents/execution_brief_template.md`, `docs/agents/execution_report_template.md`.

**Current scientific direction:** clean-slate wall-clock ABR — `docs/clean_slate_abr.md`, headline eval **`clean_v3`** (`results/clean_v3/`, report `docs/reports/clean_v3_full_metrics.md`). Server ops: `docs/server_runs.md`.

Legacy specs (archived): `docs/decision_rule.md` (OACS/VMAF), `docs/decision_rule_tube_mpc.md` (pre–clean-slate Tube-MPC).
