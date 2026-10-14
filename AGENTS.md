# Agents — CPS-Paper2

This repo uses a **two-agent** workflow documented in [`docs/agents/README.md`](docs/agents/README.md).

| Agent | Prompt / rule | Suggested model (budget) |
|-------|----------------|---------------------------|
| Academic Lead | [`docs/agents/academic_lead_system_prompt.md`](docs/agents/academic_lead_system_prompt.md) | Composer 2.5 / Sonnet / GPT-5 mini |
| Execution Arm | [`.cursor/rules/cps-paper2-execution-arm.mdc`](.cursor/rules/cps-paper2-execution-arm.mdc) | Composer 2.5 Fast (Agent) |

Handoff templates: `docs/agents/execution_brief_template.md`, `docs/agents/execution_report_template.md`.

Paper 2 uses **OACS** release naming (`oacs-r1`, `results/oacs/`) — not `v20`. See `docs/paper2_naming.md` and `docs/server_runs.md`.

**Current scientific direction (2026-10):** Tube-MPC without VMAF — `docs/decision_rule_tube_mpc.md`. Legacy OACS spec: `docs/decision_rule.md` (archived).
