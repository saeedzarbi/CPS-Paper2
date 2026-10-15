#!/usr/bin/env python3
"""Build docs/reports/clean_v3_full_metrics.md from results/clean_v3 summaries."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "clean_v3"
OUT = ROOT / "docs" / "reports" / "clean_v3_full_metrics.md"

POLICIES = ("mpc", "greedy", "bba", "bola")
TRACES = ("5g_markov", "bb")
ARMS = ("raw", "static_conformal", "tube_conformal")
METRICS = [
    ("rebuffer_total", "Mean rebuffer (s/ep)"),
    ("rebuffer_p95_chunk", "p95 chunk rebuffer (s)"),
    ("smooth_penalty_mean", "Mean smoothness (|Δlog u|)"),
    ("log_utility_mean", "Mean log-bitrate utility"),
    ("episodic_qoe", "Mean episodic QoE"),
]


def load(job: str) -> dict | None:
    p = RESULTS / job / "summary.json"
    if not p.is_file() or p.stat().st_size < 2:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def mean_metric(summary: dict, arm: str, key: str) -> float:
    try:
        return float(summary["arms"][arm][key]["mean"])
    except (KeyError, TypeError):
        return float("nan")


def p_wilcoxon(summary: dict, key: str, treat: str, ref: str) -> float:
    try:
        return float(summary["comparisons"][f"{treat}_vs_{ref}"][key]["wilcoxon_p"])
    except (KeyError, TypeError):
        return float("nan")


def table_block(title: str, metric_key: str, label: str) -> list[str]:
    lines = [f"### {title}", "", f"*{label}* — **bold** = best (rebuffer: lowest; utility/QoE: highest).", ""]
    for tr in TRACES:
        lines.append(f"#### {tr}")
        lines.append("")
        hdr = "| Policy | raw | static | tube | p(tube vs static) | p(tube vs raw) |"
        lines.append(hdr)
        lines.append("|" + "---|" * 6)
        lower_better = "rebuffer" in metric_key
        for pol in POLICIES:
            s = load(f"{pol}_{tr}")
            if not s:
                lines.append(f"| {pol} | — | — | — | — | — |")
                continue
            vals = {a: mean_metric(s, a, metric_key) for a in ARMS}
            if lower_better:
                best = min(vals, key=lambda a: vals[a])
            else:
                best = max(vals, key=lambda a: vals[a])
            cells = []
            for a in ARMS:
                v = vals[a]
                txt = f"{v:.2f}" if v == v else "n/a"
                if a == best and v == v:
                    txt = f"**{txt}**"
                cells.append(txt)
            pt_s = p_wilcoxon(s, metric_key, "tube_conformal", "static_conformal")
            pt_r = p_wilcoxon(s, metric_key, "tube_conformal", "raw")
            ps = f"{pt_s:.2e}" if pt_s == pt_s else "—"
            pr = f"{pt_r:.2e}" if pt_r == pt_r else "—"
            lines.append(f"| {pol} | {' | '.join(cells)} | {ps} | {pr} |")
        lines.append("")
    return lines


def main() -> int:
    lines = [
        "# Clean ABR `clean_v3` — full metrics",
        "",
        "Headline arms: `raw`, `static_conformal`, `tube_conformal` (asymmetric multi-horizon tube + static cap).",
        "204 paired episodes; traces: `test_traces_5g_markov`, `test_traces_v18`.",
        "",
        f"Source: `{RESULTS.relative_to(ROOT).as_posix()}/**/summary.json`",
        "",
    ]
    for key, label in METRICS:
        lines.extend(table_block(label, key, label))
    lines.append("## Rebuffer win count (tube vs static & raw)")
    lines.append("")
    wins_s = wins_r = 0
    for tr in TRACES:
        for pol in POLICIES:
            s = load(f"{pol}_{tr}")
            if not s:
                continue
            rt = mean_metric(s, "tube_conformal", "rebuffer_total")
            rs = mean_metric(s, "static_conformal", "rebuffer_total")
            rr = mean_metric(s, "raw", "rebuffer_total")
            if rt <= rs:
                wins_s += 1
            if rt <= rr:
                wins_r += 1
    lines.append(f"- Tube ≤ static on rebuffer: **{wins_s}/8** benchmarks")
    lines.append(f"- Tube ≤ raw on rebuffer: **{wins_r}/8** benchmarks")
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
