#!/usr/bin/env python3
"""Download Tube-MPC campaign summaries from server and print a compact table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/CPS-Paper2"
CAMPAIGN_DEFAULT = "tube_mpc_r1"
TRACES_DEFAULT = ("5g", "bb")
CAMPAIGN_IRISH = "tube_mpc_r1_irish5g"
TRACES_IRISH = ("5g_irish",)
ARMS = ("raw_mpc", "tube_static", "tube_aci")


def creds() -> tuple[str, str, str]:
    text = (ROOT / "server.txt").read_text(encoding="utf-8")
    host = user = password = ""
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip().lower(), v.strip()
        if k == "ip":
            host = v
        elif k == "user":
            user = v
        elif k == "pass":
            password = v
    return host, user, password


def arm_mean(summary: dict, arm: str, key: str) -> float:
    try:
        return float(summary["arms"][arm][key]["mean"])
    except (KeyError, TypeError):
        return float("nan")


def main() -> int:
    campaign = CAMPAIGN_DEFAULT
    traces = TRACES_DEFAULT
    if "--irish" in sys.argv:
        campaign = CAMPAIGN_IRISH
        traces = TRACES_IRISH
    remote_base = f"{REMOTE}/results/tube_mpc/campaigns/{campaign}"
    local_base = ROOT / "results" / "tube_mpc" / "campaigns" / campaign

    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()

    summaries: dict[str, dict] = {}
    for trace in traces:
        local_base.joinpath(trace).mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "episodes.csv"):
            remote = f"{remote_base}/{trace}/{name}"
            local = local_base / trace / name
            try:
                sftp.get(remote, str(local))
            except OSError as exc:
                if name == "summary.json":
                    print(f"MISSING {remote}: {exc}", file=sys.stderr)
        sp = local_base / trace / "summary.json"
        if sp.is_file():
            summaries[trace] = json.loads(sp.read_text(encoding="utf-8"))

    sftp.close()
    client.close()

    print(f"\nTube-MPC campaign `{campaign}` — mean rebuffer (s) and log utility\n")
    for trace in traces:
        s = summaries.get(trace)
        if not s:
            print(f"=== {trace}: no data ===")
            continue
        print(f"=== {trace} ===")
        hdr = f"{'arm':<14} {'rebuffer':>9} {'log_util':>9} {'tube_cov':>9}"
        print(hdr)
        print("-" * len(hdr))
        for arm in ARMS:
            reb = arm_mean(s, arm, "rebuffer_total")
            lu = arm_mean(s, arm, "log_utility_mean")
            cov = s.get("arms", {}).get(arm, {}).get("tube_coverage_mean", float("nan"))
            print(f"{arm:<14} {reb:9.2f} {lu:9.4f} {cov:9.3f}")
        comp = s.get("comparisons", {})
        if comp:
            print("  comparisons:")
            for k, v in comp.items():
                print(
                    f"    {k}: d_reb={v.get('rebuffer_diff_s', float('nan')):+.2f}s "
                    f"(p={v.get('rebuffer_wilcoxon_p', float('nan')):.4g}), "
                    f"d_logU={v.get('log_utility_diff', float('nan')):+.4f} "
                    f"(p={v.get('log_utility_wilcoxon_p', float('nan')):.4g})"
                )
        print()

    print(f"Saved under: {local_base}")
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
