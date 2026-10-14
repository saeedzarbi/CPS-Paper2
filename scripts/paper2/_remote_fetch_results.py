#!/usr/bin/env python3
"""Download OACS campaign summaries from server and print a compact table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/CPS-Paper2"
CAMPAIGN = "oacs-r1_mm"
REMOTE_DIR = f"{REMOTE}/results/oacs/campaigns/{CAMPAIGN}"
LOCAL_DIR = ROOT / "results" / "oacs" / "campaigns" / CAMPAIGN

JOBS = [
    "greedy_5g",
    "greedy_bb",
    "bba_5g",
    "bba_bb",
    "bola_5g",
    "bola_bb",
    "mpc_5g",
    "mpc_bb",
]


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
    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    for job in JOBS:
        remote = f"{REMOTE_DIR}/{job}/summary.json"
        local_job = LOCAL_DIR / job
        local_job.mkdir(parents=True, exist_ok=True)
        local_path = local_job / "summary.json"
        try:
            sftp.get(remote, str(local_path))
            summaries[job] = json.loads(local_path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"MISSING {remote}: {exc}", file=sys.stderr)

    # episodes.csv optional (larger)
    for job in JOBS:
        r = f"{REMOTE_DIR}/{job}/episodes.csv"
        try:
            sftp.get(r, str(LOCAL_DIR / job / "episodes.csv"))
        except OSError:
            pass

    sftp.close()
    client.close()

    print(f"\nOACS campaign `{CAMPAIGN}` — mean rebuffer (s) and VMAF by arm\n")
    hdr = f"{'job':<12} {'raw':>7} {'static':>7} {'aci_feas':>8} {'aci_oacs':>8} | {'d_oacs-cps':>10}"
    print(hdr)
    print("-" * len(hdr))
    for job in JOBS:
        s = summaries.get(job)
        if not s:
            print(f"{job:<12} (no data)")
            continue
        raw = arm_mean(s, "raw", "rebuffer_total")
        cps = arm_mean(s, "static_cps", "rebuffer_total")
        fea = arm_mean(s, "aci_feasibility", "rebuffer_total")
        oac = arm_mean(s, "aci_oacs", "rebuffer_total")
        delta = oac - cps
        print(f"{job:<12} {raw:7.2f} {cps:7.2f} {fea:8.2f} {oac:8.2f} | {delta:+10.2f}")

    print(f"\nSaved under: {LOCAL_DIR}")
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
