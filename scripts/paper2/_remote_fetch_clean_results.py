#!/usr/bin/env python3
"""Download clean_v2 campaign summaries from server and print tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/CPS-Paper2"
CAMPAIGN = "clean_v2"
REMOTE_BASE = f"{REMOTE}/results/{CAMPAIGN}"
LOCAL_BASE = ROOT / "results" / CAMPAIGN

POLICIES = ("mpc", "greedy", "bba", "bola")
TRACES = ("5g_markov", "bb")
ARMS = ("raw", "static_conformal", "tube_conformal", "two_timescale_conformal")


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


def main() -> int:
    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()

    fetched = 0
    for pol in POLICIES:
        for tr in TRACES:
            job = f"{pol}_{tr}"
            local_job = LOCAL_BASE / job
            local_job.mkdir(parents=True, exist_ok=True)
            for name in ("summary.json", "episodes.csv"):
                remote = f"{REMOTE_BASE}/{job}/{name}"
                try:
                    sftp.get(remote, str(local_job / name))
                    if name == "summary.json":
                        fetched += 1
                except OSError as exc:
                    print(f"MISSING {remote}: {exc}", file=sys.stderr)

    sftp.close()
    client.close()

    print(f"\nClean ABR `{CAMPAIGN}` — mean rebuffer (s/ep) by arm\n")
    for tr in TRACES:
        print(f"=== {tr} ===")
        hdr = f"{'policy':<8} " + " ".join(f"{a[:6]:>8}" for a in ARMS)
        print(hdr)
        print("-" * len(hdr))
        for pol in POLICIES:
            sp = LOCAL_BASE / f"{pol}_{tr}" / "summary.json"
            if not sp.is_file():
                print(f"{pol:<8} (no summary)")
                continue
            s = json.loads(sp.read_text(encoding="utf-8"))
            arms = s.get("arms", {})
            vals = []
            for arm in ARMS:
                try:
                    vals.append(f"{arms[arm]['rebuffer_total']['mean']:8.2f}")
                except (KeyError, TypeError):
                    vals.append(f"{'n/a':>8}")
            print(f"{pol:<8} " + " ".join(vals))
        print()

    print(f"Saved under: {LOCAL_BASE} ({fetched} summaries)")
    return 0 if fetched else 1


if __name__ == "__main__":
    raise SystemExit(main())
