#!/usr/bin/env python3
"""Pull results/clean_v3/*/summary.json from server (reads server.txt)."""
from __future__ import annotations

from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/CPS-Paper2"
JOBS = (
    "mpc_5g_markov",
    "mpc_bb",
    "greedy_5g_markov",
    "greedy_bb",
    "bba_5g_markov",
    "bba_bb",
    "bola_5g_markov",
    "bola_bb",
)


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
    local_base = ROOT / "results" / "clean_v3"
    local_base.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()

    ok = 0
    for job in JOBS:
        remote = f"{REMOTE}/results/clean_v3/{job}/summary.json"
        dest = local_base / job / "summary.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            sftp.get(remote, str(dest))
            ok += 1
            print(f"OK {job}")
        except OSError as e:
            print(f"SKIP {job}: {e}")

    sftp.close()
    client.close()
    print(f"Fetched {ok}/{len(JOBS)} summaries -> {local_base}")
    return 0 if ok == len(JOBS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
