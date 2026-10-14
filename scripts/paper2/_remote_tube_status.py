#!/usr/bin/env python3
"""Check/fix Tube-MPC campaign on server."""
from __future__ import annotations

import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/CPS-Paper2"


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


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> int:
    print(">>>", cmd[:140])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    print(f"[exit {code}]\n")
    return code


def main() -> int:
    fix = "--fix" in sys.argv
    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)

    venv = f"bash -lc 'cd {REMOTE} && source .venv/bin/activate && export PYTHONPATH={REMOTE}"

    run(client, f"ls -la {REMOTE}/.venv/bin/python3 2>&1")
    run(client, f"{venv} && pip show gymnasium | head -5'")
    run(client, "screen -ls")
    run(client, f"ps aux | grep eval_tube_mpc | grep -v grep | head -5")
    run(client, f"ls -la {REMOTE}/results/tube_mpc/campaigns/tube_mpc_r1/ 2>&1 || true")
    run(client, f"ls -t {REMOTE}/results/tube_mpc/logs/*.log 2>/dev/null | head -1 | xargs -r tail -40")

    if fix:
        run(client, f"{venv} && pip install -q -r requirements-smoke.txt'", 600)
        code = run(client, f"{venv} && bash experiments/paper2/run_tube_smoke.sh'", 120)
        if code != 0:
            client.close()
            return code
        run(client, "screen -S tube -X quit 2>/dev/null || true")
        run(
            client,
            f"screen -dmS tube bash -lc 'cd {REMOTE} && source .venv/bin/activate && "
            f"export PYTHONPATH={REMOTE} PYTHONIOENCODING=utf-8 TUBE_CAMPAIGN=tube_mpc_r1 && "
            f"LOG=results/tube_mpc/logs/campaign_$(date +%Y%m%d_%H%M%S).log && "
            f"bash experiments/paper2/run_tube_campaign.sh 2>&1 | tee -a $LOG'",
        )
        import time

        time.sleep(5)
        run(client, "screen -ls")
        run(client, f"ps aux | grep eval_tube_mpc | grep -v grep | head -3")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
