#!/usr/bin/env python3
"""Kill stale screen and start Tube-MPC campaign."""
from __future__ import annotations

import sys
import time
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


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[int, str]:
    print(">>>", cmd[:200])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    combined = (out + err).strip()
    if combined:
        print(combined)
    print(f"[exit {code}]\n")
    return code, combined


def main() -> int:
    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)

    run(client, "screen -ls")
    run(client, "screen -S tun -X quit 2>/dev/null; screen -S tube -X quit 2>/dev/null; true")

    starter = f"""cat > {REMOTE}/experiments/paper2/_tube_campaign_runner.sh << 'EOF'
#!/usr/bin/env bash
set -eu
cd {REMOTE}
source .venv/bin/activate
export PYTHONPATH={REMOTE}
export PYTHONIOENCODING=utf-8
export TUBE_CAMPAIGN=tube_mpc_r1
mkdir -p results/tube_mpc/logs
LOG=results/tube_mpc/logs/campaign_$(date +%Y%m%d_%H%M%S).log
exec bash experiments/paper2/run_tube_campaign.sh 2>&1 | tee -a "$LOG"
EOF"""
    run(client, starter)
    run(client, f"chmod +x {REMOTE}/experiments/paper2/_tube_campaign_runner.sh")
    run(client, f"sed -i 's/\\r$//' {REMOTE}/experiments/paper2/_tube_campaign_runner.sh")

    code, _ = run(client, f"screen -dmS tubempc {REMOTE}/experiments/paper2/_tube_campaign_runner.sh")
    if code != 0:
        client.close()
        return code

    time.sleep(6)
    run(client, "screen -ls")
    run(client, "ps aux | grep -E 'eval_tube_mpc|run_tube_campaign' | grep -v grep")
    run(client, f"ls -t {REMOTE}/results/tube_mpc/logs/*.log 2>/dev/null | head -1 | xargs -r tail -15")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
