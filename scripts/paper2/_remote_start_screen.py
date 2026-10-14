#!/usr/bin/env python3
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


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> int:
    print(">>>", cmd.split("\n")[0][:100])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    return code


def main() -> int:
    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)

    upload = [
        "requirements-smoke.txt",
        "experiments/paper2/start_oacs_screen.sh",
        "experiments/paper2/run_smoke.sh",
        "experiments/paper2/run_campaign.sh",
    ]
    sftp = client.open_sftp()
    for rel in upload:
        data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        remote = f"{REMOTE}/{rel}"
        with sftp.open(remote, "wb") as rf:
            rf.write(data)
    sftp.close()

    run(
        client,
        f"find {REMOTE}/experiments/paper2 -name '*.sh' -exec sed -i 's/\\r$//' {{}} +",
    )
    run(client, f"chmod +x {REMOTE}/experiments/paper2/*.sh")
    run(
        client,
        f"cd {REMOTE} && . .venv/bin/activate && pip install -q -r requirements-smoke.txt",
        600,
    )
    run(client, f"cd {REMOTE} && bash experiments/paper2/start_oacs_screen.sh")
    time.sleep(6)
    run(client, "screen -ls")
    run(client, f"ls -la {REMOTE}/results/oacs/logs/; tail -40 {REMOTE}/results/oacs/logs/*.log 2>/dev/null || true")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
