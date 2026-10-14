#!/usr/bin/env python3
"""Sync OACS eval to server and start long campaign in screen `oacs`."""
from __future__ import annotations

import sys
import tarfile
import tempfile
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/CPS-Paper2"

SYNC_REL = [
    "src/evaluation/eval_oacs_campaign.py",
    "src/training/aci_perceptual_shield.py",
    "src/utils/mattermost_notify.py",
    "scripts/paper2/mattermost_notify.py",
    "experiments/paper2/run_campaign.sh",
    "experiments/paper2/run_smoke.sh",
    "experiments/paper2/start_oacs_screen.sh",
    "configs/paper2_release.py",
    "mattermost_webhook.url",
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


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 86400) -> int:
    print(">>>", cmd.split("\n")[0][:120])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out[-4000:] if len(out) > 4000 else out)
    if err and code != 0:
        print(err[-2000:], file=sys.stderr)
    return code


def main() -> int:
    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)

    tmp = Path(tempfile.gettempdir()) / "oacs_sync.tgz"
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in SYNC_REL:
            p = ROOT / rel
            if p.is_file():
                tar.add(p, arcname=f"CPS-Paper2/{rel}")
    sftp = client.open_sftp()
    sftp.put(str(tmp), "/root/oacs_sync.tgz")
    sftp.close()
    tmp.unlink(missing_ok=True)
    run(client, f"cd {REMOTE} && tar -xzf /root/oacs_sync.tgz -C /root && rm -f /root/oacs_sync.tgz")
    run(client, f"find {REMOTE}/experiments/paper2 -name '*.sh' -exec sed -i 's/\\r$//' {{}} + && chmod +x {REMOTE}/experiments/paper2/*.sh")

    run(
        client,
        f"cd {REMOTE} && . .venv/bin/activate && pip install -q -r requirements-smoke.txt",
        600,
    )
    run(
        client,
        f"cd {REMOTE} && . .venv/bin/activate && export PYTHONPATH={REMOTE} && "
        f"python3 scripts/paper2/mattermost_notify.py ':rocket: OACS campaign start'",
        60,
    )
    run(client, f"screen -S oacs -X quit 2>/dev/null || true")
    run(
        client,
        f"screen -dmS oacs bash -lc 'cd {REMOTE} && source .venv/bin/activate && "
        f"export PYTHONPATH={REMOTE} PYTHONIOENCODING=utf-8 OACS_CAMPAIGN=oacs-r1_mm && "
        f"LOG=results/oacs/logs/campaign_$(date +%Y%m%d_%H%M%S).log && "
        f"bash experiments/paper2/run_campaign.sh 2>&1 | tee -a $LOG'",
    )
    time.sleep(3)
    run(client, "screen -ls")
    run(client, f"ps aux | grep -E 'eval_oacs|run_campaign' | grep -v grep | head -5")
    run(client, f"tail -15 {REMOTE}/results/oacs/logs/campaign_*.log 2>/dev/null | tail -15")
    client.close()
    print("\nCampaign running in screen `oacs`. Attach: ssh root@HOST -> screen -r oacs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
