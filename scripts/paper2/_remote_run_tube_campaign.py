#!/usr/bin/env python3
"""Sync Tube-MPC code and run campaign in screen `tube`."""
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
    "src/training/conformal_tube.py",
    "src/training/tube_mpc_policy.py",
    "src/training/aci_perceptual_shield.py",
    "src/training/certified_perceptual_shield.py",
    "src/evaluation/eval_tube_mpc_campaign.py",
    "src/evaluation/eval_certified_shield_v18.py",
    "experiments/paper2/run_tube_smoke.sh",
    "experiments/paper2/run_tube_campaign.sh",
    "src/environment/paper2_abr_env.py",
    "scripts/paper2/mattermost_notify.py",
    "src/utils/mattermost_notify.py",
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
        print(out[-5000:] if len(out) > 5000 else out)
    if err and code != 0:
        print(err[-2000:], file=sys.stderr)
    return code


def main() -> int:
    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)

    tmp = Path(tempfile.gettempdir()) / "tube_sync.tgz"
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in SYNC_REL:
            p = ROOT / rel
            if p.is_file():
                tar.add(p, arcname=f"CPS-Paper2/{rel}")
    sftp = client.open_sftp()
    sftp.put(str(tmp), "/root/tube_sync.tgz")
    sftp.close()
    tmp.unlink(missing_ok=True)

    run(client, f"cd {REMOTE} && tar -xzf /root/tube_sync.tgz -C /root && rm -f /root/tube_sync.tgz")
    run(client, f"find {REMOTE}/experiments/paper2 -name '*.sh' -exec sed -i 's/\\r$//' {{}} + && chmod +x {REMOTE}/experiments/paper2/*.sh")
    venv = f"bash -lc 'cd {REMOTE} && source .venv/bin/activate && export PYTHONPATH={REMOTE}"
    run(client, f"{venv} && pip install -q -r requirements-smoke.txt'", 600)
    if run(client, f"{venv} && bash experiments/paper2/run_tube_smoke.sh'", 120) != 0:
        print("Smoke failed; not starting campaign.", file=sys.stderr)
        client.close()
        return 1

    run(client, "screen -S tube -X quit 2>/dev/null || true")
    run(
        client,
        f"screen -dmS tube bash -lc 'cd {REMOTE} && source .venv/bin/activate && "
        f"export PYTHONPATH={REMOTE} PYTHONIOENCODING=utf-8 TUBE_CAMPAIGN=tube_mpc_r1 && "
        f"LOG=results/tube_mpc/logs/campaign_$(date +%Y%m%d_%H%M%S).log && "
        f"bash experiments/paper2/run_tube_campaign.sh 2>&1 | tee -a $LOG'",
    )
    time.sleep(4)
    run(client, "screen -ls")
    run(client, f"ps aux | grep eval_tube_mpc | grep -v grep | head -3")
    client.close()
    print("\nTube campaign in screen `tube`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
