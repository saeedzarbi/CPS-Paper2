#!/usr/bin/env python3
"""Sync clean-slate ABR stack and run full campaign in screen `clean_v2`."""
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
    "configs/paths.py",
    "configs/videos.py",
    "data/prepare_markov_5g_traces.py",
    "experiments/run_clean_campaign.py",
    "experiments/paper2/run_clean_campaign.sh",
    "experiments/paper2/run_clean_smoke.sh",
    "src/environment/clean_abr_env.py",
    "src/environment/paper2_abr_env.py",
    "src/control/two_timescale_conformal.py",
    "src/control/__init__.py",
    "src/evaluation/eval_certified_shield_v18.py",
    "src/training/certified_perceptual_shield.py",
    "src/training/aci_perceptual_shield.py",
    "src/training/conformal_tube.py",
    "src/training/tube_mpc_policy.py",
    "scripts/paper2/mattermost_notify.py",
    "src/utils/mattermost_notify.py",
    "mattermost_webhook.url",
    "requirements-smoke.txt",
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
        print(out[-8000:] if len(out) > 8000 else out)
    if err and code != 0:
        print(err[-2000:], file=sys.stderr)
    return code


def main() -> int:
    tmp = Path(tempfile.gettempdir()) / "clean_v2_sync.tgz"
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in SYNC_REL:
            p = ROOT / rel
            if p.is_file():
                tar.add(p, arcname=f"CPS-Paper2/{rel}")
    missing = [r for r in SYNC_REL if not (ROOT / r).is_file()]
    if missing:
        print("WARNING missing local files:", missing, file=sys.stderr)

    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()
    sftp.put(str(tmp), "/root/clean_v2_sync.tgz")
    sftp.close()
    tmp.unlink(missing_ok=True)

    run(client, f"cd {REMOTE} && tar -xzf /root/clean_v2_sync.tgz -C /root && rm -f /root/clean_v2_sync.tgz")
    run(
        client,
        f"find {REMOTE}/experiments/paper2 -name '*.sh' -exec sed -i 's/\\r$//' {{}} + "
        f"&& chmod +x {REMOTE}/experiments/paper2/run_clean_campaign.sh "
        f"{REMOTE}/experiments/paper2/run_clean_smoke.sh",
    )

    venv = f"bash -lc 'cd {REMOTE} && source .venv/bin/activate && export PYTHONPATH={REMOTE}"
    run(client, f"{venv} && pip install -q -r requirements-smoke.txt'", 600)
    run(client, f"{venv} && test -d data/standardized/test_traces_v18 && echo BB_OK || echo BB_MISSING'")
    if run(client, f"{venv} && bash experiments/paper2/run_clean_smoke.sh'", 180) != 0:
        client.close()
        return 1

    run(client, "screen -S clean_v2 -X quit 2>/dev/null || true")
    run(
        client,
        f"screen -dmS clean_v2 bash -lc 'cd {REMOTE} && source .venv/bin/activate && "
        f"export PYTHONPATH={REMOTE} PYTHONIOENCODING=utf-8 CLEAN_CAMPAIGN=clean_v2 && "
        f"bash experiments/paper2/run_clean_campaign.sh 2>&1 | tee -a results/clean_v2/logs/campaign_$(date +%Y%m%d_%H%M%S).log'",
    )
    time.sleep(6)
    run(client, "screen -ls")
    run(client, "ps aux | grep run_clean_campaign | grep -v grep | head -5")
    client.close()
    print("\nClean ABR campaign in screen `clean_v2`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
