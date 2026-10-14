#!/usr/bin/env python3
"""One-shot server bootstrap: sync repo, venv, start screen job. Reads ./server.txt at repo root."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE_DIR = "/root/CPS-Paper2"
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".cursor",
    "data",
    "results",
    "paper_v1",
    "node_modules",
}
SKIP_FILES = {"server.txt", ".env"}


def parse_server_txt(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8")
    host = user = password = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip().lower(), val.strip()
        if key == "ip":
            host = val
        elif key == "user":
            user = val
        elif key == "pass":
            password = val
    if not (host and user and password):
        raise SystemExit("server.txt must contain ip, user, pass lines")
    return host, user, password


def local_files_to_sync() -> list[Path]:
    out: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.name in SKIP_FILES:
            continue
        if rel.suffix in {".pyc", ".pyo"}:
            continue
        out.append(p)
    return out


def build_tarball() -> Path:
    import tarfile
    import tempfile

    tmp = Path(tempfile.gettempdir()) / "cps_paper2_deploy.tgz"
    with tarfile.open(tmp, "w:gz") as tar:
        for local in local_files_to_sync():
            rel = local.relative_to(ROOT).as_posix()
            tar.add(local, arcname=f"CPS-Paper2/{rel}")
    return tmp


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    print(f"\n>>> {cmd[:200]}{'...' if len(cmd) > 200 else ''}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out.rstrip())
    if err:
        print(err.rstrip(), file=sys.stderr)
    return code, out, err


def main() -> int:
    cred_path = ROOT / "server.txt"
    if not cred_path.exists():
        print("Missing server.txt at repo root", file=sys.stderr)
        return 1
    host, user, password = parse_server_txt(cred_path)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {user}@{host} ...")
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)

    run(
        client,
        "export DEBIAN_FRONTEND=noninteractive; "
        "command -v git >/dev/null || apt-get update -qq && apt-get install -y -qq git python3 python3-venv python3-pip screen",
        timeout=900,
    )

    run(
        client,
        f"mkdir -p {REMOTE_DIR} && "
        f"if [ ! -d {REMOTE_DIR}/.git ]; then git clone https://github.com/saeedzarbi/CPS-Paper2.git {REMOTE_DIR}; "
        f"else cd {REMOTE_DIR} && git pull --ff-only origin main || true; fi",
        timeout=300,
    )

    tgz = build_tarball()
    print(f"Uploading {tgz} ({tgz.stat().st_size // 1024} KiB) ...")
    remote_tgz = "/root/cps_paper2_deploy.tgz"
    for attempt in range(3):
        try:
            sftp = client.open_sftp()
            sftp.put(str(tgz), remote_tgz)
            sftp.close()
            break
        except Exception as exc:
            print(f"SFTP attempt {attempt + 1} failed: {exc}")
            client.close()
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)
    run(
        client,
        f"mkdir -p {REMOTE_DIR} && tar -xzf {remote_tgz} -C /root --strip-components=0 && rm -f {remote_tgz}",
        timeout=120,
    )

    setup = f"""
set -e
cd {REMOTE_DIR}
export PYTHONPATH={REMOTE_DIR}
export PYTHONIOENCODING=utf-8
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
mkdir -p results/oacs/logs results/oacs/campaigns
python scripts/paper2/verify_data.py || true
"""
    code, _, _ = run(client, setup, timeout=1200)
    if code != 0:
        print("Setup had errors (data may be missing); continuing to screen job.")

    screen_cmd = (
        f"screen -dmS oacs bash -lc '"
        f"cd {REMOTE_DIR} && source .venv/bin/activate && "
        f"export PYTHONPATH={REMOTE_DIR} PYTHONIOENCODING=utf-8 && "
        f"LOG=results/oacs/logs/run_$(date +%Y%m%d_%H%M%S).log && "
        f"echo OACS screen started at $(date) | tee -a $LOG && "
        f"bash experiments/paper2/run_smoke.sh 2>&1 | tee -a $LOG; "
        f"bash experiments/paper2/run_campaign.sh 2>&1 | tee -a $LOG; "
        f"echo DONE at $(date) | tee -a $LOG'"
    )
    run(client, screen_cmd)
    run(client, "screen -ls")
    run(client, f"tail -n 30 {REMOTE_DIR}/results/oacs/logs/*.log 2>/dev/null || echo '(log not yet written)'")

    client.close()
    print("\nServer: attach with  ssh root@HOST  then  screen -r oacs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
