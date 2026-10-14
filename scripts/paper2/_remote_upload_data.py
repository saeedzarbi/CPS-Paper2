#!/usr/bin/env python3
"""Upload minimal Paper 2 dataset to server (test traces + VMAF)."""
from __future__ import annotations

import sys
import tarfile
import tempfile
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/CPS-Paper2"

# Paths relative to repo root (under data/)
DATA_SUBDIRS = [
    "data/standardized/test_traces_5g_v18",
    "data/standardized/test_traces_v18",
    "data/vmaf_scores",
    "data/content_features",
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


def build_data_tarball() -> Path:
    tmp = Path(tempfile.gettempdir()) / "cps_paper2_data.tgz"
    missing = []
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in DATA_SUBDIRS:
            local = ROOT / rel
            if not local.exists():
                missing.append(rel)
                continue
            for f in local.rglob("*"):
                if f.is_file():
                    arc = f.relative_to(ROOT).as_posix()
                    tar.add(f, arcname=arc)
    if missing:
        print("WARNING: missing local paths:", ", ".join(missing), file=sys.stderr)
    if not tmp.exists() or tmp.stat().st_size < 1000:
        raise SystemExit("No data files packed — check local data/ folder.")
    return tmp


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 3600) -> int:
    print(">>>", cmd.split("\n")[0][:120])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out)
    if err and code != 0:
        print(err, file=sys.stderr)
    return code


def main() -> int:
    for rel in DATA_SUBDIRS:
        if not (ROOT / rel).exists():
            print(f"Local missing: {ROOT / rel}", file=sys.stderr)

    tgz = build_data_tarball()
    mib = tgz.stat().st_size / (1024 * 1024)
    print(f"Packaged {tgz} ({mib:.1f} MiB)")

    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)

    remote_tgz = "/root/cps_paper2_data.tgz"
    for attempt in range(3):
        try:
            print(f"Upload attempt {attempt + 1}...")
            sftp = client.open_sftp()
            sftp.put(str(tgz), remote_tgz, callback=lambda x, y: None)
            sftp.close()
            break
        except Exception as exc:
            print(f"Upload failed: {exc}")
            client.close()
            time.sleep(2)
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)
    else:
        raise SystemExit("Upload failed after retries")

    run(
        client,
        f"mkdir -p {REMOTE} && tar -xzf {remote_tgz} -C {REMOTE} && rm -f {remote_tgz}",
        timeout=600,
    )
    run(client, f"cd {REMOTE} && . .venv/bin/activate && export PYTHONPATH={REMOTE} && python scripts/paper2/verify_data.py")
    run(client, f"cd {REMOTE} && bash experiments/paper2/start_oacs_screen.sh")
    time.sleep(4)
    run(client, f"tail -25 {REMOTE}/results/oacs/logs/*.log 2>/dev/null | tail -25")
    client.close()
    tgz.unlink(missing_ok=True)
    print("\nDone. Server data at", f"{REMOTE}/data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
