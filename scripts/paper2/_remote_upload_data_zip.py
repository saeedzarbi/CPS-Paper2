#!/usr/bin/env python3
"""Upload repo-root data.zip (RAR) to server and extract into CPS-Paper2/data/."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data.zip"
REMOTE = "/root/CPS-Paper2"
REMOTE_ARCHIVE = f"{REMOTE}/data.zip"


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


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 1800) -> int:
    print(">>>", cmd.split("\n")[0][:140])
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
    if not ARCHIVE.is_file():
        raise SystemExit(f"Missing {ARCHIVE}")

    size = ARCHIVE.stat().st_size
    print(f"Uploading {ARCHIVE} ({size / 1024 / 1024:.1f} MiB) ...")

    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)

    def upload_via_cat() -> None:
        run(client, f"mkdir -p {REMOTE}")
        stdin, stdout, stderr = client.exec_command(f"cat > {REMOTE_ARCHIVE}")
        sent = 0
        with ARCHIVE.open("rb") as fh:
            while True:
                chunk = fh.read(256 * 1024)
                if not chunk:
                    break
                stdin.write(chunk)
                sent += len(chunk)
                if sent % (1024 * 1024) < len(chunk) or sent == size:
                    print(f"  {sent // 1024} KiB / {size // 1024} KiB")
        stdin.channel.shutdown_write()
        code = stdout.channel.recv_exit_status()
        err = stderr.read().decode("utf-8", errors="replace")
        if code != 0:
            raise RuntimeError(f"cat upload failed ({code}): {err}")

    for attempt in range(5):
        try:
            upload_via_cat()
            print("Upload complete.")
            break
        except Exception as exc:
            print(f"Upload attempt {attempt + 1} failed: {exc}")
            client.close()
            time.sleep(3)
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)
    else:
        raise SystemExit("Upload failed after retries")

    run(
        client,
        "export DEBIAN_FRONTEND=noninteractive; "
        "apt-get install -y -qq p7zip-full 2>/dev/null || apt-get install -y -qq unrar-free",
        900,
    )
    run(
        client,
        f"mkdir -p {REMOTE} && cd {REMOTE} && "
        f"(7z x -y data.zip || unrar x -o+ data.zip || tar -xf data.zip)",
        600,
    )
    run(
        client,
        f"test -d {REMOTE}/data/standardized/test_traces_5g_v18 && echo DATA_OK || ls -la {REMOTE}/data 2>/dev/null || ls -la {REMOTE}",
    )
    run(
        client,
        f"cd {REMOTE} && . .venv/bin/activate && export PYTHONPATH={REMOTE} && python scripts/paper2/verify_data.py",
    )
    run(client, f"cd {REMOTE} && bash experiments/paper2/start_oacs_screen.sh")
    time.sleep(5)
    run(client, f"tail -20 {REMOTE}/results/oacs/logs/*.log 2>/dev/null | tail -20")
    client.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
