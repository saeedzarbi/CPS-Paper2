"""Upload listed repo-relative paths to /root/CPS-Paper2 via SFTP (reads server.txt)."""
from __future__ import annotations

import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
SERVER_FILE = ROOT / "server.txt"
REMOTE_ROOT = "/root/CPS-Paper2"


def _creds():
    lines = [ln.strip() for ln in SERVER_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    user, host = lines[0].split("@", 1)
    pw = lines[1].split(":", 1)[-1].strip()
    return user, host, pw


def main() -> int:
    paths = sys.argv[1:] or []
    if not paths:
        raise SystemExit("usage: _ssh_upload.py relpath ...")
    user, host, password = _creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=45, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()
    for rel in paths:
        local = ROOT / rel
        remote = f"{REMOTE_ROOT}/{rel.replace(chr(92), '/')}"
        remote_dir = str(Path(remote).parent).replace("\\", "/")
        try:
            sftp.stat(remote_dir)
        except OSError:
            # mkdir -p style
            parts = remote_dir.split("/")
            cur = ""
            for p in parts:
                if not p:
                    continue
                cur = f"{cur}/{p}" if cur else p
                try:
                    sftp.stat(cur)
                except OSError:
                    sftp.mkdir(cur)
        sftp.put(str(local), remote)
        print("uploaded", rel)
    sftp.close()
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
