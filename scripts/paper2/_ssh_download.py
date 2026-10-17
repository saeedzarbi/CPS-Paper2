"""Download a repo-relative file from /root/CPS-Paper2 via SFTP."""
from __future__ import annotations

import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
SERVER_FILE = ROOT / "server.txt"
REMOTE_ROOT = "/root/CPS-Paper2"


def main() -> int:
    rel = sys.argv[1]
    user, host = SERVER_FILE.read_text(encoding="utf-8").splitlines()[0].strip().split("@", 1)
    pw = SERVER_FILE.read_text(encoding="utf-8").splitlines()[1].split(":", 1)[-1].strip()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=pw, timeout=45, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()
    remote = f"{REMOTE_ROOT}/{rel.replace(chr(92), '/')}"
    local = ROOT / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    sftp.close()
    client.close()
    print("downloaded", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
