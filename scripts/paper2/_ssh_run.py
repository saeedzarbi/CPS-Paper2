"""One-off remote command via server.txt (gitignored). Do not commit credentials."""
from __future__ import annotations

import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
SERVER_FILE = ROOT / "server.txt"


def _parse_server_txt() -> tuple[str, str, str]:
    lines = [ln.strip() for ln in SERVER_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise SystemExit("server.txt needs user@host and pass line")
    user, host = lines[0].split("@", 1)
    pw = lines[1]
    if pw.lower().startswith("pass:"):
        pw = pw.split(":", 1)[1].strip()
    return user, host, pw


def main() -> int:
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hostname"
    user, host, password = _parse_server_txt()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=45, allow_agent=False, look_for_keys=False)
    _, stdout, stderr = client.exec_command(cmd, get_pty=False)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, file=sys.stderr, end="" if err.endswith("\n") else "\n")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
