#!/usr/bin/env python3
from pathlib import Path
import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/CPS-Paper2"


def main() -> None:
    text = (ROOT / "server.txt").read_text(encoding="utf-8")
    creds = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            creds[k.strip().lower()] = v.strip()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        creds["ip"],
        username=creds["user"],
        password=creds["pass"],
        timeout=60,
        allow_agent=False,
        look_for_keys=False,
    )

    def run(cmd: str) -> str:
        _, stdout, _ = client.exec_command(cmd, timeout=120)
        return stdout.read().decode("utf-8", errors="replace").strip()

    print("screen:\n", run("screen -ls") or "(none)")
    proc = run("ps aux | grep run_extended_validation.py | grep -v grep")
    print("\nprocess:\n", proc or "NOT RUNNING (likely finished)")
    paths = run(f"find {REMOTE}/results/clean_v3_extended -name summary.json | sort")
    lines = [p for p in paths.splitlines() if p.strip()]
    print(f"\nsummary.json: {len(lines)} files")
    for p in lines:
        print(" ", p.replace(REMOTE, ""))
    log = run(
        f"bash -lc 'ls -t {REMOTE}/results/clean_v3_extended/logs/*.log 2>/dev/null | head -1 | xargs -r tail -20'"
    )
    print("\nlog tail:\n", log or "(no log file)")
    client.close()


if __name__ == "__main__":
    main()
