#!/usr/bin/env python3
"""Sync extended-validation code to server and start screen job."""
from __future__ import annotations

import sys
import tarfile
import tempfile
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/CPS-Paper2"

REL = [
    "configs/paths.py",
    "configs/videos.py",
    "data/prepare_markov_5g_traces.py",
    "data/prepare_eval_trace_subsets.py",
    "experiments/run_clean_campaign.py",
    "src/evaluation/run_extended_validation.py",
    "src/evaluation/clean_abr_eval.py",
    "requirements-smoke.txt",
]


def _collect() -> list[str]:
    rel = list(REL)
    for d in ("src/control", "src/environment", "src/evaluation", "src/training", "src/utils"):
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and "__pycache__" not in str(p):
                rel.append(p.relative_to(ROOT).as_posix())
    return list(dict.fromkeys(rel))


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


def main() -> int:
    tmp = Path(tempfile.gettempdir()) / "ext_v3_sync.tgz"
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in _collect():
            p = ROOT / rel
            if p.is_file():
                tar.add(p, arcname=f"CPS-Paper2/{rel}")

    host, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=60, allow_agent=False, look_for_keys=False)

    def run(cmd: str, timeout: int = 86400) -> int:
        print(">>>", cmd[:140])
        _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        if out:
            print(out[-5000:] if len(out) > 5000 else out)
        if err and code != 0:
            print(err[-1500:], file=sys.stderr)
        return code

    sftp = client.open_sftp()
    sftp.put(str(tmp), "/root/ext_v3_sync.tgz")
    sftp.close()
    tmp.unlink(missing_ok=True)

    run(f"cd {REMOTE} && tar -xzf /root/ext_v3_sync.tgz -C /root && rm -f /root/ext_v3_sync.tgz")
    wrap = f"cd {REMOTE} && source .venv/bin/activate && export PYTHONPATH={REMOTE} PYTHONIOENCODING=utf-8"
    run(f"bash -lc '{wrap} && python data/prepare_eval_trace_subsets.py'")
    run("screen -S clean_v3_ext -X quit 2>/dev/null || true")
    inner = (
        f"cd {REMOTE} && source .venv/bin/activate && export PYTHONPATH={REMOTE} PYTHONIOENCODING=utf-8 && "
        f"python src/evaluation/run_extended_validation.py --episodes 204 --latency-episodes 17 "
        f"2>&1 | tee -a results/clean_v3_extended/logs/run_$(date +%Y%m%d_%H%M%S).log"
    )
    run(f"screen -dmS clean_v3_ext bash -lc {repr(inner)}")
    time.sleep(5)
    run("screen -ls")
    run("ps aux | grep run_extended_validation | grep -v grep | head -3")
    client.close()
    print("Started extended validation in screen `clean_v3_ext`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
