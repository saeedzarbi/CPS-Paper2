#!/usr/bin/env python3
"""Deploy CPS-Paper2 + data to tools-new and start extended validation in screen."""
from __future__ import annotations

import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_DIR = "/root/CPS-Paper2"
SSH_HOST = "tools-new"

CODE_REL_PREFIXES = (
    "configs",
    "data/prepare",
    "experiments/paper2",
    "experiments/run_clean_campaign.py",
    "src",
    "scripts/paper2",
    "requirements-smoke.txt",
    "requirements.txt",
)


def _ssh(cmd: str, timeout: int = 86400) -> int:
    print(">>>", cmd[:120])
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", SSH_HOST, cmd],
        timeout=timeout,
    )
    return r.returncode


def _scp(local: Path, remote: str) -> None:
    print(">>> scp", local.name, "->", remote)
    subprocess.check_call(["scp", "-q", str(local), f"{SSH_HOST}:{remote}"])


def _tar_repo(tmp: Path) -> Path:
    archive = tmp / "cps_paper2_code.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for rel in CODE_REL_PREFIXES:
            p = ROOT / rel
            if p.is_file():
                tar.add(p, arcname=f"CPS-Paper2/{rel}")
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and "__pycache__" not in str(f):
                        tar.add(f, arcname=f"CPS-Paper2/{f.relative_to(ROOT).as_posix()}")
        for f in (ROOT / "data").glob("prepare*.py"):
            tar.add(f, arcname=f"CPS-Paper2/data/{f.name}")
    return archive


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        code = _tar_repo(tdir)
        data = tdir / "data.tgz"
        print(">>> packing local data/")
        subprocess.check_call(
            ["tar", "-czf", str(data), "-C", str(ROOT), "data"],
            shell=False,
        )

        _ssh(f"mkdir -p {REMOTE_DIR}")
        _scp(code, "/tmp/cps_paper2_code.tgz")
        _scp(data, "/tmp/cps_data.tgz")
        _ssh("tar -xzf /tmp/cps_paper2_code.tgz -C /root && rm -f /tmp/cps_paper2_code.tgz")
        _ssh(f"tar -xzf /tmp/cps_data.tgz -C {REMOTE_DIR}")
        _ssh("rm -f /tmp/cps_data.tgz")

        webhook = ROOT / "mattermost_webhook.url"
        if webhook.is_file():
            _scp(webhook, f"{REMOTE_DIR}/mattermost_webhook.url")
            print("Uploaded mattermost_webhook.url")

    _ssh(
        f"sed -i 's/\\r$//' {REMOTE_DIR}/experiments/paper2/run_extended_validation.sh "
        f"2>/dev/null || true"
    )

    setup = (
        f"cd {REMOTE_DIR} && "
        f"(test -d .venv || python3 -m venv .venv) && "
        f"source .venv/bin/activate && "
        f"pip install -q -r requirements-smoke.txt && "
        f"export PYTHONPATH={REMOTE_DIR} PYTHONIOENCODING=utf-8 && "
        f"python data/prepare_markov_5g_traces.py && "
        f"python data/prepare_eval_trace_subsets.py"
    )
    if _ssh(f"bash -lc '{setup}'", timeout=600) != 0:
        return 1

    _ssh("screen -S clean_v3_ext -X quit 2>/dev/null || true")
    run = (
        f"cd {REMOTE_DIR} && source .venv/bin/activate && "
        f"bash experiments/paper2/run_extended_validation.sh "
        f"2>&1 | tee -a results/clean_v3_extended/logs/run_$(date +%Y%m%d_%H%M%S).log"
    )
    if _ssh(f"screen -dmS clean_v3_ext bash -lc {repr(run)}") != 0:
        return 1
    _ssh("sleep 3; screen -ls; ps aux | grep run_extended_validation | grep -v grep | head -2")
    print(f"\nExtended validation started on {SSH_HOST} (screen clean_v3_ext).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
