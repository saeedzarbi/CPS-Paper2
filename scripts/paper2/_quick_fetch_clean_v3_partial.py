#!/usr/bin/env python3
import json
from pathlib import Path
import paramiko

ROOT = Path(__file__).resolve().parents[2]
text = (ROOT / "server.txt").read_text(encoding="utf-8")
h = u = p = ""
for line in text.splitlines():
    if ":" not in line:
        continue
    k, v = line.split(":", 1)
    k, v = k.strip().lower(), v.strip()
    if k == "ip":
        h = v
    elif k == "user":
        u = v
    elif k == "pass":
        p = v

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(h, username=u, password=p, timeout=60, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
jobs = ["greedy_5g_markov", "greedy_bb", "mpc_5g_markov", "mpc_bb"]
for job in jobs:
    remote = f"/root/CPS-Paper2/results/clean_v3/{job}/summary.json"
    local = ROOT / f"results/clean_v3/{job}/summary.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        sftp.get(remote, str(local))
    except OSError:
        print(job, "MISSING")
        continue
    s = json.loads(local.read_text(encoding="utf-8"))
    arms = s["arms"]

    def reb(a):
        return arms[a]["rebuffer_total"]["mean"]

    print(
        f"{job:18} raw={reb('raw'):7.2f}  static={reb('static_conformal'):7.2f}  "
        f"tube={reb('tube_conformal'):7.2f}  (tube wins: {reb('tube_conformal') <= min(reb('raw'), reb('static_conformal'))})"
    )
sftp.close()
c.close()
