# Real 5G traces for Paper 2

## Recommended dataset: Irish operator 5G (MMSys 2020)

| | |
|---|---|
| **What** | Client-side KPI logs from a commercial **5G** deployment (Ireland), including `DL_bitrate`, RSRP/RSRQ/CQI, static + driving, Netflix/Prime/download workloads |
| **Size** | ~2 MB zip, 83 CSV sessions |
| **License** | GPL-3.0 in [uccmisl/5Gdataset](https://github.com/uccmisl/5Gdataset) |
| **Citation** | D. Raca, D. Leahy, C.J. Sreenan, J.J. Quinlan, *Beyond Throughput, The Next Generation: A 5G Dataset with Channel and Context Metrics*, MMSys 2020 |

### Download

```bash
mkdir -p data/external
curl -L -o data/external/5G-production-dataset.zip \
  https://github.com/uccmisl/5Gdataset/raw/master/5G-production-dataset.zip
unzip -d data/external/5g_irish_mmsys2020 data/external/5G-production-dataset.zip
```

Windows (PowerShell): use `Invoke-WebRequest` and `Expand-Archive` on the same paths.

### Convert to simulator format

```bash
python scripts/paper2/prepare_traces_5g_irish_mmsys20.py --all-test
```

(`--all-test` puts every valid session in the eval folder; omit for a 75/25 group split with train co-design.)

Outputs (under `data/standardized/`):

- `test_traces_5g_irish_mmsys20/` — hold-out scenario groups (~25% of groups)
- `train_traces_5g_irish_mmsys20/` — remaining sessions
- `trace_stats_5g_irish_mmsys20.json` — provenance manifest

Each trace JSON matches the ABR env: `{"throughput_kbps": [ ... ]}` at **1 Hz**, built from `DL_bitrate` with 5G-only rows, per-second max, zero → forward-fill.

**Eval:** point `--trace-dir` at the test folder (same paired seeds as v18). Keep `test_traces_5g_v18` as **synthetic stress** ablation if desired.

### Limitations (disclose in the paper)

- Throughput is **application-layer DL bitrate** from G-NetTrack, not a controlled iperf sweep.
- Geographic / operator coverage is **single country, one operator (anonymized as “B”)**, 2019–2020.
- Not a substitute for FCC/Norway broadband traces; use **`test_traces_v18`** for fixed/wireline.

## Alternatives (not wired in this repo yet)

| Dataset | Notes |
|---------|--------|
| [MONROE Nettest open dataset](https://faculty.e-ce.uth.gr/anargyr/papers/jnl_2020_comnet_preprint.pdf) | Large-scale LTE measurements; JSON time series per speedtest |
| [MONROE @ PoliTO 2018–2019](https://smartdata.polito.it/mobile-networks-open-dataset/) | ~46 GB, TCP/UDP logs; more preprocessing |
| [GreenLv/Chorus `bw_traces/`](https://github.com/GreenLv/Chorus) | 52 traces, some **5G**, CSV Mbps — good second source for robustness |
| [4G LTE Irish dataset](https://github.com/uccmisl/4Gdataset) (MMSys 2018) | Same team; use if reviewers want NR + LTE comparison |

## Server

```bash
python scripts/paper2/prepare_traces_5g_irish_mmsys20.py --all-test
python scripts/paper2/_remote_run_tube_irish_campaign.py
```

Fetch results: `python scripts/paper2/_remote_fetch_tube_results.py --irish`
