# Canonical trace manifests

`canonical_traces.json` lists every Tube-v2 **train / test / val / calib** directory with **sha256** per `*.json` trace file and generator metadata where applicable.

- **Regenerate** after intentional trace changes:  
  `python scripts/paper2/build_trace_manifests.py --write`
- **Verify** before smoke / Phase 2:  
  `python scripts/paper2/build_trace_manifests.py --verify`  
  (also called automatically from `tube_v2_campaign.py --smoke` and `run_tube_v2_phase2.py`)

Markov train/test are defined by `scripts/paper2/gen_markov_traces.py` (seed **4242**, deterministic JSON). A fresh box must either match this manifest or regenerate traces with that script and commit an updated manifest.
