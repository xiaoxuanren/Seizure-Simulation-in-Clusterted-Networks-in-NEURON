# CHTC seed-sweep dataset generation

Generates the cluster-count × topology-seed sweep on UW–Madison's CHTC
(HTCondor): each **job = one 60 s recording** of one network, ~70–90 min on a
single core. The sweep declared in [`sweep_config.json`](sweep_config.json):

| group | `num_clusters` | `space_size` | topology seeds | recordings each |
|-------|---------------|--------------|----------------|-----------------|
| `sweep_c50` | 50 | 15.0 | 1–10 | 200 |
| `sweep_c40` | 40 | 13.416 (= 15·√(40/50), constant neuron density) | 11–20 | 200 |

= 20 networks × 200 recordings = **4,000 jobs** (~17 min each observed), full
voltage (`inline_npz`, ~105–130 MB/recording, **~470 GB total**). Each job
returns its outputs as ONE uniquely-named tar via normal HTCondor file
transfer into the submit directory on `/home` (per CHTC guidance for sub-GB
outputs); `/staging` only hosts the container image.

**Reproducibility / provenance.** Everything is deterministic from the config:
topology from `(num_clusters, space_size, seed)`; background noise per
recording from `Random123(noise_seed_base, gid, rec_idx)` with
`noise_seed_base = 1000 + 100·topology_seed`, so recordings differ within a
network and noise streams differ across networks. `topology_seed`,
`noise_seed_base`, `num_clusters`, `space_size`, the builder kwargs, the full
resolved state/build parameters, and library versions are stored **inside
`network_<session>.npz`** and in `session_provenance.json` /
`session_metadata.json`.

**Operating point.** `build_overrides` in the config carries
`noise_weight = 0.007` — the flagship operating point used by
`dataset_generation.ipynb` and measured in `MODEL_CHARACTERIZATION.md` — which
is deliberately NOT the registry default (0.004). Remove or change it only if
you *intend* to run at a different drive strength; the value used is recorded
in every session's provenance either way.

## One-time setup (CHTC submit node)

You need a CHTC account, a **`/home` quota that fits the outputs** (the full
sweep is ~470 GB; CHTC granted 1.1 TB here on request — for sub-GB output
files they prefer `/home` + HTCondor file transfer over `/staging`), and a
default [/staging allocation](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata)
which only needs to hold the ~1 GB container image. With a default-sized
`/home` (40 GB), run session-filtered waves instead (pass session names to
`submit_all.sh`) and download + delete the returned tars between waves.

```bash
# on the submit node
apptainer build neuron-sim.sif neuron.def
cp neuron-sim.sif /staging/YOUR_NETID/
mkdir -p /staging/YOUR_NETID/neuron_sweeps logs
```

Then edit two placeholders:
- `job.sh` → `STAGING_DIR="/staging/YOUR_NETID/neuron_sweeps"`
- `generate.sub` → `container_image = file:///staging/YOUR_NETID/neuron-sim.sif`

## Submit

```bash
# copy the repo (or just chtc/ + neuron_simulation/) to the submit node, then:
python3 chtc/make_manifest.py        # writes jobs.txt (4,000 lines) + repo.tar.gz
cd chtc
mkdir -p logs                        # generate.sub writes per-job logs here
condor_submit generate.sub
condor_q                             # watch progress
```

(Line endings are handled: `job.sh` normalizes CRLF in the `.mod` files before
compiling, so a tarball built on Windows works.)

Failed/evicted jobs retry automatically (`max_retries = 3`). To resubmit only
whatever is still missing after the run:

```bash
python3 chtc/make_manifest.py --done-root /staging/YOUR_NETID/neuron_sweeps
condor_submit generate.sub
```

## Collect

Each finished job leaves `<session>_r<idx>.tar` in the `chtc/` submit
directory. Download them to your machine (remote glob works with scp):

```bash
scp "YOUR_NETID@ap2001.chtc.wisc.edu:Seizure-Simulation-in-Clusterted-Networks-in-NEURON/chtc/*_r*.tar" "D:/path/to/download_dir"
```

then assemble into the repo's session layout and validate (`collect.py`
auto-extracts any job tars it finds under `--src`):

```bash
python chtc/collect.py --src "D:/path/to/download_dir"
```

After a session collects OK, delete its tars on the access point to free
`/home` (`rm chtc/<session>_r*.tar`).

This creates `notebooks/NEURON data parallel/<session>/normal/` with the
recordings, `network_<session>.npz`, and a `session_metadata.json` per
session. **Every** recording is validated (readable npz with spikes) before it
enters the dataset — corrupt files are reported and left out, so the printed
`make_manifest.py --done-root` command re-emits exactly the jobs that need to
run again. Uploads are atomic on the job side (temp name + `mv`), so partial
files cannot appear under a final recording name in the first place.
After that every analysis script sees the sweeps:

```bash
python analysis/session_paths.py                 # lists all 20 sessions
python scripts/run_inference.py --session "notebooks/NEURON data parallel/sweep_c50_seed01/normal" glm
```

Note: the new sessions are raw data, so git ignores them by default
(committing a dataset is a deliberate `git add -f`, per `.gitignore`).

## Local smoke test (before burning core-hours)

Any machine with NEURON installed (no Condor needed):

```bash
python chtc/generate_one.py --sweep chtc/sweep_config.json --list-sessions
python chtc/generate_one.py --sweep chtc/sweep_config.json \
    --session sweep_c50_seed01 --rec-idx 0 --out smoke_out
```

For a fast end-to-end check, temporarily lower `recording_ms` (e.g. 5000) in a
copy of the config. Verify the seeds landed in the network file:

```python
import numpy as np
d = np.load("smoke_out/sweep_c50_seed01/network_sweep_c50_seed01.npz", allow_pickle=True)
print(int(d["topology_seed"]), int(d["noise_seed_base"]), int(d["num_clusters"]), float(d["space_size"]))
```

## Files

| file | role |
|------|------|
| `sweep_config.json` | the sweep declaration (edit this to change the grid) |
| `generate_one.py` | worker: builds the network deterministically, runs one recording |
| `make_manifest.py` | writes `jobs.txt` (+ `repo.tar.gz`); `--done-root` for resubmits |
| `job.sh` | in-job wrapper: compile mechanisms → run → copy to `/staging` |
| `generate.sub` | HTCondor submit file (1 CPU / 4 GB / 4 GB disk per job) |
| `neuron.def` | Apptainer recipe (Python 3.11 + `pip install neuron`) |
| `collect.py` | assemble staged outputs into the repo layout + validate + metadata |
