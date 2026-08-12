# Seizure Simulation in Clustered Networks (NEURON)

A biophysical spiking-network model of spontaneous **network bursts** and their
transition into a **seizure (ictal) state**, in a clustered culture built in
[NEURON](https://neuron.yale.edu/). Single-compartment Hodgkin–Huxley neurons
carry a custom **A-type / Kv4-like potassium current** (`kA`), a **dynamic
extracellular-potassium** mechanism (`kdyn`), and a **two-timescale
spike-frequency-adaptation** mechanism (`sAHP`). The network is driven only by
weak per-neuron Poisson background input and produces discrete,
high-participation network bursts.

**The seizure knob is a slow-AHP deficit (`sahp_ainc_slow`), not impaired K⁺
clearance and not a reduced A-current.** `sahp_ainc_slow` is the Ca²⁺-dependent
slow-AHP per-spike conductance increment (a KCa conductance, *not* the
Kv7/M-current). Normal is **pinned at 0.01 µS**; any **lower** value is a
seizure state (the default 0.004 is a convenience, not a commitment). This
models an **acquired-epilepsy sAHP deficit** — the adaptation-deficit,
mild-[K⁺]ₒ bursting phenotype. K⁺ clearance is **held fixed** at
`tau_k = 200 ms`; the impaired-clearance route survives only as
`kclearance_seizure_state` (the alternative high-[K⁺]ₒ ictal phenotype), and
the reduced-`gbar_kA` "4-AP" route remains a **phenomenological knob** (see
below).

The project deliberately mirrors the companion **LIF project**
([LIF-Project](https://github.com/xiaoxuanren/LIF-Project),
branch `chore/repo-cleanup`) in its saved **data format**. The live
connectivity-inference pipeline is now the in-repo **sparse GLM**
(`sparse_glm.py` + `glm_connectivity.py`, driven by
`scripts/run_inference.py`); the vendored CCG + learned-LIF pipeline is
**retired** to `archive/inference/` (its results are preserved below as
historical record).

---

## Repository layout

```
neuron_simulation/          # the simulator package
  mechanisms/
    kA.mod                  # A-type K+ current (Kv4-like), fast htau (20 ms)
    kdyn.mod                # dynamic [K+]o accumulation -> ek (clearance route; tau_k held fixed)
    sAHP.mod                # two-timescale adaptation: fast M-like + slow KCa (THE SEIZURE KNOB)
    AmpaNmda.mod            # AMPA+NMDA excitatory synapse with depression (default exc model)
    DepSyn.mod              # single-exponential depressing excitatory synapse (alternative)
    README.md               # how to compile: `nrnivmodl mechanisms`
  neurons.py                # single-compartment HH + kA + kdyn + sAHP cell builders (E and I)
  topology.py               # clustered+hub AND log-normal-degree builders
  network_builder.py        # assemble a NEURON net from a topology (cells, synapses, NetCons, noise)
  noise.py                  # Poisson background; per-recording Random123 streams
  states.py                 # normal vs seizure (sahp_ainc_slow); kclearance + gbar_block alternatives
  parameters.py             # the parameter registry (single source of defaults)
  simulation.py             # run(): finitialize/continuerun, spike + optional voltage + [K+]o recording
  workflows.py              # topology->network->run->save spikes + ground truth; dataset generation
  analysis.py               # participation-based network-burst detection, burst stats
  plotting.py               # raster, population activity, degree distribution, topology map
  io.py                     # save/load spikes, ground-truth connectivity, metadata (LIF-format)
sparse_glm.py               # memory-efficient sparse lag-resolved ridge GLM (the live inference core)
glm_connectivity.py         # GLM edge prediction / typing (extracted from the archived package)
scripts/
  run_inference.py          # GLM inference CLI (--session is required; no default)
  glm_sweep.py              # GLM hyperparameter sweeps
  verify_readout.py         # readout verification
analysis/                   # ~40 standalone analysis/dataset scripts + session_paths.py (path registry)
notebooks/
  dataset_generation.ipynb            # CURRENT dataset driver (thin UI over analysis/dataset_nb.py)
  neuron_network_simulation.ipynb     # original walkthrough (final inference cells now archived)
  NEURON data parallel/               # committed sessions: <session>/<state>/recordingNNN.npz
                                      #   + <session>/results/<state>/{glm,bursts,...}
  NEURON data/                        # old sequential pilot sessions
tests/
  test_parameter_drift.py
figures/                    # 6 curated PNGs (referenced from analysis/README.md)
archive/                    # historical material incl. inference/ (retired learned-LIF/CCG pipeline);
                            #   see archive/README.md for an item-by-item inventory
README.md, MODEL_CHARACTERIZATION.md
```

---

## Quick start

### 1. Environment

NEURON, NumPy, SciPy, Matplotlib, scikit-learn, h5py, and (only if you want to
re-run the *archived* learned-LIF pipeline) PyTorch — the live GLM inference
does not need torch:

```bash
pip install neuron numpy scipy matplotlib scikit-learn h5py torch jupyterlab ipykernel
```

**The interpreter must match NEURON's bundled `hoc` bindings.** NEURON only ships
`hoc` for specific Python versions, so `import neuron` fails on a mismatched
interpreter (e.g. a newer system Python) with
`ModuleNotFoundError: No module named 'hoc'` / `neuron.hocNNN`.

#### Validated Windows setup (this machine)

- NEURON **8.0** installed at `C:\nrn`.
- Runs on **Python 3.9** (`...\Programs\Python\Python39\python.exe`) — with
  numpy 2.0.2, scipy, matplotlib, scikit-learn, h5py, torch 2.5.1+cu121.
- The system **Python 3.12 cannot run NEURON** here (`No module named
  'neuron.hoc312'`), so use the 3.9 interpreter for everything.
- **Running** the notebook needs **no PATH change** — NEURON 8.0 registers its
  own DLL directory on `import neuron` (verified: the kernel builds a network
  from a plain inherited PATH). `C:\nrn\bin` on `PATH` is only needed for
  **compiling** mechanisms with the bare `nrnivmodl` command (step 2) — or call
  it by full path, `& 'C:\nrn\bin\nrnivmodl.bat' mechanisms`.

#### Register a Jupyter kernel for the NEURON interpreter (once)

The `.ipynb` needs a kernel backed by the NEURON-capable Python. On this machine:

```powershell
$py = 'C:\Users\<you>\AppData\Local\Programs\Python\Python39\python.exe'
& $py -m pip install jupyterlab ipykernel
& $py -m ipykernel install --user --name neuron-py39 --display-name "Python 3.9 (NEURON)"
& $py -m jupyter lab
```

Then open the notebook and pick the **"Python 3.9 (NEURON)"** kernel — it runs
NEURON out of the box (no PATH setup). You can also execute the notebook's cells
headlessly with the same interpreter directly.

### 2. Compile the mechanisms (once)

The custom mechanisms (`kA`, `kdyn`, `DepSyn`, `AmpaNmda`, `sAHP`) **must be
compiled before running** (and recompiled whenever a `.mod` is added or changed):

```bash
cd neuron_simulation
nrnivmodl mechanisms
```

This produces `nrnmech.dll` (Windows) or an `x86_64/` build directory
(Linux/macOS). See [`neuron_simulation/mechanisms/README.md`](neuron_simulation/mechanisms/README.md).
`build_network(...)` calls `load_mechanisms()` for you.

### 3. Run a notebook

- [`notebooks/dataset_generation.ipynb`](notebooks/dataset_generation.ipynb) —
  the **current dataset driver**: a thin UI over `analysis/dataset_nb.py` for
  generating inference-ready sessions.
- [`notebooks/neuron_network_simulation.ipynb`](notebooks/neuron_network_simulation.ipynb) —
  the original end-to-end walkthrough (topology, normal-state bursts,
  dose-response, dataset save). Its **final inference cells reference the
  now-archived adapter and no longer run**; use `scripts/run_inference.py`
  instead (step 4).

### 4. Or from Python

```python
from neuron_simulation import topology, workflows, states

topo = topology.build_topology_lognormal(seed=1)          # preferred builder

# Normal vs seizure: SAME network, one lower sahp_ainc_slow value.
normal  = workflows.run_single_state(topo, state=states.normal_state())   # sahp_ainc_slow = 0.01
seizure = workflows.run_single_state(topo, state=states.seizure_state())  # default 0.004; any value < 0.01
# Lower knob -> more firing (0.29 -> 0.63 Hz on the 926-cell reference network);
# [K+]o stays ~4 mM in BOTH states (this is the mild-[K+]o phenotype).

# Generate an inference-ready dataset (normal state):
meta, session_dir = workflows.generate_dataset(n_recordings=3, recording_duration=15000)
```

Then run the sparse-GLM inference against a session (`--session` is required
and goes *before* the subcommand; quote paths — they contain spaces):

```bash
python scripts/run_inference.py --session "notebooks/NEURON data parallel/IC-locked_flagship_spikeonly_50rec/normal" glm --readout sum4
```

---

## The biophysics

- **Neurons.** Single-compartment soma (`L = diam = 20 µm`) with NEURON's
  built-in `hh` (Na⁺/K⁺/leak), the custom `kA` A-current, the `kdyn` dynamic
  [K⁺]ₒ mechanism, and the `sAHP` two-timescale spike-frequency adaptation
  (fast M-current/Kv7-like component, `tau 300 ms`, plus the slow
  Ca²⁺-dependent KCa component, `tau 6500 ms` — the slow increment is the
  seizure knob). 80% excitatory, 20% inhibitory. `celsius` is configurable
  (default **6.3 °C**, keeping squid `hh` kinetics); a mammalian variant at 34 °C
  runs with faster `kA` kinetics via a q10 factor.

- **A-current (`kA.mod`).** Fast activation `m` and inactivation `h`
  (`htau0 = 20 ms`), reversal at `ek`; `I_kA = gbar · m⁴ · h · (v − ek)`. It
  shapes crisp discrete bursts. `gbar_kA` is retained only as a phenomenological
  knob (`gbar_block_state`), **not** a faithful 4-AP model — see below.

- **Dynamic [K⁺]ₒ (`kdyn.mod`).** `ek` is written from the Nernst equation on a
  state variable `[K⁺]ₒ` that rises with K⁺ efflux (firing) and is cleared with
  time constant `tau_k`. `ki = 72 mM` fixes resting E_K = −77 mV. **`tau_k` is
  held fixed at 200 ms** — it is the substrate of the *alternative*
  high-[K⁺]ₒ ictal route (`kclearance_seizure_state`), not the project's
  seizure knob.

- **Synapses.** The default excitatory synapse is `AmpaNmda` (fast AMPA,
  `tau 5 ms`, plus slow NMDA, `tau 350 ms` at NMDA/AMPA ratio 3.0 — the NMDA
  component carries burst reverberation), with Tsodyks–Markram short-term
  depression (`d = 0.2`, `tau_d = 500 ms`). `DepSyn` (single fast exponential
  with the same depression) is the alternative via `synapse_model="depsyn"`.
  Inhibitory synapses are `ExpSyn` with reversal −75 mV. Dale's law is
  enforced: a neuron's E/I identity fixes the sign of every synapse it makes.

- **Drive.** Each neuron receives an independent Poisson `NetStim → ExpSyn`
  background. **This is the only drive** — no current injection, no stimulation,
  no tonic drivers. Each generator uses its own reproducible `Random123` stream.

### Background noise: the single-event effect depends on adaptation state

The flagship ran at noise_weight = 0.007 uS. Measured rheobase at zero
adaptation is 0.000863 uS, so a single background event is strongly
SUPRATHRESHOLD on a rested cell (dV +98.8 mV at 0.004 uS; it fires).

RETRACTED: an earlier version of this file claimed a single noise event was
subthreshold (~5.6 mV). That was true of the older 0.0008 uS weight only.

At the network's actual operating point the picture reverses. Tonic sAHP load
(~0.021 uS) holds V_rest at -84.6 mV and raises the adapted rheobase to
0.01435 uS, so one event is 0.49x threshold -- dV +14.2 mV, no spike.

The functional claim -- background drive integrates rather than detonating --
therefore holds, but it is delivered by ADAPTATION, not by event amplitude.
Cells are suprathreshold-sensitive only in the first ~1 s after finitialize,
before sAHP loads.

### Tuned defaults (validated)

> These are the *registry* defaults (`neuron_simulation/parameters.py`), not
> necessarily the flagship run's values: the flagship used
> `noise_weight = 0.007` (vs the registry 0.004); its `exc_weight_scale = 2.0`
> and `inh_weight_scale = 2.5` match the registry. See
> `MODEL_CHARACTERIZATION.md` for the measured operating point.

| parameter | default | role |
|-----------|---------|------|
| `sahp_ainc_slow` | `0.01` µS (normal, pinned) | **the seizure knob** — slow-AHP (KCa) per-spike increment; any lower value is a seizure |
| `noise_weight` | `0.004` µS | single background event — suprathreshold on a rested cell, ~0.49× the adapted rheobase at the operating point (see above) |
| `noise_rate` | `5.0` Hz | per-neuron Poisson background (the sole drive) |
| `exc_weight_scale` | `2.0` | recurrent gain |
| `inh_weight_scale` | `2.5` | recurrent inhibition |
| `gbar_kA_exc` / `gbar_kA_inh` | `0.006` / `0.004` S/cm² | A-current density (identical across states) |
| `htau0_kA` | `20 ms` | fast A-current inactivation (crisp bursts) |
| `tau_k` | `200 ms` (held fixed) | K⁺ clearance — **not** the seizure knob |

### Network bursts

RETRACTED: this file previously reported a network burst rate of 1.3-1.5 Hz and
carried an honest-caveat that this was fast relative to dissociated cultures
(~0.03 Hz). Both are wrong, and the caveat is wrong in the opposite direction.

Measured over 200 recordings at the project's 0.35 participation gate:

  spontaneous burst rate   0.0166 Hz over the full record
                           0.0184 Hz over the eligible 6-60 s window
  spikes in bursts         7.30% of all spikes (3.64% spontaneous-only)
  burst time               0.177% of the record

So the model is within 1.6-1.8x of the cited culture rate, and within 1.07x
counting all 0.35-gate bursts. It does not have a fast-burst problem; if
anything it is slightly slow.

Note that bursts fall into two classes -- an initialization-locked event at
4.60-5.34 s present in 137 of 336, and 199 genuinely spontaneous ones uniformly
distributed over 8.63-59.85 s. The rates above are the spontaneous class. Full
per-class statistics are pending.

### Recurrent coupling and the sharp HH threshold (honest caveat)

The single-compartment HH point-neuron has a **razor-sharp single-event
rheobase** (~`0.00085 µS` ≈ a 6 mV peak EPSP): a synaptic event is either
**≤ 5.6 mV (subthreshold)** or triggers a **full ~101 mV spike** — there is no
"moderately suprathreshold" middle. Consequently:

- A single **noise** event: the subthreshold claim that stood here has been
  **retracted** — see "Background noise" above.
- A single **recurrent** excitatory event, at any weight strong enough to sustain
  network bursts, is **suprathreshold** (fires the postsynaptic cell). We verified
  by sweep (density 0.04–0.12, `exc_tau` 3–6 ms, `noise_rate` 2–30 Hz, `tau_d`
  250–800 ms) that a **genuinely subthreshold recurrent weight set does NOT
  sustain bursts** — the network is silent at low noise and asynchronously tonic
  at high noise. So **subthreshold recurrent coupling and the burst gate are
  mutually exclusive** in this cell.

The recurrent weights were nonetheless **reduced** from the earlier calibration
(`exc_weight_scale` 4.0 → 1.5 at the time of that sweep; the registry default
has since settled at 2.0) to near the minimum that still bursts. Making them
fully subthreshold would require a better-integrating cell (e.g. a
multi-compartment neuron), which is out of scope here.

### The seizure mechanism (slow-AHP deficit)

Seizure is modelled by a **single knob**: `sahp_ainc_slow`, the Ca²⁺-dependent
slow-AHP per-spike conductance increment. One fixed network, one parameter, two
phenotypes (authoritative source: the `neuron_simulation/states.py` module
docstring):

- **Normal** — `sahp_ainc_slow = 0.01` µS (strong slow adaptation; **pinned**:
  it is what the shipped 50-minute flagship session was generated with). Quiet,
  sparse loose bursts; [K⁺]ₒ stays ~4 mM. `states.normal_state()`.
- **Seizure** — any **lower** `sahp_ainc_slow` (weak slow adaptation; more
  firing, denser bursts). Seizure is defined as "less slow adaptation than
  normal", not as one blessed number — the default `0.004` is a convenience.
  Measured on the 926-cell seed-1 reference network (60 s): `0.010` → 0.29 Hz
  firing, 8 loose bursts (0.13 Hz), participation 0.93; `0.004` → 0.63 Hz,
  10 bursts (0.17 Hz), participation 1.00 — lower knob → more firing,
  monotonically. `states.seizure_state(value)`;
  `states.seizure_dose_response()` sweeps the knob.

**Channel identity (mind this before citing the knob).** With
`tau_slow = 6500 ms`, the knob is by both kinetics and pharmacology the
**Ca²⁺-dependent slow AHP** (a KCa conductance), **not** the Kv7/KCNQ
M-current — that is `sahp_ainc_fast` (`tau 300 ms`), which is **held fixed**
across states. Lowering the knob therefore models an **acquired-epilepsy sAHP
deficit** (the KCa3.1-like reduction seen in post-status-epilepticus
hippocampus), *not* a KCNQ2/3 channelopathy. This is the
**adaptation-deficit, mild-[K⁺]ₒ bursting** phenotype — not a high-[K⁺]ₒ
ictal state.

Everything else is held fixed across the two states by design: `tau_k = 200 ms`,
`sahp_ainc_fast = 0.005`, `gbar_kA_exc/inh` — so any activity difference is
attributable to the one parameter.

Two alternative routes are **retained but are NOT the project's seizure model**:

- **`states.kclearance_seizure_state(severity)`** — impaired glial/diffusive
  K⁺ clearance: an elevated `tau_k` (reference impaired value `12000 ms`) lets
  firing-driven [K⁺]ₒ accumulate, E_K depolarizes (Nernst), and positive
  feedback drives a genuine **high-[K⁺]ₒ ictal state** rather than the
  mild-[K⁺]ₒ bursting phenotype. Kept for comparison; it moves `tau_k`, so it
  is not a single-knob state.
- **`states.gbar_block_state`** (alias `four_ap_state`) — the reduced-A-current
  "4-AP" knob. On the realistic log-normal topology this does **not**
  faithfully reproduce seizure (the dramatic reduced-A-current effect was
  specific to the dense discrete-hub topology). Phenomenological option only.

### Two bug fixes

1. **Mis-calibrated weights → hyperexcitability.** Previously a single background
   event (`noise_weight = 0.0016 µS`) caused a ~103 mV deflection — one
   presynaptic spike was suprathreshold, so the network chain-reacted to
   near-continuous firing and swamped the A-current. The fix at the time cut the
   **noise** weight to `0.0008 µS` (single noise EPSP ~5.6 mV, subthreshold) and
   reduced the **recurrent** gain (`exc_weight_scale` 4.0 → 1.5). Both were
   later retuned upward — the registry now sits at `noise_weight = 0.004` and
   `exc_weight_scale = 2.0` — with runaway prevented by sAHP adaptation rather
   than event amplitude (see *Background noise* above). See *Recurrent coupling
   and the sharp HH threshold* above for why the recurrent EPSP remains
   suprathreshold in any bursting regime (an intrinsic property of this HH
   point-neuron).
2. **Per-recording noise seed had no effect → identical recordings.** The old
   `NetStim.seed()` path produced byte-identical recordings. Each generator now
   draws from a `Random123(base_seed, gid, recording_index)` stream via
   `noiseFromRandom`, re-keyed per recording (`noise.reseed_noise`), so every
   recording is a distinct trial. **Verified:** a 3-recording session gives
   pairwise-different spike trains.

### Roles of noise and depression

- **Noise is an ignition seed, not a driver.** Below a threshold noise level the
  network is silent; above it, recurrent excitation ignites synchronized bursts.
- **Short-term depression is the burst terminator / brake against runaway** in
  the tuned regime; turning depression off at the default recurrent strength
  drives continuous firing rather than discrete bursts.

---

## Topology options

Both builders use a clustered spatial layout with distance-dependent
connectivity (the LIF `create_clustered_network` backbone).

- **`build_topology_lognormal` — preferred / biologically defensible.** Adds a
  *continuous* per-neuron log-normal connection propensity, giving a heavy-tailed
  degree distribution with **no bimodal gap** at a realistic sparse density
  (~3–4%). This is the notebook default.
- **`build_topology` — discrete hub class.** Clustered layout plus a few densely,
  long-range-connected hub neurons. **Caveat:** this yields a **bimodal,
  unrealistically concentrated** degree distribution and a higher density. Use it
  only when you want a densely-coupled, always-fully-recruiting network.

`plotting.plot_degree_distribution` visualizes the heavy-tailed (log-normal) vs
bimodal (discrete-hub) contrast.

---

## Inference linkage

The **live pipeline is the in-repo sparse GLM**:

- **`sparse_glm.py`** — a memory-efficient sparse **lag-resolved ridge GLM**
  library (the inference core).
- **`glm_connectivity.py`** — GLM **edge prediction / typing** on top of it.
  (Extracted from the archived `inference/lif_inference/` package, where it
  was a local addition of this project — it is *not* vendored LIF code.)
- **`scripts/run_inference.py`** — the CLI driver (GLM only; the old `lif`
  subcommand was removed, and `--session` is **required**, with no default).
  The `analysis/` scripts run the same pipeline for the paper-style studies.

```bash
python scripts/run_inference.py --session "notebooks/NEURON data parallel/IC-locked_flagship_spikeonly_50rec/normal" glm --readout sum4
```

The simulator still writes each session in the **exact LIF layout**
(`neuron_simulation/io.py` replicates `save_network_structure` /
`save_recording_data`). The **ground truth is the exact wired graph** (the
`connections` table). The first ~1 s startup transient is discarded at save
time, so inference sees steady-state data. Optional downsampled voltage
recording is still saved in the LIF-compatible format.

### Historical results — the retired learned-LIF / CCG pipeline

The vendored CCG + learned-LIF pipeline (and its `adapter.py`) is **retired**
and lives at [`archive/inference/`](archive/inference/). The results below were
produced by that pipeline before retirement and are kept as scientific record;
they cannot be regenerated with the live GLM tooling.

**Verified end-to-end** on the then-deliverable session (148 neurons,
log-normal topology, **20×60 s** recordings, ~456k spikes), straight out of the
vendored pipeline against the ground-truth wiring:

| model | AUC | notes |
|-------|-----|-------|
| learned-LIF (spike-only) | **0.921** | FDR 0.47 (see the FDR note above) |
| CCG baseline (all bins) | **0.840** | FDR 0.56 |
| CCG baseline (our bursts excluded) | **0.866** | 11% of bins excluded |

Excluding the burst-dominated bins (our participation-based windows) does **not**
lower the CCG AUC — it slightly *raises* it (0.840 → 0.866) — so the CCG score is
**not** inflated by burst common-input here. (Note: the inference pipeline's own
default burst detector flags 0 windows on these short ~100 ms bursts, so its
built-in exclusion is a no-op on this data; the test above uses the project's
participation-based burst windows.)

**On the FDR (~0.6).** A high FDR alongside a good AUC does **not** mean "AUC is
insensitive to sparse spikes" — that conflates two things. It means the model
*ranks* edges well (good AUC) but the **surrogate-derived threshold is
miscalibrated**: too many false positives are admitted at the chosen cutoff. This
is the **same surrogate-FDR calibration problem already documented in the LIF
project** (where the true FDR was ~0.63 at the chosen threshold), carried over
here unchanged — not a new NEURON-specific artifact, and not "expected/fine."
It was never fixed before the pipeline was retired; it would be
threshold-calibration work in the archived inference package, independent of
the simulator.

### Session layout (raw files must match the LIF format exactly)

`analysis/session_paths.py` is the **single source of truth** for the data
tree; the analysis scripts accept `DATASET_SESSION` / `DATASET_STATE`
environment variables.

```
notebooks/NEURON data parallel/<session>/<state>/     # raw data (LIF format)
  network_*.npz               # ground-truth topology (saved once)
  recording000.npz            # per recording: spike_times (ms), resampled raster, optional voltage
  recording001.npz ...
  session_metadata.json
notebooks/NEURON data parallel/<session>/results/<state>/{glm,bursts,ic_artifact,figures,other}/
                              # analysis output, grouped by kind
```

Committed sessions: `IC-locked_flagship_spikeonly_50rec` (normal: **50 raw
spike-only recordings — the runnable dataset**),
`IC-locked_zeroedge_control_15rec` (normal: 15 raw),
`IC-locked_flagship_200rec` (**results/ only** — the raw 200-recording data was
never committed), and `dataset_noise7_random2` (results/ only).
`notebooks/NEURON data/` holds the old sequential pilot sessions.

Neuron ids are the row index `0..N-1`, consistent across the network and
recording files; spike times are in **milliseconds**. Inference-critical fields
(`connections`, `spike_times`, `neuron_positions`, `cluster_assignments`,
`resampled_*`) are never renamed — the log-normal/hub/E-I metadata is added as
*new* fields only.

---

## Honest caveats

- **Burst rate.** The "burst rate is fast" caveat that stood here has been
  **retracted** — see "Network bursts" above and `MODEL_CHARACTERIZATION.md` §C.
- **Reduced-A-current is not a faithful 4-AP model here.** The dramatic
  reduced-`gbar_kA` effect was specific to the dense discrete-hub topology; on the
  realistic log-normal topology it changes burst frequency in a
  topology-dependent, sometimes wrong-signed way. Use the slow-AHP-deficit
  (`sahp_ainc_slow`) seizure model instead; `gbar_block_state` is kept only as
  a phenomenological knob (mainly useful with the discrete-hub builder).
- **Squid HH kinetics.** The default 6.3 °C `hh` is the classic squid model, not
  mammalian cortex. The 34 °C variant speeds kinetics but is still a caricature.
- **`kdyn` is a lumped caricature.** [K⁺]ₒ is a single well-mixed pool per soma
  with a lumped efflux coupling and fixed `ki` — enough to reproduce the
  accumulation → depolarization → runaway loop, not a spatially-resolved ion
  model.
- **The A-current is a compact caricature**, not a fit to a specific Kv4 channel;
  parameters were tuned for network bursting, not channel realism.

---

## Attribution

The vendored learned-LIF/CCG inference package came from the LIF project and is
now **archived** at `archive/inference/lif_inference/` (see its `SOURCE.md`).
The saved data format still follows the LIF project's layout
(`neuron_simulation/io.py` replicates it), so LIF-side tooling continues to
read this repository's sessions unmodified.
