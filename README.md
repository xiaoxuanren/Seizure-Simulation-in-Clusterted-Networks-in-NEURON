# Seizure Simulation in Clustered Networks (NEURON)

A biophysical spiking-network model of spontaneous **network bursts** and their
transition into a **seizure (ictal) state**, in a clustered culture built in
[NEURON](https://neuron.yale.edu/). Single-compartment Hodgkin–Huxley neurons
carry a **dynamic extracellular-potassium** mechanism (`kdyn`) and two-timescale
spike-triggered adaptation (`sAHP`). A custom **A-type / Kv4-like potassium
current** (`kA`) is present but **inert at its shipped parameters** — see
[The A-current is inert](#the-a-current-is-inert-and-gbar_ka-is-a-dead-parameter).
The network is driven only by weak per-neuron Poisson background input and
produces discrete, high-participation network bursts.

**The seizure knob is impaired glial K⁺ clearance (`tau_k`), not a reduced
A-current.** Firing raises [K⁺]ₒ → the K⁺ reversal E_K depolarizes (Nernst) →
positive feedback toward an ictal state; glial/diffusive clearance is the
negative feedback that terminates it. Weak clearance (large `tau_k`) is the
epilepsy model. The old reduced-`gbar_kA` "4-AP" route is **non-functional**: the
A-current is inert at its shipped parameters, so blocking it does nothing at any
dose (see below).

The project deliberately mirrors the companion **LIF project**
([LIF-Project](https://github.com/xiaoxuanren/LIF-Project),
branch `chore/repo-cleanup`) in structure, naming, and — crucially — its saved
**data format**, so the same connectivity-inference pipeline (CCG baseline +
learned-LIF, spike-only and voltage-augmented) runs against NEURON output
unmodified.

---

## Repository layout

```
neuron_simulation/
  mechanisms/
    kA.mod              # A-type K+ current (Kv4-like) -- INERT at shipped params
    kdyn.mod            # dynamic [K+]o accumulation -> ek (the SEIZURE substrate)
    DepSyn.mod          # depressing excitatory synapse (short-term depression)
    README.md           # how to compile: `nrnivmodl mechanisms`
  neurons.py            # single-compartment HH + kA + kdyn cell builders (E and I)
  topology.py           # clustered+hub AND log-normal-degree builders
  network_builder.py    # assemble a NEURON net from a topology (cells, synapses, NetCons, noise)
  noise.py              # Poisson background (NetStim -> ExpSyn); per-recording Random123 streams
  states.py             # normal vs seizure (tau_k) states; inert gbar_block knob
  simulation.py         # run(): finitialize/continuerun, spike + optional voltage + [K+]o recording
  workflows.py          # topology->network->run->save spikes + ground truth; dataset generation
  analysis.py           # participation-based network-burst detection (post burn-in), burst stats
  plotting.py           # raster, population activity, degree distribution, topology map, comparisons
  io.py                 # save/load spikes, ground-truth connectivity, metadata (LIF-format)
inference/
  adapter.py            # load NEURON output -> run vendored CCG + learned-LIF -> AUC/FDR
  lif_inference/        # VENDORED copy of the LIF inference package (see SOURCE.md)
notebooks/
  neuron_network_simulation.ipynb
README.md
```

---

## Quick start

### 1. Environment

NEURON, NumPy, SciPy, Matplotlib, scikit-learn, h5py, and (for the learned-LIF
inference) PyTorch:

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

### 3. Run the notebook

Open [`notebooks/neuron_network_simulation.ipynb`](notebooks/neuron_network_simulation.ipynb).
It builds a log-normal topology, **verifies the network bursts in the normal
state**, compares normal vs seizure across the single `sahp_ainc_slow` knob,
sweeps that knob, saves a dataset from the same wiring, and finally runs
inference on the generated data.

### 4. Or from Python

```python
from neuron_simulation import topology, workflows, states

topo = topology.build_topology_lognormal(seed=1)          # preferred builder

# Normal vs seizure (the tuned defaults already give clean bursts):
normal  = workflows.run_single_state(topo, state=states.normal_state())
seizure = workflows.run_single_state(topo, state=states.seizure_state(1.0))
print(normal["burst_stats"], normal["ko_data"]["mean_ko"].max())    # ~4 mM, discrete bursts
print(seizure["ko_data"]["mean_ko"].max())                          # ~12 mM, ictal

# Generate an inference-ready dataset (normal state), then run inference:
meta, session_dir = workflows.generate_dataset(n_recordings=3, recording_duration=15000)
```

```bash
python inference/adapter.py latest        # CCG + learned-LIF, reports AUC / FDR
```

---

## The biophysics

- **Neurons.** Single-compartment soma (`L = diam = 20 µm`) with NEURON's
  built-in `hh` (Na⁺/K⁺/leak), the custom `kA` A-current, and the `kdyn` dynamic
  [K⁺]ₒ mechanism. 80% excitatory, 20% inhibitory. `celsius` is configurable
  (default **6.3 °C**, keeping squid `hh` kinetics); a mammalian variant at 34 °C
  runs with faster `kA` kinetics via a q10 factor.

- **A-current (`kA.mod`).** Fast activation `m` and inactivation `h`
  (`htau0 = 20 ms`), reversal at `ek`; `I_kA = gbar · m⁴ · h · (v − ek)`. **It is
  inert at its shipped parameters and shapes nothing.** The `m⁴` exponent puts the
  conductance V½ at `vhalfm + 1.665·km` = **−0.4 mV** (`vhalfm = -27`, `km = 16`),
  where `h` is fully inactivated: peak steady-state window conductance is
  `1.88e-6 S/cm²` (**0.005%** of the `hh` gK) and over −65…−50 mV `g_kA` is
  **under 2%** of the leak conductance. `gbar_kA` is a **dead parameter** — reducing it does
  nothing at any dose, on any topology. See
  [The A-current is inert](#the-a-current-is-inert-and-gbar_ka-is-a-dead-parameter).

- **Dynamic [K⁺]ₒ (`kdyn.mod`).** `ek` is written from the Nernst equation on a
  state variable `[K⁺]ₒ` that rises with K⁺ efflux (firing) and is cleared with
  time constant `tau_k`. `ki = 72 mM` fixes resting E_K = −77 mV. **`tau_k` is
  the seizure knob.**

- **Synapses.** Excitatory synapses are `DepSyn` (short-term depression,
  `d = 0.5`, `tau_d = 800 ms`) or static (`d = 0`). Inhibitory synapses are
  `ExpSyn` with reversal −75 mV. Dale's law is enforced: a neuron's E/I identity
  fixes the sign of every synapse it makes.

- **Drive.** Each neuron receives an independent Poisson `NetStim → ExpSyn`
  background. **This is the only drive** — no current injection, no stimulation,
  no tonic drivers. Each generator uses its own reproducible `Random123` stream.

### Tuned defaults (validated)

The `build_network` defaults keep the **single background (noise) event
subthreshold** so the network integrates rather than chain-reacting from noise:

| parameter | default | role |
|-----------|---------|------|
| `noise_weight` | `0.0008` µS | single noise EPSP ≈ 5.6 mV (subthreshold) |
| `noise_rate` | `2.5` Hz | sparse ignition seed |
| `exc_weight_scale` | `1.5` | recurrent gain |
| `inh_weight_scale` | `1.5` | recurrent inhibition |
| `htau0_kA` | `20 ms` | A-current inactivation — **no effect** (`kA` is inert) |
| `tau_k` | `200 ms` (normal) | K⁺ clearance (seizure knob) |

**Verified normal state:** mean rate **2.6 Hz**, network bursts (**93%
participation**) at **1.5 Hz**, ~**77% of spikes in bursts**, [K⁺]ₒ ≈ 4.0–4.2 mM.

### Recurrent coupling and the sharp HH threshold (honest caveat)

The single-compartment HH point-neuron has a **razor-sharp single-event
rheobase** (~`0.00085 µS` ≈ a 6 mV peak EPSP): a synaptic event is either
**≤ 5.6 mV (subthreshold)** or triggers a **full ~101 mV spike** — there is no
"moderately suprathreshold" middle. Consequently:

- A single **noise** event (`0.0008 µS`) is subthreshold (~5.6 mV) — noise
  integrates, it does not detonate.
- A single **recurrent** excitatory event, at any weight strong enough to sustain
  network bursts, is **suprathreshold** (fires the postsynaptic cell). We verified
  by sweep (density 0.04–0.12, `exc_tau` 3–6 ms, `noise_rate` 2–30 Hz, `tau_d`
  250–800 ms) that a **genuinely subthreshold recurrent weight set does NOT
  sustain bursts** — the network is silent at low noise and asynchronously tonic
  at high noise. So **subthreshold recurrent coupling and the burst gate are
  mutually exclusive** in this cell.

The recurrent weights were nonetheless **reduced** from the earlier calibration
(`exc_weight_scale` 4.0 → 1.5) to the minimum that still bursts. Making them fully
subthreshold would require a better-integrating cell (e.g. a multi-compartment
neuron), which is out of scope here.

### The seizure mechanism (K⁺ accumulation)

Seizure is modelled by **impaired glial/diffusive K⁺ clearance**, not a reduced
A-current:

- **Normal** — `tau_k = 200 ms` (strong buffering). [K⁺]ₒ stays ~4 mM; the
  network produces discrete bursts. `states.normal_state()`.
- **Seizure** — large `tau_k` (e.g. `2500 ms`, impaired buffering). Firing-driven
  [K⁺]ₒ accumulates, E_K depolarizes, and positive feedback drives an ictal
  state. **Verified:** [K⁺]ₒ rises to **~12–14 mM**, firing ~**8 Hz**, discrete
  bursts merge into sustained activity. `states.seizure_state(severity)`;
  `states.seizure_dose_response()` sweeps `tau_k`.
- **Non-functional `gbar_kA` route** — `states.gbar_block_state` (alias
  `four_ap_state`) still reduces the `gbar_kA` *number*, but the A-current is inert
  at its shipped parameters, so nothing changes at any dose. Kept for API
  compatibility only. See
  [The A-current is inert](#the-a-current-is-inert-and-gbar_ka-is-a-dead-parameter).

### The A-current is inert (and `gbar_kA` is a dead parameter)

`kA.mod` computes `g = gbar · m⁴ · h`. The `m⁴` exponent means the **conductance**
half-activates at `vhalfm + 1.665·km` — **26.6 mV depolarized of the m-gate** (`km = 16`).
The shipped `vhalfm = -27 mV` therefore puts the conductance V½ at **−0.4 mV**, where the
inactivation gate `h` (V½ = −60 mV, `kh = 6`) is fully closed.

| quantity | value |
|---|---|
| peak steady-state window conductance | `1.88e-6 S/cm²` = **0.005%** of `hh` gK |
| `g_kA` at −65…−50 mV (subthreshold) | **<2%** of leak (`3e-4 S/cm²`) |
| effect of reducing `gbar_kA` at any dose | **none** on the burst phenotype (rate, IBI, participation) |

**It is not, however, bitwise neutral — do not "clean up" `gbar_kA` to zero.**
Inertness is a *subthreshold* statement; `m⁴` is still non-negligible at the spike
peak, so zeroing `gbar_kA` perturbs spike waveforms, and this chaotic recurrent
network amplifies that into a different spike train. Measured by
[`scripts/check_ka_contribution.py`](scripts/check_ka_contribution.py), which runs
the notebook's network twice changing only `gbar_kA`: the arms are bit-identical
until **t = 2175.5 ms** and then diverge, and by 20 s **all 926** spike trains
differ (6180 vs 4442 spikes) — while the burst statistics stay put (mean IBI
7062 → 7077 ms, participation 0.92 vs 0.95). So the *phenotype* is untouched, the
*bits* are not, and the shipped values are retained so existing datasets reproduce.

This has been true since the initial commit: `git log` shows `vhalfm` has never held
another value, and `build_network(kA_globals=...)` has no call sites. **The A-current has
been inert on every topology this repo has ever run.** A previously-documented claim that
"the dramatic reduced-A-current effect was specific to the dense discrete-hub topology"
was therefore unsupported and has been removed — an inert mechanism cannot produce a
topology-dependent effect. If such an effect was observed, it came from the companion
[LIF-Project](https://github.com/xiaoxuanren/LIF-Project)'s separate A-current
implementation, not from this `kA.mod`. Pinned by `tests/test_kA_characterization.py`.

**Why this is not simply patched.** Setting `vhalfm = -54` restores the documented
behaviour (conductance V½ = −27.4 mV) and yields a functional current at the single-cell
level — under 0.1 nA injection, control fires 1 spike vs 33 with the current fully
blocked, i.e. it gates rheobase as an A-current should. But it still produces **no network
dose-response** (mean rate 0.298 / 0.317 / 0.296 Hz at 0 / 50 / 100% block — within
seed-to-seed noise), because the `sAHP` per-spike increment is **1.2–2.6× the leak
conductance** and decays over 4–6.5 s, dominating the adaptation budget on the seconds
timescale that sets burst rate. A functional `gbar_kA` knob requires re-balancing `sAHP`
against `I_A`, not just a gating fix — and the gating fix alone shifts the tuned baseline
(control burst rate 0.182 → 0.136 Hz), forcing a re-tune and dataset regeneration. The
patch is tracked separately and is **not** applied on this branch.

Independently: 4-AP's epileptogenic action in slice is thought to be substantially
**presynaptic** (AP broadening → enhanced transmitter release). This model has point
neurons with event-driven synapses and structurally cannot reproduce that, so a faithful
4-AP model would need synaptic weights scaled alongside `gbar_kA` regardless.

### Two bug fixes

1. **Mis-calibrated weights → hyperexcitability.** Previously a single background
   event (`noise_weight = 0.0016 µS`) caused a ~103 mV deflection — one
   presynaptic spike was suprathreshold, so the network chain-reacted to
   near-continuous firing and swamped the A-current. The **noise** weight is now
   `0.0008 µS` (single noise EPSP ~5.6 mV, subthreshold), and the **recurrent**
   gain was reduced (`exc_weight_scale` 4.0 → 1.5). See *Recurrent coupling and
   the sharp HH threshold* above for why the recurrent EPSP remains suprathreshold
   in any bursting regime (an intrinsic property of this HH point-neuron).
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

The simulator writes each session in the **exact LIF layout**
(`neuron_simulation/io.py` replicates `save_network_structure` /
`save_recording_data`), so the vendored inference package
(`inference/lif_inference/`, a verbatim copy of the LIF project's
`lif_inference`) consumes it unmodified. `inference/adapter.py`:

1. **validates** the saved format (fields, shapes, dtypes),
2. runs the **spike-only learned-LIF** model,
3. runs the **training-free CCG baseline** on the same surfaced inputs, and
4. reports **AUC** and **FDR** against the ground-truth wiring.

Optional downsampled voltage recording enables the **voltage-augmented**
inference mode. The **ground truth is the exact wired graph** (the `connections`
table). The first ~1 s startup transient is discarded at save time, so inference
sees steady-state data.

**Verified end-to-end** on the deliverable session (148 neurons, log-normal
topology, **20×60 s** recordings, ~456k spikes), straight out of the vendored
pipeline against the ground-truth wiring:

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
is the **same, still-open surrogate-FDR calibration problem already documented in
the LIF project** (where the true FDR was ~0.63 at the chosen threshold), carried
over here unchanged — not a new NEURON-specific artifact, and not "expected/fine."
Fixing it is threshold-calibration work in the inference package, independent of
the simulator.

### Session layout (must match the LIF pipeline exactly)

```
<save_dir>/<timestamp>/
  network_<timestamp>.npz     # ground-truth topology (saved once)
  recording000.npz            # per recording: spike_times (ms), resampled raster, optional voltage
  recording001.npz ...
  recording000_voltage.h5     # optional external voltage sidecar
  session_metadata.json
```

Neuron ids are the row index `0..N-1`, consistent across the network and
recording files; spike times are in **milliseconds**. Inference-critical fields
(`connections`, `spike_times`, `neuron_positions`, `cluster_assignments`,
`resampled_*`) are never renamed — the log-normal/hub/E-I metadata is added as
*new* fields only.

---

## Honest caveats

- **Burst rate is fast.** The verified normal regime bursts at ~1.3 Hz, far
  faster than the ~0.03 Hz (tens of seconds between bursts) seen in real
  dissociated cultures. The fast regime is convenient for generating many bursts
  quickly for inference; slowing it toward culture-realistic spacing would need
  weaker drive and stronger slow adaptation.
- **Reduced-A-current is not a 4-AP model here — it does nothing at all.** The
  A-current is inert at its shipped parameters, so `gbar_kA` is a dead parameter on
  every topology. Use the K⁺-clearance (`tau_k`) seizure model instead;
  `gbar_block_state` is kept for API compatibility only. See
  [The A-current is inert](#the-a-current-is-inert-and-gbar_ka-is-a-dead-parameter).
- **Squid HH kinetics.** The default 6.3 °C `hh` is the classic squid model, not
  mammalian cortex. The 34 °C variant speeds kinetics but is still a caricature.
- **`kdyn` is a lumped caricature.** [K⁺]ₒ is a single well-mixed pool per soma
  with a lumped efflux coupling and fixed `ki` — enough to reproduce the
  accumulation → depolarization → runaway loop, not a spatially-resolved ion
  model.
- **The A-current is a compact caricature**, not a fit to a specific Kv4 channel —
  and at its shipped gating it is inert, so it contributes nothing to the tuned
  bursting regime.

---

## Attribution

The `inference/lif_inference/` package and the saved data format are from the
LIF project (see `inference/lif_inference/SOURCE.md`). The LIF project remains
the source of truth for the inference code; it is vendored here so this
repository is self-contained.
