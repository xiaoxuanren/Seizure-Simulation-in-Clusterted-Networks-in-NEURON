# Seizure Simulation in Clustered Networks (NEURON)

A biophysical spiking-network model of spontaneous **network bursts** and their
transition into a **seizure (ictal) state**, in a clustered culture built in
[NEURON](https://neuron.yale.edu/). Single-compartment Hodgkin–Huxley neurons
carry a custom **A-type / Kv4-like potassium current** (`kA`) and a **dynamic
extracellular-potassium** mechanism (`kdyn`). The network is driven only by weak
per-neuron Poisson background input and produces discrete, high-participation
network bursts.

**The seizure knob is impaired glial K⁺ clearance (`tau_k`), not a reduced
A-current.** Firing raises [K⁺]ₒ → the K⁺ reversal E_K depolarizes (Nernst) →
positive feedback toward an ictal state; glial/diffusive clearance is the
negative feedback that terminates it. Weak clearance (large `tau_k`) is the
epilepsy model. The old reduced-`gbar_kA` "4-AP" route is **deprecated to a
phenomenological knob** (see below).

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
    kA.mod              # A-type K+ current (Kv4-like), fast htau (20 ms)
    kdyn.mod            # dynamic [K+]o accumulation -> ek (the SEIZURE substrate)
    DepSyn.mod          # depressing excitatory synapse (short-term depression)
    README.md           # how to compile: `nrnivmodl mechanisms`
  neurons.py            # single-compartment HH + kA + kdyn cell builders (E and I)
  topology.py           # clustered+hub AND log-normal-degree builders
  network_builder.py    # assemble a NEURON net from a topology (cells, synapses, NetCons, noise)
  noise.py              # Poisson background (NetStim -> ExpSyn); per-recording Random123 streams
  states.py             # normal vs seizure (tau_k) states; deprecated gbar_block knob
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
pip install neuron numpy scipy matplotlib scikit-learn h5py torch jupyter
```

> On Windows, NEURON is typically installed from the official installer; use the
> Python interpreter that matches the bundled `hoc` extension (this project was
> validated with NEURON 8.0 + Python 3.9).

### 2. Compile the mechanisms (once)

The custom `kA` and `DepSyn` mechanisms **must be compiled before running**:

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
state**, compares normal vs 4-AP, traces a dose-response curve, saves a dataset,
and finally runs inference on the generated data.

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
  (`htau0 = 20 ms`), reversal at `ek`; `I_kA = gbar · m⁴ · h · (v − ek)`. It
  shapes crisp discrete bursts. `gbar_kA` is retained only as a phenomenological
  knob (`gbar_block_state`), **not** a faithful 4-AP model — see below.

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
| `htau0_kA` | `20 ms` | fast A-current inactivation (crisp bursts) |
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
- **Deprecated `gbar_kA` route** — `states.gbar_block_state` (alias
  `four_ap_state`) still reduces the A-current, but on the realistic log-normal
  topology this does **not** faithfully reproduce seizure (the dramatic
  reduced-A-current effect was specific to the dense discrete-hub topology). Kept
  as a phenomenological option only.

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
topology, 20×60 s recordings): the learned-LIF and CCG models run against NEURON
output straight out of the vendored pipeline, scoring AUC well above the 0.5
chance level. See PR #1 for the exact learned-LIF and CCG AUCs, including the CCG
baseline **with and without burst exclusion** (burst-dominated recordings are
where CCG can inflate from common-input confounds).

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
- **Reduced-A-current is not a faithful 4-AP model here.** The dramatic
  reduced-`gbar_kA` effect was specific to the dense discrete-hub topology; on the
  realistic log-normal topology it changes burst frequency in a
  topology-dependent, sometimes wrong-signed way. Use the K⁺-clearance
  (`tau_k`) seizure model instead; `gbar_block_state` is kept only as a
  phenomenological knob (mainly useful with the discrete-hub builder).
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

The `inference/lif_inference/` package and the saved data format are from the
LIF project (see `inference/lif_inference/SOURCE.md`). The LIF project remains
the source of truth for the inference code; it is vendored here so this
repository is self-contained.
