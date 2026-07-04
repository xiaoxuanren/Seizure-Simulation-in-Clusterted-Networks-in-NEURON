# Seizure Simulation in Clustered Networks (NEURON)

A biophysical spiking-network model of spontaneous **network bursts** (a
seizure-like synchronization) in a clustered culture, built in
[NEURON](https://neuron.yale.edu/). Single-compartment Hodgkin–Huxley neurons
carry a custom **A-type / Kv4-like potassium current** whose density `gbar_kA`
is the pharmacological knob for **4-aminopyridine (4-AP)**. The network is driven
only by weak per-neuron Poisson background input and produces discrete,
high-participation network bursts whose frequency rises as the A-current is
partially blocked.

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
    kA.mod              # A-type K+ current (Kv4-like); the 4-AP target
    DepSyn.mod          # depressing excitatory synapse (short-term depression)
    README.md           # how to compile: `nrnivmodl mechanisms`
  neurons.py            # single-compartment HH + A-current cell builders (E and I)
  topology.py           # clustered+hub AND log-normal-degree builders
  network_builder.py    # assemble a NEURON net from a topology (cells, synapses, NetCons, noise)
  noise.py              # Poisson background input (NetStim -> ExpSyn) -- the sole drive
  states.py             # normal vs 4-AP configs (gbar_kA) + dose-response helper
  simulation.py         # run(): finitialize/continuerun, spike (+ optional voltage) recording
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
result = workflows.run_single_state(                       # verify bursting
    topo, state=states.normal_state(),
    build_kwargs=dict(noise_weight=0.0006, noise_rate=20.0, exc_weight_scale=3.0),
)
print(result["burst_stats"])                               # network-burst rate, participation

# Generate an inference-ready dataset, then run inference:
meta, session_dir = workflows.generate_dataset(
    n_recordings=5, recording_duration=30000,
    build_kwargs=dict(noise_weight=0.0006, noise_rate=20.0, exc_weight_scale=3.0),
)
```

```bash
python inference/adapter.py latest        # CCG + learned-LIF, reports AUC / FDR
```

---

## The biophysics

- **Neurons.** Single-compartment soma (`L = diam = 20 µm`) with NEURON's
  built-in `hh` (Na⁺/K⁺/leak) plus the custom `kA` A-current. 80% excitatory,
  20% inhibitory. `celsius` is configurable (default **6.3 °C**, keeping squid
  `hh` kinetics). A **mammalian variant at 34 °C** runs automatically with faster
  kinetics — the `kA` mechanism scales its time constants by a q10 factor.

- **A-current (`kA.mod`).** Fast voltage-gated activation `m` and **slow**
  inactivation `h`, reversal at `ek`; `I_kA = gbar · m⁴ · h · (v − ek)`.
  **`gbar_kA` is the 4-AP knob.** Normal ≈ **0.006 S/cm²**. 4-AP is a **partial**
  reduction of `gbar_kA`.

- **Synapses.** Excitatory synapses are `DepSyn` (short-term depression,
  `d = 0.5`, `tau_d = 800 ms`) or static (`d = 0`). Inhibitory synapses are
  `ExpSyn` with reversal −75 mV. Dale's law is enforced: a neuron's E/I identity
  fixes the sign of every synapse it makes.

- **Drive.** Each neuron receives an independent Poisson `NetStim → ExpSyn`
  background (~16–22 Hz). **This is the only drive** — no current injection, no
  stimulation, no tonic drivers.

### The 4-AP mapping (validated)

`gbar_kA` sets the strength of the burst brake:

- **Normal** (`gbar_kA ≈ 0.006`): discrete, well-separated network bursts
  (verified regime: ~2–5 Hz burst rate, ~90% participation, depending on network
  size).
- **4-AP, partial block** (reduce `gbar_kA` toward ~0.0045–0.005): the brake
  weakens, so **burst frequency rises** — there is a dose window where the burst
  rate increases with block strength.
- **Strong block** (`gbar_kA` → 0): the terminator is gone and discrete bursts
  **collapse into continuous firing**. 4-AP must stay in the partial regime.

See `states.py` (`normal_state`, `four_ap_state`, `dose_response_gbar`).

### Roles of noise, the A-current, and depression

- **Noise is an ignition seed, not a driver.** Alone (recurrent synapses
  removed) it makes each neuron fire only ~0.07 Hz. Below a threshold noise level
  the network is silent; above it, recurrent excitation ignites synchronized
  bursts.
- **The A-current (`gbar_kA`) controls the ignition threshold and burst
  frequency** — it is the 4-AP knob. Reducing it raises the burst rate (see the
  dose-response above).
- **Short-term depression is the burst terminator / brake against runaway.** In
  this implementation's tuned regime it is *required* to keep bursts discrete:
  with the default recurrent strength, turning depression **off** (static
  synapses) drives the network into **continuous ~75 Hz firing** rather than
  discrete bursts. This is exactly the "backup brake against runaway" role the
  spec calls out for the weakened-A-current (4-AP) state — here it is load-bearing
  at baseline too.

  > **Honest note / deviation from the LIF-based spec.** The reference finding
  > was that the network *still bursts with static synapses* (A-current as the
  > sole terminator). That was validated for a different, hand-provided `kA.mod`
  > and a weaker-coupling operating point. The A-current mechanism here was
  > written from the spec (the reference `.mod` files were not supplied) and
  > tuned for robust bursting *with* depression; at that operating point the
  > A-current alone does not terminate bursts. A static-synapse bursting regime
  > is reachable at lower `exc_weight_scale`, but is not the default.

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

**Verified end-to-end** on a generated session (148 neurons, log-normal
topology, 2×12 s recordings): the learned-LIF model reached **AUC ≈ 0.85** and
the CCG baseline **AUC ≈ 0.84** against the ground-truth wiring, straight out of
the vendored pipeline. (FDR at the chosen threshold is high, ~0.4–0.5, because
the fast bursting leaves sparse inter-burst spikes — AUC, which is
threshold-free, is the more meaningful score here.)

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

- **Burst rate is fast.** The verified regime bursts at ~2–5 Hz, far faster than
  the ~0.03 Hz (tens of seconds between bursts) seen in real dissociated
  cultures. Slowing it toward culture-realistic spacing would require stronger
  adaptation/depression recovery and weaker drive; the fast regime is convenient
  for generating many bursts quickly for inference. A side effect of the fast
  bursting is that inter-burst spiking is sparse, which makes the monosynaptic
  connectivity signal (what inference reads between bursts) weaker than in the
  slower LIF cultures.
- **Squid HH kinetics.** The default 6.3 °C `hh` is the classic squid model, not
  mammalian cortex. The 34 °C variant speeds kinetics but is still a caricature.
- **Sparse topology gives a weaker/messier 4-AP effect.** The realistic sparse
  log-normal network produces a less clean, noisier 4-AP dose response than a
  dense discrete-hub network, where near-full recruitment makes the effect
  crisp. Use the discrete-hub builder if you want the sharpest 4-AP contrast.
- **The A-current here is a compact caricature**, not a fit to a specific Kv4
  channel; parameters were tuned for network bursting, not channel realism.

---

## Attribution

The `inference/lif_inference/` package and the saved data format are from the
LIF project (see `inference/lif_inference/SOURCE.md`). The LIF project remains
the source of truth for the inference code; it is vendored here so this
repository is self-contained.
