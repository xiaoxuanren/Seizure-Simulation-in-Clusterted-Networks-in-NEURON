# NMODL mechanisms

Two custom NEURON mechanisms back the biophysics of this project:

| File | Mechanism | Role |
|------|-----------|------|
| `kA.mod` | `kA` (density) | A-type / Kv4-like transient K⁺ current. **`gbar_kA` is the 4-AP knob.** |
| `DepSyn.mod` | `DepSyn` (point process) | Depressing excitatory synapse (short-term depression; `d=0` ⇒ static). |

Inhibitory synapses use NEURON's built-in `ExpSyn` (reversal −75 mV) and the
soma uses the built-in `hh` mechanism, so only these two files need compiling.

## Compile

Compile **once** before running anything. From the `neuron_simulation/`
directory:

```bash
nrnivmodl mechanisms
```

- **Linux / macOS** — produces an architecture sub-directory (e.g. `x86_64/`,
  `arm64/`) next to where you ran the command. NEURON auto-discovers it when you
  launch from that directory, or load it explicitly via
  `neuron_simulation.neurons.load_mechanisms()`.
- **Windows** — produces `nrnmech.dll`. NEURON auto-loads a `nrnmech.dll` found
  in the current working directory; `load_mechanisms()` also finds and loads it
  from the package directory if you run from elsewhere.

`neuron_simulation.neurons.load_mechanisms()` is called automatically by
`build_network(...)`; it guards against double-loading, so you rarely call it
yourself.

## Recompiling after edits

If you edit a `.mod` file, re-run `nrnivmodl mechanisms`. Delete stale build
artifacts first if the rebuild misbehaves:

```bash
# Linux/macOS
rm -rf x86_64 arm64
# Windows
rm -f nrnmech.dll mechanisms/*.c mechanisms/*.o mechanisms/mod_func.*
```

The generated build artifacts (`x86_64/`, `nrnmech.dll`, `*.c`, `*.o`) are
git-ignored — only the `.mod` source is tracked.

## The 4-AP mapping (important)

`gbar_kA` is the A-current density in S/cm². 4-aminopyridine (4-AP) blocks
A-type K⁺ channels, so 4-AP is modelled as a **partial reduction** of `gbar_kA`:

- Normal (drug-free): `gbar_kA ≈ 0.006` S/cm² → discrete network bursts.
- 4-AP (partial block): reduce toward ~`0.0045–0.005` S/cm² → bursts become
  **more frequent** (there is a dose window where burst frequency rises).
- **Do not** drop `gbar_kA` near zero: a strong block removes the burst
  terminator and collapses discrete bursts into continuous firing.

See `neuron_simulation/states.py` for the `normal_state` / `four_ap_state`
helpers and the dose-response sweep.
