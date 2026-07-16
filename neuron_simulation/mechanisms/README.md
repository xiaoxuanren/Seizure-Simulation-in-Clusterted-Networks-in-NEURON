# NMODL mechanisms

Three custom NEURON mechanisms back the biophysics of this project:

| File | Mechanism | Role |
|------|-----------|------|
| `kA.mod` | `kA` (density) | A-type / Kv4-like transient K⁺ current (`htau0 = 20 ms`). **Inert at its shipped parameters** — shapes nothing; `gbar` is a dead parameter. |
| `kdyn.mod` | `kdyn` (density) | Dynamic [K⁺]ₒ accumulation → writes `ek` (Nernst). **`tau_k` is the seizure knob.** |
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

## The seizure knob is `tau_k` (K⁺ clearance), not `gbar_kA`

Seizure is modelled with `kdyn`: firing raises [K⁺]ₒ → `ek` depolarizes (Nernst)
→ positive feedback. Glial/diffusive clearance (`tau_k`) is the negative feedback.

- Normal: `tau_k = 200 ms` (strong buffering) → [K⁺]ₒ ~4 mM → discrete bursts.
- Seizure: `tau_k = 2500 ms` (impaired buffering) → [K⁺]ₒ climbs to ~12 mM →
  ictal runaway. `ki = 72 mM` fixes resting E_K = −77 mV.

`gbar_kA` (A-current density, S/cm²) is a **dead parameter**, not a 4-AP model:
the `kA` mechanism is **inert at its shipped parameters**, so reducing `gbar_kA`
does nothing at any dose, on any topology. `states.gbar_block_state` is retained
for API compatibility only. See the inertness note in `kA.mod`, the README section
"The A-current is inert", and `tests/test_kA_characterization.py`.

See `neuron_simulation/states.py` for `normal_state` / `seizure_state` /
`seizure_dose_response` (and the inert `gbar_block_state`).
