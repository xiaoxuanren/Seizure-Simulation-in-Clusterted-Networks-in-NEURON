# NMODL mechanisms

Five custom NEURON mechanisms back the biophysics of this project:

| File | Mechanism | Role |
|------|-----------|------|
| `kA.mod` | `kA` (density) | A-type / Kv4-like transient K⁺ current (`htau0 = 20 ms`); shapes crisp bursts. |
| `kdyn.mod` | `kdyn` (density) | Dynamic [K⁺]ₒ accumulation → writes `ek` (Nernst). `tau_k` is the retained *alternative* ictal route, **not** the project's seizure knob (see below). |
| `DepSyn.mod` | `DepSyn` (point process) | Depressing excitatory synapse (Tsodyks–Markram short-term depression; `d=0` ⇒ static). Fallback when `synapse_model != "ampa_nmda"`. |
| `AmpaNmda.mod` | `AmpaNmda` (point process) | Two-component (fast AMPA + slow voltage-dependent NMDA) depressing excitatory synapse — the **default** excitatory synapse (`synapse_model="ampa_nmda"` in `network_builder.py`); the NMDA component sustains realistic 200–800 ms bursts. |
| `sAHP.mod` | `sAHP` (point process) | Two-timescale spike-triggered adaptation: fast SFA (M-current/Kv7-like) + slow Ca²⁺-dependent AHP. **The slow component's per-spike increment `sahp_ainc_slow` is the project's seizure knob.** |

Inhibitory synapses use NEURON's built-in `ExpSyn` (reversal −75 mV) and the
soma uses the built-in `hh` mechanism, so these five files are all that need
compiling.

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

## The seizure knob is `sahp_ainc_slow` (in `sAHP.mod`), not `tau_k` or `gbar_kA`

The project's seizure mechanism is a single knob: the Ca²⁺-dependent slow-AHP
per-spike increment `sahp_ainc_slow` in `sAHP.mod` (a KCa conductance, **not**
the M-current/Kv7 — that is the fast component `sahp_ainc_fast`, held fixed).

- Normal: `sahp_ainc_slow = 0.01` (strong slow adaptation; pinned — the shipped
  flagship session was generated with it).
- Seizure: any **lower** value (weak slow adaptation → more firing, denser
  bursts; the default 0.004 is a convenience, not a commitment). This models an
  acquired-epilepsy sAHP deficit — the adaptation-deficit, mild-[K⁺]ₒ phenotype.

`kdyn` still supplies the [K⁺]ₒ substrate (firing raises [K⁺]ₒ → `ek`
depolarizes (Nernst) → positive feedback; clearance `tau_k` is the negative
feedback; `ki = 72 mM` fixes resting E_K = −77 mV), but `tau_k = 200 ms` is
**held fixed** across states. The impaired-K⁺-clearance (`tau_k`) route
survives only as `states.kclearance_seizure_state` — the alternative
high-[K⁺]ₒ ictal phenotype, not the project's model.

`gbar_kA` (A-current density, S/cm²) is retained only as a **phenomenological**
knob (`states.gbar_block_state`), **not** a faithful 4-AP model: on the realistic
log-normal topology, reducing it does not reproduce the seizure phenotype.

See `neuron_simulation/states.py` for `normal_state` / `seizure_state`
(and the retained alternatives `kclearance_seizure_state` / `gbar_block_state`).
