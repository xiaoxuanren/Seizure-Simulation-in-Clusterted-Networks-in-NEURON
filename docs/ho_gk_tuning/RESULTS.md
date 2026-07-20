# Ho gK cells — Stage-A reference tuning

Branch `sim/ho-gk-cells`. Goal: a non-epileptic **reference** operating point for the
Ho et al. 2025 conductance cells at **normal gK** (PY 15 / FS 10, `tau_k=200`), scored
by **synchrony S**, not by firing rate.

Key reframing (from tuning): these single-compartment Ho cells are **mean-driven, not
fluctuation-driven** — background noise is a dead knob — so the reference is defined by
**low synchrony S ≈ 0**, not by a target rate or high CV.

## Locked reference config
| Parameter | Value | Note |
|---|---|---|
| `dt` | **0.025 ms** | Ho mechanisms diverge at 0.05 (|Vm|→1e4 mV) |
| `gK_exc` / `gK_inh` | 15 / 10 | normal (reference) |
| `tau_k` | 200 ms | K+ clearance held healthy |
| `ikpumpmax` PY / FS | **20 / 8** | FS lowered from 30 — at 30 FS is unfireable |
| `kdyn_epsilon` | **0.002** | 0.06 let baseline [K+]o run away under drive |
| `iext_exc` / `iext_inh` | **2 / 0** | operating point (FS is EPSP-driven) |
| `iext_sigma` | **2** | per-cell iext heterogeneity — needed |
| `exc_weight_scale` / `inh_weight_scale` | 2.0 / 2.5 | recurrent gains |
| noise | — | dead knob (mean-driven cells) |

## Chosen reference — `iext_exc = 2`
- **S ≈ 0.0045** — asynchronous (async floor ≈ 1/√N ≈ 0.05; ictal → ~1).
- `[K+]o` **flat**: 3.28 mM, drift 0.004 mM/s (no drift flag).
- participation 0.057, **unimodal** (no all-or-none recruitment).
- **FS not stone-dead**: fires 0.004–0.013 Hz on transient exc upticks, quiet between.
- exc ~16 Hz (rate is *not* a criterion for this mean-driven model).

Every point in the 12-run S-grid had **S = 0.004–0.010** (see `stageA_grid_S.csv` /
`07_stageA_grid_S_montage.png`). Alternative reference: `iext_exc = 3` (S ≈ 0.008, hotter
baseline ~28 Hz, FS engages harder) — equally valid on S.

## What made the reference reachable (three fixes + one enabler)
1. **`dt` 0.05 → 0.025** — numerical stability.
2. **FS pump 30 → 8** — at 30, FS never fired at any `iext` (0–22), so there was no
   inhibition and exc ran away; at 8 FS is silenced at rest yet EPSP-driven in-network
   (see `04_fs_pump_isolated_FI.png`, `05_fs_revival_network.png`).
3. **`kdyn_epsilon` 0.06 → 0.002** — flattens baseline `[K+]o` under `iext` drive.
4. **`iext_sigma` = 2** — heterogeneity keeps participation unimodal (breaks the
   cell-level knife-edge at the population level).

## Cell property (validated, INTRINSIC — do not "fix", do not retune gNaP/gKCa)
The isolated Ho cells are knife-edge: **silent → ~6 Hz → ~300 Hz → depolarization block**,
with no graded DC F-I. This is dt-converged and intrinsic (persistent Na, `isodiumP`) —
it is **not** kdyn and **not** the K-leak (`02_kdyn_freeze_FI.png`, `03_kleak_FI.png`).

## Figures
| File | What |
|---|---|
| `01_rheobase_FI.png` | isolated PY/FS F-I — the knife-edge |
| `02_kdyn_freeze_FI.png` | freezing kdyn does not graduate it (not the cause) |
| `03_kleak_FI.png` | K-leak only shifts the knife-edge (not the cause) |
| `04_fs_pump_isolated_FI.png` | isolated FS vs `ikpumpmax` (silence-vs-fireable) |
| `05_fs_revival_network.png` | FS revives in-network at pump 8 (E/I balanced) |
| `06_iext_onset_calib_prefix.png` | in-network onset calibration (**pre-fix; superseded**) |
| `07_stageA_grid_S_montage.png` | S-scored Stage-A grid (all S ≈ 0.004–0.010) |
| `08_reference_iE2.png` | the chosen reference run |
| `09_reference_alt_iE3.png` | alternative reference (hotter baseline) |
| `stageA_grid_S.csv` | the 12-run S-scored grid table |

## Reproduce
Pipeline: `neuron_simulation/{tuning.py, network_builder.py, neurons_ho.py}` +
mechanisms `{iext,ikpump,kbalance}.mod`. Run one point:
```
python -m neuron_simulation.tuning <params.json> <label> <outdir>
```
Ho's synchrony **S**, per-population `[K+]o` + drift, participation bimodality, and rates
are computed on every run (see `tuning.py`).

**Stage B/C (the gK flip: normal-gK reference vs PY gK→0.3 ictal) not yet run.**
The Stage-C success metric is the *same* S rising from ~0 toward 1 as gK falls.
