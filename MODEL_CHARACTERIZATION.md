# Model characterization — what this model *does*

Measured behaviour of the network as built by `build_network` with the flagship
configuration. **No design intentions in this file.** Every number is either
measured here or traceable to a named script and run; where a number is a
literature comparison it is marked as such, and where a claim could not be
sourced it is left out rather than softened.

Flagship configuration is the one that produced
`notebooks/NEURON data parallel/normal/20260721_163430` (200 recordings × 60 s,
926 neurons, 13 356 edges). Note it ran at **`noise_weight = 0.007 µS`**, not the
registry default of 0.004 — several older numbers in the repo were derived at
0.004 and do not describe this dataset.

Provenance for anything below: `analysis/measure_characterization.py`
(`characterization_measurements.json`), `analysis/burst_windows_p035.py`,
`analysis/burst_gate_sensitivity.py`, `analysis/decoupled_control.py`,
`analysis/compare_control_windows.py`, `analysis/burstexcl_glm_arm.py`.

---

## A. Measured operating point

### Resting potential

| quantity | value |
|---|---|
| V_rest, per-neuron mean (926 neurons) | **−83.31 mV** (5/50/95: −84.53 / −83.26 / −82.24) |
| V_rest, independent 21 s network run, subthreshold samples, 40 probe cells | **−83.16 mV** (5/50/95: −85.72 / −83.67 / −79.25) |
| neurons resting below −77 mV | **100%** of 926 |

Two independent measurements agree to 0.15 mV.

### Why V_rest proves sAHP carries the resting conductance

This is provable from recorded voltage alone, without instrumenting any
mechanism.

`kdyn` writes `ek` from the Nernst equation with `ki = 72 mM` held fixed. At
resting `ko = ko_rest = 4 mM`:

```
ek = 26.64 · ln(4/72) = −77.00 mV      (exact)
```

Every other reversal in the model is at or above −77 mV: `hh` leak −54.3 mV,
`ena` +50 mV, inhibitory `ExpSyn` −75 mV, excitatory `AmpaNmda` 0 mV. `sAHP` is
the sole exception — it declares `NONSPECIFIC_CURRENT` with a **private
`ek = −90 mV`** and no `USEION`, so its reversal is not tied to the potassium
pool.

A passive conductance mixture cannot rest below its most negative reversal. The
network rests at −83.2 mV, which is 6.2 mV below −77 mV. Solving the conductance
divider with every non-sAHP conductance assigned its most negative possible
reversal (−77 mV, the case most favourable to them):

```
g_sAHP / g_total  ≥  (77 − 83.16) / (90 − 83.16)  =  47%
```

So **sAHP carries at least 47% of resting conductance** (≥48% using the −83.31 mV
figure). This is a strict lower bound: every other conductance actually reverses
*above* −77 mV, which requires more sAHP, not less.

### sAHP is rate-tracking, and the load law

| quantity | value |
|---|---|
| V_rest vs each neuron's own firing rate | Pearson **−0.937**, slope **−11.18 mV/Hz** |
| tonic `g_slow`, measured (21 s network run) | **0.02046 µS** |
| tonic `g_fast`, measured | **0.00041 µS** |

The standing conductance follows `g ≈ a_inc · rate · τ`, verified exact in
stationary firing (ratios 1.00 / 0.97 / 1.09).

> **Caveat on the cross-check.** The 21 s network run above gives
> `0.01 × 0.2667 Hz × 6.5 s = 0.01734 µS` predicted against 0.02046 µS measured,
> a ratio of 1.18. The window is only 3.2 τ_slow long and contains the
> initialization burst (§C), whose contribution is smeared forward over ~6.5 s
> and so inflates mean `g_slow` relative to mean rate. The load law should be
> re-checked on a burst-free window ≥10 τ_slow before being called exact at the
> network level.

### Potassium currents at the operating point

| quantity | value |
|---|---|
| `ik` (hh + kA), all samples | **+0.000360 mA/cm²** |
| `ik` (hh + kA), subthreshold samples only | **−0.000009 mA/cm²** |
| sAHP current | **+0.13867 nA** = **+0.011035 mA/cm²** |
| sAHP / mean total `ik` | **30.7×** |
| measured [K⁺]ₒ | **3.9972 mM** mean, **4.2117 mM** max (4.29 mM max over a 60 s run) |

The subthreshold `ik` is **negative** — potassium flows *inward* between spikes,
because sAHP holds the cell 6 mV below E_K. Any ratio against it is
ill-conditioned; the 30.7× figure uses mean total `ik` and should be quoted with
that caveat, or replaced by the [K⁺]ₒ consequence in §D.2.

---

## B. Transmission

Measured spike-transmission efficacy. The unconnected-network figure is the
common-input floor — what the same measurement returns when no synapse exists.

| condition | efficacy |
|---|---|
| isolated pair, connected | **4.2%** |
| unconnected control (common-input floor) | **0.8%** |
| **synaptic excess** | **3.5%** |
| in-burst, connected | **21.0%** |
| in-burst control | **8.5%** |
| in-burst synaptic excess | **12.5%** (ratio **3.6×**) |

Per-pair efficacy, pooled: median **2.4%**, mean **4.0%**, max **28%**.

### Efficacy is voltage-gated

Efficacy varies **~300×** between −85 mV and −65 mV. The consequence is that
transmission is carried by a small minority of spikes:

- **92.6%** of subthreshold time is spent below −80 mV, where efficacy is ~1.2%.
- Pre-synaptic spikes arriving above −80 mV are **33%** of all spikes but carry
  **86%** of transmissions.

---

## C. Dynamics and bursts

### How a burst is measured, and what that costs

`detect_network_bursts` ([analysis.py:64](neuron_simulation/analysis.py:64)) runs
in two stages with two independent parameters, and **both** move the burst count.

*Stage 1 — bracket.* Spikes are binned at `activity_bin_ms = 5`. A bin is "on"
when the fraction of distinct neurons firing in it clears
`onset_active_frac = 0.05` (47 of 926). Contiguous on-runs become windows;
windows < `merge_gap_ms = 50` apart are merged; windows < `min_event_ms = 8` are
dropped; survivors are padded ±`pad_ms = 10`.

*Stage 2 — accept.* Count *distinct* neurons firing anywhere in the padded
window. Accept iff `participation > participation_threshold` (strictly greater).

Three properties that must travel with any burst number:

1. **The window is data-defined, not fixed-width** — measured median duration
   55 ms (spontaneous 56.4 ± 9.2 ms, IC-locked 72.8 ± 19.2 ms).
2. **Participation is counted over the whole window, not in one bin.** A 500 ms
   sliding window on the same events reports up to 94% where the detector's
   ~60 ms window reports ~0.6. Both are correct; they measure different spans.
3. **`onset_active_frac` is the binding constraint, not the gate.** The
   active-fraction signal has median 0.00108 per 5 ms bin, p99 = 0.0151,
   p99.9 = 0.0551 — so 0.05 sits at roughly the **99.87th percentile** and only
   0.13% of bins clear it. Only **421 candidate events exist across all 200
   recordings (2.10 per recording)**, while the rasters show ~14 synchronous
   stripes each. The unbracketed stripes are real synchrony at 1–3%
   instantaneous density, too temporally diffuse to bracket.

**Burst rate is a two-parameter quantity** (40-recording subset, event counts):

| `onset_active_frac` | gate 0.20 | 0.35 | 0.50 | 0.80 | rate @ gate 0.35 |
|---|---|---|---|---|---|
| 0.01 | 286 | 168 | 91 | 25 | 0.0700 Hz |
| 0.02 | 200 | 131 | 63 | 18 | 0.0546 Hz |
| 0.03 | 160 | 108 | 55 | 14 | 0.0450 Hz |
| **0.05 (default)** | 87 | **72** | 32 | 8 | **0.0300 Hz** |
| 0.08 | 21 | 21 | 13 | 2 | 0.0088 Hz |

Across defensible settings the rate spans **0.0088–0.0700 Hz**. No single burst
rate can be quoted without both parameters.

Participation of bracketed events is a **continuum, not two modes**: median
0.449, 5–95th percentile 0.273–0.826, and no bracketed event falls below ~0.20.
There is no valley to place a threshold in, which is why 0.35 vs 0.80 changes the
count 30-fold (336 → 26).

### Two burst classes

At the project's 0.35 gate, 336 bursts over 200 recordings:

| class | n | start range | participation | duration |
|---|---|---|---|---|
| **IC-locked** | 137 | 4.60–5.34 s | 0.615 ± 0.160 | 72.8 ± 19.2 ms |
| **spontaneous** | 199 | 8.63–59.85 s | 0.457 ± 0.078 | 56.4 ± 9.2 ms |

**IC-locked bursts are an initialization artifact.** `h.finitialize(-65)` starts
every cell at zero adaptation and `ko = ko_rest` simultaneously — peak
excitability for the whole population at once. Evidence: the event is
**r = 0.977 at zero lag** between two independent noise seeds in a
zero-recurrence control, i.e. its timing is set by initial conditions and not by
the noise.

**Spontaneous bursts are genuine and not phase-locked**: starts are uniform on
[6, 60] s, KS D = 0.054, p = 0.589.

**198 of the 199 spontaneous bursts are absent from the stored 0.8-gate windows.**
The single exception is recording 035 at 26.53 s (86% participation).

The stored `mean_participation = 0.86` at the 0.8 gate is **not comparable** to
the 0.35-gate mean of 0.457/0.615 — the 0.8 gate reports the mean of a top slice,
conditioned on passing.

### Burst rates, with denominators

| definition | rate | IBI |
|---|---|---|
| all 0.35-gate bursts | **0.0280 Hz** | 35.7 s |
| spontaneous only, per full 60 s | **0.01658 Hz** | 60.3 s |
| spontaneous only, per eligible 6–60 s window | **0.01843 Hz** | 54.3 s |
| stored 0.8-gate | **0.00217 Hz** | 461.5 s |

Spikes inside burst windows at the 0.35 gate: **7.30%** of all spikes (3.64%
spontaneous-only, 3.65% IC-locked). Total burst time is **0.177%** of the record.

### Rate decomposition — recurrence buys synchrony, not rate

Both arms measured with identical preprocessing (1 s transient discarded),
identical topology and noise seeds; only `exc_weight_scale`/`inh_weight_scale`
differ.

| | rate |
|---|---|
| coupled (flagship, 200 recordings) | **0.2789 Hz** (sd 0.0044 across recordings) |
| decoupled (`exc = inh = 0`, 15 recordings) | **0.2355 Hz** (sd 16 spikes/rec) |
| **noise-driven fraction** | **84.4%** |
| recurrence contributes | **+0.0434 Hz** (15.6%) |

What recurrence *does* buy, same two arms:

| | coupled | decoupled |
|---|---|---|
| peak participation (500 ms window) | **94.3%** | 47.9% |
| sharpness (50 ms peak ÷ 500 ms peak) | **0.593** | 0.126 |
| population Vm excursion | **10.30 mV** | 1.53 mV |
| event period (autocorrelation) | 6.08 s (r = 0.31) | 4.38 s (r = 0.78) |

The decoupled arm still rings at 4.38 s from the shared initial condition,
detectable above baseline to ~35 s, and that ring is fully IC-determined
(r = 0.977 across seeds). It is not a network burst: half the participation, a
fifth the sharpness, and no population Vm signature.

---

## D. Deliberate simplifications

Each: what it is, why it is defensible, what it costs.

### D.1 The [K⁺]ₒ route is held inert, deliberately

`tau_k = 200 ms` makes the `kdyn` positive-feedback loop contribute very little.
`kclearance_seizure_state()` exists as the alternative route with
`tau_k = 12000 ms`.

**Why defensible — experimental control.** Two seizure mechanisms appear in the
literature: loss of slow adaptation, and failure of K⁺ clearance. This model
isolates the first by pinning the second. Measured effect of the knob: rate
**2.81×**, [K⁺]ₒ **+0.44%**, with 98% of even that mediated by spike count.

**Cost.** The [K⁺]ₒ readout carries almost no dynamics in the shipped regime;
measured range is 3.99–4.29 mM, never ictal.

**Phrasing.** `tau_k` **is** a knob that reaches [K⁺]ₒ — it is held fixed. It is
`sahp_ainc_slow`, the single knob, that cannot reach [K⁺]ₒ directly.

### D.2 `sAHP` is `NONSPECIFIC_CURRENT` with a private `ek = −90 mV`

So its current never enters `ik` and is invisible to the potassium pool `kdyn`
integrates. Electrically it behaves as a K⁺ conductance and it is what sets
V_rest (§A).

**Why defensible.** The declaration affects only ion bookkeeping, not membrane
dynamics. Fixing it would not produce ictal [K⁺]ₒ: adding the measured sAHP
current to `ik` moves steady-state [K⁺]ₒ by

```
Δko = ε · τ_k · Δi = 0.06 × 200 ms × 0.011035 mA/cm² = +0.133 mM
```

i.e. **3.9972 → 4.1297 mM**, still far below the 8–12 mM ictal range.
**`tau_k` is the binding constraint, not the declaration.**

**Cost.** [K⁺]ₒ under-reports true K⁺ efflux by roughly the above. The sAHP
current is 30.7× mean total `ik`, but see §A — subthreshold `ik` is near zero and
slightly inward, so that ratio is ill-conditioned and the +0.133 mM figure is the
honest statement.

### D.3 A single background event is suprathreshold — on a rested cell only

Measured at the flagship `noise_weight = 0.007 µS`:

| condition | ΔV | outcome |
|---|---|---|
| rested (zero adaptation), 0.004 µS | +98.79 mV | fires |
| rested (zero adaptation), 0.007 µS | +99.20 mV | fires |
| operating point (`g_slow` 0.021 µS, rest −84.56 mV), 0.004 µS | +8.64 mV | no spike |
| **operating point, 0.007 µS** | **+14.20 mV** | **no spike** |

| rheobase (single event) | value | flagship 0.007 relative to it |
|---|---|---|
| rested | 0.000863 µS | 8.11× — suprathreshold |
| **at the operating point** | **0.01435 µS** | **0.49× — subthreshold** |

**Why defensible.** The README's *literal* claim (a single noise event is
subthreshold) is **false on a rested cell**. Its *functional* claim — noise
integrates rather than detonating — **holds**, delivered by adaptation rather
than by amplitude: at the operating point a single event reaches 49% of
threshold.

**Cost.** The margin is a dynamic property, not a static one. A cell that has
been quiet long enough for `g_slow` to decay is detonable by one background
event. With τ_slow = 6.5 s that is the state every cell is in at `t = 0`, which
is the mechanism behind the IC-locked burst in §C.

### D.4 Bare-soma geometry inflates unitary EPSPs

Single 20 × 20 µm compartment, no dendritic load.

| quantity | value |
|---|---|
| R_in, active at rest (hh + kA + kdyn) | **73.9 MΩ** |
| R_in, leak only (`gnabar = gkbar = gbar_kA = 0`) | **265.3 MΩ** |
| unitary EPSP at the operating point, median scaled weight 0.00234 µS | **6.27 mV** |
| literature unitary EPSP, L5A pyramidal pairs, ~4 weeks | ~0.65 mV *(lit.)* |
| **inflation** | **~9.6×** |

Note the raw topology weights are 0.001–0.0022 µS (1–2.2 nS), but
`exc_weight_scale = 2.0` means the **delivered** peak AMPA conductance is
2–4.4 nS (median 2.34 nS) — above the 1–2.2 nS range, so the conductance cannot
be described as literature-matched without the scale factor.

**Why defensible.** The measured *effective* coupling is what enters the
inference problem, and it is much smaller than the unitary amplitude suggests:
the deep resting potential leaves cells ~20 mV further from threshold than a real
neuron, and measured synaptic excess in spike transmission is **3.5%** (§B).

> **Unsourced comparison, deliberately omitted.** The claim that 3.5% "lands
> inside the 1–5% range reported for real cortical pyramidal pairs" could **not
> be sourced**. Literature searches returned pyramid→interneuron transmission at
> 46 ± 27%, CA3 interpyramid up to 80%, and *no detected* pyramid→pyramid spike
> transmission in human L2/3 — none of which supports a 1–5% range. The measures
> also differ in kind: paired-recording spike-transmission probability is not the
> same quantity as correlogram efficacy-above-common-input-floor. **Do not put
> this comparison in the thesis without a citation.** The model-side number
> (3.5%) stands on its own.

**Cost.** Unitary EPSP amplitude is unphysiological by ~an order of magnitude.

### D.5 `AmpaNmda` runs outside its own cited ranges

| parameter | mod-file comment | `build_network` | verdict |
|---|---|---|---|
| `tau_nmda` | 50–150 ms | **350 ms** | outside |
| `nmda_ratio` | ~0.5–1 | **3.0** | outside |
| `depression_d` | `d = 0.5` (Silver 2002) | **0.2** | outside, opposite direction |
| `tau_d` | 600 ms (`AmpaNmda`), 800 ms (`DepSyn`) | **500 ms** | at the edge |

Mg²⁺ block, `B(v) = 1/(1 + exp(−0.062 v)·mg/3.57)` with `mg = 1`:

| v | B(v) | peak NMDA/AMPA | **charge NMDA/AMPA** |
|---|---|---|---|
| −85 | 0.0180 | 0.054 | **3.79** |
| −83.31 | 0.0200 | 0.060 | **4.20** |
| −70 | 0.0445 | 0.133 | 9.34 |
| −55 | 0.1055 | 0.317 | 22.16 |
| −40 | 0.2302 | 0.690 | 48.33 |

**The peak ratio is the wrong statistic.** `tau_nmda = 350 ms` is 70× `exc_tau =
5 ms`, so the charge ratio is `nmda_ratio · B · τ_N/τ_A = 210·B`. Measured
directly on a unitary event at the operating point: **NMDA carries 80.2% of the
synaptic charge**, while removing it changes the peak by only −0.50 mV
(6.27 → 5.77 mV).

**Why defensible — restated.** The inflated parameters supply slow depolarizing
*charge* that matters for integration over hundreds of ms — which is what
sustains burst reverberation — while contributing almost nothing to individual
EPSP peaks.

**What is not defensible.** "NMDA is silent at rest and active only in bursts" is
false: it dominates charge transfer at every voltage measured, including
−85 mV. The peak-conductance framing (6% at rest) understates its role by more
than an order of magnitude.

**Cost.** Two synapse parameters sit outside the ranges their own file cites, and
the file is not annotated to say so.

---

## Known documentation defects (see git history for fixes)

- `plotting.py` claimed [K⁺]ₒ "accumulates into the ictal range (~8–12 mM)".
  Measured max is **4.26 mM**. See §D.1.
- `README.md` "Verified normal state" claimed mean rate 2.6 Hz, bursts at 1.5 Hz,
  77% of spikes in bursts. Measured: **0.2789 Hz**, **0.0280 Hz** at the 0.35
  gate, **7.30%** of spikes.
- `README.md` honest-caveat claimed the burst rate is "far faster" than the
  ~0.03 Hz of dissociated cultures. Measured **0.0088–0.0700 Hz** depending on
  detector settings, with the defaults landing on **0.0280–0.0300 Hz** — i.e. the
  model brackets the cited culture value. The caveat described an earlier regime.
- `noise.py` docstring claimed ~0.07 Hz noise-only firing, unmeasured and derived
  at an older weight. Measured **0.2355 Hz** at `noise_weight = 0.007`.
