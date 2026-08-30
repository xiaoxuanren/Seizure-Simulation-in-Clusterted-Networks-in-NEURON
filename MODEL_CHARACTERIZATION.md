# Model characterization — what this model *does*

Measured behaviour of the network as built by `build_network` with the flagship
configuration. **No design intentions in this file.** Every number is either
measured here or traceable to a named script and run; where a number is a
literature comparison it is marked as such, and where a claim could not be
sourced it is left out rather than softened.

Flagship configuration is the one that produced the
`IC-locked_flagship_200rec` session (200 recordings × 60 s, 926 neurons,
13 356 edges). The raw 200-recording data is not committed to this repo — only
its `notebooks/NEURON data parallel/IC-locked_flagship_200rec/results/` tree is.
The committed runnable subset is
`notebooks/NEURON data parallel/IC-locked_flagship_spikeonly_50rec/normal`
(50 spike-only recordings). Note the flagship ran at **`noise_weight = 0.007 µS`**, not the
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

`detect_network_bursts` (`neuron_simulation/analysis.py` line 64) runs
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
the sAHP knob (`sahp_ainc_slow` + `sahp_tau_slow`) that cannot reach [K⁺]ₒ
directly.

**Verified 2026-08 (20 networks, ladder runs).** Raising `tau_k` on top of the
seizure state does drive [K⁺]ₒ to 12–16 mM, but it *suppresses* network
bursting rather than producing ictal events — participation falls 0.95 → 0.16
and inter-burst intervals stretch from 2.4 s to ~18 s, the signature of
depolarization block. See `analysis/tauk_ictal_ladder.py` and
`sweep_summary/ladder_ictal_test.png`.

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

### D.6 The seizure knob is the Ca²⁺-dependent sAHP, not Kv7/KCNQ

`sahp_ainc_slow` was documented in `states.py` and `parameters.py` as the
"slow-AHP / M-current (Kv7/KCNQ)" increment, and lowering it as "KCNQ2/3
loss-of-function". That attribution is wrong, and `sAHP.mod` never made it — the
mod file has always assigned Kv7 to the *fast* component.

**The literature splits the AHP by timescale.**

| component | timescale | carrier | this model |
|---|---|---|---|
| mAHP | 50–100 ms | Kv7/KCNQ (I_M) + SK | `sahp_ainc_fast`, τ 300 ms — **held fixed** |
| sAHP | 1–5 s (I_sAHP τ ≈ 2.9 s) | Ca²⁺-dependent KCa, likely KCa3.1 | `sahp_ainc_slow` + `sahp_tau_slow` (6500 ms normal / 3000 ms seizure) — **the two-parameter knob** |

Kv7/M channels activate subthreshold from about −60 mV with gating on tens of
milliseconds and do not inactivate; they cannot produce a 6.5 s decay. The
Ca²⁺-dependent sAHP survives apamin, XE-991 (Kv7 block) and Cs⁺, which is the
pharmacological separation. The one link between KCNQ and the sAHP —
Tzingounis & Nicoll 2008, *PNAS* 105:19974 — is specific to dentate granule
cells and is not the mainline story.

**The corrected attribution is stronger, not weaker.** Reduced sAHP is an
established *acquired*-epilepsy mechanism: in post-status-epilepticus
hippocampus, principal neurons generate markedly more spikes per depolarization,
and the main cause is sAHP suppression via PKA-mediated inhibition of KCa3.1
(Tamir et al. 2017; Tiwari et al. 2019, *J Neurosci* 39:9914; CRF/CRF1R route,
*J Neurosci* 42:5843, 2022; KCa3.1 in L5 neocortex, Roshchin et al. 2020,
*Sci Rep* 10:14484). That is one reduced adaptation conductance on fixed wiring
— exactly this experiment. KCNQ2/3 loss-of-function is a genetic neonatal
syndrome acting mainly through the mAHP, i.e. through the parameter this project
holds constant.

**Recommended phrasing:** "an acquired-epilepsy adaptation deficit — reduced
Ca²⁺-dependent sAHP, KCa3.1-like."

### D.7 The adaptation model is linear, unsaturating, and temperature-inconsistent

`sAHP.mod` is the textbook spike-triggered adaptation *conductance* — the
conductance-based sibling of AdEx's `b` (Brette & Gerstner 2005), a special case
of Benda & Herz's universal SFA model. The network-burst phenotype that follows
is the expected one for this model class: coupled adaptive neurons with
spike-driven adaptation form a relaxation oscillator, with the adaptation
variable decaying through the quiescent phase until the network re-ignites
(Frontiers Neurosci 12:41, 2018). §C's ~4.9 s ignition is that oscillator.

Three consequences worth stating before the knob is cited as biophysical:

1. **The load is unbounded in rate.** `g_ss = ainc · rate · τ` with no ceiling,
   whereas the real I_sAHP saturates through a nonlinear Ca²⁺ sensor
   (hippocalcin; Tzingounis et al. 2007, *Neuron* 53:487). This is why cutting
   `ainc` 2.5× lowers the measured load only ~18% — rate rises 2.05× and nearly
   compensates (§ Part-3 runs).
2. **The conductance is large relative to the cell.** One spike adds 0.796
   mS/cm² (0.01 µS over the 1256.6 µm² soma) = **2.65× the entire leak**
   (`gl_hh` 0.3 mS/cm²); the 0.29 Hz steady state is 1.50 mS/cm² = **5× leak**.
   That is why sAHP sets V_rest (§A) — a real sAHP modulates firing, it does not
   set the resting potential.
3. **Kinetics and temperature disagree.** `h.celsius = 6.3` (squid HH) and
   `kA.mod` carries a `q10 = 3` `tadj`, but `sAHP.mod` and `kdyn.mod` have no
   temperature scaling. Read as Q10 = 3 kinetics measured at 35 °C, τ_fast
   300 ms ↔ ~13 ms (the real I_M range) and τ_slow 6500 ms ↔ ~278 ms. The two
   timescales are coherent only if they are *not* temperature-corrected.

**Why defensible.** None of this changes the experimental logic: one parameter
moves, everything else is pinned, and the ground-truth wiring is exact. The
model is a caricature of adaptation chosen to produce culture-like bursting,
which it does.

**Cost.** `ainc_slow` is µS *per spike*, not a channel density, so it bundles
channel density, Ca²⁺ influx per spike and Ca²⁺ sensitivity into one number. A
channelopathy would move only the first — "one knob = one channel" is a
modelling convenience, not a literal mapping.

---

## Known documentation defects (see git history for fixes)

- `states.py` and `parameters.py` attributed `sahp_ainc_slow` to the M-current
  (Kv7/KCNQ) and called lowering it "KCNQ2/3 loss-of-function". By kinetics and
  pharmacology the knob is the **Ca²⁺-dependent sAHP**; Kv7 belongs to
  `sahp_ainc_fast`, which is held fixed. See §D.6.
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

---

## G. Scope: what this model reproduces relative to the 4-AP literature

Measured 2026-08 with the mechanism ladders (`analysis/fourap_dose_response.py`,
`analysis/tauk_ictal_ladder.py`, `chtc/ladder_mechanisms.json`; 24 parameter
points × 20 networks, 180 s recordings, figures in `sweep_summary/`).

**The seizure knob is not 4-AP, and that was tested.** 4-AP blocks voltage-gated
Kv channels — Kv1/D-type at the low-µM concentrations used in slice models, and
Kv4/A-type only near 1 mM (IC₅₀ ≈ 1 mM; Kd 0.9 ± 0.07 mM for Kv4.2 tonic block
at −80 mV). The project's knob is the Ca²⁺-dependent slow AHP, a KCa
conductance that is classically 4-AP-**insensitive**. Simulating 4-AP directly
(A-current block, 0 → 90%, ≈ 0–9 mM equivalent) left every phenotype axis flat
across 20 networks: firing rate 0.30 → 0.30 Hz, event duration 146 → 156 ms.
There is therefore **no 4-AP dose equivalent** for the seizure state.

**What the model does reproduce: the 4-AP culture/MEA signature.** The published
culture response to 4-AP is higher mean firing rate, more frequent network
bursts, and greater synchrony; the clearest quantitative anchor is MFR
0.53 → 1.90 Hz (≈3.6×). The sAHP severity axis reproduces this: 3.2× at
severity 0.75 and 5.8× at the shipped seizure state, with burst rate
9.4 → 25.5 per 60 s and participation 0.51 → 0.95.

**What it does not reproduce: the 4-AP slice ictal regime.** Slice studies
report ictal discharges of 31–103 s recurring on ~26 s intervals, with
interictal events of 1–2 s. This model's events top out below 1 s (seizure
mean 442 ms, longest observed ~830 ms). Impaired K⁺ clearance — the mechanism
slice ictogenesis depends on, together with interneuron depolarization block —
does raise [K⁺]ₒ into the ictal range (12–16 mM) but **suppresses** bursting
rather than sustaining it (participation 0.95 → 0.16, IBI 2.4 s → ~18 s).

**Interpretation.** The model reaches the ingredients of slice ictogenesis
(high [K⁺]ₒ, depolarization block) without producing sustained discharges.
Sustained ictal events plausibly require mechanisms this single-compartment
model lacks: dendritic compartments, activity-dependent GABA reversal shifts,
and spatially resolved K⁺ diffusion. Claims from this model should be scoped to
**culture-scale epileptiform dynamics**, not slice ictal phenomenology.
