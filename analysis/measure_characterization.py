"""Measure the quantities that go into MODEL_CHARACTERIZATION.md.

Every number this prints is measured from the model as built by
``build_network`` with the FLAGSHIP config (notably noise_weight = 0.007, not the
registry's 0.004). Nothing here is a hand calculation.

Sections:
  1  operating point: ik (hh+kA) vs the sAHP current, and the [K+]o counterfactual
  2  single background event at 0.007 uS: rested vs adapted; rheobase
  3  input resistance, active at rest and leak-only
  4  unitary EPSP amplitude, and the AMPA/NMDA charge split
  5  NMDA Mg-block gate vs voltage (peak AND charge weighting)

Writes ``characterization_measurements.json`` next to this script.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from neuron import h  # noqa: E402
from neuron_simulation.neurons import build_cell, load_mechanisms  # noqa: E402
from neuron_simulation.network_builder import build_network  # noqa: E402
from neuron_simulation.simulation import run_simulation  # noqa: E402

FLAGSHIP_CFG = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal",
                            "20260721_163430", "_worker_config.pkl")
AREA_CM2 = math.pi * (20e-4) * (20e-4)          # pi * diam * L, both 20 um
NA_TO_MA_PER_CM2 = 1e-6 / AREA_CM2               # nA -> mA/cm2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "characterization_measurements.json")
R = {}


def cfg():
    return pickle.load(open(FLAGSHIP_CFG, "rb"))


# --------------------------------------------------------------------------- #
# 1. operating point: ik vs sAHP current
# --------------------------------------------------------------------------- #
def section1(duration_ms, n_probe=40):
    print("\n=== 1. operating point: ik (hh+kA) vs sAHP current ===", flush=True)
    c = cfg()
    net = build_network(c["topology"], noise_seed=c["noise_seed_base"],
                        report_deviations=False, **c["build_kwargs"])
    for g in net.noise:
        g.reseed(0)

    probe = np.linspace(0, net.n_neurons - 1, n_probe).astype(int)
    dt_rec = 1.0
    rec = {k: [] for k in ("v", "ik", "sahp_i", "g_slow", "g_fast", "ko")}
    for gid in probe:
        cell = net.cells[int(gid)]
        seg = cell.soma(0.5)
        rec["v"].append(h.Vector().record(seg._ref_v, dt_rec))
        rec["ik"].append(h.Vector().record(seg._ref_ik, dt_rec))
        rec["sahp_i"].append(h.Vector().record(cell.sahp._ref_i, dt_rec))
        rec["g_slow"].append(h.Vector().record(cell.sahp._ref_g_slow, dt_rec))
        rec["g_fast"].append(h.Vector().record(cell.sahp._ref_g_fast, dt_rec))
        rec["ko"].append(h.Vector().record(seg.kdyn._ref_ko, dt_rec))
    tv = h.Vector().record(h._ref_t, dt_rec)

    t0 = time.time()
    spike_data, _, _ = run_simulation(net, duration=duration_ms, dt=c["dt"],
                                     discard_transient_ms=1000.0,
                                     record_voltage=False, record_ko=False)
    print("  ran %.0f ms in %.0f s" % (duration_ms, time.time() - t0), flush=True)

    t = np.asarray(tv)
    keep = t >= 1000.0                      # match the flagship's discard
    A = {k: np.array([np.asarray(v)[keep] for v in rec[k]]) for k in rec}

    rate = sum(len(v) for v in spike_data.values()) / (net.n_neurons * duration_ms / 1000.0)
    # Subthreshold mask: exclude spikes so "resting" quantities mean what they say.
    sub = A["v"] < -60.0

    mean_ik_all = float(A["ik"].mean())
    mean_ik_sub = float(A["ik"][sub].mean())
    mean_sahp_i = float(A["sahp_i"].mean())
    mean_sahp_i_sub = float(A["sahp_i"][sub].mean())
    sahp_dens = mean_sahp_i * NA_TO_MA_PER_CM2
    ik_dens = mean_ik_all                    # already mA/cm2

    # kdyn steady state: ko - ko_rest = epsilon * tau_k * (ik - ik_rest)
    eps, tau_k, ik_rest = 0.06, float(c["build_kwargs"]["tau_k"]), 0.0006
    dko_if_counted = eps * tau_k * sahp_dens

    print("  mean rate                      %.4f Hz" % rate, flush=True)
    print("  V (subthreshold)               %.2f mV  [5/50/95 %.2f/%.2f/%.2f]"
          % (A["v"][sub].mean(), *np.percentile(A["v"][sub], [5, 50, 95])), flush=True)
    print("  g_slow (tonic)                 %.5f uS   g_fast %.5f uS"
          % (A["g_slow"].mean(), A["g_fast"].mean()), flush=True)
    print("  ik  (hh+kA), all samples       %+.6f mA/cm2" % mean_ik_all, flush=True)
    print("  ik  (hh+kA), subthreshold only %+.6f mA/cm2  <- NEGATIVE means K+ flows"
          " INWARD (cell rests below E_K)" % mean_ik_sub, flush=True)
    print("  sAHP current                   %+.5f nA = %+.6f mA/cm2"
          % (mean_sahp_i, sahp_dens), flush=True)
    print("  |sAHP| / |ik(all)|             %.1fx" % abs(sahp_dens / mean_ik_all),
          flush=True)
    print("  measured [K+]o                 %.4f mM (mean) / %.4f max"
          % (A["ko"].mean(), A["ko"].max()), flush=True)
    print("  [K+]o if sAHP were counted     %.4f + %.4f = %.4f mM"
          % (A["ko"].mean(), dko_if_counted, A["ko"].mean() + dko_if_counted),
          flush=True)

    R["operating_point"] = dict(
        duration_ms=duration_ms, n_probe=int(n_probe), rate_hz=rate,
        v_sub_mean=float(A["v"][sub].mean()),
        v_sub_pct=[float(x) for x in np.percentile(A["v"][sub], [5, 50, 95])],
        g_slow_mean_uS=float(A["g_slow"].mean()),
        g_fast_mean_uS=float(A["g_fast"].mean()),
        ik_mean_mA_cm2=mean_ik_all, ik_sub_mean_mA_cm2=mean_ik_sub,
        sahp_i_mean_nA=mean_sahp_i, sahp_i_sub_mean_nA=mean_sahp_i_sub,
        sahp_i_mean_mA_cm2=sahp_dens,
        sahp_over_ik_ratio=float(abs(sahp_dens / mean_ik_all)),
        ko_mean_mM=float(A["ko"].mean()), ko_max_mM=float(A["ko"].max()),
        dko_if_sahp_counted_mM=float(dko_if_counted),
        ko_if_sahp_counted_mM=float(A["ko"].mean() + dko_if_counted),
        epsilon=eps, tau_k_ms=tau_k, ik_rest_mA_cm2=ik_rest,
        area_cm2=AREA_CM2)
    return R["operating_point"]


# --------------------------------------------------------------------------- #
# helpers for the single-cell sections
# --------------------------------------------------------------------------- #
def solo_cell(adapt=True, g_slow=0.0, freeze_sahp=True):
    """One flagship-spec cell. Optionally hold sAHP at a standing conductance.

    ``freeze_sahp`` sets tau_slow/tau_fast enormous so a standing g does not
    decay over the measurement window -- this isolates the electrotonic question
    from the adaptation dynamics.
    """
    c = cfg()
    bk = c["build_kwargs"]
    load_mechanisms()
    h.celsius = 6.3
    cell = build_cell(0, is_inhibitory=False, gbar_kA=bk["gbar_kA_exc"],
                      adapt=adapt, sahp_ainc_fast=bk["sahp_ainc_fast"],
                      sahp_tau_fast=bk["sahp_tau_fast"],
                      sahp_ainc_slow=bk["sahp_ainc_slow"],
                      sahp_tau_slow=bk["sahp_tau_slow"],
                      sahp_ek=bk.get("sahp_ek", -90.0))
    cell.soma(0.5).kdyn.tau_k = bk["tau_k"]
    if adapt and freeze_sahp:
        cell.sahp.tau_slow = 1e9
        cell.sahp.tau_fast = 1e9
    if adapt:
        cell.sahp.g_slow = g_slow
    return cell


def settle(cell, ms=800.0, dt=0.05, v_init=-65.0, g_slow=None):
    h.dt = dt
    h.finitialize(v_init)
    if g_slow is not None and cell.sahp is not None:
        cell.sahp.g_slow = g_slow
    h.continuerun(ms)
    return cell.soma(0.5).v


def one_event(cell, weight, tau=3.0, e_rev=0.0, settle_ms=800.0, g_slow=None,
              window_ms=60.0):
    """Deliver a single synaptic event; return (v_rest, v_peak, dV, fired)."""
    syn = h.ExpSyn(cell.soma(0.5))
    syn.tau, syn.e = tau, e_rev
    stim = h.NetStim()
    stim.number, stim.start, stim.noise = 1, settle_ms, 0
    nc = h.NetCon(stim, syn)
    nc.weight[0], nc.delay = weight, 0.0
    vv = h.Vector().record(cell.soma(0.5)._ref_v, 0.025)
    n0 = len(cell.spike_times)
    settle(cell, ms=settle_ms + window_ms, g_slow=g_slow)
    v = np.asarray(vv)
    n_settle = int(settle_ms / 0.025)
    v_rest = float(v[max(0, n_settle - 40):n_settle].mean())
    v_peak = float(v[n_settle:].max())
    fired = len(cell.spike_times) > n0
    del nc, stim, syn
    return v_rest, v_peak, v_peak - v_rest, bool(fired)


# --------------------------------------------------------------------------- #
# 2. single background event at the flagship weight
# --------------------------------------------------------------------------- #
def section2(g_slow_tonic):
    print("\n=== 2. single background event (noise_weight = 0.007 uS) ===",
          flush=True)
    W_FLAG, W_REG = 0.007, 0.004
    out = {"g_slow_tonic_uS": g_slow_tonic}

    for label, gs in (("rested (zero adaptation)", 0.0),
                      ("operating point (tonic sAHP)", g_slow_tonic)):
        for w in (W_REG, W_FLAG):
            cell = solo_cell(adapt=True, g_slow=gs)
            vr, vp, dv, fired = one_event(cell, w, g_slow=gs)
            print("  %-30s w=%.4f uS: rest %.2f -> peak %.2f  dV %+.2f mV  %s"
                  % (label, w, vr, vp, dv, "FIRES" if fired else "no spike"),
                  flush=True)
            out["%s_w%.4f" % ("rested" if gs == 0.0 else "adapted", w)] = dict(
                v_rest=vr, v_peak=vp, dV=dv, fired=fired, weight_uS=w,
                g_slow_uS=gs)
            del cell

    # rheobase: smallest single-event weight that fires, at each adaptation state
    for label, gs in (("rested", 0.0), ("adapted", g_slow_tonic)):
        lo, hi = 1e-5, 0.05
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cell = solo_cell(adapt=True, g_slow=gs)
            _, _, _, fired = one_event(cell, mid, g_slow=gs)
            del cell
            if fired:
                hi = mid
            else:
                lo = mid
        print("  rheobase (%s): %.6f uS  -> flagship 0.007 is %.2fx rheobase"
              % (label, hi, W_FLAG / hi), flush=True)
        out["rheobase_%s_uS" % label] = hi
        out["flagship_over_rheobase_%s" % label] = W_FLAG / hi

    R["single_event"] = out
    return out


# --------------------------------------------------------------------------- #
# 3. input resistance
# --------------------------------------------------------------------------- #
def section3():
    print("\n=== 3. input resistance ===", flush=True)
    out = {}
    for label, strip in (("active (hh+kA+kdyn, at rest)", False),
                         ("leak only (gna=gk=gkA=0)", True)):
        cell = solo_cell(adapt=False)
        if strip:
            cell.soma(0.5).hh.gnabar = 0.0
            cell.soma(0.5).hh.gkbar = 0.0
            cell.soma(0.5).kA.gbar = 0.0
        ic = h.IClamp(cell.soma(0.5))
        ic.delay, ic.dur, ic.amp = 1000.0, 800.0, -0.010    # -10 pA
        vv = h.Vector().record(cell.soma(0.5)._ref_v, 1.0)
        h.dt = 0.05
        h.finitialize(-65.0)
        h.continuerun(2000.0)
        v = np.asarray(vv)
        v_base = v[900:1000].mean()
        v_ss = v[1700:1800].mean()
        rin = (v_ss - v_base) / ic.amp        # mV / nA = MOhm
        print("  %-32s rest %.2f mV, dV %.3f mV at -10 pA -> R_in %.1f MOhm"
              % (label, v_base, v_ss - v_base, rin), flush=True)
        out[label] = dict(v_rest=float(v_base), dV=float(v_ss - v_base),
                          R_in_MOhm=float(rin))
        del ic, cell
    R["input_resistance"] = out
    return out


# --------------------------------------------------------------------------- #
# 4. unitary EPSP + AMPA/NMDA charge split
# --------------------------------------------------------------------------- #
def section4(g_slow_tonic):
    print("\n=== 4. unitary EPSP (AmpaNmda, flagship params) ===", flush=True)
    c = cfg()
    bk = c["build_kwargs"]
    conns = c["topology"]["connections"]
    w_exc = np.array([float(r[2]) for r in conns if str(r[3]) == "exc"])
    scaled = w_exc * bk["exc_weight_scale"]
    print("  excitatory weights: n=%d  raw median %.5f uS  x%.1f -> median %.5f uS"
          % (len(w_exc), np.median(w_exc), bk["exc_weight_scale"],
             np.median(scaled)), flush=True)

    out = {"n_exc_edges": int(len(w_exc)),
           "w_raw_median_uS": float(np.median(w_exc)),
           "w_scaled_median_uS": float(np.median(scaled)),
           "w_scaled_mean_uS": float(scaled.mean()),
           "exc_weight_scale": float(bk["exc_weight_scale"])}

    for label, gs in (("unadapted rest", 0.0),
                      ("operating point", g_slow_tonic)):
        for nmda_on in (True, False):
            cell = solo_cell(adapt=True, g_slow=gs)
            syn = h.AmpaNmda(cell.soma(0.5))
            syn.tau_ampa = bk["exc_tau"]
            syn.tau_nmda = bk["tau_nmda"]
            syn.nmda_ratio = bk["nmda_ratio"] if nmda_on else 0.0
            syn.d = bk["depression_d"]
            syn.tau_d = bk["tau_d"]
            stim = h.NetStim()
            stim.number, stim.start, stim.noise = 1, 800.0, 0
            nc = h.NetCon(stim, syn)
            nc.weight[0], nc.delay = float(np.median(scaled)), 0.0
            vv = h.Vector().record(cell.soma(0.5)._ref_v, 0.025)
            iv = h.Vector().record(syn._ref_i, 0.025)
            settle(cell, ms=2400.0, g_slow=gs)
            v, i = np.asarray(vv), np.asarray(iv)
            n0 = int(800.0 / 0.025)
            vr = v[n0 - 40:n0].mean()
            amp = v[n0:].max() - vr
            charge = float(-i[n0:].sum() * 0.025)     # pA*ms ~ fC (sign: inward)
            print("  %-16s NMDA %-3s: rest %.2f mV  EPSP %+.3f mV  charge %.1f fC"
                  % (label, "on" if nmda_on else "off", vr, amp, charge), flush=True)
            out["%s_nmda%s" % ("rest" if gs == 0.0 else "op",
                               "On" if nmda_on else "Off")] = dict(
                v_rest=float(vr), epsp_mV=float(amp), charge_fC=charge)
            del nc, stim, syn, cell

    for tag in ("rest", "op"):
        on = out["%s_nmdaOn" % tag]
        off = out["%s_nmdaOff" % tag]
        out["%s_nmda_charge_fraction" % tag] = 1.0 - off["charge_fC"] / on["charge_fC"]
        out["%s_nmda_peak_effect_mV" % tag] = on["epsp_mV"] - off["epsp_mV"]
        print("  %s: NMDA carries %.1f%% of unitary charge; adds %+.3f mV to peak"
              % (tag, 100 * out["%s_nmda_charge_fraction" % tag],
                 out["%s_nmda_peak_effect_mV" % tag]), flush=True)

    R["unitary_epsp"] = out
    return out


# --------------------------------------------------------------------------- #
# 5. NMDA Mg-block gate
# --------------------------------------------------------------------------- #
def section5():
    print("\n=== 5. NMDA Mg-block gate B(v) ===", flush=True)
    c = cfg()
    bk = c["build_kwargs"]
    ratio, tn, ta, mg = bk["nmda_ratio"], bk["tau_nmda"], bk["exc_tau"], 1.0
    rows = []
    print("     v(mV)    B(v)   peak NMDA/AMPA   charge NMDA/AMPA", flush=True)
    for v in (-85, -83.31, -80, -70, -60, -55, -50, -40, -30):
        B = 1.0 / (1.0 + math.exp(-0.062 * v) * mg / 3.57)
        rows.append(dict(v_mV=v, B=B, peak_ratio=ratio * B,
                         charge_ratio=ratio * B * tn / ta))
        print("   %7.2f  %.4f      %6.3f          %7.2f"
              % (v, B, ratio * B, ratio * B * tn / ta), flush=True)
    R["nmda_gate"] = dict(nmda_ratio=ratio, tau_nmda=tn, tau_ampa=ta, mg=mg,
                          rows=rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=21000.0,
                    help="network run length for section 1 (ms, incl. 1 s discard)")
    ap.add_argument("--skip-network", action="store_true")
    a = ap.parse_args()

    g_slow_tonic = 0.0
    if not a.skip_network:
        op = section1(a.duration)
        g_slow_tonic = op["g_slow_mean_uS"]
    else:
        print("skipping section 1; using nominal g_slow", flush=True)
        g_slow_tonic = 0.021

    section2(g_slow_tonic)
    section3()
    section4(g_slow_tonic)
    section5()

    json.dump(R, open(OUT, "w"), indent=2)
    print("\nsaved -> %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
