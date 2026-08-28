"""4-AP dose-response arm: is the seizure state reachable by A-current block?

WHY THIS EXISTS. The project's seizure knob is ``sahp_ainc_slow`` -- the
Ca2+-dependent slow AHP, a KCa conductance that is classically 4-AP
INSENSITIVE. 4-AP blocks voltage-gated Kv channels; in this model the only
4-AP-sensitive current is the A-type ``kA`` (states.gbar_block_state). So a
"4-AP dose for the seizure state" cannot be read off the knob -- it has to be
established (or refuted) FUNCTIONALLY: simulate a 4-AP dose ladder, measure
the same phenotype axes used everywhere else, and ask whether any dose
reproduces the seizure phenotype.

DOSE MAPPING. Fractional A-current block f = [4-AP] / ([4-AP] + IC50) with
IC50 = 1.0 mM for Kv4/I_A (Kd 0.9 +- 0.07 mM for Kv4.2 tonic block at -80 mV;
Tigerholm et al. 2017 J Pharmacol Exp Ther; ~1 mM IA IC50, cf. Bio-Techne
compound data). CAVEAT to state whenever these numbers are shown: the low-uM
concentrations used in slice epilepsy models act mainly on Kv1/D-type
currents, which this single-compartment model does not contain -- so these
simulations speak only to 4-AP's A-current component.

Runs one 60 s recording per dose per network (NEURON required), measures
firing rate / event rate / participation / burst duration with the same
detector as the sweep, and writes results/<state>/other/fourap_dose.json plus
the figure fourap_dose_response.png in sweep_summary/ with the normal and
seizure phenotypes overlaid as horizontal bands.

    py -3.9 analysis/fourap_dose_response.py sweep_c50_seed01 sweep_c50_seed09
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from session_paths import DATA, resolve, results_dir  # noqa: E402

IC50_MM = 1.0
DOSES_MM = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0]      # -> f = 0, .2, .33, .5, .67, .8, .9

# --- sAHP-deficit severity axis (the project's own seizure knobs) -----------
# s = 0 -> normal state; s = 1 -> the chosen seizure state. Both knobs move
# together along s, so intermediate values are graded severities of the same
# manipulation rather than an arbitrary two-parameter grid.
SAHP_SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]
SAHP_AINC_NORMAL, SAHP_AINC_SEIZURE = 0.010, 0.004
SAHP_TAU_NORMAL, SAHP_TAU_SEIZURE = 6500.0, 3000.0


def sahp_params(severity):
    """(ainc_slow, tau_slow) at severity s in [0, 1] (linear interpolation)."""
    a = SAHP_AINC_NORMAL + severity * (SAHP_AINC_SEIZURE - SAHP_AINC_NORMAL)
    t = SAHP_TAU_NORMAL + severity * (SAHP_TAU_SEIZURE - SAHP_TAU_NORMAL)
    return float(a), float(t)

_spec = importlib.util.spec_from_file_location(
    "_nsim_analysis", os.path.join(REPO, "neuron_simulation", "analysis.py"))
_an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_an)


def block_fraction(dose_mm):
    return dose_mm / (dose_mm + IC50_MM)


def phenotype(spike_data, n, duration_ms):
    bursts, _ = _an.detect_network_bursts_all(spike_data, n, duration_ms)
    parts = [b["participation"] for b in bursts]
    durs = [b["duration_ms"] for b in bursts]
    n_spikes = sum(len(s) for s in spike_data.values())
    return dict(
        rate_hz=n_spikes / n / (duration_ms / 1000.0),
        n_events=len(bursts),
        events_per_min=len(bursts) / (duration_ms / 60000.0),
        mean_participation=float(np.mean(parts)) if parts else float("nan"),
        mean_duration_ms=float(np.mean(durs)) if durs else float("nan"),
        n_full=int(sum(1 for b in bursts if b["burst_class"] == "full")))


def session_cfg(session):
    """Rebuild the session's network config from the CHTC sweep config + the
    seeds recorded in its network npz (CHTC sessions carry no worker pickle)."""
    sys.path.insert(0, os.path.join(REPO, "chtc"))
    import generate_one as g
    sweep = g.load_sweep(os.path.join(REPO, "chtc", "sweep_config.json"))
    cfg, prov = g.build_cfg(sweep, session)
    return cfg, sweep


def run_dose(session, dose_mm, duration_ms, out_dir, _cache={}):
    """One 60 s run of the session's own network with gbar_kA reduced."""
    from neuron_simulation import states as st
    from neuron_simulation.network_builder import build_network
    from neuron_simulation.simulation import run_simulation
    from neuron_simulation.noise import reseed_noise

    if session not in _cache:
        _cache[session] = session_cfg(session)
    cfg, sweep = _cache[session]
    f = block_fraction(dose_mm)
    build_kwargs = dict(cfg["build_kwargs"])
    if f > 0:
        blocked = st.gbar_block_state(f)
        build_kwargs["gbar_kA_exc"] = blocked["gbar_kA_exc"]
        build_kwargs["gbar_kA_inh"] = blocked["gbar_kA_inh"]
    net = build_network(cfg["topology"], noise_seed=cfg["noise_seed_base"], **build_kwargs)
    reseed_noise(net.noise, 0)
    spikes, _v, _k = run_simulation(
        net, duration=duration_ms, dt=cfg["dt"],
        discard_transient_ms=cfg["discard_transient_ms"], record_voltage=False)
    ph = phenotype(spikes, net.n_neurons, duration_ms)
    ph.update(dose_mm=dose_mm, block_fraction=f,
              gbar_kA_exc=build_kwargs.get("gbar_kA_exc"))
    print("  %5.2f mM (f=%.2f): %.2f Hz | %.1f events/min | part %.2f | dur %.0f ms"
          % (dose_mm, f, ph["rate_hz"], ph["events_per_min"],
             ph["mean_participation"], ph["mean_duration_ms"]), flush=True)
    return ph


def run_severity(session, severity, duration_ms, _cache={}):
    """One 60 s run with both sAHP knobs set to the given severity."""
    from neuron_simulation.network_builder import build_network
    from neuron_simulation.simulation import run_simulation
    from neuron_simulation.noise import reseed_noise

    if session not in _cache:
        _cache[session] = session_cfg(session)
    cfg, _sweep = _cache[session]
    ainc, tau = sahp_params(severity)
    build_kwargs = dict(cfg["build_kwargs"])
    build_kwargs["sahp_ainc_slow"] = ainc
    build_kwargs["sahp_tau_slow"] = tau
    net = build_network(cfg["topology"], noise_seed=cfg["noise_seed_base"], **build_kwargs)
    reseed_noise(net.noise, 0)
    spikes, _v, _k = run_simulation(
        net, duration=duration_ms, dt=cfg["dt"],
        discard_transient_ms=cfg["discard_transient_ms"], record_voltage=False)
    ph = phenotype(spikes, net.n_neurons, duration_ms)
    ph.update(severity=severity, sahp_ainc_slow=ainc, sahp_tau_slow=tau)
    print("  s=%.2f (ainc %.4f, tau %.0f): %.2f Hz | %.1f events/min | part %.2f | dur %.0f ms"
          % (severity, ainc, tau, ph["rate_hz"], ph["events_per_min"],
             ph["mean_participation"], ph["mean_duration_ms"]), flush=True)
    return ph


def reference_phenotype(session, state):
    """Phenotype of the session's saved recording000 in the given state."""
    d = np.load(os.path.join(resolve(session, state), "recording000.npz"),
                allow_pickle=True)
    stimes = d["spike_times"]
    n = len(stimes)
    sd = {i: np.asarray(stimes[i], float) for i in range(n)}
    return phenotype(sd, n, float(d["duration"]))


def figure(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT = os.path.join(DATA, "sweep_summary")
    os.makedirs(OUT, exist_ok=True)
    metrics = [("rate_hz", "firing rate (Hz)"),
               ("events_per_min", "events per minute"),
               ("mean_participation", "mean participation"),
               ("mean_duration_ms", "mean burst duration (ms)")]
    markers = ["o", "s", "^", "D"]
    # rows = metrics, columns = the two manipulations on their own dose axes
    fig, axes = plt.subplots(4, 2, figsize=(11.6, 13.4))
    for row, (key, label) in enumerate(metrics):
        ax4, axs = axes[row][0], axes[row][1]
        for m, (session, res) in zip(markers, sorted(all_results.items())):
            ax4.plot([p["dose_mm"] for p in res["doses"]],
                     [p[key] for p in res["doses"]], m + "-", ms=5, lw=1.5,
                     color="#7b3294", label=session)
            if res.get("severities"):
                axs.plot([p["severity"] for p in res["severities"]],
                         [p[key] for p in res["severities"]], m + "-", ms=5, lw=1.5,
                         color="#008837", label=session)
        for ax in (ax4, axs):
            for state, col, ls in (("normal", "#1f5fd0", "--"), ("seizure", "#c0392b", ":")):
                vals = [r.get(state, {}).get(key) for r in all_results.values()]
                vals = [v for v in vals if v is not None and np.isfinite(v)]
                if vals:
                    ax.axhline(float(np.mean(vals)), color=col, ls=ls, lw=1.4, alpha=0.75)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        ax4.set_xlabel("[4-AP] (mM)  --  A-current block, IC50 %.1f mM" % IC50_MM)
        axs.set_xlabel("sAHP-deficit severity s  (0 = normal, 1 = seizure state)")
    axes[0][0].set_title("4-AP: block the A-current (Kv4)", fontsize=11)
    axes[0][1].set_title("sAHP deficit: weaken + speed the slow AHP (KCa)", fontsize=11)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Two routes to an epileptiform phenotype, same networks and noise\n"
                 "dashed blue = measured normal state, dotted red = measured seizure state",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(OUT, "fourap_vs_sahp_dose_response.png")
    fig.savefig(out, dpi=140, facecolor="white")
    print("figure -> %s" % out)


def main_point(a):
    """Run ONE (session, dose|severity) point and write it to --point-out.

    Each 60 s simulation costs ~70 CPU-minutes, so the ladder is run as one
    process per point in parallel rather than sequentially; --combine then
    merges the point files into the per-session JSON and the figure.
    """
    if a.point_kind == "dose":
        ph = run_dose(a.point_session, a.point_value, a.duration, None)
    else:
        ph = run_severity(a.point_session, a.point_value, a.duration)
    ph["session"] = a.point_session
    ph["kind"] = a.point_kind
    with open(a.point_out, "w", encoding="utf-8") as fh:
        json.dump(ph, fh, indent=1)
    print("point saved -> %s" % a.point_out)


def main_combine(a):
    """Merge point files from --combine into per-session JSONs + the figure."""
    import glob as _g
    points = []
    for p in sorted(_g.glob(os.path.join(a.combine, "point_*.json"))):
        with open(p, encoding="utf-8") as fh:
            points.append(json.load(fh))
    all_results = {}
    for ph in points:
        res = all_results.setdefault(ph["session"], {"session": ph["session"],
                                                     "ic50_mm": IC50_MM,
                                                     "doses": [], "severities": []})
        res["doses" if ph["kind"] == "dose" else "severities"].append(ph)
    for session, res in all_results.items():
        res["doses"].sort(key=lambda p: p["dose_mm"])
        res["severities"].sort(key=lambda p: p["severity"])
        for state in ("normal", "seizure"):
            try:
                res[state] = reference_phenotype(session, state)
            except Exception as exc:
                print("  (no %s reference for %s: %s)" % (state, session, exc))
        out = os.path.join(results_dir(session, "normal", "other"), "fourap_dose.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print("%s: %d doses + %d severities -> %s"
              % (session, len(res["doses"]), len(res["severities"]), out))
    figure(all_results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--point-session")
    ap.add_argument("--point-kind", choices=["dose", "severity"])
    ap.add_argument("--point-value", type=float)
    ap.add_argument("--point-out")
    ap.add_argument("--combine", help="directory of point_*.json files to merge")
    ap.add_argument("--duration", type=float, default=60000.0)
    ap.add_argument("--doses", type=float, nargs="*", default=None)
    ap.add_argument("--severities", type=float, nargs="*", default=None)
    ap.add_argument("--skip-4ap", action="store_true",
                    help="only run the sAHP severity ladder (reuse saved 4-AP results)")
    a = ap.parse_args()
    if a.point_session:
        return main_point(a)
    if a.combine:
        return main_combine(a)
    doses = a.doses if a.doses else DOSES_MM
    severities = a.severities if a.severities else SAHP_SEVERITIES

    all_results = {}
    for session in a.sessions:
        print("%s (IC50 %.1f mM)" % (session, IC50_MM), flush=True)
        prev = {}
        prev_path = os.path.join(results_dir(session, "normal", "other"), "fourap_dose.json")
        if a.skip_4ap and os.path.exists(prev_path):
            with open(prev_path, encoding="utf-8") as fh:
                prev = json.load(fh)
        res = {"session": session, "ic50_mm": IC50_MM,
               "doses": prev.get("doses") if a.skip_4ap and prev.get("doses")
                        else [run_dose(session, d, a.duration, None) for d in doses]}
        res["severities"] = [run_severity(session, s, a.duration) for s in severities]
        for state in ("normal", "seizure"):
            try:
                res[state] = reference_phenotype(session, state)
            except Exception as exc:
                print("  (no %s reference: %s)" % (state, exc))
        out_dir = results_dir(session, "normal", "other")
        with open(os.path.join(out_dir, "fourap_dose.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print("  saved -> %s" % os.path.join(out_dir, "fourap_dose.json"))
        all_results[session] = res
    figure(all_results)


if __name__ == "__main__":
    main()
