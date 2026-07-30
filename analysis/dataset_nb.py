"""Helpers behind the dataset-generation notebook.

The notebook stays declarative -- one CONFIG dict and five one-line calls. All
subprocess plumbing, path handling and progress polling lives here.

    import dataset_nb as nb
    nb.build_topology(CONFIG)     # build the graph, write the session config
    pre = nb.preflight(CONFIG)
    nb.build_libraries(CONFIG, pre)
    nb.generate(CONFIG, pre)
    nb.validate(CONFIG)

CONFIG is the single source of truth. Everything the workers need is written
once into ``<session>/_session_config.pkl``, so the warm-start libraries and the
recordings provably used identical settings.
"""

import glob
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_ROOT = os.path.join(REPO, "notebooks", "NEURON data parallel")

#: measured on this machine: NEURON runs a ~900-cell network ~68x slower than realtime
REALTIME_FACTOR = 68.4


def session_dir(cfg):
    return os.path.join(DATA_ROOT, cfg["session"])


def config_path(cfg):
    return os.path.join(session_dir(cfg), "_session_config.pkl")


def out_dir(cfg, state):
    return os.path.join(session_dir(cfg), state)


def library_path(cfg, state):
    return os.path.join(session_dir(cfg), "_state_library_%s.npz" % state)


# --------------------------------------------------------------------------- #
# step 0: topology + session config
# --------------------------------------------------------------------------- #
def build_topology(cfg, rebuild=False):
    """Build the network graph and write the session config.

    Cached: the warm-start libraries and the recordings must share one graph, so
    rebuilding it midway would silently invalidate every library. Pass
    ``rebuild=True`` only when you intend to start the session over.
    """
    sd = session_dir(cfg)
    os.makedirs(sd, exist_ok=True)
    cp = config_path(cfg)

    if os.path.exists(cp) and not rebuild:
        with open(cp, "rb") as fh:
            s = pickle.load(fh)
        t = s["topology"]
        print("session '%s': existing graph, %d neurons, %d edges"
              % (cfg["session"], t["n_neurons"], len(t["connections"])))
        stale = _config_drift(s["config"], cfg)
        if stale:
            print("  WARNING: CONFIG has changed since this session was created:")
            for k, (was, now) in sorted(stale.items()):
                print("    %-26s %r -> %r" % (k, was, now))
            print("  Recordings already generated used the OLD values. Either")
            print("  revert, or pick a new 'session' name, or rebuild=True.")
        return cp

    sys.path.insert(0, REPO)
    from neuron_simulation.topology import build_topology_lognormal, build_topology as _bt
    kind = cfg.get("topology_kind", "lognormal")
    print("session '%s': building a %s graph (seed=%s) ..."
          % (cfg["session"], kind, cfg["topology"].get("seed")), flush=True)
    topo = (build_topology_lognormal(**cfg["topology"]) if kind == "lognormal"
            else _bt(**cfg["topology"]))
    with open(cp, "wb") as fh:
        pickle.dump(dict(config=cfg, topology=topo,
                         cluster_info=topo["cluster_info"], session_dir=sd), fh)
    print("  %d neurons, %d edges, %d clusters -> %s"
          % (topo["n_neurons"], len(topo["connections"]),
             len(topo["cluster_info"]["cluster_neuron_groups"]),
             os.path.relpath(cp, REPO)), flush=True)
    return cp


def _config_drift(old, new):
    """Parameters that changed between the stored session config and CONFIG."""
    out = {}
    for section in ("build", "sim"):
        for k, v in new.get(section, {}).items():
            if old.get(section, {}).get(k) != v:
                out["%s.%s" % (section, k)] = (old.get(section, {}).get(k), v)
    for k in ("states", "recording_ms", "noise_seed_base", "warmup_ms",
              "snapshot_times", "discard_extra_ms", "voltage"):
        if old.get(k) != new.get(k):
            out[k] = (old.get(k), new.get(k))
    return out


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def preflight(cfg):
    py = sys.executable
    ok, need_library, todo = True, [], {}
    print("python  : %s (%s)" % (py, sys.version.split()[0]))
    if subprocess.run([py, "-c", "import neuron"], capture_output=True).returncode:
        ok = False
        print("FAIL    : this kernel cannot import neuron -- select the Python 3.9 "
              "interpreter that has it")
    if subprocess.run(
            [py, "-c", 'import sys; sys.path.insert(0, r"%s"); '
             "from neuron_simulation.neurons import load_mechanisms; "
             "load_mechanisms()" % REPO], capture_output=True, cwd=REPO).returncode:
        ok = False
        print("FAIL    : mechanisms not built -- run  "
              "cd neuron_simulation && nrnivmodl mechanisms")

    cp = config_path(cfg)
    if not os.path.exists(cp):
        print("FAIL    : no session config -- run the build_topology cell first")
        return dict(ok=False, need_library=list(cfg["states_to_run"]), todo={},
                    config_path=cp)
    with open(cp, "rb") as fh:
        s = pickle.load(fh)
    t = s["topology"]
    print("session : %s  (%d neurons, %d edges)"
          % (cfg["session"], t["n_neurons"], len(t["connections"])))
    drift = _config_drift(s["config"], cfg)
    if drift:
        ok = False
        print("FAIL    : CONFIG differs from the stored session config in %d "
              "parameter(s) -- rerun build_topology or pick a new session name"
              % len(drift))
        for k, (was, now) in sorted(drift.items()):
            print("            %-24s %r -> %r" % (k, was, now))

    print("")
    print("%-9s %-8s %-30s %s" % ("state", "knob", "warm-start library", "recordings"))
    for st in cfg["states_to_run"]:
        lp = library_path(cfg, st)
        if os.path.exists(lp):
            lb = np.load(lp)
            k = float(lb["sahp_ainc_slow"])
            if abs(k - cfg["states"][st]) > 1e-12:
                txt, ok = "MISMATCH: built at %.4f" % k, False
            else:
                txt = "%d snapshots, sd %.5f uS" % (
                    lb["g_slow"].shape[0], lb["g_slow"].std(axis=1).mean())
        else:
            txt = "MISSING -> run the library cell"
            need_library.append(st)
        d = out_dir(cfg, st)
        todo[st] = [r for r in range(cfg["n_recordings"])
                    if not os.path.exists(os.path.join(d, "recording%03d.npz" % r))]
        print("%-9s %-8.4f %-30s %d have / %d to go"
              % (st, cfg["states"][st], txt,
                 cfg["n_recordings"] - len(todo[st]), len(todo[st])))

    n = sum(len(v) for v in todo.values())
    per_min = ((cfg["recording_ms"] + cfg["sim"]["discard_transient_ms"]
                + cfg["discard_extra_ms"]) / 1000.0 * REALTIME_FACTOR / 60.0)
    gb = n * {"all": 77.0, "probe": 3.0, "none": 0.6}[cfg["voltage"]] / 1024.0
    print("")
    print("generate: %d recordings -> ~%.1f h with %d workers, ~%.2f GB"
          % (n, n * per_min / 60.0 / max(1, cfg["n_workers"]), cfg["n_workers"], gb))
    try:                       # memory-aware sanity check on n_workers
        sys.path.insert(0, REPO)
        from neuron_simulation.parallel_dataset import pick_worker_count
        safe, free_gb, cpu = pick_worker_count(max(n, 1), max_workers=None)
        print("machine : %d cores, %.1f GB free -> at most ~%d workers is safe"
              % (cpu, free_gb, safe))
        if cfg["n_workers"] > safe:
            print("          WARNING: n_workers=%d exceeds that; consider lowering it"
                  % cfg["n_workers"])
    except Exception:
        pass
    if need_library:
        print("library : %d to build -> ~%.1f h (they run concurrently)"
              % (len(need_library),
                 cfg["warmup_ms"] / 1000.0 * REALTIME_FACTOR / 3600.0))
    print("")
    print("PREFLIGHT %s" % ("OK" if ok else "FAILED -- fix the above first"))
    return dict(ok=ok, need_library=need_library, todo=todo, config_path=cp)


# --------------------------------------------------------------------------- #
# subprocess driver
# --------------------------------------------------------------------------- #
def _run_jobs(jobs, n_workers, poll_s=120, stagger_s=15, label="job"):
    queue, running, done, t0 = list(jobs), [], [], time.time()
    while queue or running:
        while queue and len(running) < n_workers:
            name, cmd, log = queue.pop(0)
            running.append((name, subprocess.Popen(
                cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT), log))
            print("  launched %s" % name, flush=True)
            time.sleep(stagger_s)
        still = []
        for name, p, log in running:
            if p.poll() is None:
                still.append((name, p, log))
            else:
                done.append((name, p.returncode, log))
                print("  finished %s (exit %s)" % (name, p.returncode), flush=True)
        running = still
        if running or queue:
            print("  [%5.1f min] %d running, %d queued"
                  % ((time.time() - t0) / 60, len(running), len(queue)), flush=True)
            time.sleep(poll_s)
    bad = [(n, rc, lg) for n, rc, lg in done if rc]
    for n, rc, lg in bad:
        print("  FAILED %s (exit %s):" % (n, rc))
        for line in open(lg).read().strip().splitlines()[-6:]:
            print("      " + line)
    print("all %ss done in %.1f min%s"
          % (label, (time.time() - t0) / 60,
             ", %d FAILED" % len(bad) if bad else ""))
    return done


# --------------------------------------------------------------------------- #
# the long-running steps
# --------------------------------------------------------------------------- #
def build_libraries(cfg, pre=None):
    pre = pre or preflight(cfg)
    if not pre["need_library"]:
        print("all libraries present -- nothing to do")
        return []
    jobs = []
    for st in pre["need_library"]:
        log = os.path.join(HERE, "_ds_lib_%s_%s.log" % (cfg["session"], st))
        jobs.append(("library:%s" % st,
                     [sys.executable, "-u",
                      os.path.join(HERE, "dataset_warmstart.py"),
                      "--config", pre["config_path"], "--state", st], log))
    return _run_jobs(jobs, len(jobs), label="library")


def generate(cfg, pre=None):
    pre = pre or preflight(cfg)
    assert pre["ok"], "preflight failed -- do not run this"
    assert not pre["need_library"], "build the warm-start libraries first"
    jobs = []
    for st in cfg["states_to_run"]:
        todo = pre["todo"][st]
        if not todo:
            print("%s: nothing to do" % st)
            continue
        per = int(np.ceil(len(todo) / max(1, cfg["n_workers"])))
        for w in range(cfg["n_workers"]):
            chunk = todo[w * per:(w + 1) * per]
            if not chunk:
                continue
            log = os.path.join(HERE, "_ds_%s_%s_w%d.log" % (cfg["session"], st, w))
            jobs.append(("%s w%d (rec %d-%d)" % (st, w, chunk[0], chunk[-1]),
                         [sys.executable, "-u",
                          os.path.join(HERE, "dataset_generate.py"),
                          "--config", pre["config_path"], "--state", st,
                          "--start", str(chunk[0]), "--count", str(len(chunk))],
                         log))
    done = _run_jobs(jobs, cfg["n_workers"], label="worker")
    for st in cfg["states_to_run"]:
        d = out_dir(cfg, st)
        print("%-9s %d recordings, %d rasters, %d summaries"
              % (st, len(glob.glob(os.path.join(d, "recording*.npz"))),
                 len(glob.glob(os.path.join(d, "recording*_raster*.png"))),
                 len(glob.glob(os.path.join(d, "_summary_*.json")))))
    return done


def validate(cfg):
    cp = config_path(cfg)
    for st in cfg["states_to_run"]:
        print("=" * 70)
        print("%s  (sahp_ainc_slow = %.4f)" % (st, cfg["states"][st]))
        print("=" * 70)
        r = subprocess.run(
            [sys.executable, "-u", os.path.join(HERE, "dataset_validate.py"),
             "--config", cp, "--state", st], capture_output=True, text=True)
        print(r.stdout[-4000:])
        if r.returncode:
            print("STDERR:", r.stderr[-1500:])


# --------------------------------------------------------------------------- #
# inspection: the views neuron_network_simulation.ipynb shows
# --------------------------------------------------------------------------- #
def topology_report(cfg):
    """4-panel graph overview + printed connectivity stats.

    Spatial layout, hub fan-out, cluster-sorted connection matrix, out-degree
    distribution. Run after build_topology; costs nothing to look at.
    """
    sys.path.insert(0, REPO)
    import matplotlib.pyplot as plt
    from neuron_simulation import plotting
    with open(config_path(cfg), "rb") as fh:
        topo = pickle.load(fh)["topology"]
    stats, fig = plotting.topology_report(topo)
    plt.show()
    return stats


def preview(cfg, duration_ms=20000.0, states=None):
    """Short run per state -> raster + [K+]o, before committing hours.

    Uses the same build parameters and the same total discard as the real
    recordings, so the kept window is representative. This is a parameter check:
    is the firing rate sane, is it bursting, does [K+]o move.
    """
    sys.path.insert(0, REPO)
    import matplotlib.pyplot as plt
    from neuron_simulation import analysis as _an, plotting, states as _st, workflows
    with open(config_path(cfg), "rb") as fh:
        topo = pickle.load(fh)["topology"]
    n = topo["n_neurons"]
    discard = cfg["sim"]["discard_transient_ms"] + cfg["discard_extra_ms"]
    gate = float(cfg["sim"]["participation_threshold"])
    out = {}
    for state in (states or cfg["states_to_run"]):
        bk = dict(cfg["build"])
        bk["sahp_ainc_slow"] = float(cfg["states"][state])
        print("running %.0f s preview of '%s' (sahp_ainc_slow=%.4f) ..."
              % (duration_ms / 1000, state, bk["sahp_ainc_slow"]), flush=True)
        r = workflows.run_single_state(
            topo, state=_st.normal_state(), build_kwargs=bk,
            duration=duration_ms, record_ko=True, dt=cfg["sim"]["dt"],
            discard_transient_ms=discard, noise_seed=cfg["noise_seed_base"])
        bursts = _an.detect_network_bursts(r["spike_data"], n, duration_ms,
                                           participation_threshold=gate,
                                           burn_in_ms=0.0)
        st = _an.burst_statistics(bursts, duration_ms, burn_in_ms=0.0)
        rate = _an.firing_rate_summary(r["spike_data"], duration_ms,
                                       burn_in_ms=0.0)["mean_rate_hz"]
        ko = r["ko_data"]["mean_ko"]
        print("  %-8s firing %.3f Hz | bursts %d (%.3f Hz, participation %.2f) "
              "| [K+]o %.2f-%.2f mM"
              % (state, rate, st["n_bursts"], st["burst_rate_hz"],
                 st["mean_participation"], float(ko.min()), float(ko.max())),
              flush=True)
        for rr, tag in ((False, "cluster-sorted"), (True, "randomized rows")):
            plotting.plot_raster_with_ko(
                r["spike_data"], n, duration_ms, r["ko_data"],
                is_inhibitory=topo["neuron_is_inhibitory"],
                cluster_assignments=topo["cluster_assignments"],
                title="%s preview (sahp_ainc_slow=%.3f, %s)"
                      % (state.upper(), bk["sahp_ainc_slow"], tag),
                randomize_rows=rr)
            plt.show()
        out[state] = dict(rate_hz=rate, n_bursts=int(st["n_bursts"]),
                          burst_rate_hz=float(st["burst_rate_hz"]),
                          participation=float(st["mean_participation"]),
                          ko_min=float(ko.min()), ko_max=float(ko.max()))
    return out


def knob_sweep(cfg, values=None, duration_ms=20000.0):
    """Burst rate vs sahp_ainc_slow -- the graded seizure/normal transition."""
    sys.path.insert(0, REPO)
    import matplotlib.pyplot as plt
    from neuron_simulation import analysis as _an, states as _st, workflows
    with open(config_path(cfg), "rb") as fh:
        topo = pickle.load(fh)["topology"]
    n = topo["n_neurons"]
    vals = sorted(values or set(cfg["states"].values()))
    discard = cfg["sim"]["discard_transient_ms"] + cfg["discard_extra_ms"]
    gate = float(cfg["sim"]["participation_threshold"])
    rates, parts = [], []
    for v in vals:
        bk = dict(cfg["build"]); bk["sahp_ainc_slow"] = float(v)
        r = workflows.run_single_state(
            topo, state=_st.normal_state(), build_kwargs=bk,
            duration=duration_ms, record_ko=False, dt=cfg["sim"]["dt"],
            discard_transient_ms=discard, noise_seed=cfg["noise_seed_base"])
        st = _an.burst_statistics(
            _an.detect_network_bursts(r["spike_data"], n, duration_ms,
                                      participation_threshold=gate, burn_in_ms=0.0),
            duration_ms, burn_in_ms=0.0)
        rates.append(st["burst_rate_hz"]); parts.append(st["mean_participation"])
        print("  sahp_ainc_slow=%.4f -> burst rate %.3f Hz, participation %.2f"
              % (v, st["burst_rate_hz"], st["mean_participation"]), flush=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(vals, rates, "o-", color="#c0392b")
    for v, rr in zip(vals, rates):
        for nm, kv in cfg["states"].items():
            if abs(kv - v) < 1e-12:
                ax.annotate(nm, (v, rr), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("sahp_ainc_slow  (slow-AHP / M-current strength)")
    ax.set_ylabel("burst rate (Hz)")
    ax.set_title("Single knob: adaptation strength sets burst density\n"
                 "(left = weak adaptation / more excitable)")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    plt.show()
    return dict(values=list(vals), burst_rate_hz=rates, participation=parts)


def show_rasters(cfg, n_each=2, shuffled=False):
    """Display generated rasters inline, a few per state."""
    from IPython.display import Image, display
    for st in cfg["states_to_run"]:
        pat = "recording*_raster%s.png" % ("_shuffled" if shuffled else "")
        fs = sorted(glob.glob(os.path.join(out_dir(cfg, st), pat)))
        print("%s: %d rasters, showing %d" % (st, len(fs), min(n_each, len(fs))))
        for f in fs[:n_each]:
            display(Image(filename=f))
