"""Helpers behind the dephased-IC generation notebook.

The notebook stays declarative -- a CONFIG dict and four one-line calls. All the
subprocess plumbing, path handling and progress polling lives here so the
notebook cells read as intent rather than machinery.

    import dephase_nb as nb
    nb.preflight(CONFIG)
    nb.build_libraries(CONFIG)
    nb.generate(CONFIG)
    nb.validate(CONFIG)

Every function takes the same CONFIG dict and returns a plain result dict, so a
cell can inspect what happened without re-deriving it.
"""

import glob
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FLAGSHIP_CFG = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal",
                            "20260721_163430", "_worker_config.pkl")
DEPHASED = os.path.join(REPO, "notebooks", "NEURON data parallel", "dephased_ic")

#: measured on this machine: NEURON runs this network ~68.4x slower than realtime
REALTIME_FACTOR = 68.4


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def library_path(state, tag=""):
    return os.path.join(HERE, "dephase_state_library_%s%s.npz" % (state, tag))


def out_dir(cfg, state):
    return os.path.join(DEPHASED, cfg.get("session", "") or "", state).replace(
        os.sep + os.sep, os.sep)


def topology_path(cfg):
    """Where this CONFIG's topology lives (flagship pickle, or a built one)."""
    if cfg.get("use_flagship_topology", True):
        return FLAGSHIP_CFG
    return os.path.join(HERE, "dephase_topology_%s.pkl"
                        % cfg.get("topology_tag", "custom"))


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #
def ensure_topology(cfg, verbose=True):
    """Return the path to a worker-config pickle for this CONFIG's topology.

    With ``use_flagship_topology=True`` (default) this is the flagship's own
    pickle, so the graph is identical to the 200-recording dataset and a
    flagship-vs-dephased comparison isolates the initial condition.

    Set it False to build a fresh graph from ``CONFIG['topology']``. The result
    is cached, because the warm-start library and the recordings MUST share one
    graph -- rebuilding it between them would silently invalidate the library.
    """
    path = topology_path(cfg)
    if cfg.get("use_flagship_topology", True):
        if verbose:
            print("topology: flagship graph (identical to the 200-rec dataset)")
        return path
    if os.path.exists(path):
        if verbose:
            with open(path, "rb") as fh:
                t = pickle.load(fh)["topology"]
            print("topology: cached custom graph (%d neurons, %d edges) -> %s"
                  % (t["n_neurons"], len(t["connections"]), os.path.basename(path)))
        return path

    sys.path.insert(0, REPO)
    from neuron_simulation.topology import build_topology_lognormal, build_topology
    kind = cfg.get("topology_kind", "lognormal")
    tk = dict(cfg["topology"])
    if verbose:
        print("topology: building a NEW %s graph (seed=%s) ..."
              % (kind, tk.get("seed")), flush=True)
    topo = (build_topology_lognormal(**tk) if kind == "lognormal"
            else build_topology(**tk))

    base = pickle.load(open(FLAGSHIP_CFG, "rb"))
    base["topology"] = topo
    base["cluster_info"] = topo["cluster_info"]
    with open(path, "wb") as fh:
        pickle.dump(base, fh)
    if verbose:
        print("  built %d neurons, %d edges -> %s"
              % (topo["n_neurons"], len(topo["connections"]),
                 os.path.basename(path)), flush=True)
    return path


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def preflight(cfg):
    """Check everything that could waste compute, and report the plan."""
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
             "load_mechanisms()" % REPO],
            capture_output=True, cwd=REPO).returncode:
        ok = False
        print("FAIL    : mechanisms not built -- run  "
              "cd neuron_simulation && nrnivmodl mechanisms")

    topo_pkl = ensure_topology(cfg)

    flag = pickle.load(open(FLAGSHIP_CFG, "rb"))["build_kwargs"]
    diff = {k: (flag.get(k), v) for k, v in cfg["build"].items()
            if k != "sahp_ainc_slow" and flag.get(k) != v}
    if diff:
        print("build   : %d parameter(s) changed from the flagship" % len(diff))
        for k, (was, now) in sorted(diff.items()):
            print("            %-20s %r -> %r" % (k, was, now))
        print("          (allowed, but the dataset is then not an IC-only "
              "comparison)")
    else:
        print("build   : matches the flagship (IC-only comparison)")

    print("")
    print("%-9s %-8s %-30s %s" % ("state", "knob", "warm-start library", "recordings"))
    for s in cfg["states_to_run"]:
        lp = library_path(s, cfg.get("library_tag", ""))
        if os.path.exists(lp):
            lb = np.load(lp)
            k = float(lb["sahp_ainc_slow"]) if "sahp_ainc_slow" in lb else None
            if k is not None and abs(k - cfg["states"][s]) > 1e-12:
                txt, ok = "MISMATCH: built at %.4f" % k, False
            else:
                txt = "%d snapshots, sd %.5f uS" % (
                    lb["g_slow"].shape[0], lb["g_slow"].std(axis=1).mean())
        else:
            txt = "MISSING -> run the library cell"
            need_library.append(s)
        d = out_dir(cfg, s)
        todo[s] = [r for r in range(cfg["n_recordings"])
                   if not os.path.exists(os.path.join(d, "recording%03d.npz" % r))]
        print("%-9s %-8.4f %-30s %d have / %d to go"
              % (s, cfg["states"][s], txt, cfg["n_recordings"] - len(todo[s]),
                 len(todo[s])))

    n = sum(len(v) for v in todo.values())
    per_min = ((cfg["recording_ms"] + 1000.0 + cfg["discard_extra_ms"]) / 1000.0
               * REALTIME_FACTOR / 60.0)
    gb = n * {"all": 77.0, "probe": 3.0, "none": 0.6}[cfg["voltage"]] / 1024.0
    print("")
    print("generate: %d recordings -> ~%.1f h with %d workers, ~%.2f GB"
          % (n, n * per_min / 60.0 / max(1, cfg["n_workers"]), cfg["n_workers"], gb))
    if need_library:
        print("library : %d to build -> ~%.1f h (they run concurrently)"
              % (len(need_library),
                 cfg["warmup_ms"] / 1000.0 * REALTIME_FACTOR / 3600.0))
    print("")
    print("PREFLIGHT %s" % ("OK" if ok else "FAILED -- fix the above first"))
    return dict(ok=ok, need_library=need_library, todo=todo, topology_pkl=topo_pkl)


# --------------------------------------------------------------------------- #
# subprocess driver shared by build_libraries and generate
# --------------------------------------------------------------------------- #
def _run_jobs(jobs, n_workers, poll_s=120, stagger_s=15, label="job"):
    """Run (name, cmd, logpath) jobs at most ``n_workers`` at a time."""
    queue, running, done, t0 = list(jobs), [], [], time.time()
    while queue or running:
        while queue and len(running) < n_workers:
            name, cmd, log = queue.pop(0)
            p = subprocess.Popen(cmd, stdout=open(log, "w"),
                                 stderr=subprocess.STDOUT)
            running.append((name, p, log))
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


def _common_args(cfg, state, topo_pkl):
    a = ["--state", state,
         "--sahp-ainc-slow", str(cfg["states"][state]),
         "--noise-seed-base", str(cfg["noise_seed_base"]),
         "--build-overrides", json.dumps(cfg["build"])]
    if topo_pkl and topo_pkl != FLAGSHIP_CFG:
        a += ["--topology-pkl", topo_pkl]
    return a


# --------------------------------------------------------------------------- #
# the three long-running steps
# --------------------------------------------------------------------------- #
def build_libraries(cfg, pre=None):
    """One warm-up run per state that still needs a state library."""
    pre = pre or preflight(cfg)
    if not pre["need_library"]:
        print("all libraries present -- nothing to do")
        return []
    jobs = []
    for s in pre["need_library"]:
        log = os.path.join(HERE, "_dephase_lib_%s.log" % s)
        cmd = ([sys.executable, "-u", os.path.join(HERE, "dephase_snapshot.py")]
               + _common_args(cfg, s, pre["topology_pkl"])
               + ["--duration", str(cfg["warmup_ms"]), "--snapshots"]
               + [str(int(t)) for t in cfg["snapshot_times"]])
        jobs.append(("library:%s" % s, cmd, log))
    return _run_jobs(jobs, len(jobs), label="library")


def generate(cfg, pre=None):
    """Generate every missing recording, N_WORKERS at a time, across states."""
    pre = pre or preflight(cfg)
    assert pre["ok"], "preflight failed -- do not run this"
    assert not pre["need_library"], "build the warm-start libraries first"

    jobs = []
    for s in cfg["states_to_run"]:
        todo = pre["todo"][s]
        if not todo:
            print("%s: nothing to do" % s)
            continue
        per = int(np.ceil(len(todo) / max(1, cfg["n_workers"])))
        for w in range(cfg["n_workers"]):
            chunk = todo[w * per:(w + 1) * per]
            if not chunk:
                continue
            log = os.path.join(HERE, "_dephase_nb_%s_w%d.log" % (s, w))
            cmd = ([sys.executable, "-u", os.path.join(HERE, "dephase_generate.py")]
                   + _common_args(cfg, s, pre["topology_pkl"])
                   + ["--start", str(chunk[0]), "--count", str(len(chunk)),
                      "--duration", str(cfg["recording_ms"]),
                      "--voltage", cfg["voltage"],
                      "--voltage-probe-n", str(cfg["voltage_probe_n"]),
                      "--voltage-dt", str(cfg["voltage_dt"]),
                      "--discard-extra-ms", str(cfg["discard_extra_ms"])])
            jobs.append(("%s w%d (rec %d-%d)" % (s, w, chunk[0], chunk[-1]),
                         cmd, log))
    done = _run_jobs(jobs, cfg["n_workers"], label="worker")
    for s in cfg["states_to_run"]:
        d = out_dir(cfg, s)
        print("%-9s %d recordings, %d rasters, %d summaries"
              % (s, len(glob.glob(os.path.join(d, "recording*.npz"))),
                 len(glob.glob(os.path.join(d, "recording*_raster*.png"))),
                 len(glob.glob(os.path.join(d, "_summary_*.json")))))
    return done


def validate(cfg):
    """Run the four gating checks for each state."""
    for s in cfg["states_to_run"]:
        print("=" * 70)
        print("%s  (sahp_ainc_slow = %.4f)" % (s, cfg["states"][s]))
        print("=" * 70)
        r = subprocess.run(
            [sys.executable, "-u", os.path.join(HERE, "dephase_validate.py"),
             "--state", s], capture_output=True, text=True)
        print(r.stdout[-4000:])
        if r.returncode:
            print("STDERR:", r.stderr[-1500:])
