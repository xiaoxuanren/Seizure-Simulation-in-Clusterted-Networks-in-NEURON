"""Build the baseline-sweep manifest + input tarballs.  Run LOCALLY.

Writes, into chtc/:
  baselines_jobs.txt               '<session> <state> <method>' per line
                                   (20 sessions x 2 states x |methods|)
  repo_baselines.tar.gz            the code the job needs (sparse_glm,
                                   glm_connectivity, chtc/analysis_one.py,
                                   analysis/session_paths.py,
                                   analysis/baselines/*.py,
                                   neuron_simulation/analysis.py)
  spikes/spikes_<session>_<state>.tar.gz   per (session, state): the
                                   spike-only recording*.npz + network npz

Then upload chtc/baselines_jobs.txt, chtc/repo_baselines.tar.gz,
chtc/spikes/, chtc/baselines_job.sh, chtc/baselines.sub to the access point
and submit with  SWEEP=baselines bash chtc/submit_all.sh YOUR_NETID.

    python chtc/make_baselines_manifest.py
    python chtc/make_baselines_manifest.py --methods tier1 --states normal
    python chtc/make_baselines_manifest.py --done-root "notebooks/NEURON data parallel"
"""
import argparse
import glob
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from baselines import registry  # noqa: E402
from baselines import common as cm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cm.DATA)
    ap.add_argument("--states", default="normal,seizure")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--n-recordings", type=int, default=200)
    ap.add_argument("--done-root", default=None,
                    help="skip (session, state, method) whose "
                         "<root>/<session>/results/<state>/baselines/<method>__burst_excluded__n<NNN>.json exists")
    ap.add_argument("--no-tars", action="store_true")
    a = ap.parse_args()

    sessions = a.sessions or cm.sweep_sessions()
    states = [s.strip() for s in a.states.split(",")]
    methods = registry.resolve(a.methods)
    n_tag = "n%03d" % a.n_recordings

    lines, skipped = [], 0
    for s in sessions:
        for st in states:
            for m in methods:
                if a.done_root and os.path.exists(os.path.join(
                        a.done_root, s, "results", st, "baselines",
                        "%s__burst_excluded__%s.json" % (m, n_tag))):
                    skipped += 1
                    continue
                lines.append("%s %s %s" % (s, st, m))
    out = os.path.join(HERE, "baselines_jobs.txt")
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    print("%s: %d jobs%s (%d sessions x %d states x %d methods)"
          % (out, len(lines), " (%d done, skipped)" % skipped if skipped else "",
             len(sessions), len(states), len(methods)))
    if a.no_tars:
        return

    tar_path = os.path.join(HERE, "repo_baselines.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel in ("sparse_glm.py", "glm_connectivity.py", "chtc/analysis_one.py",
                    "chtc/baselines_one.py", "analysis/session_paths.py",
                    "neuron_simulation/analysis.py"):
            tar.add(os.path.join(REPO, rel), arcname="repo/" + rel)
        for p in sorted(glob.glob(os.path.join(REPO, "analysis", "baselines", "*.py"))):
            tar.add(p, arcname="repo/analysis/baselines/" + os.path.basename(p))
    print("%s: %.2f MB" % (tar_path, os.path.getsize(tar_path) / 1e6))

    spikes_dir = os.path.join(HERE, "spikes")
    os.makedirs(spikes_dir, exist_ok=True)
    for s in sessions:
        for st in states:
            dst = os.path.join(spikes_dir, "spikes_%s_%s.tar.gz" % (s, st))
            if os.path.exists(dst):
                print("%s: exists, kept" % dst)
                continue
            src = os.path.join(a.data, s, st)
            recs = [p for p in sorted(glob.glob(os.path.join(src, "recording*.npz")))
                    if "raster" not in os.path.basename(p)]
            nets = sorted(glob.glob(os.path.join(src, "network_*.npz")))
            if not recs or not nets:
                raise SystemExit("missing recordings or network npz in %s" % src)
            with tarfile.open(dst, "w:gz") as tar:
                for p in recs + nets[:1]:
                    tar.add(p, arcname="data/%s/%s/%s" % (s, st, os.path.basename(p)))
            print("%s: %d recs, %.1f MB" % (dst, len(recs), os.path.getsize(dst) / 1e6))


if __name__ == "__main__":
    main()
