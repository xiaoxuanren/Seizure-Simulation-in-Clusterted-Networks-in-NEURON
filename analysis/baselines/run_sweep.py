"""Local parallel driver: run_dataset.py over every (session, state) pair.

    python analysis/baselines/run_sweep.py [--sessions all|a,b] [--states normal,seizure]
        [--n-recordings 200] [--methods all] [--arms full,burst_excluded]
        [--workers 8] [--threads 3] [--force]

One subprocess per (session, state): memory is released between jobs and
BLAS threads are capped per process so ``workers x threads`` fits the box.
Logs go to sweep_summary/baselines/logs/<session>_<state>_n<NNN>.log; a job
whose outputs all exist is skipped unless --force.  The CHTC kit
(chtc/make_baselines_manifest.py) is the cluster equivalent, one job per
(dataset, method).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(ANALYSIS)
for _p in (REPO, ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from baselines import common as cm  # noqa: E402
from baselines import registry  # noqa: E402


def outputs_exist(session, state, n_tag, methods, arms):
    d = cm.baselines_results_dir(session, state, create=False)
    return all(os.path.exists(os.path.join(d, "%s__%s__%s.json" % (m, a, n_tag)))
               for m in methods for a in arms)


def run_job(session, state, n_rec, methods, arms, threads, force, log_dir):
    n_tag = "n%03d" % n_rec
    log = os.path.join(log_dir, "%s_%s_%s.log" % (session, state, n_tag))
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(threads)
    cmd = [sys.executable, os.path.join(HERE, "run_dataset.py"), "--session", session,
           "--state", state, "--n-recordings", str(n_rec), "--methods", ",".join(methods),
           "--arms", ",".join(arms)]
    if force:
        cmd.append("--force")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n=== %s ===\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
    return session, state, rc, time.time() - t0, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", default="all")
    ap.add_argument("--states", default="normal,seizure")
    ap.add_argument("--n-recordings", type=int, default=200)
    ap.add_argument("--methods", default="all")
    ap.add_argument("--arms", default="full,burst_excluded")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    sessions = cm.sweep_sessions() if a.sessions == "all" else [s.strip() for s in a.sessions.split(",")]
    states = [s.strip() for s in a.states.split(",")]
    methods = registry.resolve(a.methods)
    arms = [x.strip() for x in a.arms.split(",")]
    n_tag = "n%03d" % a.n_recordings
    log_dir = os.path.join(cm.summary_dir(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    jobs = []
    for s in sessions:
        for st in states:
            if not a.force and outputs_exist(s, st, n_tag, methods, arms):
                print("skip %s/%s (done)" % (s, st))
                continue
            jobs.append((s, st))
    print("%d jobs x %d methods x %d arms | %d workers x %d threads | logs -> %s"
          % (len(jobs), len(methods), len(arms), a.workers, a.threads, log_dir), flush=True)
    t0 = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(run_job, s, st, a.n_recordings, methods, arms, a.threads, a.force,
                            log_dir): (s, st) for s, st in jobs}
        done = 0
        for f in as_completed(futs):
            s, st, rc, dt, log = f.result()
            done += 1
            status = "ok" if rc == 0 else "FAILED rc=%d" % rc
            if rc != 0:
                failed.append((s, st, log))
            print("[%d/%d] %s/%s %s (%.0f min) | elapsed %.0f min"
                  % (done, len(jobs), s, st, status, dt / 60, (time.time() - t0) / 60), flush=True)
    print("done in %.1f h; %d failed" % ((time.time() - t0) / 3600, len(failed)))
    for s, st, log in failed:
        print("  FAILED %s/%s -> %s" % (s, st, log))


if __name__ == "__main__":
    main()
