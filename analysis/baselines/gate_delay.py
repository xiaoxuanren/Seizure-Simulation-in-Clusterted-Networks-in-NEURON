"""Delay-recovery validation gate.

For every method that exposes ``lag_profile``, take the true EXCITATORY edges,
find the lag at which the method's evidence peaks, and correlate it with the
edge's true conduction delay (1.5 + 2.0 x distance ms, 1.5-13.5 ms).  The
shipped GLM's peak readout reaches r = 0.807 on the flagship.  A method with
r < 0.3 is transposed or broken; the transposed statistic is printed next to
it so a swap is unmistakable (a transposed method shows r_transposed >> r).

    python analysis/baselines/gate_delay.py --session sweep_c50_seed01 --state normal \
        --n-recordings 50 --methods tier1

Writes sweep_summary/baselines/delay_gate_<session>_<state>_n<NNN>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(ANALYSIS)
for _p in (REPO, ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from baselines import common as cm  # noqa: E402
from baselines import evaluate as ev  # noqa: E402
from baselines import registry  # noqa: E402

R_MIN = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sweep_c50_seed01")
    ap.add_argument("--state", default="normal")
    ap.add_argument("--n-recordings", type=int, default=50)
    ap.add_argument("--methods", default="tier1")
    a = ap.parse_args()

    sd = cm.dataset_dir(a.session, a.state)
    gt = ev.GroundTruth(sd)
    sess = cm.Session(sd, a.n_recordings)
    methods = registry.resolve(a.methods)
    out = dict(session=a.session, state=a.state, n_recordings=a.n_recordings,
               n_true_exc=gt.n_exc, r_min=R_MIN, methods={})
    print("delay gate: %s/%s n=%d | %d true excitatory edges | delays %.1f-%.1f ms"
          % (a.session, a.state, a.n_recordings, gt.n_exc,
             gt.delay_ms[gt.A_exc].min(), gt.delay_ms[gt.A_exc].max()), flush=True)
    print("  %-18s %8s %8s %8s %10s  %s" % ("method", "r", "r_T", "res_ms", "MAE_ms", "verdict"))
    for name in methods:
        mod = registry.get(name)
        if not hasattr(mod, "lag_profile"):
            out["methods"][name] = dict(applicable=False,
                                        reason="no lag profile (undirected)" if not mod.DIRECTED
                                        else "no lag profile")
            print("  %-18s %8s %8s %8s %10s  n/a (%s)" % (name, "-", "-", "-", "-",
                                                         out["methods"][name]["reason"]), flush=True)
            continue
        t0 = time.time()
        lags_ms, E = mod.lag_profile(sess, gt.n)
        r = ev.delay_recovery(E, lags_ms, gt, signed=True)
        r["elapsed_s"] = time.time() - t0
        r["applicable"] = True
        r["lags_ms"] = np.asarray(lags_ms, float).tolist()
        verdict = "PASS" if r["r"] >= R_MIN else "FAIL"
        if r["r_transposed"] > r["r"] + 0.1:
            verdict += " (TRANSPOSED?)"
        r["verdict"] = verdict
        out["methods"][name] = r
        print("  %-18s %8.3f %8.3f %8.2f %10.2f  %s  (%.0fs)"
              % (name, r["r"], r["r_transposed"], r["lag_resolution_ms"],
                 r["mean_abs_error_ms"], verdict, r["elapsed_s"]), flush=True)
        del E
    p = os.path.join(cm.summary_dir(), "delay_gate_%s_%s_n%03d.json"
                     % (a.session, a.state, a.n_recordings))
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("-> %s" % p)
    bad = [n for n, r in out["methods"].items() if r.get("applicable") and r["r"] < R_MIN]
    if bad:
        print("GATE FAILED for: %s" % ", ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
