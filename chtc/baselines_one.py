"""Run ONE (dataset, method) baseline job on CHTC (spike-only inputs).

    python chtc/baselines_one.py --session sweep_c50_seed01 --state normal \
        --method ccg_jitter --n-recordings 200 --data data --out out

Inputs: the session's spike-only recordings + network npz, shipped as
spikes_<session>_<state>.tar.gz and extracted to data/<session>/<state>/,
plus repo_baselines.tar.gz (the code).  Both arms (full, burst_excluded) are
run; the ground truth is read only by the evaluation step, exactly as in the
local runner.  Output: out/<method>__<arm>__n<NNN>.{json,npz} and the burst
window cache, packed by baselines_job.sh into one tar per job.
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines.run_dataset import run_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n-recordings", type=int, default=200)
    ap.add_argument("--arms", default="full,burst_excluded")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    sd = os.path.join(a.data, a.session, a.state)
    if not glob.glob(os.path.join(sd, "recording*.npz")):
        sd = os.path.join(a.data, a.session)
    os.makedirs(a.out, exist_ok=True)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    summary = run_one(a.session, a.state, a.n_recordings, [a.method], arms,
                      force=True, out_dir=a.out, session_dir=sd)
    if not all(s["ok"] for s in summary):
        raise SystemExit("job failed: %s" % summary)


if __name__ == "__main__":
    main()
