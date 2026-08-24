"""Split-half reliability of the GLM edge ranking, per sweep session.

Fits sum4 |W| separately on the even- and odd-indexed recordings and reports
the Spearman rank correlation of the off-diagonal |W| scores -- a LABEL-FREE
reliability number computable on real data too.

Writes results/<state>/glm/splithalf.json per session and the sweep figure
splithalf_reliability.png.

    python analysis/sweep_splithalf.py [session ...]         # default: all sweep_*
"""
import importlib
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

C50, C40 = "#1f5fd0", "#c0392b"


def subset(M, bnd, recs):
    """Columns of the given recording indices, boundaries rebuilt."""
    blocks, nb = [], [0]
    for r in recs:
        blocks.append(M[:, bnd[r]:bnd[r + 1]])
        nb.append(nb[-1] + (bnd[r + 1] - bnd[r]))
    return sp.hstack(blocks, format="csr"), nb


def run_session(session_name):
    os.environ["DATASET_SESSION"] = session_name
    import burstexcl_glm_arm as bx
    importlib.reload(bx)
    import sparse_glm as sg
    from session_paths import results_dir

    M, bnd = sg.load_session(bx.SESSION, bin_ms=bx.BIN_MS)
    n = M.shape[0]
    n_rec = len(bnd) - 1
    off = ~np.eye(n, dtype=bool)

    halves = []
    for recs in (range(0, n_rec, 2), range(1, n_rec, 2)):
        Mh, bh = subset(M, bnd, list(recs))
        halves.append(np.abs(bx.sum4_W(Mh, bh))[off])
    rho = float(spearmanr(halves[0], halves[1]).statistic)
    print("%s: split-half Spearman rho = %.4f" % (session_name, rho), flush=True)

    out = os.path.join(results_dir(session_name, os.environ.get("DATASET_STATE", "normal"),
                                   "glm"), "splithalf.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(session=session_name, n_recordings=n_rec,
                       spearman_rho=rho), fh, indent=1)
    return rho


def figure(results):
    from session_paths import DATA, resolve
    OUT = os.path.join(DATA, "sweep_summary")
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for session, rho in results.items():
        with open(os.path.join(resolve(session, "normal"), "session_metadata.json"),
                  encoding="utf-8") as fh:
            m = json.load(fh)
        b = np.mean([r.get("n_bursts", 0) for r in m["recordings"]])
        ax.plot(b, rho, "o", ms=8, color=C50 if "_c50_" in session else C40)
    ax.set_xlabel("bursts per recording")
    ax.set_ylabel("split-half Spearman rho of |W| ranking")
    ax.set_title("Edge-ranking reliability (100 vs 100 recordings; blue c50, red c40)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "splithalf_reliability.png"),
                                    dpi=140, facecolor="white"); plt.close(fig)
    print("figure -> %s" % os.path.join(OUT, "splithalf_reliability.png"))


def main():
    from session_paths import list_sessions
    sessions = sys.argv[1:] or sorted(s for s in list_sessions() if s.startswith("sweep_"))
    results = {}
    for s in sessions:
        results[s] = run_session(s)
    if len(results) > 1:
        figure(results)


if __name__ == "__main__":
    main()
