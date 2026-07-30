"""Normal vs seizure: does the label-free operating point transfer?

Both states are the SAME network (926 neurons, 13356 edges, identical ground
truth); they differ only in ``sahp_ainc_slow`` (0.01 vs 0.004). So any difference
here is caused by the activity regime, not by the connectivity being inferred.

Draws the comparison the per-state figures cannot show:

    (a) realized vs nominal FDR at n=200 -- the calibration gap, per state
    (b) realized FDR vs duration at the shipped target 0.70
    (c) F1 vs duration at the shipped target
    (d) best-F1 target vs duration -- where the optimum actually sits

Writes ``compare_states_glm.png`` + ``compare_states_glm.json`` to the session's
results root (they belong to neither state alone).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import results_dir, session_dir  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SESSION = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
STATES = ("normal", "seizure")
COLORS = {"normal": "#1f4e79", "seizure": "#c0392b"}
SHIPPED = 0.70


def load(state):
    p = os.path.join(results_dir(SESSION, state, "glm", create=False),
                     "fdrdur10to200_metrics.json")
    if not os.path.exists(p):
        raise SystemExit("missing %s -- run fdr_duration_200_* for %s first"
                         % (p, state))
    d = json.load(open(p))
    return (np.array(d["sizes"], float), np.array(d["targets"], float),
            {k: np.array(v) for k, v in d["grids"].items()})


def main():
    data = {s: load(s) for s in STATES}
    out = {}

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) realized vs nominal at the largest common duration
    for s in STATES:
        x, tg, G = data[s]
        i = len(x) - 1
        ax[0, 0].plot(tg, G["realized_fdr"][:, i], "-o", ms=4, lw=1.6,
                      color=COLORS[s], label="%s (n=%d)" % (s, int(x[i])))
    ax[0, 0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax[0, 0].axvline(SHIPPED, color="gray", ls=":", lw=1.2)
    ax[0, 0].set_xlabel("nominal target FDR")
    ax[0, 0].set_ylabel("realized FDR")
    ax[0, 0].set_title("(a) the jitter null is conservative on normal,\n"
                       "roughly honest on seizure")
    ax[0, 0].legend(fontsize=8)

    # (b) realized FDR vs duration at the shipped target
    for s in STATES:
        x, tg, G = data[s]
        ti = int(np.argmin(np.abs(tg - SHIPPED)))
        y = G["realized_fdr"][ti]
        ax[0, 1].plot(x, y, "-o", ms=4, lw=1.6, color=COLORS[s], label=s)
        out.setdefault(s, {})["realized_fdr_at_shipped"] = y.tolist()
    ax[0, 1].axhline(0.10, color="gray", ls=":", lw=1)
    ax[0, 1].set_xlabel("recordings (= minutes)")
    ax[0, 1].set_ylabel("realized FDR")
    ax[0, 1].set_title("(b) at the shipped target %.2f, realized FDR\n"
                       "FALLS with data on normal, RISES on seizure" % SHIPPED)
    ax[0, 1].legend(fontsize=8)

    # (c) F1 vs duration at the shipped target
    for s in STATES:
        x, tg, G = data[s]
        ti = int(np.argmin(np.abs(tg - SHIPPED)))
        y = G["f1"][ti]
        ax[1, 0].plot(x, y, "-o", ms=4, lw=1.6, color=COLORS[s], label=s)
        out[s]["f1_at_shipped"] = y.tolist()
    ax[1, 0].set_xlabel("recordings (= minutes)")
    ax[1, 0].set_ylabel("F1 (excitatory)")
    ax[1, 0].set_title("(c) F1 at the shipped target")
    ax[1, 0].legend(fontsize=8)

    # (d) where the optimum actually is
    for s in STATES:
        x, tg, G = data[s]
        best = tg[np.nanargmax(G["f1"], axis=0)]
        ax[1, 1].plot(x, best, "-o", ms=4, lw=1.6, color=COLORS[s], label=s)
        out[s]["best_f1_target"] = best.tolist()
        out[s]["best_f1"] = np.nanmax(G["f1"], axis=0).tolist()
        out[s]["sizes"] = x.tolist()
    ax[1, 1].axhline(SHIPPED, color="gray", ls=":", lw=1.2,
                     label="shipped %.2f" % SHIPPED)
    ax[1, 1].set_ylim(0, 1.02)
    ax[1, 1].set_xlabel("recordings (= minutes)")
    ax[1, 1].set_ylabel("best-F1 nominal target")
    ax[1, 1].set_title("(d) the optimum target is 0.70 on normal but\n"
                       "slides to 0.10 on seizure as data accumulates")
    ax[1, 1].legend(fontsize=8)

    for a in ax.ravel():
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same network, same ground truth, one parameter different "
                 "(sahp_ainc_slow 0.01 vs 0.004)\n"
                 "The label-free operating point does NOT transfer between "
                 "activity regimes", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    root = os.path.join(session_dir(SESSION), "results")
    os.makedirs(root, exist_ok=True)
    png = os.path.join(root, "compare_states_glm.png")
    fig.savefig(png, dpi=145, facecolor="white", bbox_inches="tight")
    json.dump(dict(session=SESSION, shipped_target=SHIPPED, states=out),
              open(os.path.join(root, "compare_states_glm.json"), "w"), indent=2)

    print("shipped target %.2f, at the largest duration:" % SHIPPED)
    for s in STATES:
        x, tg, G = data[s]
        ti = int(np.argmin(np.abs(tg - SHIPPED)))
        print("  %-8s realized FDR %.4f | F1 %.4f | best-F1 target %.2f (F1 %.4f)"
              % (s, G["realized_fdr"][ti, -1], G["f1"][ti, -1],
                 tg[np.nanargmax(G["f1"][:, -1])], np.nanmax(G["f1"][:, -1])))
    print("\nsaved -> results/compare_states_glm.png / .json")


if __name__ == "__main__":
    main()
