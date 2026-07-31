"""Write FIGURES.md: which pipeline produced each figure, and with what settings.

Every entry names the generating script and the operating point, so a figure in a
talk or a paper can be traced back to the code and re-made. Run after
regenerating figures.

    python write_figures_index.py [--session ... --states normal seizure]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import results_dir, session_dir  # noqa: E402

#: figure basename -> (generating script, what it shows, pipeline notes)
FIGURES = {
    "glm_labelfree_scaling_vs_duration.png": (
        "analysis/glm_labelfree_scaling.py",
        "Edge recovery vs recording duration, three layers "
        "(excitatory / inhibitory / all edges).",
        "LABEL-FREE inference. Sparse lag-resolved ridge GLM (bin 5 ms, lags 1-6, "
        "l2=2.0), sum4 readout. Threshold from a spike-jitter null "
        "(+/-25 ms, 8 surrogates, seed 1) at target FDR 0.70 -- ground truth is "
        "used only to score, never to threshold. Inhibitory layer restricted to "
        "pairs whose presynaptic neuron is typed inhibitory by the RANK rule "
        "(infer_inhibitory typing='rank', fraction=0.25) -- the A1 fix."),
    "glm_scaling_vs_duration.png": (
        "analysis/glm_scaling.py",
        "ORACLE upper bounds vs duration: best achievable F1, and recall at a "
        "true 10% FDR.",
        "Same GLM fit, but thresholds are chosen USING ground truth. Not a "
        "deployable result -- it is the ceiling the label-free curve is measured "
        "against. Excitatory layer only."),
    "glm_distance_recovery.png": (
        "analysis/glm_distance_recovery.py",
        "Distance-resolved recovery: (a) recall/precision vs inter-neuron "
        "distance, (b) within- vs between-cluster recall, (c) GLM score for true "
        "edges vs background, (d) edge-count density vs distance.",
        "Reads glm_connectivity_sum4_5ms.npz produced by "
        "analysis/glm_fit_current.py -- the REVISED pipeline including the A1 "
        "typing fix. Earlier versions of this figure used a pre-fix file whose "
        "inhibitory layer was effectively empty (4 neurons typed, 8 edges)."),
    "fdrdur10to200_calibration.png": (
        "analysis/fdr_duration_200_worker.py + _combine.py",
        "Realized vs nominal FDR per duration; calibration error; and the "
        "realized-FDR heatmap over (target x duration).",
        "Same label-free operating point, swept over 13 durations x 10 nominal "
        "targets. Nominal target 1.0 is excluded from the PLOTS (it admits every "
        "candidate pair by construction); the CSV and JSON keep it."),
    "fdrdur10to200_performance.png": (
        "analysis/fdr_duration_200_worker.py + _combine.py",
        "F1, recall, precision, realized FDR, TP and predicted-edge count vs "
        "duration, one curve per nominal target.",
        "As above."),
    "burstexcl_arms.png": (
        "analysis/burstexcl_glm_arm.py + burstexcl_combine.py",
        "Three-arm test of whether the initialization-locked burst drives edge "
        "recovery: full data vs dropping the burst window vs dropping a matched "
        "window elsewhere.",
        "Same GLM operating point and the SAME jitter seed in all three arms. "
        "Boundaries are recomputed per arm so lagged features cannot cross a cut."),
    "burstwindows_p035_starts.png": (
        "analysis/burst_windows_p035.py",
        "Burst start times across recordings, split into initialization-locked "
        "and spontaneous classes.",
        "No inference. Burst detection only: participation > 0.35 of distinct "
        "neurons over a data-defined event window."),
    "burst_gate_sensitivity.png": (
        "analysis/burst_gate_sensitivity.py",
        "How the burst count and rate depend on the acceptance gate.",
        "No inference. Same detector with the acceptance gate swept; the "
        "bracketing stage is held fixed."),
    "glm_labelfree_fdr_duration.png": (
        "analysis/glm_labelfree_fdr_duration.py",
        "SUPERSEDED -- the earlier FDR target x duration sweep, 5-100 recordings.",
        "Replaced by fdrdur10to200_*.png, which covers 10-200 recordings and "
        "excludes the degenerate target 1.0 from its plots. Kept for provenance; "
        "do not mix it with the newer figures in one presentation, since the "
        "duration axis stops at 100."),
    "glm_predicted_topology.png": (
        "analysis/glm_predicted_topology.py",
        "Predicted vs true wiring, whole network: spatial map with every edge "
        "drawn (TP green, FP red, FN orange dotted) and the cluster-sorted "
        "adjacency matrix.",
        "Reads glm_connectivity_sum4_5ms.npz from analysis/glm_fit_current.py "
        "(label-free sum4 @FDR 0.70, A1 typing fix). At 926 neurons individual "
        "neurons are not resolvable -- use the zoom figures for that."),
    "glm_topology_zoom_combined.png": (
        "analysis/glm_topology_zoom.py",
        "The three largest clusters, one row each: spatial zoom with neurons "
        "labelled by global id, and the within-cluster adjacency submatrix.",
        "Same npz and operating point as glm_predicted_topology.png; this is a "
        "crop of it, not a re-fit. Only WITHIN-cluster pairs appear, so the "
        "precision and recall printed per cluster are local, not the global "
        "P=0.87 / R=0.85."),
    "glm_topology_zoom_group1.png": (
        "analysis/glm_topology_zoom.py",
        "Four spatially adjacent clusters together: between-cluster edges are "
        "visible, and the adjacency matrix is blocked by cluster.",
        "Same npz and operating point. Diagonal blocks are within-cluster pairs, "
        "off-diagonal blocks between-cluster; the title reports recall split the "
        "same way. Groups are chosen by centroid proximity, seeded from the "
        "largest clusters (glm_topology_zoom.py --group-size / --n-groups)."),
    "compare_states_glm.png": (
        "analysis/compare_states_glm.py",
        "Normal vs seizure: does the label-free operating point transfer between "
        "activity regimes?",
        "Reads both states' FDR x duration grids. Same network and ground truth "
        "in both; the states differ only in sahp_ainc_slow (0.01 vs 0.004)."),
}

#: numbered/parameterised figures share one entry; longest prefix wins
FAMILIES = {
    "glm_topology_zoom_group": "glm_topology_zoom_group1.png",
    "glm_topology_zoom_cluster": "glm_topology_zoom_combined.png",
}


def describe(basename):
    if basename in FIGURES:
        return FIGURES[basename]
    for pre in sorted(FAMILIES, key=len, reverse=True):
        if basename.startswith(pre):
            return FIGURES[FAMILIES[pre]]
    return ("(unknown)", "(undocumented figure)", "")


HEADER = """# Figures — {session}

Which pipeline produced each figure, and at what operating point.

**Two kinds of result appear here.** *Label-free* means the detection threshold
came from a spike-jitter surrogate null with no access to ground truth; ground
truth is used only to score the result afterwards. *Oracle* means the threshold
was chosen using ground truth — an upper bound, not a deployable method. Only
`glm_scaling_vs_duration.png` is oracle.

Regenerate any figure with:

```bash
DATASET_SESSION={session} DATASET_STATE=<state> python <script>
```

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="IC-locked_flagship_200rec")
    ap.add_argument("--states", nargs="+", default=["normal", "seizure"])
    a = ap.parse_args()

    lines = [HEADER.format(session=a.session)]
    for state in a.states:
        fd = results_dir(a.session, state, "figures", create=False)
        if not os.path.isdir(fd):
            continue
        present = sorted(f for f in os.listdir(fd) if f.endswith(".png"))
        if not present:
            continue
        # recording count actually used, where the metrics record it
        n_used = ""
        mp = os.path.join(results_dir(a.session, state, "glm", create=False),
                          "glm_labelfree_scaling_metrics.json")
        if os.path.exists(mp):
            rows = json.load(open(mp)).get("rows", [])
            if rows:
                n_used = "  (scaling curves run to %d recordings)" % max(
                    r["n_rec"] for r in rows)
        lines.append("## %s%s\n" % (state, n_used))
        for f in present:
            script, shows, notes = describe(f)
            lines.append("### `%s`\n" % f)
            lines.append("**Generated by:** `%s`\n" % script)
            lines.append("%s\n" % shows)
            if notes:
                lines.append("*Pipeline:* %s\n" % notes)
        lines.append("")

    root = os.path.join(session_dir(a.session), "results")
    extra = os.path.join(root, "compare_states_glm.png")
    if os.path.exists(extra):
        script, shows, notes = FIGURES["compare_states_glm.png"]
        lines.append("## cross-state\n")
        lines.append("### `compare_states_glm.png`\n")
        lines.append("**Generated by:** `%s`\n" % script)
        lines.append("%s\n" % shows)
        lines.append("*Pipeline:* %s\n" % notes)

    p = os.path.join(root, "FIGURES.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("wrote %s" % p)
    undoc = []
    for state in a.states:
        fd = results_dir(a.session, state, "figures", create=False)
        if os.path.isdir(fd):
            undoc += [f for f in os.listdir(fd)
                      if f.endswith(".png") and describe(f)[0] == "(unknown)"]
    if undoc:
        print("  UNDOCUMENTED (add to FIGURES dict): %s" % sorted(set(undoc)))


if __name__ == "__main__":
    main()
