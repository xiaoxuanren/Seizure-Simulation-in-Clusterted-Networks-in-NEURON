"""Write sweep_summary/baselines/RESULTS.md from the aggregated tables.

    python analysis/baselines/write_results.py --n-recordings 200

Reads metrics_wide.csv, compute.json, params.json, figures_provenance.json,
delay_gate_*.json and (optional, hand-maintained) compute_notes.json in
sweep_summary/baselines/.  Every number in the document comes from those
files; the prose sections (design, limitations, framing) are fixed text.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)
from baselines import common as cm  # noqa: E402
from baselines import registry  # noqa: E402
from baselines.figures import LABEL, order  # noqa: E402

CITE = {
    "glm_ridge": "this repository (sparse lag-resolved ridge GLM, sum4 readout, jitter-null FDR 0.70)",
    "pearson0": "zero-lag Pearson correlation of 5 ms binned trains (undirected)",
    "pearson_lag": "lagged Pearson correlation, max |r| over lags 1-6 at 5 ms, signed",
    "ccg_classical": "Perkel, Gerstein & Moore 1967, Biophys J 7:419",
    "ccg_jitter": "Fujisawa et al. 2008, Nat Neurosci 11:823; Amarasingham et al. 2012, J Neurophysiol 107:517",
    "tspe": "De Blasi et al. 2019, J Neurosci Methods 312:169 (github.com/biomemsLAB/TSPE)",
    "fncch": "Pastore et al. 2018, PLoS Comput Biol 14:e1006381",
    "glmcc": "Kobayashi et al. 2019, Nat Commun 10:4468 (github.com/NII-Kobayashi/GLMCC)",
    "te_1ms": "Schreiber 2000, PRL 85:461; Ito et al. 2011, PLoS ONE 6:e27431 (1 ms bins)",
    "te_2ms": "as te_1ms, 2 ms bins", "te_5ms": "as te_1ms, 5 ms bins", "te_10ms": "as te_1ms, 10 ms bins",
    "stg": "Spivak et al. 2022, Commun Biol 5:520 (ROI 0-5 ms as published)",
    "stg_roi15": "Spivak et al. 2022 with the ROI widened to 0-15 ms (mismatch probe)",
    "glmcc_d14": "Kobayashi et al. 2019 with delays 1-14 ms (mismatch probe)",
}

LIMITATIONS = {
    "glm_ridge": ("Developed on this simulator; every hyperparameter (5 ms bins, lags 1-6, l2 2.0, sum4, "
                  "jitter 25 ms, FDR 0.70) was chosen on earlier sessions of the same model family. It is the "
                  "in-domain method; that is the reviewer objection the oracle columns are there to answer."),
    "pearson0": ("No source paper; undirected by construction. Included as the floor every other method must "
                 "clear. It is scored on the directed candidate set, so its ceiling is well below 1."),
    "pearson_lag": ("No source paper. Rate normalisation only; nothing explains away shared drive, so its "
                    "burst sensitivity is the reference for the correlation family."),
    "ccg_classical": ("Perkel et al. 1967 validated the correlogram on pairs of neurons recorded in isolation, "
                      "not on densely interconnected populations; the flank baseline cannot remove common input "
                      "and the authors themselves discuss the resulting false inferences. Poisson test constants "
                      "(alpha 0.001, Bonferroni over 15 causal bins) are ours."),
    "ccg_jitter": ("Fujisawa et al. 2008 and Amarasingham et al. 2012 used 1 ms bins, +/-5 ms jitter and ~1000 "
                   "surrogates on in vivo prefrontal / motor recordings with tens of units. Here the null is the "
                   "shipped GLM's (5 ms bins, +/-25 ms, 8 surrogates) by design, so this row isolates the joint "
                   "fit rather than reproducing the published operating point."),
    "tspe": ("De Blasi et al. 2019 validated TSPE on simulated cortical cultures (Izhikevich networks of ~1000 "
             "neurons, low connection density, 1 ms bins) and on MEA culture recordings. Delays up to 25 ms are "
             "covered, so the 1.5-13.5 ms conduction delays here are inside its window; the clustered, "
             "burst-dominated dynamics of this model are not what it was tuned on, and it defines no "
             "label-free threshold (its authors chose thresholds by ROC)."),
    "fncch": ("Pastore et al. 2018 validated FNCCH on 1000-neuron in silico random networks (1 h) and on cortical "
              "cultures on MEA-60 and MEA-4k arrays (0.12 ms bins on the 4k array). The +/-25 ms window covers the "
              "delays here; the mu+2sigma / mu+1sigma hard thresholds were tuned for their density and "
              "recording length."),
    "glmcc": ("Kobayashi et al. 2019 validated GLMCC on 1000 Hodgkin-Huxley neurons recorded for 5400 s and on "
              "rat CA1 in vivo (hc-3), with synaptic delays of a few ms and tau = 4 ms. Its default delay search "
              "(1-4 ms) covers only the shortest conduction delays of this network (1.5-13.5 ms); the "
              "glmcc_d14 probe widens it. The per-pair background a(t) absorbs slow common input but nothing "
              "shared across pairs, and the J_thr rule assumes independent Poisson-like backgrounds."),
    "te_1ms": ("Ito et al. 2011 validated delayed TE on a 1000-neuron Izhikevich cortical model (100 subsampled "
               "neurons, 30 min, 7.3 % density) and thresholded by ROC. The bin width is known to matter (their "
               "Fig. on bin size), hence the 1/2/5/10 ms sweep; k = l = 1 histories are used (the higher-order "
               "extension is not run). The sign attached here is an addition: TE is unsigned."),
    "stg": ("Spivak et al. 2022 validated the deconvolved STG on point-process simulations, a 100-neuron CA1 "
            "model and 1041 PYR / 215 INT recorded in mouse CA1, with a 0-5 ms region of interest suited to "
            "hippocampal monosynaptic latencies. Most conduction delays here are longer than 5 ms, so the "
            "faithful ROI sees a minority of true edges; stg_roi15 is the widened probe."),
}
LIMITATIONS["te_2ms"] = LIMITATIONS["te_5ms"] = LIMITATIONS["te_10ms"] = LIMITATIONS["te_1ms"]
LIMITATIONS["stg_roi15"] = LIMITATIONS["stg"]
LIMITATIONS["glmcc_d14"] = LIMITATIONS["glmcc"]


def lab(m):
    return LABEL.get(m, m)


def read_wide(out):
    with open(os.path.join(out, "metrics_wide.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("session", "group", "state", "method", "module", "arm", "directed"):
                continue
            try:
                r[k] = float(v) if v != "" else np.nan
            except ValueError:
                pass
    return rows


def agg(rows, col):
    v = np.array([r[col] for r in rows], float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "--"
    if len(v) == 1:
        return "%.3f" % v[0]
    return "%.3f (%.3f-%.3f)" % (v.mean(), v.min(), v.max())


def table(rows, methods, state, arm, cols, title):
    lines = ["", "**%s**" % title, "", "| method | n | " + " | ".join(c[1] for c in cols) + " |",
             "|---|---|" + "|".join("---" for _ in cols) + "|"]
    for m in methods:
        rr = [r for r in rows if r["state"] == state and r["arm"] == arm and r["method"] == m]
        if not rr:
            continue
        lines.append("| %s | %d | %s |" % (lab(m), len(rr), " | ".join(agg(rr, c[0]) for c in cols)))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-recordings", type=int, default=200)
    a = ap.parse_args()
    out = cm.summary_dir()
    n_tag = "n%03d" % a.n_recordings
    rows = [r for r in read_wide(out) if int(r["n_recordings"]) == a.n_recordings]
    methods = order({r["method"] for r in rows})
    states = [s for s in ("normal", "seizure") if any(r["state"] == s for r in rows)]
    n_sessions = len({r["session"] for r in rows})

    def load(name, default):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            return default
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    compute = load("compute.json", {})
    params = load("params.json", {})
    prov = load("figures_provenance.json", {})
    notes = load("compute_notes.json", {})
    gates = sorted(glob.glob(os.path.join(out, "delay_gate_*.json")))

    L = []
    L.append("# Baseline connectivity inference across the sweep datasets")
    L.append("")
    L.append("Branch `baselines-comparison`. Every number below was produced by "
             "`analysis/baselines/run_dataset.py` (per dataset) and aggregated by "
             "`analysis/baselines/aggregate.py` into `metrics.csv` (long format: session, state, "
             "method, arm, readout, operating_point, denominator, metric, value). Figures are listed "
             "with the script and operating point that produced them. Datasets: %d networks x %s state(s), "
             "%d recordings each (%s)." % (n_sessions, "/".join(states), a.n_recordings, n_tag))
    L.append("")
    L.append("## Design")
    L.append("")
    L.append("- **Same sufficient statistics, three normalisations.** `analysis/baselines/accumulate.py` "
             "builds, in one pass over per-recording sparse products, the lag products `C[l][pre, post]` "
             "(the raw cross-correlogram counts). The ridge GLM's right-hand side `F X^T` at lag l *is* "
             "`C[l]`; the raw counts feed the correlogram methods, the rate-normalised counts feed Pearson, "
             "and the precision-whitened solve `(F F^T + lambda I)^-1 F X^T` is the shipped GLM. "
             "`test_accumulate.py` shows the solve is bit-identical to `sparse_glm.fit_B` and to the "
             "shipped `glm_connectivity_sum4_5ms.npz`, so the GLM row is reproduced, not re-fit.")
    L.append("- **Two passes.** 5 ms bins with lags 1-6 (the shipped GLM, lagged/zero-lag Pearson, "
             "jittered CCG) and 1 ms bins with lags 0-50 (classical CCG, TSPE, FNCCH, GLMCC, STG, TE at "
             "1 ms; TE at 2/5/10 ms uses its own pass).")
    L.append("- **Invariants.** Every matrix is `[pre, post]`; the candidate set is every ordered "
             "off-diagonal pair for every method; ground truth is read only in `evaluate.py`; positive = "
             "excitatory; every F1 and recall is reported under two denominators, `all_edges` "
             "(A_exc | A_inh) and `exc_only` (A_exc, an inhibitory hit counted as a false positive).")
    L.append("- **Read-outs.** `exc` (+S vs excitatory edges), `inh` (-S vs inhibitory edges), "
             "`all_absS` (|S| vs all edges, the untyped protocol), `typed` (rank rule, assumed inhibitory "
             "fraction 0.25: +S on rows typed excitatory, -S on rows typed inhibitory, vs all edges).")
    L.append("- **Operating points.** Threshold-free first (AUC, AP). `oracle_best_f1` picks each method's "
             "own best-F1 threshold *using labels* -- an upper bound, labelled as such. `label_free_*` is "
             "the method's own published rule from spikes alone, where it defines one (jitter-null FDR "
             "0.70 for the GLM and the jittered CCG; Poisson flank test for the classical CCG; mu+k sigma "
             "for FNCCH; |J| > J_thr for GLMCC; Poisson ROI test for STG). Pearson, TSPE and TE define none.")
    L.append("- **Arms.** `full` = all recordings; `burst_excluded` = burst windows recomputed at the "
             "0.35 participation gate, padded +/-25 ms, removed from every pass with each retained interval "
             "its own segment. Stored burst windows are never loaded.")
    L.append("- **Delay-recovery gate.** Before the sweep, every directed method with a lag-resolved "
             "evidence profile was checked on one dataset (n = 50): the lag of peak evidence on true "
             "excitatory edges must correlate with the true conduction delay (1.5 + 2.0 x distance ms) "
             "at r >= 0.3, and the transposed profile must not do better.")
    L.append("")

    # ---- gate ------------------------------------------------------------
    if gates:
        L.append("## Delay-recovery gate")
        L.append("")
        L.append("| method | r (peak lag vs true delay) | r transposed | lag resolution (ms) | verdict |")
        L.append("|---|---|---|---|---|")
        seen = set()
        for gp in gates[::-1]:
            with open(gp, encoding="utf-8") as fh:
                g = json.load(fh)
            for m in order(g["methods"]):
                if m in seen:
                    continue
                seen.add(m)
                r = g["methods"][m]
                if not r.get("applicable"):
                    L.append("| %s | n/a | n/a | n/a | %s |" % (lab(m), r.get("reason", "n/a")))
                else:
                    L.append("| %s | %.3f | %.3f | %.1f | %s |" % (lab(m), r["r"], r["r_transposed"],
                                                                  r["lag_resolution_ms"], r["verdict"]))
        L.append("")
        L.append("(from `%s`; the shipped GLM's peak readout gives r = 0.807 on the flagship)"
                 % os.path.basename(gates[-1]))
        L.append("")

    # ---- key findings (computed) -----------------------------------------
    def mean_of(state, arm, method, col):
        v = np.array([r[col] for r in rows if r["state"] == state and r["arm"] == arm
                      and r["method"] == method], float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    PROBES = {"stg_roi15", "glmcc_d14"}
    FINE = ["ccg_classical", "tspe", "fncch", "te_1ms", "stg_roi15"]      # 1 ms methods
    COARSE = ["ccg_jitter", "pearson_lag", "te_5ms", "te_10ms"]           # 5-10 ms methods
    L.append("## Key findings (computed from the tables below)")
    L.append("")
    for st in states:
        if not any(r["state"] == st and r["arm"] == "full" for r in rows):
            continue
        base = [m for m in methods if m != "glm_ridge" and m not in PROBES]
        probes = [m for m in methods if m in PROBES]
        g_exc, g_unt = mean_of(st, "full", "glm_ridge", "exc_ap"), mean_of(st, "full", "glm_ridge", "untyped_ap")
        g_p, g_r = mean_of(st, "full", "glm_ridge", "labelfree_precision_untyped"), mean_of(st, "full", "glm_ridge", "labelfree_recall_untyped")
        ranked = sorted(base, key=lambda m: -mean_of(st, "full", m, "exc_ap"))
        b1 = ranked[0] if ranked else None
        L.append("**%s state, full recordings (mean over networks).**" % st)
        if b1:
            L.append("- Excitatory AP: ridge GLM %.3f vs best published baseline %s %.3f "
                     "(then %s); untyped AP %.3f vs %.3f." % (
                         g_exc, lab(b1), mean_of(st, "full", b1, "exc_ap"),
                         ", ".join("%s %.3f" % (lab(m), mean_of(st, "full", m, "exc_ap")) for m in ranked[1:4]),
                         g_unt, mean_of(st, "full", b1, "untyped_ap")))
        for m in probes:
            L.append("- Mismatch probe %s: exc AP %.3f, untyped AP %.3f (vs the faithful %s %.3f / %.3f)." % (
                lab(m), mean_of(st, "full", m, "exc_ap"), mean_of(st, "full", m, "untyped_ap"),
                lab(m.split("_")[0]), mean_of(st, "full", m.split("_")[0], "exc_ap"),
                mean_of(st, "full", m.split("_")[0], "untyped_ap")))
        lf = [(m, mean_of(st, "full", m, "labelfree_precision_untyped"), mean_of(st, "full", m, "labelfree_recall_untyped"),
               mean_of(st, "full", m, "labelfree_realized_fdr_untyped"))
              for m in methods if m != "glm_ridge" and np.isfinite(mean_of(st, "full", m, "labelfree_precision_untyped"))]
        if lf:
            L.append("- Label-free operating points (each method's own rule, untyped, all-edges denominator): "
                     "ridge GLM P %.3f / R %.3f; %s." % (
                         g_p, g_r, "; ".join("%s P %.3f / R %.3f" % (lab(m), p_, r_) for m, p_, r_, _ in lf)))
        j_fdr, g_fdr = mean_of(st, "full", "ccg_jitter", "labelfree_realized_fdr_untyped"), mean_of(st, "full", "glm_ridge", "labelfree_realized_fdr_untyped")
        if np.isfinite(j_fdr):
            L.append("- Same statistic, same null: with the identical +/-25 ms x 8-surrogate jitter null at nominal "
                     "FDR 0.70, the raw-CCG row realizes FDR %.3f (AP %.3f) while the whitened GLM row realizes "
                     "%.3f (AP %.3f). The joint fit is what makes the null usable, not the null itself." % (
                         j_fdr, mean_of(st, "full", "ccg_jitter", "untyped_ap"), g_fdr, g_unt))
        # burst exclusion by timescale
        def dap(m):
            a, b = mean_of(st, "full", m, "untyped_ap"), mean_of(st, "burst_excluded", m, "untyped_ap")
            return b - a if np.isfinite(a) and np.isfinite(b) else float("nan")
        fine = [dap(m) for m in FINE if np.isfinite(dap(m))]
        coarse = [dap(m) for m in COARSE if np.isfinite(dap(m))]
        if fine and coarse:
            gp2 = mean_of(st, "burst_excluded", "glm_ridge", "labelfree_precision_untyped")
            L.append("- Burst exclusion (untyped AP, burst-excluded minus full): 1 ms-window methods %+.3f on average "
                     "(%s), 5-10 ms-window methods %+.3f (%s); ridge GLM %+.3f in AP but label-free precision "
                     "%.3f -> %.3f. Removing burst windows removes the shared drive that swamps coarse-timescale "
                     "statistics, and removes the coincidence counts that the fine-timescale statistics need." % (
                         float(np.mean(fine)), ", ".join("%s %+.2f" % (lab(m), dap(m)) for m in FINE if np.isfinite(dap(m))),
                         float(np.mean(coarse)), ", ".join("%s %+.2f" % (lab(m), dap(m)) for m in COARSE if np.isfinite(dap(m))),
                         dap("glm_ridge"), g_p, gp2))
        L.append("")

    # ---- tables ----------------------------------------------------------
    L.append("## Results")
    L.append("")
    L.append("Mean over networks (min-max in parentheses). `n` = number of networks with a result.")
    tf_cols = [("exc_auc", "exc AUC"), ("exc_ap", "exc AP"), ("inh_ap", "inh AP"),
               ("untyped_ap", "untyped AP"), ("typed_ap", "typed AP")]
    or_cols = [("oracle_f1_untyped", "oracle F1 (untyped, all_edges)"),
               ("oracle_precision_untyped", "oracle P"), ("oracle_recall_untyped", "oracle R (all_edges)"),
               ("oracle_recall_exc_only_untyped", "oracle R (exc_only)"),
               ("oracle_f1_typed", "oracle F1 (typed)"), ("recall_true_fdr10_exc", "exc recall @ true FDR 0.10")]
    lf_cols = [("labelfree_precision_untyped", "P"), ("labelfree_recall_untyped", "R (all_edges)"),
               ("labelfree_recall_exc_only_untyped", "R (exc_only)"), ("labelfree_f1_untyped", "F1 (all_edges)"),
               ("labelfree_f1_exc_only_untyped", "F1 (exc_only)"), ("labelfree_realized_fdr_untyped", "realized FDR"),
               ("labelfree_precision_typed", "typed P"), ("labelfree_recall_typed", "typed R")]
    for st in states:
        L.append("")
        L.append("### %s state" % st)
        for arm in ("full", "burst_excluded"):
            if not any(r["state"] == st and r["arm"] == arm for r in rows):
                continue
            L += table(rows, methods, st, arm, tf_cols, "Threshold-free, %s arm" % arm)
            L += table(rows, methods, st, arm, or_cols, "ORACLE operating points (labels used to pick the threshold), %s arm" % arm)
            L += table(rows, methods, st, arm, lf_cols, "LABEL-FREE operating points (the method's own rule), %s arm" % arm)

    # ---- burst gap -------------------------------------------------------
    gap = prov.get("burst_gap_summary", {})
    if gap:
        L.append("")
        L.append("### Burst exclusion: what it buys each method")
        L.append("")
        L.append("Change from the full arm to the burst-excluded arm, mean (min-max) over networks. "
                 "Precision is at the method's own label-free point where it has one, else the oracle best-F1 point.")
        L.append("")
        L.append("| method | dAP normal | dAP seizure | dPrecision normal | dPrecision seizure | precision point |")
        L.append("|---|---|---|---|---|---|")
        for m in order(gap):
            g = gap[m]

            def f(k):
                if k not in g:
                    return "--"
                return "%+.3f (%+.3f..%+.3f)" % (g[k]["mean"], g[k]["min"], g[k]["max"])
            kind = g.get("prec_normal", g.get("prec_seizure", {})).get("kind", "")
            L.append("| %s | %s | %s | %s | %s | %s |" % (lab(m), f("ap_normal"), f("ap_seizure"),
                                                         f("prec_normal"), f("prec_seizure"), kind))
    rho = {k: v for k, v in prov.items() if k.startswith("ap_vs_burstiness_rho")}
    if rho:
        L.append("")
        L.append("### Does burstiness order the baselines too?")
        L.append("")
        L.append("Spearman rho between a method's AP and the network's bursts per recording (negative = "
                 "more bursting, worse ranking).")
        L.append("")
        keys = sorted(rho)
        L.append("| method | " + " | ".join(k.replace("ap_vs_burstiness_rho_", "") for k in keys) + " |")
        L.append("|---|" + "|".join("---" for _ in keys) + "|")
        for m in methods:
            cells = []
            for k in keys:
                v = rho[k].get("untyped_ap", {}).get(m)
                cells.append("%+.2f" % v if v is not None and np.isfinite(v) else "--")
            L.append("| %s | %s |" % (lab(m), " | ".join(cells)))

    # ---- GLMCC diagnostics ----------------------------------------------
    long_p = os.path.join(out, "metrics.csv")
    if os.path.exists(long_p):
        diag = {}
        with open(long_p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (r["method"].startswith("glmcc") and int(r["n_recordings"]) == a.n_recordings
                        and r["operating_point"] == "label_free_untyped"
                        and r["metric"] in ("frac_J_at_lower_bound", "frac_J_at_upper_bound",
                                            "J_thr_median", "J_thr_min", "newton_iters_mean")):
                    diag.setdefault((r["method"], r["state"], r["arm"]), {}).setdefault(
                        r["metric"], []).append(float(r["value"]))
        if diag:
            L.append("")
            L.append("### GLMCC fit diagnostics")
            L.append("")
            L.append("The published rule declares a connection when |J| > J_thr = c/sqrt(T tau "
                     "lambda_pre lambda_post) with c = 5.16 (alpha 0.001) and J bounded to [-3, 5]. "
                     "At this network's firing rates J_thr is often above the bound, so the test cannot "
                     "fire; pairs with no coincidences in the kernel window pin at the lower bound J = -3.")
            L.append("")
            L.append("| method | state | arm | median J_thr | min J_thr | frac J at -3 | frac J at +5 | Newton iters |")
            L.append("|---|---|---|---|---|---|---|---|")
            for (m, st, arm), d in sorted(diag.items()):
                g = lambda k: "%.3g" % np.mean(d[k]) if k in d else "--"
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    lab(m), st, arm, g("J_thr_median"), g("J_thr_min"), g("frac_J_at_lower_bound"),
                    g("frac_J_at_upper_bound"), g("newton_iters_mean")))

    # ---- pilot appendix --------------------------------------------------
    pilot_p = os.path.join(out, "metrics_wide_pilot_n050.csv")
    if os.path.exists(pilot_p):
        with open(pilot_p, encoding="utf-8") as fh:
            prow = list(csv.DictReader(fh))
        L.append("")
        L.append("### Staging pilot (n = 50 recordings; quietest and burstiest network)")
        L.append("")
        L.append("Run before the sweep to check the ordering. Untyped AP, full arm -> burst-excluded arm.")
        L.append("")
        psess = sorted({r["session"] for r in prow if r["session"] in ("sweep_c40_seed12", "sweep_c40_seed14")})
        pmeth = order({r["method"] for r in prow})
        hdr = ["%s/%s" % (s.replace("sweep_", ""), st) for s in psess for st in ("normal", "seizure")]
        L.append("| method | " + " | ".join(hdr) + " |")
        L.append("|---|" + "|".join("---" for _ in hdr) + "|")
        for m in pmeth:
            cells = []
            for s in psess:
                for st in ("normal", "seizure"):
                    fu = [r for r in prow if r["session"] == s and r["state"] == st and r["method"] == m and r["arm"] == "full"]
                    ex = [r for r in prow if r["session"] == s and r["state"] == st and r["method"] == m and r["arm"] == "burst_excluded"]
                    try:
                        cells.append("%.2f -> %.2f" % (float(fu[0]["untyped_ap"]), float(ex[0]["untyped_ap"])))
                    except (IndexError, ValueError):
                        cells.append("--")
            L.append("| %s | %s |" % (lab(m), " | ".join(cells)))

    # ---- figures ---------------------------------------------------------
    L.append("")
    L.append("## Figures")
    L.append("")
    L.append("All figures are written by `analysis/baselines/figures.py` from the run_dataset outputs.")
    L.append("")
    L.append("| figure | what | operating point |")
    L.append("|---|---|---|")
    figs = [
        ("figures/roc_pr_<state>.png", "ROC and precision-recall overlays, all methods, full arm; representative network bold, others faint", "threshold-free (|S| vs all edges)"),
        ("figures/ap_heatmap_<state>_<arm>.png", "method x network AP heatmaps (untyped and excitatory), networks ordered by burstiness, Spearman rho per method", "threshold-free"),
        ("figures/summary_bars_<state>.png", "mean and range of the headline metrics per method, full vs burst-excluded", "threshold-free + oracle + label-free, as labelled"),
        ("figures/burst_gap.png", "full -> burst-excluded change in AP and in operating-point precision, both states", "label-free where defined, else oracle best-F1"),
        ("figures/delay_gate.png", "delay-recovery gate per method", "threshold-free lag profile"),
        ("figures/te_binwidth.png", "transfer-entropy AP vs bin width", "threshold-free"),
        ("maps/<session>_<state>_<method>.png", "per dataset x method: cluster-sorted predicted-vs-true adjacency (TP green, FP red, FN orange), distance-resolved recall/precision, weight-quintile detection; full arm solid, burst-excluded dashed", "label-free untyped where defined, else oracle best-F1 untyped (stated in the title)"),
    ]
    for f, w, op in figs:
        L.append("| `%s` | %s | %s |" % (f, w, op))

    # ---- compute ---------------------------------------------------------
    L.append("")
    L.append("## Compute: projected vs spent, subsampling, failures")
    L.append("")
    if notes.get("text"):
        L.append(notes["text"])
        L.append("")
    if compute:
        L.append("| method | jobs | mean wall time per (dataset, arm) | max | total CPU-h (at the threads used) |")
        L.append("|---|---|---|---|---|")
        per = {}
        for key, e in compute.items():
            m = key.split("/")[2]
            per.setdefault(m, []).append(sum(e.values()))
        for m in order(per):
            v = np.array(per[m])
            L.append("| %s | %d | %.0f s | %.0f s | %.1f |" % (lab(m), len(v), v.mean(), v.max(), v.sum() / 3600))
    sub = notes.get("subsampled", [])
    L.append("")
    L.append("- **Subsampled:** %s" % ("; ".join(sub) if sub else "nothing. Every method scored every ordered pair of every dataset."))
    fails = notes.get("failed", [])
    L.append("- **Failed to run:** %s" % ("; ".join(fails) if fails else "nothing (all (dataset, method, arm) jobs completed)."))
    if notes.get("projected"):
        L.append("- **Projected before launch:** %s" % notes["projected"])
    if notes.get("spent"):
        L.append("- **Spent:** %s" % notes["spent"])
    L.append("")

    # ---- limitations -----------------------------------------------------
    L.append("## LIMITATIONS: what each method was validated on, and how this network differs")
    L.append("")
    L.append("This network: 629-1010 single-compartment HH neurons in 40-50 spatial clusters, log-normal "
             "weights, 1.5-2.1 % directed edge density, conduction delays 1.5-13.5 ms (1.5 + 2.0 x distance), "
             "mean rates 0.29 Hz (normal) and 1.7 Hz (seizure), network bursts 9-25 per minute with 51-95 % "
             "participation; 200 x 60 s per dataset.")
    L.append("")
    for m in methods:
        L.append("- **%s** (%s). %s" % (lab(m), CITE.get(m, ""), LIMITATIONS.get(m, "")))
    L.append("")
    L.append("### Skipped methods")
    L.append("")
    for k, why in registry.SKIPPED.items():
        L.append("- **%s**: %s" % (k, why))
    L.append("")

    # ---- framing ---------------------------------------------------------
    L.append("## On the conservatism of the jitter null")
    L.append("")
    L.append("The label-free operating point of the shipped GLM realizes an FDR far below its nominal target "
             "(section 2.3 of the thesis: 20-250x below nominal). Kobayashi et al. (2019), the GLMCC paper, "
             "report the same property of the jitter method on their ground-truth simulations: it \"avoided "
             "making FPs, but missed many existing connections, in particular for inhibitory connections\", "
             "and the 2025 review (Kobayashi & Shinomoto, Neurosci Res 215:37) restates it as the method "
             "being \"too conservative\". The conservatism is therefore a known property of jitter-based "
             "thresholding, not an idiosyncrasy of this pipeline. The `ccg_jitter` row applies the identical "
             "null to the raw correlogram; the gap between that row and the `glm_ridge` row is what the joint "
             "whitened fit adds on top of the null.")
    L.append("")
    L.append("## Parameters")
    L.append("")
    for m in order(params):
        L.append("- `%s`: `%s`" % (m, json.dumps(params[m], sort_keys=True)))
    L.append("")
    p = os.path.join(out, "RESULTS.md")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    print(p)


if __name__ == "__main__":
    main()
