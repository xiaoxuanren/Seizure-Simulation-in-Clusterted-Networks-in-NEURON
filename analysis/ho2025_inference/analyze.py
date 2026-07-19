#!/usr/bin/env python3
"""Recompute every headline number for the Ho2025 learned-LIF inference analysis
from the committed data in ``inputs/`` — no rerun of the (multi-hour) inference needed.

Emits into ``results/``:
    results.json                overall + directional AUCs, per-class confusion, rates
    results_summary.csv         one row per condition (AUC/AP/F1/FDR/rates)
    confusion_by_class.csv      TP/FP/FN per E/I connection class + population pair
    per_neuron_outcomes.csv     per-neuron rate, population, TP/FP as pre & post, predicted-out
                                (this file is what the two plot_*.py scripts consume)

Definitions match the inference pipeline exactly (inference/lif_inference/
connectivity_metrics.py::evaluate_connectivity): for each postsynaptic neuron p and
each candidate presynaptic id in neighbor_indices[p], score = |connectivity_matrix[p,pre]|,
predicted = score >= threshold, label = real directed edge pre->p.

Run:  python analyze.py           (needs numpy + scikit-learn)
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "inputs")
RESULTS = os.path.join(HERE, "results")
CONDS = ("normal", "mauve")
POP = {0: "E2", 1: "E5", 2: "I2"}          # to_session.py: sorted(['E2','E5','I2'])
INH = {"I2"}


def load(cond):
    d = np.load(os.path.join(DATA, f"connectivity_{cond}.npz"), allow_pickle=True)
    conn, nbr, thr = d["connectivity_matrix"], d["neighbor_indices"], float(d["threshold"])
    net = np.load(os.path.join(DATA, f"network_{cond}.npz"), allow_pickle=True)
    N = len(net["neuron_positions"])
    pops = np.array([POP[int(c)] for c in net["cluster_assignments"]])
    true_bin = np.zeros((N, N), dtype=np.int8)          # [post, pre]
    real_edges, inh_edges = set(), set()
    for r in net["connections"]:
        pre, post, typ = int(r[0]), int(r[1]), str(r[3])
        true_bin[post, pre] = 1
        real_edges.add((pre, post))
        if typ == "inh":
            inh_edges.add((pre, post))
    rec = np.load(os.path.join(DATA, f"recording_{cond}.npz"), allow_pickle=True)
    dur_s = float(rec["duration"]) / 1000.0
    st = rec["spike_times"]
    rate = np.array([len(np.asarray(st[i])) / dur_s for i in range(N)])
    return dict(conn=conn, nbr=nbr, thr=thr, N=N, pops=pops, true_bin=true_bin,
                real=real_edges, inh=inh_edges, rate=rate)


def analyze(cond):
    from sklearn.metrics import roc_auc_score, average_precision_score
    g = load(cond)
    conn, nbr, thr, N = g["conn"], g["nbr"], g["thr"], g["N"]
    pops, true_bin = g["pops"], g["true_bin"]

    scores, labels = [], []
    s_inh, y_inh, s_exc, y_exc = [], [], [], []
    cls = ["E2", "E5", "I2"]
    conf = {(a, b): [0, 0, 0] for a in cls for b in cls}          # tp,fp,fn
    ei = {"E->E": [0, 0, 0], "E->I": [0, 0, 0], "I->E": [0, 0, 0], "I->I": [0, 0, 0]}
    tp_pre = np.zeros(N); fp_pre = np.zeros(N); tp_post = np.zeros(N); fp_post = np.zeros(N)
    TP = FP = FN = TN = 0
    for post in range(N):
        for pre in nbr[post]:
            pre = int(pre)
            w = float(conn[post, pre]); s = abs(w)
            lab = int(true_bin[post, pre])
            scores.append(s); labels.append(lab)
            # directional (signed) scores over the same candidates
            edge = (pre, post)
            is_inh = edge in g["inh"]; is_exc = (edge in g["real"]) and not is_inh
            s_inh.append(-w); y_inh.append(1 if is_inh else 0)
            s_exc.append(w); y_exc.append(1 if is_exc else 0)
            pred = 1 if s >= thr else 0
            pp, qp = pops[pre], pops[post]
            pe = "I" if pp in INH else "E"; qe = "I" if qp in INH else "E"
            if pred and lab:
                TP += 1; conf[(pp, qp)][0] += 1; ei[f"{pe}->{qe}"][0] += 1
                tp_pre[pre] += 1; tp_post[post] += 1
            elif pred and not lab:
                FP += 1; conf[(pp, qp)][1] += 1; ei[f"{pe}->{qe}"][1] += 1
                fp_pre[pre] += 1; fp_post[post] += 1
            elif (not pred) and lab:
                FN += 1; conf[(pp, qp)][2] += 1; ei[f"{pe}->{qe}"][2] += 1
            else:
                TN += 1
    scores = np.array(scores); labels = np.array(labels)
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    prec = TP / (TP + FP + 1e-10); rec = TP / (TP + FN + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    fdr = FP / (TP + FP) if (TP + FP) else 0.0
    auc_inh = float(roc_auc_score(y_inh, s_inh)) if len(set(y_inh)) > 1 else None
    auc_exc = float(roc_auc_score(y_exc, s_exc)) if len(set(y_exc)) > 1 else None

    rate = g["rate"]
    rates_by_pop = {p: dict(n=int((pops == p).sum()),
                            mean_hz=float(rate[pops == p].mean()),
                            sd_hz=float(rate[pops == p].std()),
                            median_hz=float(np.median(rate[pops == p])))
                    for p in ("E2", "E5", "I2")}
    res = dict(
        condition=cond, N=N, threshold=thr,
        overall=dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=round(prec, 4),
                     recall=round(rec, 4), f1=round(f1, 4), fdr=round(fdr, 4),
                     auc=round(auc, 4), ap=round(ap, 4)),
        directional=dict(auc_inhibitory=None if auc_inh is None else round(auc_inh, 4),
                         auc_excitatory=None if auc_exc is None else round(auc_exc, 4)),
        confusion_ei={k: dict(tp=v[0], fp=v[1], fn=v[2]) for k, v in ei.items()},
        confusion_pop={f"{a}->{b}": dict(tp=conf[(a, b)][0], fp=conf[(a, b)][1], fn=conf[(a, b)][2])
                       for a in cls for b in cls},
        firing_rates_hz=rates_by_pop,
    )
    per_neuron = [dict(condition=cond, neuron=i, population=str(pops[i]),
                       rate_hz=round(float(rate[i]), 4),
                       tp_pre=int(tp_pre[i]), fp_pre=int(fp_pre[i]),
                       tp_post=int(tp_post[i]), fp_post=int(fp_post[i]),
                       predicted_out=int(tp_pre[i] + fp_pre[i]))
                  for i in range(N)]
    return res, per_neuron


def main():
    os.makedirs(RESULTS, exist_ok=True)
    all_res, all_pn = {}, []
    for cond in CONDS:
        res, pn = analyze(cond)
        all_res[cond] = res
        all_pn.extend(pn)
        o = res["overall"]; dR = res["directional"]
        print(f"[{cond}] AUC={o['auc']} FDR={o['fdr']} F1={o['f1']} "
              f"TP={o['TP']} FP={o['FP']} FN={o['FN']} | inhAUC={dR['auc_inhibitory']} excAUC={dR['auc_excitatory']}")

    with open(os.path.join(RESULTS, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)

    with open(os.path.join(RESULTS, "results_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "N", "auc", "ap", "f1", "fdr", "precision", "recall",
                    "TP", "FP", "FN", "auc_inhibitory", "auc_excitatory",
                    "rate_E2_hz", "rate_E5_hz", "rate_I2_hz"])
        for c in CONDS:
            r = all_res[c]; o = r["overall"]; fr = r["firing_rates_hz"]
            w.writerow([c, r["N"], o["auc"], o["ap"], o["f1"], o["fdr"], o["precision"], o["recall"],
                        o["TP"], o["FP"], o["FN"], r["directional"]["auc_inhibitory"],
                        r["directional"]["auc_excitatory"],
                        round(fr["E2"]["mean_hz"], 3), round(fr["E5"]["mean_hz"], 3), round(fr["I2"]["mean_hz"], 3)])

    with open(os.path.join(RESULTS, "confusion_by_class.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "class", "kind", "tp", "fp", "fn"])
        for c in CONDS:
            for k, v in all_res[c]["confusion_ei"].items():
                w.writerow([c, k, "ei", v["tp"], v["fp"], v["fn"]])
            for k, v in all_res[c]["confusion_pop"].items():
                w.writerow([c, k, "pop_pair", v["tp"], v["fp"], v["fn"]])

    with open(os.path.join(RESULTS, "per_neuron_outcomes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_pn[0].keys()))
        w.writeheader(); w.writerows(all_pn)

    print(f"\nwrote results.json + 3 CSVs to {RESULTS}")


if __name__ == "__main__":
    main()
