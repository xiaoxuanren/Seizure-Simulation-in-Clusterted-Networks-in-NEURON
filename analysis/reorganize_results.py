"""Move analysis output out of the raw-data folders into ``results/``.

Before::

    <session>/<state>/   200 recordings + 400 rasters + 200 summaries
                         + ~50 analysis outputs, all in one 1000-file directory

After::

    <session>/<state>/            raw only: recordings, rasters, summaries,
                                  network_*.npz, _worker_config.pkl
    <session>/results/<state>/
        glm/          connectivity fits, typing tables, scaling + FDR metrics
        bursts/       burst windows, gate sensitivity
        ic_artifact/  the three-arm initialization-burst exclusion test
        figures/      every .png
        other/        earlier work (learned-LIF, p3, transmission, Vm, ...)

Prints a manifest of every move, so anything that breaks is one lookup away.
An INDEX.md is written next to the results describing each file.

    python reorganize_results.py --dry-run
    python reorganize_results.py
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import DATA, results_dir  # noqa: E402

#: filename prefix/suffix -> category. First match wins; order matters.
RULES = [
    (("burstexcl_",), "ic_artifact"),
    (("burstwindows_", "burst_gate_sensitivity"), "bursts"),
    (("a1_typing_fix", "glm_", "fdrdur"), "glm"),
    (("learned_lif", "lif_checkpoints", "p3_", "sta80_", "vrest_analysis",
      "transmission_corrected", "efficacy_vs_vm", "sahp_load_table"), "other"),
]

#: never moved -- these are the raw dataset
KEEP = ("recording", "_summary_", "network_", "_worker_config",
        "session_metadata", "_state_library_", "_session_config")


def categorize(name):
    if any(name.startswith(k) for k in KEEP):
        return None
    if name.lower().endswith(".png"):
        return "figures"
    for prefixes, cat in RULES:
        if any(name.startswith(p) for p in prefixes):
            return cat
    return "other"


def plan_for(session, state):
    src = os.path.join(DATA, session, state)
    if not os.path.isdir(src):
        return []
    moves = []
    for name in sorted(os.listdir(src)):
        cat = categorize(name)
        if cat is None:
            continue
        moves.append((os.path.join(src, name),
                      os.path.join(results_dir(session, state, cat, create=False),
                                   name),
                      cat, name))
    return moves


INDEX = """# Results — {session} / {state}

Analysis output for this dataset. Raw recordings, rasters and per-recording
summaries stay in `../../{state}/`.

Regenerate anything here by pointing the producing script at this dataset:

```bash
DATASET_SESSION={session} DATASET_STATE={state} python analysis/<script>.py
```

| folder | what |
|---|---|
| `glm/` | connectivity fits, E/I typing tables, scaling and FDR-calibration metrics |
| `bursts/` | burst windows at the 0.35 gate, acceptance-gate sensitivity |
| `ic_artifact/` | three-arm test of whether the initialization burst drives edge recovery |
| `figures/` | every figure |
| `other/` | earlier work: learned-LIF, p3 sweeps, transmission, Vm analyses |

## Files

{files}
"""

DESCRIPTIONS = {
    "a1_typing_fix_results": "E/I typing fix: four result tables (JSON)",
    "a1_typing_fix_tables": "E/I typing fix: same tables as CSV",
    "glm_connectivity": "fitted signed connectivity matrix W",
    "glm_labelfree_scaling": "sum4 @FDR0.70 edge recovery vs recording duration",
    "glm_labelfree_fdr_duration": "FDR target x duration sweep (5-100 recordings)",
    "glm_scaling": "oracle upper bounds (best-F1, @10% FDR) vs duration",
    "glm_lag_sweep": "per-lag exc/inh AUC and AP",
    "fdrdur10to200": "FDR target x duration grid, 10-200 recordings",
    "burstwindows_p035": "burst windows recomputed at the 0.35 participation gate",
    "burst_gate_sensitivity": "how burst count depends on the acceptance gate",
    "burstexcl": "three-arm initialization-burst exclusion test",
}


def describe(name):
    for k, v in DESCRIPTIONS.items():
        if name.startswith(k):
            return v
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--session", default="IC-locked_flagship_200rec")
    ap.add_argument("--states", nargs="+", default=["normal", "seizure"])
    a = ap.parse_args()

    total = 0
    for state in a.states:
        moves = plan_for(a.session, state)
        print("=== %s / %s : %d file(s) to move ===" % (a.session, state, len(moves)))
        by_cat = {}
        for src, dst, cat, name in moves:
            by_cat.setdefault(cat, []).append(name)
        for cat in sorted(by_cat):
            print("  %-12s %d" % (cat + "/", len(by_cat[cat])))
            for n in by_cat[cat]:
                print("      %s" % n)
        total += len(moves)

        if not a.dry_run:
            for src, dst, cat, name in moves:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    print("  [skip] exists: %s" % dst)
                    continue
                shutil.move(src, dst)
            # index everything present, not just what moved this run
            rroot = results_dir(a.session, state, create=False)
            present = {}
            if os.path.isdir(rroot):
                for cat in sorted(os.listdir(rroot)):
                    cp = os.path.join(rroot, cat)
                    if os.path.isdir(cp):
                        present[cat] = sorted(os.listdir(cp))
            rows = []
            for cat in sorted(present):
                for n in present[cat]:
                    rows.append("| `%s/%s` | %s |" % (cat, n, describe(n)))
            idx = INDEX.format(session=a.session, state=state,
                               files="| file | description |\n|---|---|\n"
                                     + "\n".join(rows))
            with open(os.path.join(results_dir(a.session, state), "INDEX.md"),
                      "w", encoding="utf-8") as fh:
                fh.write(idx)
            print("  wrote INDEX.md")
        print("")

    print("%s %d file(s)" % ("WOULD MOVE" if a.dry_run else "MOVED", total))
    if a.dry_run:
        print("(dry run - nothing changed)")


if __name__ == "__main__":
    main()
