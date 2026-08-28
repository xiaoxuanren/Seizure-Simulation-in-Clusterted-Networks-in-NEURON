"""Write ladder_jobs.txt ('<session> <point_idx>' per line) + repo.tar.gz.

    python3 chtc/make_ladder_manifest.py                       # all networks
    python3 chtc/make_ladder_manifest.py --sessions sweep_c50_seed01
    python3 chtc/make_ladder_manifest.py --done-root results   # skip finished
"""
import argparse
import json
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default=os.path.join(HERE, "ladder_mechanisms.json"))
    ap.add_argument("--sweep", default=os.path.join(HERE, "sweep_config.json"))
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "ladder_jobs.txt"))
    ap.add_argument("--done-root", default=None,
                    help="skip points whose ladder_<session>_p<idx>.json exists there")
    ap.add_argument("--no-tar", action="store_true")
    a = ap.parse_args()

    with open(a.ladder, encoding="utf-8") as fh:
        lad = json.load(fh)
    with open(a.sweep, encoding="utf-8") as fh:
        sweep = json.load(fh)

    sessions = a.sessions or ["%s_seed%02d" % (g["prefix"], s)
                              for g in sweep["groups"] for s in g["seeds"]]
    lines, skipped = [], 0
    for session in sessions:
        for idx in range(len(lad["points"])):
            if a.done_root and os.path.exists(os.path.join(
                    a.done_root, "ladder_%s_p%d.json" % (session, idx))):
                skipped += 1
                continue
            lines.append("%s %d" % (session, idx))
    with open(a.out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    print("%s: %d jobs%s (%d points x %d networks)"
          % (a.out, len(lines), " (%d done, skipped)" % skipped if skipped else "",
             len(lad["points"]), len(sessions)))

    if not a.no_tar:
        tar_path = os.path.join(HERE, "repo.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(os.path.join(REPO, "neuron_simulation"), arcname="repo/neuron_simulation",
                    filter=lambda ti: None if ("__pycache__" in ti.name or
                                               ti.name.endswith((".o", ".c", ".dll"))) else ti)
            for f in ("ladder_one.py", "generate_one.py"):
                tar.add(os.path.join(HERE, f), arcname="repo/chtc/" + f)
            tar.add(a.ladder, arcname="repo/chtc/" + os.path.basename(a.ladder))
            tar.add(a.sweep, arcname="repo/chtc/" + os.path.basename(a.sweep))
        print("%s: %.1f MB" % (tar_path, os.path.getsize(tar_path) / 1e6))


if __name__ == "__main__":
    main()
