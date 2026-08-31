"""Build the duration-grid analysis manifest + input tarballs. Run LOCALLY.

Writes, into chtc/:
  analysis_jobs.txt          '<session> <n_recordings>' per line (20 x 13 = 260)
  repo_analysis.tar.gz       the three source files the job needs (~50 KB)
  spikes/spikes_<session>.tar.gz   one per session: the NORMAL-state
                             recording*.npz (already voltage-stripped) +
                             network npz (~55 MB each, ~1.1 GB total)

Then upload chtc/analysis_jobs.txt, chtc/repo_analysis.tar.gz, chtc/spikes/,
chtc/analysis_job.sh, chtc/analysis.sub to the access point and submit.

    python chtc/make_analysis_manifest.py
    python chtc/make_analysis_manifest.py --sessions sweep_c50_seed01
    python chtc/make_analysis_manifest.py --done-root sweep_summary/durgrid
"""
import argparse
import glob
import json
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DURATIONS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        REPO, "notebooks", "NEURON data parallel"))
    ap.add_argument("--state", default="normal")
    ap.add_argument("--sweep", default=os.path.join(HERE, "sweep_config.json"))
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--done-root", default=None,
                    help="skip points whose durgrid_<session>_n<NNN>.json exists there")
    ap.add_argument("--no-tars", action="store_true")
    a = ap.parse_args()

    with open(a.sweep, encoding="utf-8") as fh:
        sweep = json.load(fh)
    sessions = a.sessions or ["%s_seed%02d" % (g["prefix"], s)
                              for g in sweep["groups"] for s in g["seeds"]]

    lines, skipped = [], 0
    for session in sessions:
        for n in DURATIONS:
            if a.done_root and os.path.exists(os.path.join(
                    a.done_root, "durgrid_%s_n%03d.json" % (session, n))):
                skipped += 1
                continue
            lines.append("%s %d" % (session, n))
    out = os.path.join(HERE, "analysis_jobs.txt")
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    print("%s: %d jobs%s (%d durations x %d sessions)"
          % (out, len(lines), " (%d done, skipped)" % skipped if skipped else "",
             len(DURATIONS), len(sessions)))

    if a.no_tars:
        return

    tar_path = os.path.join(HERE, "repo_analysis.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(os.path.join(REPO, "sparse_glm.py"), arcname="repo/sparse_glm.py")
        tar.add(os.path.join(REPO, "glm_connectivity.py"),
                arcname="repo/glm_connectivity.py")
        tar.add(os.path.join(HERE, "analysis_one.py"),
                arcname="repo/chtc/analysis_one.py")
    print("%s: %.2f MB" % (tar_path, os.path.getsize(tar_path) / 1e6))

    spikes_dir = os.path.join(HERE, "spikes")
    os.makedirs(spikes_dir, exist_ok=True)
    for session in sessions:
        dst = os.path.join(spikes_dir, "spikes_%s.tar.gz" % session)
        if os.path.exists(dst):
            print("%s: exists, kept" % dst)
            continue
        src = os.path.join(a.data, session, a.state)
        recs = [p for p in sorted(glob.glob(os.path.join(src, "recording*.npz")))
                if "raster" not in os.path.basename(p)]
        nets = sorted(glob.glob(os.path.join(src, "network_*.npz")))
        if not recs or not nets:
            raise SystemExit("missing recordings or network npz in %s" % src)
        with tarfile.open(dst, "w:gz") as tar:
            for p in recs + nets[:1]:
                tar.add(p, arcname="data/%s/%s" % (session, os.path.basename(p)))
        print("%s: %d recs, %.1f MB"
              % (dst, len(recs), os.path.getsize(dst) / 1e6))


if __name__ == "__main__":
    main()
