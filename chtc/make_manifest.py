"""Write jobs.txt (one line per recording: '<session> <rec_idx>') for generate.sub.

Full manifest (4,000 lines for the default sweep):
    python chtc/make_manifest.py --sweep chtc/sweep_config.json

Resubmit only what is missing, by scanning a directory that holds the
already-produced sessions (the staging download, or the assembled data tree):
    python chtc/make_manifest.py --sweep chtc/sweep_config.json --done-root /path/to/out

Also writes repo.tar.gz (the code the jobs need) unless --no-tar.
"""
import argparse
import json
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=os.path.join(HERE, "sweep_config.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "jobs.txt"))
    ap.add_argument("--done-root", default=None,
                    help="skip recordings whose recordingNNN.npz already exists "
                         "under <done-root>/<session>/")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="only these session names (e.g. sweep_c50_seed01) -- "
                         "for staged waves under a small /staging quota")
    ap.add_argument("--no-tar", action="store_true")
    a = ap.parse_args()

    with open(a.sweep, "r", encoding="utf-8") as fh:
        sweep = json.load(fh)

    state = sweep.get("state", "normal")

    def done(session, *names):
        """True if every named file exists in the done-root, in the raw layout
        (<session>/f), the assembled one (<session>/<state>/f), or -- for the
        HTCondor-transfer flow -- as an unextracted <session>_r<idx>.tar."""
        return all(
            os.path.exists(os.path.join(a.done_root, session, nm)) or
            os.path.exists(os.path.join(a.done_root, session, state, nm))
            for nm in names)

    def tar_done(session, rec):
        return a.done_root and os.path.exists(
            os.path.join(a.done_root, "%s_r%d.tar" % (session, rec)))

    only = set(a.sessions) if a.sessions else None
    lines, skipped = [], 0
    for group in sweep["groups"]:
        for seed in group["seeds"]:
            session = "%s_seed%02d" % (group["prefix"], seed)
            if only is not None and session not in only:
                continue
            for rec in range(int(group["n_recordings"])):
                need = ["recording%03d.npz" % rec]
                if rec == 0:
                    # the rec-0 job also produces the network npz + provenance;
                    # if either is missing the job must run again
                    need += ["network_%s.npz" % session, "session_provenance.json"]
                if a.done_root and (done(session, *need) or tar_done(session, rec)):
                    skipped += 1
                    continue
                lines.append("%s %d" % (session, rec))

    with open(a.out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    print("%s: %d jobs%s" % (a.out, len(lines),
                             " (%d already done, skipped)" % skipped if skipped else ""))

    if not a.no_tar:
        tar_path = os.path.join(HERE, "repo.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(os.path.join(REPO, "neuron_simulation"), arcname="repo/neuron_simulation",
                    filter=lambda ti: None if ("__pycache__" in ti.name or
                                               ti.name.endswith((".o", ".c", ".dll"))) else ti)
            tar.add(os.path.join(HERE, "generate_one.py"), arcname="repo/chtc/generate_one.py")
            tar.add(a.sweep, arcname="repo/chtc/sweep_config.json")
        print("%s: %.1f MB" % (tar_path, os.path.getsize(tar_path) / 1e6))


if __name__ == "__main__":
    main()
