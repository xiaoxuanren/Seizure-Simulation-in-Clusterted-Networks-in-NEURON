"""Strip voltage traces from a session's recordings, in place, to save disk.

Rewrites each recordingNNN.npz WITHOUT its voltage_* fields (spike data,
bursts, rasters, resampled arrays and all other keys are preserved
byte-for-byte), via a temp file + atomic replace. Typically shrinks a
full-voltage recording ~40x. The session's metadata gains
"voltage_stripped_locally": true.

SAFETY: refuses to run unless --archived-at names a directory that holds a
same-named copy of every recording with size >= the local one (your
full-voltage archive, e.g. on ResearchDrive) -- or you pass --force.

    python scripts/strip_voltage.py --session sweep_c50_seed01 \
        --archived-at "Z:/Users/Xiaoxuan/NEURON_seizure_sweep/sweep_c50_seed01/normal"
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
from session_paths import resolve  # noqa: E402


def strip_one(path):
    d = np.load(path, allow_pickle=True)
    keep = {k: d[k] for k in d.files if not k.startswith("voltage")}
    dropped = [k for k in d.files if k.startswith("voltage")]
    if not dropped:
        return 0, 0
    before = os.path.getsize(path)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".npz.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **keep)
        # verify the stripped file before replacing the original
        chk = np.load(tmp, allow_pickle=True)
        assert sorted(chk.files) == sorted(keep.keys())
        n_old = sum(len(s) for s in d["spike_times"])
        n_new = sum(len(s) for s in chk["spike_times"])
        assert n_old == n_new, "spike count changed (%d != %d)" % (n_old, n_new)
        del d, chk
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return before, os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--state", default="normal")
    ap.add_argument("--archived-at", default=None,
                    help="directory holding the full-voltage archive copy of "
                         "this session's recordings (required unless --force)")
    ap.add_argument("--force", action="store_true",
                    help="strip WITHOUT verifying an archive copy exists")
    a = ap.parse_args()

    sd = resolve(a.session, a.state)
    recs = [p for p in sorted(glob.glob(os.path.join(sd, "recording*.npz")))
            if "raster" not in os.path.basename(p)]
    if not recs:
        raise SystemExit("no recordings in %s" % sd)

    if not a.force:
        if not a.archived_at or not os.path.isdir(a.archived_at):
            raise SystemExit("--archived-at is required (or --force): refusing "
                             "to delete the only full-voltage copy")
        missing = []
        for p in recs:
            q = os.path.join(a.archived_at, os.path.basename(p))
            if not os.path.exists(q) or os.path.getsize(q) < os.path.getsize(p):
                missing.append(os.path.basename(p))
        if missing:
            raise SystemExit(
                "archive at %s is missing or smaller for %d recording(s) "
                "(e.g. %s) -- finish the archive copy first"
                % (a.archived_at, len(missing), missing[:5]))

    total_before = total_after = n_stripped = 0
    for p in recs:
        b, af = strip_one(p)
        if b:
            n_stripped += 1
            total_before += b
            total_after += af
    print("%s/%s: stripped voltage from %d/%d recordings, %.1f GB -> %.2f GB"
          % (a.session, a.state, n_stripped, len(recs),
             total_before / 1e9, total_after / 1e9))

    meta_path = os.path.join(sd, "session_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["voltage_stripped_locally"] = True
        meta["voltage_archive"] = a.archived_at
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
