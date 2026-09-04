"""Unpack the baselines_<session>_<state>_<method>.tar files returned by the
CHTC jobs into the repo layout (<session>/results/<state>/baselines/).

    python chtc/collect_baselines.py --src "D:/path/to/download_dir"
"""
import argparse
import glob
import os
import re
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from baselines import common as cm  # noqa: E402

PAT = re.compile(r"^baselines_(sweep_c\d0_seed\d\d)_(normal|seizure)_([A-Za-z0-9_]+)\.tar$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=HERE)
    a = ap.parse_args()
    n = 0
    for p in sorted(glob.glob(os.path.join(a.src, "baselines_*.tar"))):
        m = PAT.match(os.path.basename(p))
        if not m:
            print("skip (name?)", p)
            continue
        session, state, method = m.groups()
        dst = cm.baselines_results_dir(session, state)
        with tarfile.open(p) as tar:
            tar.extractall(dst)
        n += 1
        print("%s/%s %s -> %s" % (session, state, method, dst))
    print("%d job tars collected" % n)


if __name__ == "__main__":
    main()
