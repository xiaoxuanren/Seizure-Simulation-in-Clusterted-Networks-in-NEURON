"""One place that knows where datasets live.

Layout::

    notebooks/NEURON data parallel/<session>/<state>/recordingNNN.npz

Analysis scripts take ``--session`` and ``--state`` and call :func:`resolve` to
get the directory holding the recordings. That directory also holds the
ground-truth ``network_*.npz``, so it can be handed straight to
``sparse_glm.load_session`` and ``sparse_glm.load_ground_truth``.

    from session_paths import resolve, add_args
    ap = argparse.ArgumentParser(); add_args(ap)
    a = ap.parse_args()
    data_dir = resolve(a.session, a.state)
"""

import argparse
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "notebooks", "NEURON data parallel")

#: what analysis defaults to when nothing is specified
DEFAULT_SESSION = "IC-locked_flagship_200rec"
DEFAULT_STATE = "normal"


def session_dir(session=DEFAULT_SESSION):
    return os.path.join(DATA, session)


def resolve(session=DEFAULT_SESSION, state=DEFAULT_STATE):
    """Directory holding recordings for ``session``/``state``.

    Accepts an absolute path for ``session`` too, so callers can point at
    anything without going through the registry.
    """
    root = session if os.path.isabs(session) else session_dir(session)
    d = os.path.join(root, state) if state else root
    if not os.path.isdir(d):
        raise SystemExit(
            "no such dataset: %s\n  available:\n%s" % (d, available_text()))
    if not glob.glob(os.path.join(d, "recording*.npz")):
        raise SystemExit("no recordings in %s" % d)
    return d


def list_sessions():
    """``{session: [state, ...]}`` for everything on disk."""
    out = {}
    if not os.path.isdir(DATA):
        return out
    for s in sorted(os.listdir(DATA)):
        sp = os.path.join(DATA, s)
        if not os.path.isdir(sp):
            continue
        states = sorted(
            st for st in os.listdir(sp)
            if os.path.isdir(os.path.join(sp, st))
            and glob.glob(os.path.join(sp, st, "recording*.npz")))
        if states:
            out[s] = states
    return out


def available_text():
    lines = []
    for s, states in list_sessions().items():
        for st in states:
            n = len(glob.glob(os.path.join(DATA, s, st, "recording*.npz")))
            lines.append("    --session %-38s --state %-24s (%d recordings)"
                         % (s, st, n))
    return "\n".join(lines) or "    (none found)"


def add_args(ap, default_session=DEFAULT_SESSION, default_state=DEFAULT_STATE):
    ap.add_argument("--session", default=default_session,
                    help="dataset session name or absolute path "
                         "(default: %s)" % default_session)
    ap.add_argument("--state", default=default_state,
                    help="state folder within the session (default: %s)"
                         % default_state)
    return ap


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="List available datasets.")
    p.parse_args()
    print("data root: %s\n" % DATA)
    print(available_text())
