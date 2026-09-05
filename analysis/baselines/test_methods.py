"""Mechanical invariant tests for EVERY registered method (no data needed).

    python analysis/baselines/test_methods.py [--methods all] [--verbose]

For each registry name:
  1. CONTRACT   -- registry.get() works; NAME / TIER / DIRECTED / CITATION / PARAMS present.
  2. LEAKAGE    -- the module source never mentions ground-truth identifiers
                   (network_, A_exc, A_inh, is_inhibitory, load_ground_truth, evaluate).
  3. ORIENTATION -- on a synthetic 3-neuron session with two segments where
                   neuron 0 fires and neuron 1 fires exactly 3 ms later (neuron 2
                   independent), score() must give S[0,1] > S[1,0] and
                   S[0,1] > S[0,2]; the diagonal must be 0 and S finite; for
                   undirected methods S[0,1] == S[1,0] > S[0,2] is required instead.
  4. LAG PROFILE -- if defined, lags_ms and E agree in length and the peak lag
                   for (0, 1) lies within one bin of 3 ms.
  5. LABEL-FREE  -- if defined, returns boolean [N, N] masks with a false
                   diagonal (content is not judged here).
  6. SILENT UNIT -- a neuron with zero spikes yields finite scores.
The synthetic session exposes .at(bin_ms) at any width (rebinned from a
0.1 ms master clock) and .surrogates(...) built with the shared jitter.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import traceback

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(ANALYSIS)
for _p in (REPO, ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from baselines import accumulate as acc  # noqa: E402
from baselines import common as cm  # noqa: E402
from baselines import registry  # noqa: E402

LEAK = re.compile(r"network_|A_exc|A_inh|is_inhibitory|load_ground_truth|baselines\.evaluate|from baselines import evaluate|GroundTruth")


def code_only(path):
    """Module source with every string literal and comment removed, so the
    leakage grep sees code, not docstrings that *mention* ground truth."""
    import io
    import tokenize
    out = []
    with open(path, encoding="utf-8") as fh:
        toks = tokenize.generate_tokens(io.StringIO(fh.read()).readline)
        for tok in toks:
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                continue
            out.append(tok.string)
    return " ".join(out)


class SyntheticSession:
    """Spike times in ms -> passes at any bin width, boundary-respecting."""

    def __init__(self, spikes_ms, seg_ends_ms, n_neurons):
        self.spikes_ms = spikes_ms
        self.seg_ends_ms = list(seg_ends_ms)
        self.n_neurons = n_neurons
        self._passes, self._surr = {}, {}
        self.session_dir = "<synthetic>"
        self.n_recordings = len(seg_ends_ms)
        self.arm = "full"
        self.pad_ms = 0.0
        self.exclude_windows_ms = None

    def at(self, bin_ms):
        key = float(bin_ms)
        if key not in self._passes:
            rows, cols, bnd = [], [], [0]
            start = 0.0
            for end in self.seg_ends_ms:
                T = int(round((end - start) / key))
                for i, t in enumerate(self.spikes_ms):
                    t = np.asarray(t, float)
                    t = t[(t >= start) & (t < end)] - start
                    b = np.unique(np.clip((t / key).astype(np.int64), 0, T - 1))
                    rows.append(np.full(len(b), i, np.int32))
                    cols.append(b + bnd[-1])
                bnd.append(bnd[-1] + T)
                start = end
            rows, cols = np.concatenate(rows), np.concatenate(cols)
            M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                              shape=(self.n_neurons, bnd[-1]))
            self._passes[key] = cm.SpikeData(M, bnd, key)
        return self._passes[key]

    def surrogates(self, bin_ms, jitter_ms=cm.JITTER_MS, n_surr=cm.N_SURR, seed=cm.SEED):
        key = (float(bin_ms), float(jitter_ms), int(n_surr), int(seed))
        if key not in self._surr:
            base = self.at(bin_ms)
            jb = max(1, int(round(jitter_ms / bin_ms)))
            rng = np.random.default_rng(seed)
            self._surr[key] = [base.jittered(jb, rng) for _ in range(n_surr)]
        return self._surr[key]

    def dropped(self, bin_ms):
        return (0, self.at(bin_ms).T)

    def release(self):
        pass


def make_session(seed=0, rate_hz=8.0, seg_s=(60.0, 60.0), delay_ms=3.0, p_transmit=0.6,
                 silent=False):
    rng = np.random.default_rng(seed)
    total = sum(seg_s) * 1000.0
    t0 = np.sort(rng.uniform(0, total, int(rate_hz * total / 1000.0)))
    t0 = t0[np.diff(np.r_[-10.0, t0]) > 2.0]                    # 2 ms refractory
    follow = t0[rng.random(len(t0)) < p_transmit] + delay_ms + rng.normal(0, 0.15, None)
    t1_bg = np.sort(rng.uniform(0, total, int(0.5 * rate_hz * total / 1000.0)))
    t1 = np.sort(np.r_[follow, t1_bg])
    t1 = t1[np.diff(np.r_[-10.0, t1]) > 2.0]
    t2 = np.sort(rng.uniform(0, total, int(rate_hz * total / 1000.0)))
    t2 = t2[np.diff(np.r_[-10.0, t2]) > 2.0]
    spikes = [t0, t1, t2]
    if silent:
        spikes.append(np.array([], float))
    ends = np.cumsum(np.asarray(seg_s) * 1000.0)
    return SyntheticSession(spikes, ends, len(spikes))


def check(name, verbose=False):
    out = dict(name=name, ok=True, problems=[])
    try:
        m = registry.get(name)
    except Exception as e:
        out["ok"] = False
        out["problems"].append("contract: %s" % e)
        return out
    src = code_only(os.path.join(HERE, m.MODULE + ".py"))
    hits = sorted(set(LEAK.findall(src)))
    if hits:
        out["ok"] = False
        out["problems"].append("leakage identifiers in source: %s" % hits)
    sess = make_session()
    n = sess.n_neurons
    try:
        S = np.asarray(m.score(sess, n), float)
        assert S.shape == (n, n), "shape %s" % (S.shape,)
        assert np.isfinite(S).all(), "non-finite"
        assert np.allclose(np.diag(S), 0.0), "diagonal not zero"
        if m.DIRECTED:
            if not (S[0, 1] > S[1, 0] and S[0, 1] > S[0, 2]):
                out["ok"] = False
                out["problems"].append("ORIENTATION: S[0,1]=%.3g S[1,0]=%.3g S[0,2]=%.3g" % (S[0, 1], S[1, 0], S[0, 2]))
        else:
            if not (np.isclose(S[0, 1], S[1, 0]) and S[0, 1] > S[0, 2]):
                out["ok"] = False
                out["problems"].append("undirected symmetry/ordering: S[0,1]=%.3g S[1,0]=%.3g S[0,2]=%.3g" % (S[0, 1], S[1, 0], S[0, 2]))
        out["S01"], out["S10"], out["S02"] = float(S[0, 1]), float(S[1, 0]), float(S[0, 2])
    except Exception as e:
        out["ok"] = False
        out["problems"].append("score raised %s: %s" % (type(e).__name__, e))
        if verbose:
            traceback.print_exc()
        return out
    if hasattr(m, "lag_profile"):
        try:
            lags, E = m.lag_profile(sess, n)
            lags, E = np.asarray(lags, float), np.asarray(E, float)
            assert E.shape == (len(lags), n, n), "lag profile shape"
            peak = float(lags[np.argmax(E[:, 0, 1])])
            out["peak_lag_ms"] = peak
            res = float(np.min(np.diff(lags))) if len(lags) > 1 else 1.0
            if abs(peak - 3.0) > max(res, 1.0) + 1e-9:
                out["ok"] = False
                out["problems"].append("lag profile peak for 0->1 at %.1f ms (expected ~3 ms, resolution %.1f)" % (peak, res))
        except Exception as e:
            out["ok"] = False
            out["problems"].append("lag_profile raised %s: %s" % (type(e).__name__, e))
            if verbose:
                traceback.print_exc()
    if hasattr(m, "label_free"):
        try:
            lf = m.label_free(sess, n, S)
            for key in ("untyped", "typed"):
                item = lf.get(key)
                if item is None:
                    continue
                pred, info = item
                pred = np.asarray(pred)
                assert pred.dtype == bool and pred.shape == (n, n), "%s mask type/shape" % key
                assert not np.diag(pred).any(), "%s mask has a true diagonal" % key
            out["lf_untyped_01"] = bool(lf["untyped"][0][0, 1]) if lf.get("untyped") else None
        except Exception as e:
            out["ok"] = False
            out["problems"].append("label_free raised %s: %s" % (type(e).__name__, e))
            if verbose:
                traceback.print_exc()
    try:
        S2 = np.asarray(m.score(make_session(seed=1, silent=True), n + 1), float)
        assert np.isfinite(S2).all(), "non-finite with a silent unit"
    except Exception as e:
        out["ok"] = False
        out["problems"].append("silent unit: %s: %s" % (type(e).__name__, e))
        if verbose:
            traceback.print_exc()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="all")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    names = registry.resolve(a.methods)
    bad = 0
    print("%-16s %-4s %10s %10s %10s %8s %6s  problems" % ("method", "ok", "S[0,1]", "S[1,0]", "S[0,2]", "peak_ms", "lf01"))
    for name in names:
        r = check(name, a.verbose)
        bad += not r["ok"]
        print("%-16s %-4s %10.3g %10.3g %10.3g %8s %6s  %s"
              % (name, "OK" if r["ok"] else "FAIL", r.get("S01", np.nan), r.get("S10", np.nan),
                 r.get("S02", np.nan), ("%.1f" % r["peak_lag_ms"]) if "peak_lag_ms" in r else "-",
                 str(r.get("lf_untyped_01", "-")), "; ".join(r["problems"])), flush=True)
    print("%d/%d methods pass" % (len(names) - bad, len(names)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
