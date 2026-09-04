"""Invariant tests for the accumulator.  Run from the repo root:

    python analysis/baselines/test_accumulate.py [--session sweep_c50_seed01]

1. ORIENTATION: on a synthetic pair where neuron 0 always fires 2 bins before
   neuron 1, ``C[2][0, 1]`` carries the coincidences and ``C[2][1, 0]`` is 0.
2. SUFFICIENT STATISTIC: the ridge right-hand side ``F X^T`` row block a
   equals the raw lag-(a+1) cross-correlogram ``C[a+1]`` exactly.
3. REPRODUCTION: ``solve_ridge(gram(...))`` is bit-identical to
   ``sparse_glm.fit_B`` on 50 recordings, and the sum4 readout on all 200
   recordings is bit-identical to the shipped
   ``results/normal/glm/glm_connectivity_sum4_5ms.npz`` W.
4. BOUNDARY: no lag product counts a pair that straddles two recordings.
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import sparse_glm as sg  # noqa: E402
from baselines import accumulate as acc  # noqa: E402
from session_paths import resolve, results_dir  # noqa: E402


def test_orientation():
    T = 200
    M = np.zeros((3, T), np.float32)
    t0 = np.arange(5, T - 5, 7)
    M[0, t0] = 1                    # neuron 0 fires first ...
    M[1, t0 + 2] = 1                # ... neuron 1 two bins later
    M[2, t0 + 4] = 1                # ... neuron 2 four bins later
    C = acc.lag_products(sp.csr_matrix(M), [0, T], 4)
    assert C[2][0, 1] == len(t0) and C[2][1, 0] == 0, "orientation broken"
    assert C[4][0, 2] == len(t0) and C[2][1, 2] == len(t0) and C[2][2, 1] == 0
    assert C[0][0, 0] == len(t0)
    r = acc.pearson_from_counts(C[2], np.asarray(M.sum(1)), T - 2)
    # reverse direction is genuinely ANTI-correlated at lag 2 (-0.16), so only
    # require a large forward/backward gap
    assert r[0, 1] > 0.9 and r[1, 0] < 0.0, "pearson orientation broken"
    print("[1] orientation: C[l][pre, post] confirmed on synthetic pair")


def test_boundary():
    T = 50
    M = np.zeros((2, 2 * T), np.float32)
    M[0, T - 1] = 1                  # last bin of recording 1
    M[1, T + 1] = 1                  # second bin of recording 2 (lag 2 across the cut)
    C = acc.lag_products(sp.csr_matrix(M), [0, T, 2 * T], 3)
    assert C[2][0, 1] == 0, "lag product bridged a recording boundary"
    print("[4] boundary: no count bridges two recordings")


def test_reproduction(session, state, n_small=50):
    sd = resolve(session, state)
    t0 = time.time()
    M, bnd = acc.load_session(sd, 5.0, n_recordings=n_small)
    G, RHS = acc.gram(M, bnd, 6)
    B = acc.solve_ridge(G, RHS, 2.0, 6)
    B_ref = sg.fit_B(M, bnd, max_lag=6, l2=2.0)
    assert np.array_equal(B, B_ref), "solve_ridge != sparse_glm.fit_B"
    C = acc.lag_products(M, bnd, 6)
    n = M.shape[0]
    for a in range(6):
        assert np.array_equal(RHS[a * n:(a + 1) * n], C[a + 1]), "RHS block != lag product"
    print("[2] F X^T row block a == C[a+1] for a=0..5 (n=%d, %.0fs)" % (n_small, time.time() - t0))
    print("[3a] solve_ridge(gram) bit-identical to sparse_glm.fit_B at n=%d" % n_small)

    shipped = os.path.join(results_dir(session, state, "glm", create=False),
                           "glm_connectivity_sum4_5ms.npz")
    if not os.path.exists(shipped):
        print("[3b] no shipped npz for %s/%s -- skipped" % (session, state))
        return
    t0 = time.time()
    M, bnd = acc.load_session(sd, 5.0)
    G, RHS = acc.gram(M, bnd, 6)
    W = acc.readout_sum_k(acc.solve_ridge(G, RHS, 2.0, 6), 4)
    Ws = np.load(shipped, allow_pickle=True)["W"]
    d = float(np.abs(W - Ws).max())
    assert np.array_equal(W, Ws), "sum4 W differs from shipped npz (max |diff| %.3e)" % d
    print("[3b] sum4 W on %d recordings bit-identical to shipped npz (%.0fs)"
          % (len(bnd) - 1, time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sweep_c50_seed01")
    ap.add_argument("--state", default="normal")
    ap.add_argument("--skip-data", action="store_true")
    a = ap.parse_args()
    test_orientation()
    test_boundary()
    if not a.skip_data:
        test_reproduction(a.session, a.state)
    print("OK")
