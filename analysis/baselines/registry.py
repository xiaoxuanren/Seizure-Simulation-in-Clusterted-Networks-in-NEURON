"""Method registry.  The runner is method-agnostic: it asks :func:`get` for a
:class:`Method` and calls the contract below.

METHOD MODULE CONTRACT  (analysis/baselines/<module>.py)
--------------------------------------------------------
Module attributes::

    NAME      = "<module name>"
    TIER      = 1 | 2
    DIRECTED  = True | False       # False only for zero-lag Pearson
    CITATION  = "..."              # source paper, in the module docstring too
    PARAMS    = {...}              # every default, so the CSV can record them

Functions::

    score(spikes, n_neurons, **params) -> np.ndarray [N, N] float64
        spikes : baselines.common.Session (use spikes.at(bin_ms) for a pass,
                 spikes.surrogates(bin_ms) for the shared jitter surrogates).
        Signed: positive = excitatory evidence, negative = inhibitory,
        indexed [pre, post], diagonal 0, finite everywhere.  Uses SPIKES ONLY.

    lag_profile(spikes, n_neurons, **params) -> (lags_ms [L], E [L, N, N])
        Optional.  Evidence per lag, same orientation, for the delay gate.
        E should be signed so that argmax over L finds the excitatory peak.

    label_free(spikes, n_neurons, S, **params) -> dict or None
        Optional.  The method's OWN operating point from spikes alone:
        {"untyped": (pred [N,N] bool, info), "typed": (pred, info) or None,
         "typing_inh": [N] bool or None, "rule": "<one line>"}.
        Never touches ground truth.

    typing_inhibitory(spikes, n_neurons, **params) -> [N] bool
        Optional.  The method's own E/I typing (default: rank rule on -S.sum(1)).

Every method scores the same candidate set (all ordered off-diagonal pairs);
the runner enforces this by zeroing the diagonal and evaluating on ``off``.

REGISTRY NAMES.  A registry name is either a module name or a VARIANT: the
same module with parameter overrides (e.g. transfer entropy at several bin
widths).  ``get(name)`` returns a :class:`Method` wrapper whose ``.PARAMS``
are the merged defaults and whose ``score``/``lag_profile``/``label_free``
already carry the overrides.
"""
from __future__ import annotations

import importlib

#: registry name -> (module, parameter overrides)
VARIANTS = {
    "te_1ms": ("transfer_entropy", dict(bin_ms=1.0)),
    "te_2ms": ("transfer_entropy", dict(bin_ms=2.0)),
    "te_5ms": ("transfer_entropy", dict(bin_ms=5.0)),
    "te_10ms": ("transfer_entropy", dict(bin_ms=10.0)),
    "stg_roi15": ("stg", dict(roi_ms=(0.0, 15.0))),
    "glmcc_d14": ("glmcc", dict(delays_ms=tuple(range(1, 15)))),
}

TIER1 = ["pearson0", "pearson_lag", "ccg_classical", "ccg_jitter", "tspe", "fncch", "glm_ridge"]
TIER2 = ["glmcc", "te_1ms", "te_2ms", "te_5ms", "te_10ms", "stg"]
#: extra variants run in the pilot only (documented mismatch probes)
PILOT_EXTRA = ["stg_roi15", "glmcc_d14"]
SKIPPED = {
    "connect": "CoNNECT needs a convolutional net pre-trained on statistics matched to the "
               "target network; no matched model exists for this simulator.",
    "ising": "Ising / kinetic-Ising couplings estimate synchrony, are bin-width unstable, and "
             "are not directed in the sense the ground truth requires.",
    "granger": "The review's own STG validation found Granger causality inconsistent with "
               "known physiology where the GLM was consistent.",
}
ALL = TIER1 + TIER2


class Method:
    """A registered method: module + merged parameters."""

    def __init__(self, name, module, params):
        self.name = name
        self.module = module
        self.PARAMS = dict(module.PARAMS)
        self.PARAMS.update(params)
        self.TIER = module.TIER
        self.DIRECTED = module.DIRECTED
        self.CITATION = module.CITATION
        self.MODULE = module.NAME
        for fn in ("lag_profile", "label_free", "typing_inhibitory"):
            if hasattr(module, fn):
                setattr(self, fn, self._bind(getattr(module, fn)))

    def _bind(self, fn):
        def call(*args, **kw):
            p = dict(self.PARAMS)
            p.update(kw)
            return fn(*args, **p)
        return call

    def score(self, spikes, n_neurons, **kw):
        p = dict(self.PARAMS)
        p.update(kw)
        return self.module.score(spikes, n_neurons, **p)


def get(name):
    """Resolve a registry name to a :class:`Method` (validates the contract)."""
    module_name, params = VARIANTS.get(name, (name, {}))
    mod = importlib.import_module("baselines." + module_name)
    for attr in ("NAME", "TIER", "DIRECTED", "CITATION", "PARAMS", "score"):
        if not hasattr(mod, attr):
            raise AttributeError("%s lacks %s" % (module_name, attr))
    if mod.NAME != module_name:
        raise ValueError("module %s registers NAME=%r" % (module_name, mod.NAME))
    return Method(name, mod, params)


def resolve(names):
    """'all' | 'tier1' | 'tier2' | 'pilot' | comma list -> list of names."""
    if isinstance(names, str):
        names = names.split(",")
    out = []
    for n in names:
        n = n.strip()
        if n == "all":
            out += ALL
        elif n == "tier1":
            out += TIER1
        elif n == "tier2":
            out += TIER2
        elif n == "pilot":
            out += ALL + PILOT_EXTRA
        elif n:
            out.append(n)
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq
