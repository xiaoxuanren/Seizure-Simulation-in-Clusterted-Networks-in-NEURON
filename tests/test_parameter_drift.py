"""Guard: library signature defaults must match the parameter registry.

The operating point once lived in four places and no two agreed. The registry
(``neuron_simulation/parameters.py``) is now the single source of truth, and this
test fails loudly if any function signature drifts away from it.

Parses source with ``ast`` rather than importing, so it runs in CI without
NEURON installed. Run directly (``python tests/test_parameter_drift.py``) or
under pytest.
"""

import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the registry BY PATH: importing the package would pull in NEURON, and this
# guard must run in CI where NEURON is not installed.
_spec = importlib.util.spec_from_file_location(
    "_parameters", os.path.join(REPO, "neuron_simulation", "parameters.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def test_signatures_match_registry():
    """Every synced signature default equals its registry default."""
    problems = []
    for rel_path, func_name, group in P.SYNCED_SIGNATURES:
        drift = P.verify_source_defaults(os.path.join(REPO, rel_path), func_name, group)
        for name, (source_default, registry_default) in sorted(drift.items()):
            problems.append(
                f"  {func_name}(): {name}={source_default!r} but registry says "
                f"{registry_default!r}"
            )
    assert not problems, (
        "Signature defaults have drifted from neuron_simulation/parameters.py.\n"
        "Update the signature (or the registry, if the registry is what changed):\n"
        + "\n".join(problems)
    )


def test_registry_is_fully_documented():
    """No parameter may ship without units, a description, and an effect."""
    incomplete = [
        p.name for p in P.PARAMETERS.values()
        if not p.units or not p.description or not p.effect_of_increasing
    ]
    assert not incomplete, f"Undocumented parameters: {incomplete}"


def test_groups_are_known():
    """Every parameter belongs to a declared group."""
    bad = [p.name for p in P.PARAMETERS.values() if p.group not in P.GROUPS]
    assert not bad, f"Parameters in unknown groups: {bad}"


if __name__ == "__main__":
    failures = 0
    for fn in (test_signatures_match_registry,
               test_registry_is_fully_documented,
               test_groups_are_known):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n{exc}")
    n_verify = len(P.needs_verification())
    print(f"\n{len(P.PARAMETERS)} parameters registered "
          f"({n_verify} with effect directions awaiting empirical verification)")
    sys.exit(1 if failures else 0)
