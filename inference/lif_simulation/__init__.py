"""Minimal vendored shim of the LIF ``lif_simulation`` package.

Only the pieces the vendored ``lif_inference`` package imports are included --
currently just :mod:`lif_simulation.voltage_storage` (used by the
voltage-augmented learned-LIF pipeline to resolve inline vs external-HDF5
voltage traces).

Source: LIF-Project (``LIF-simulation/lif_simulation/``, branch
``chore/repo-cleanup``). The LIF project remains the source of truth; this shim
exists only so the vendored inference package imports cleanly without cloning
the full LIF simulation package.
"""
