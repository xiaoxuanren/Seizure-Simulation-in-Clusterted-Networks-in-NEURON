"""NEURON biophysical clustered-network simulation package.

A single-compartment Hodgkin-Huxley + A-current (Kv4-like) clustered network
that produces spontaneous synchronized network bursts under Poisson background
drive, with the A-current density ``gbar_kA`` as a 4-AP knob. Structured to
mirror the LIF project's ``lif_simulation`` package and to write session data in
the exact format its inference pipeline consumes.

Public API (see the individual modules for full docstrings):

* topology: :func:`build_topology_lognormal` (preferred), :func:`build_topology`
* neurons: :func:`build_cell`, :func:`load_mechanisms`, :class:`Cell`
* network_builder: :func:`build_network`, :class:`Network`
* noise: :func:`add_poisson_noise`
* states: :func:`normal_state`, :func:`four_ap_state`, :func:`dose_response_gbar`
* simulation: :func:`run_simulation`
* analysis: :func:`detect_network_bursts`, :func:`burst_statistics`
* io: :func:`save_network_structure`, :func:`save_recording_data`, loaders
* workflows: :func:`generate_dataset`, :func:`run_single_state`
"""

from . import analysis, io, plotting, states, topology
from .network_builder import Network, build_network
from .neurons import Cell, build_cell, load_mechanisms
from .noise import add_poisson_noise
from .simulation import run_simulation
from .topology import build_topology, build_topology_lognormal

__all__ = [
    "topology",
    "states",
    "analysis",
    "io",
    "plotting",
    "build_topology",
    "build_topology_lognormal",
    "build_network",
    "Network",
    "build_cell",
    "Cell",
    "load_mechanisms",
    "add_poisson_noise",
    "run_simulation",
]
