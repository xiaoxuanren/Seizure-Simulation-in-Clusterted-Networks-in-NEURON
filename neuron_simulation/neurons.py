"""Single-compartment Hodgkin-Huxley + A-current cell builders.

This module is the NEURON biophysical analogue of the LIF project's
``models.py``. Instead of a reduced integrate-and-fire point neuron it builds a
single-compartment soma with:

* NEURON's built-in ``hh`` mechanism (fast Na+, delayed-rectifier K+, leak), and
* the custom ``kA`` mechanism (A-type / Kv4-like transient K+ current), which is
  **inert at its shipped parameters**: ``gbar_kA`` is a dead parameter and does
  not model 4-AP. Reducing it does nothing to the burst phenotype, at any dose,
  on any topology -- see :mod:`neuron_simulation.states` and
  ``tests/test_kA_characterization.py``. (It is not bitwise neutral, though; the
  values are retained so existing datasets reproduce. See
  :data:`DEFAULT_GBAR_KA`.)

Excitatory and inhibitory cells share the same membrane machinery; they differ
only in their default A-current density and in the sign of the synapses they
*make* (enforced elsewhere by Dale's law). Every cell also carries its own
spike detector so the simulator can read spike times without post-processing
the voltage trace.

The compiled mechanisms (``kA``, ``DepSyn``) must be built first with
``nrnivmodl mechanisms`` -- see ``mechanisms/README.md``. Call
:func:`load_mechanisms` once before constructing any cell.
"""

import os

from neuron import h

# NEURON's standard run library provides ``finitialize``/``continuerun``.
h.load_file("stdrun.hoc")


# --------------------------------------------------------------------------- #
# Mechanism loading
# --------------------------------------------------------------------------- #
def _mechanism_available(name="kA"):
    """Return whether a density mechanism is already registered with NEURON.

    Args:
        name: Name of the density mechanism to probe (default ``"kA"``).

    Returns:
        ``True`` if a throwaway section can insert ``name`` (i.e. the compiled
        mechanism is already loaded), otherwise ``False``.
    """
    try:
        probe = h.Section(name="_mech_probe")
        try:
            probe.insert(name)
            return True
        finally:
            del probe
    except Exception:
        return False


def load_mechanisms(mechanisms_dir=None):
    """Load the compiled ``kA``/``DepSyn`` mechanisms exactly once.

    NEURON auto-loads a ``nrnmech.dll`` found in the current working directory,
    so this helper first checks whether the mechanisms are already registered
    and returns quietly if so (re-loading the same mechanisms raises a HOC
    "name already exists" error). Otherwise it searches the package directory,
    the ``mechanisms/`` sub-directory, and the working directory for a compiled
    artifact and loads it in a platform-appropriate way.

    Args:
        mechanisms_dir: Optional explicit directory containing the compiled
            mechanisms (a ``nrnmech.dll`` on Windows or an ``x86_64``/``arm64``
            build sub-directory on Linux/macOS). Defaults to the ``mechanisms``
            folder next to this module.

    Returns:
        None.

    Raises:
        RuntimeError: If no compiled mechanisms can be found or loaded. The
            message points at ``nrnivmodl mechanisms``.
    """
    if _mechanism_available("kA"):
        return

    here = os.path.dirname(os.path.abspath(__file__))
    if mechanisms_dir is None:
        mechanisms_dir = os.path.join(here, "mechanisms")

    # Windows: look for an explicit nrnmech.dll to load with nrn_load_dll.
    dll_candidates = [
        os.path.join(here, "nrnmech.dll"),
        os.path.join(mechanisms_dir, "nrnmech.dll"),
        os.path.join(os.getcwd(), "nrnmech.dll"),
    ]
    for dll in dll_candidates:
        if os.path.exists(dll):
            h.nrn_load_dll(dll)
            if _mechanism_available("kA"):
                return

    # Linux/macOS: neuron.load_mechanisms() consumes the directory that holds
    # the compiled build sub-folder and tracks already-loaded directories.
    import neuron

    for directory in (here, mechanisms_dir, os.getcwd()):
        try:
            if neuron.load_mechanisms(directory):
                if _mechanism_available("kA"):
                    return
        except Exception:
            continue

    raise RuntimeError(
        "Compiled kA/DepSyn mechanisms not found. Build them first with "
        "`nrnivmodl mechanisms` from the neuron_simulation/ directory "
        "(see mechanisms/README.md)."
    )


# --------------------------------------------------------------------------- #
# Default biophysical parameters
# --------------------------------------------------------------------------- #
#: Default A-current density (S/cm2) for the "normal" (drug-free) state. ``kA`` is
#: inert at its shipped gating (see :mod:`neuron_simulation.states`), so this value
#: does not shape the network's behaviour and reducing it is not a 4-AP block.
#:
#: It is nonetheless RETAINED, not zeroed, specifically so the existing datasets
#: reproduce bit-for-bit. Inertness is a statement about the *subthreshold* current;
#: m^4 is still non-negligible at the spike peak, so zeroing gbar perturbs spike
#: waveforms, and this recurrent network is chaotic enough to amplify that into a
#: different spike train. Verified, not assumed: ``scripts/check_ka_contribution.py``
#: runs the notebook's network twice changing only gbar_kA, and the arms diverge at
#: t = 2175.5 ms -- by 20 s all 926 spike trains differ. Do not "clean this up" to 0.0.
#:
#: (That run's spike counts, 6180 vs 4442, are NOT a -28% rate effect: 83% of spikes
#: are in bursts at ~1703 spikes/burst, so the gap is one burst crossing the window
#: edge -- arm A's third burst peaks at 19380 ms, arm B's has not fired by 20000 ms.
#: The window yields 2 inter-burst intervals for A and 1 for B, so it settles bitwise
#: identity only; no phenotype conclusion follows from it, and none is needed here.)
#:
#: (That check ran on the notebook's current topology, ~926 neurons at 1.56%
#: density. The sessions under ``NEURON data/`` were generated from a denser
#: 10-cluster variant -- 3.61% density -- and their ``session_metadata.json``
#: records ``gbar_kA_exc = 0.006`` / ``gbar_kA_inh = 0.004``, so these exact
#: defaults are what reproduces them. Chaotic divergence under a gbar change is a
#: generic property of the recurrent dynamics, not specific to either topology.)
DEFAULT_GBAR_KA = 0.006
#: Inhibitory cells carry a nominally weaker A-current. This was intended to make
#: them recruit a touch earlier than excitatory cells, matching fast-spiking
#: interneuron behaviour, but ``kA`` is inert so the E/I difference has no effect
#: either -- inhibitory cells recruit earlier only via their synaptic drive.
DEFAULT_GBAR_KA_INH = 0.004


class Cell:
    """A single-compartment HH + A-current neuron with a built-in spike detector.

    The soma is a 20 x 20 um cylinder carrying the ``hh`` and ``kA`` density
    mechanisms. Recurrent and background synapses are *not* created here; the
    network builder attaches one point-process synapse per incoming connection
    so that short-term depression and the ground-truth wiring stay per-edge.

    Args:
        gid: Global integer id, consistent with the topology and saved outputs.
        is_inhibitory: Whether this cell is an inhibitory interneuron. Only the
            default A-current density differs; the sign of the synapses the cell
            *makes* is enforced by the network builder (Dale's law).
        gbar_kA: A-current density (S/cm2). Defaults to :data:`DEFAULT_GBAR_KA`
            for excitatory cells and :data:`DEFAULT_GBAR_KA_INH` for inhibitory
            cells. Inert at the shipped ``kA`` parameters (see
            :mod:`neuron_simulation.states`).
        spike_threshold: Membrane voltage (mV) whose upward crossing the built-in
            ``NetCon`` records as a spike time.
        cluster_id: Optional cluster index carried for bookkeeping/metadata.

    Returns:
        An initialized ``Cell`` whose ``soma`` can be wired into a network.
    """

    def __init__(
        self,
        gid,
        is_inhibitory=False,
        gbar_kA=None,
        spike_threshold=0.0,
        cluster_id=-1,
        adapt=False,
        sahp_ainc_fast=0.009,
        sahp_tau_fast=200.0,
        sahp_ainc_slow=0.001,
        sahp_tau_slow=2000.0,
        sahp_ek=-90.0,
    ):
        self.gid = int(gid)
        self.is_inhibitory = bool(is_inhibitory)
        self.cluster_id = int(cluster_id)

        if gbar_kA is None:
            gbar_kA = DEFAULT_GBAR_KA_INH if is_inhibitory else DEFAULT_GBAR_KA
        self.gbar_kA = float(gbar_kA)

        # --- membrane ---
        self.soma = h.Section(name="soma_%d" % self.gid)
        self.soma.L = 20.0
        self.soma.diam = 20.0
        self.soma.cm = 1.0
        self.soma.Ra = 100.0
        self.soma.insert("hh")
        self.soma.insert("kA")
        self.soma(0.5).kA.gbar = self.gbar_kA

        # Dynamic extracellular K+ (kdyn) writes ek from [K+]o, which hh/kA read.
        # This is the seizure substrate: firing raises [K+]o -> ek depolarizes ->
        # positive feedback; glial clearance (tau_k, set by the network builder)
        # is the negative feedback. ki defaults to 72 mM so resting E_K = -77 mV,
        # matching the tuned regime. (Physically cleaner: ki=140 -> E_K_rest
        # ~-94 mV, but then re-tune exc_weight_scale up ~1.5x.)
        self.soma.insert("kdyn")

        # --- per-cell synapse receivers ---
        # Incoming recurrent synapses (one point process per edge) are appended
        # here by the network builder; the background-noise synapse is created
        # by noise.py. Kept as lists so the whole graph stays introspectable.
        self.recurrent_synapses = []
        self.netcons = []
        self.noise = None

        # --- spike detector ---
        self.spike_times = h.Vector()
        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self._spike_detector.threshold = float(spike_threshold)
        self._spike_detector.record(self.spike_times)

        # --- slow spike-triggered adaptation (sAHP) ---
        # Optional intrinsic slow K+ conductance driven by the cell's OWN spikes
        # (a self-NetCon from the soma voltage). Each spike bumps the conductance;
        # its seconds-scale decay (sahp_tau) provides spike-frequency adaptation
        # and sets the multi-second inter-burst interval. Off by default; enabled
        # via ``build_network(adapt=True)``.
        self.sahp = None
        self._adapt_nc = None
        if adapt:
            self.sahp = h.sAHP(self.soma(0.5))
            self.sahp.ainc_fast = float(sahp_ainc_fast)
            self.sahp.tau_fast = float(sahp_tau_fast)
            self.sahp.ainc_slow = float(sahp_ainc_slow)
            self.sahp.tau_slow = float(sahp_tau_slow)
            self.sahp.ek = float(sahp_ek)
            self._adapt_nc = h.NetCon(self.soma(0.5)._ref_v, self.sahp, sec=self.soma)
            self._adapt_nc.threshold = float(spike_threshold)
            self._adapt_nc.delay = 0.0
            self._adapt_nc.weight[0] = 1.0

    def set_gbar_kA(self, gbar_kA):
        """Set the A-current density for this cell.

        The ``kA`` mechanism is inert at its shipped parameters, so this does not
        move the burst phenotype at any value. It still perturbs the spike
        waveform slightly, which a recurrent network amplifies into a different
        spike train -- so this is not a no-op on saved output. See
        :mod:`neuron_simulation.states` and :data:`DEFAULT_GBAR_KA`.

        Args:
            gbar_kA: New A-current density in S/cm2.

        Returns:
            None. The soma's ``kA.gbar`` is updated in place.
        """
        self.gbar_kA = float(gbar_kA)
        self.soma(0.5).kA.gbar = self.gbar_kA

    def get_spike_times(self):
        """Return the recorded spike times as a plain list of milliseconds.

        Args:
            None.

        Returns:
            A list of spike times (ms). Empty until the simulation has run.
        """
        return list(self.spike_times)


def build_cell(gid, is_inhibitory=False, gbar_kA=None, cluster_id=-1, spike_threshold=0.0,
               adapt=False, sahp_ainc_fast=0.003, sahp_tau_fast=200.0,
               sahp_ainc_slow=0.001, sahp_tau_slow=2000.0, sahp_ek=-90.0):
    """Construct one HH + A-current cell (thin wrapper over :class:`Cell`).

    Args:
        gid: Global integer id for the cell.
        is_inhibitory: Whether to build an inhibitory interneuron.
        gbar_kA: Optional A-current density override in S/cm2.
        cluster_id: Optional cluster index carried into the cell for metadata.
        spike_threshold: Spike-detection voltage threshold in mV.
        adapt: Whether to add the intrinsic two-timescale adaptation current (sAHP).
        sahp_ainc_fast: Per-spike increment of the fast SFA component (uS).
        sahp_tau_fast: Fast SFA decay time constant (ms).
        sahp_ainc_slow: Per-spike increment of the slow AHP component (uS).
        sahp_tau_slow: Slow AHP decay time constant (ms).
        sahp_ek: sAHP K+ reversal potential (mV).

    Returns:
        An initialized :class:`Cell`.
    """
    return Cell(
        gid,
        is_inhibitory=is_inhibitory,
        gbar_kA=gbar_kA,
        cluster_id=cluster_id,
        spike_threshold=spike_threshold,
        adapt=adapt,
        sahp_ainc_fast=sahp_ainc_fast,
        sahp_tau_fast=sahp_tau_fast,
        sahp_ainc_slow=sahp_ainc_slow,
        sahp_tau_slow=sahp_tau_slow,
        sahp_ek=sahp_ek,
    )
