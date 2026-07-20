"""Ho-et-al.-2025 conductance-based cells for the clustered NEURON network.

Drop-in replacement for ``neuron_simulation/neurons.py``. Instead of the
built-in ``hh`` + ``kA`` membrane, each soma carries the full Ho et al. (2025,
J. Neurosci.; ModelDB 2018263) intrinsic conductance set, so that
downmodulating the delayed-rectifier ``g_ipotassium`` (gK) reproduces the 4-AP
cascade: spike broadening -> Ca2+ influx -> I_KCa recruitment -> rheobase shift
-> noise-robust ~3 Hz almost-periodic ictal rhythm, plus the PY<->FS
differential (FS depolarization block under strong reduction).

WHY THIS SWAP (not just cutting gkbar_hh):
    ``hh`` has no calcium current and no Ca2+-activated K+ current. The entire
    Ho phenotype runs through I_KCa (bundled with I_Ca in ``ikCa.mod``), so the
    A-current/gkbar route cannot reproduce it. The FS vs PY difference is
    exactly g_ikCa = 0 (FS) vs 9 (PY).

[K+]o FEEDBACK + RESTING PUMP:
    ``kdyn.mod`` is the DYNAMIC [K+]o integrator (READs the summed ik, integrates
    its own internal ko, WRITEs the dynamic ek that every Ho K+ current READs) --
    it carries the seizure [K+]o -> E_K positive feedback. SEPARATELY, Ho's
    ``ikpump`` + ``kbalance`` are RE-ADDED to restore the electrogenic resting
    OUTWARD current that keeps cells quiescent at rest; without it these single-
    compartment cells self-fire and have a knife-edge F-I (no graded low-rate
    region). ikpump reads the STATIC NEURON-ion ko pinned by kbalance's init
    (~3 mM) -- used ONLY by the pump, so it does NOT touch kdyn's dynamic ek and
    there is no conflict. Pump strength is per-population (``ikpumpmax`` PY 20 /
    FS 30). kdyn.ki ~133 and ko_rest ~3 match Ho's E_K regime. For a *faithful*
    reproduction, delete kdyn and rebuild Ho's RxD layer instead.

MECHANISMS TO COMPILE (copy from the Ho repo into mechanisms/, then nrnivmodl):
    isodium.mod isodiumP.mod ipotassium.mod ipotassiumM.mod ikCa.mod
    ikleak.mod ileak.mod iext.mod ikpump.mod kbalance.mod
    (+ keep your kdyn.mod; keep AmpaNmda/DepSyn/ExpSyn)
    Optional for the "full cascade": GradedSyn.mod (Fig-7 AP-broadening synapse).

INTEGRATION:
    - Compatible with the existing ``build_cell(...)`` call signature: it still
      accepts ``is_inhibitory`` (selects FS vs PY), ``gid``, ``cluster_id``,
      ``spike_threshold``; legacy ``gbar_kA``/``adapt``/``sahp_*`` kwargs are
      accepted and ignored so callers don't break.
    - The 4-AP knob is now ``set_gK`` (writes soma.ipotassium.g), replacing
      ``set_gbar_kA``. In network_builder.build_network, pass gK values instead
      of gbar_kA and keep setting ``kdyn.tau_k`` as before.
    - Move spike_threshold to ~ -15 mV: broadened spikes + FS depolarization
      block plateaus misbehave at a 0 mV crossing.

NUMERICAL (important): integrate at dt <= 0.025 ms. These mechanisms are
UNSTABLE at dt = 0.05 (the usual HH value): the membrane voltage diverges
(|Vm| -> 1e3-1e4 mV) and firing rates / [K+]o become garbage. dt = 0.025 is
stable and converged; use 0.01 for high-rate or final runs.
"""

import os
from neuron import h

h.load_file("stdrun.hoc")

# --- normal (drug-free) reference conductances, mS/cm2 (Ho Table 1) --------- #
# gK here is the 4-AP knob; sweep PY 15 -> 0.3, FS 10 -> 7 -> 0.03.
PY_PARAMS = dict(
    diam=45.0, L=130.0 / 3.14159265,
    g_ipotassium=15.0,     # delayed rectifier gK (KNOB)
    g_ipotassiumM=0.02,    # I_Km
    g_isodium=35.0,        # I_Na
    g_isodiumP=0.2,        # I_NaP
    g_ikCa=9.0,            # gKCa (Ca2+-activated K+) -- nonzero in PY
    gCa_ikCa=0.1,          # Ca2+ conductance feeding I_KCa
    glcl_ileak=0.15,       # Cl- leak
    ikpumpmax=20.0,        # Na/K pump strength (electrogenic resting outward bias)
)
FS_PARAMS = dict(
    diam=55.0, L=130.0 / 3.14159265,
    g_ipotassium=10.0,     # delayed rectifier gK (KNOB)
    g_ipotassiumM=0.0,     # no I_Km in FS
    g_isodium=35.0,
    g_isodiumP=0.0,        # no I_NaP in FS
    g_ikCa=0.0,            # gKCa = 0  <-- source of FS depolarization block
    gCa_ikCa=0.03,
    glcl_ileak=0.1,
    ikpumpmax=30.0,        # stronger pump in FS (kept quiescent at rest)
)

# kdyn tuning so resting E_K matches Ho's regime (ki=133, ko~3).
KDYN_KI = 133.0
KDYN_KO_REST = 3.0


def _mech_available(name="ipotassium"):
    try:
        probe = h.Section(name="_probe")
        try:
            probe.insert(name)
            return True
        finally:
            del probe
    except Exception:
        return False


def load_mechanisms(mechanisms_dir=None):
    """Load compiled Ho mechanisms once (probes for ``ipotassium``)."""
    if _mech_available("ipotassium"):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    mechanisms_dir = mechanisms_dir or os.path.join(here, "mechanisms")
    import neuron
    for d in (here, mechanisms_dir, os.getcwd()):
        try:
            if neuron.load_mechanisms(d) and _mech_available("ipotassium"):
                return
        except Exception:
            continue
    raise RuntimeError(
        "Ho mechanisms not found. Copy isodium/isodiumP/ipotassium/ipotassiumM/"
        "ikCa/ikleak/ileak (+ kdyn) into mechanisms/ and run `nrnivmodl mechanisms`."
    )


class Cell:
    """Single-compartment Ho-et-al. PY or FS cell with a built-in spike detector.

    Args:
        gid: Global integer id.
        is_inhibitory: Selects the FS parameter set (gKCa=0, no I_NaP/I_Km);
            otherwise the PY set is used.
        gK: Optional delayed-rectifier density override (mS/cm2), the 4-AP knob.
            Defaults to the PY (15) or FS (10) normal reference.
        spike_threshold: Upward-crossing voltage (mV) recorded as a spike.
            Use ~ -15 mV, not 0, for broadened / depolarization-block spikes.
        cluster_id: Cluster index carried for metadata.
    """

    def __init__(self, gid, is_inhibitory=False, gK=None, iext=0.0, ikpumpmax=None,
                 spike_threshold=-15.0, cluster_id=-1, **legacy_ignored):
        self.gid = int(gid)
        self.is_inhibitory = bool(is_inhibitory)
        self.cluster_id = int(cluster_id)
        params = FS_PARAMS if is_inhibitory else PY_PARAMS

        # --- membrane geometry ---
        self.soma = h.Section(name="soma_%d" % self.gid)
        self.soma.nseg = 1
        self.soma.L = params["L"]
        self.soma.diam = params["diam"]
        self.soma.cm = 1.0

        # --- Ho intrinsic currents ---
        for mech in ("ileak", "ikleak", "isodium", "isodiumP",
                     "ipotassium", "ipotassiumM", "ikCa", "iext",
                     "kbalance", "ikpump"):
            self.soma.insert(mech)
        seg = self.soma(0.5)
        seg.isodium.g = params["g_isodium"]
        seg.isodiumP.g = params["g_isodiumP"]
        seg.ipotassium.g = params["g_ipotassium"] if gK is None else float(gK)
        seg.ipotassiumM.g = params["g_ipotassiumM"]
        seg.ikCa.g = params["g_ikCa"]
        seg.ikCa.gCa = params["gCa_ikCa"]
        seg.ileak.glcl = params["glcl_ileak"]
        # ikleak.g defaults to gL|K = 0.035 in the .mod (Ho Table 1); leave as-is.
        seg.iext.iext = float(iext)   # external DC bias current (uA/cm2); 0 = off
        self.gK = float(seg.ipotassium.g)
        self.iext = float(seg.iext.iext)

        # --- Na/K pump: electrogenic resting OUTWARD bias that restores Ho's
        #     quiescent resting point (without it these single-compartment cells
        #     self-fire and have a knife-edge F-I). ikpump reads a STATIC ion ko
        #     pinned by kbalance's init (~3 mM), used ONLY by the pump -- it does
        #     NOT touch kdyn's dynamic ek, so there is no conflict. Per-population.
        seg.kbalance.ko_init = KDYN_KO_REST
        seg.kbalance.ki_init = KDYN_KI
        seg.ikpump.koeq = KDYN_KO_REST
        seg.ikpump.ikpumpmax = params["ikpumpmax"] if ikpumpmax is None else float(ikpumpmax)
        self.ikpumpmax = float(seg.ikpump.ikpumpmax)

        # --- [K+]o feedback (hybrid): kdyn integrates the summed ik -> ek ---
        # Drop ikpump/kbalance; kdyn.tau_k lumps pump+glia clearance.
        self.soma.insert("kdyn")
        seg.kdyn.ki = KDYN_KI
        seg.kdyn.ko_rest = KDYN_KO_REST

        # --- per-cell synapse receivers (network builder appends per edge) ---
        self.recurrent_synapses = []
        self.netcons = []
        self.noise = None

        # --- spike detector ---
        self.spike_times = h.Vector()
        self._spike_detector = h.NetCon(seg._ref_v, None, sec=self.soma)
        self._spike_detector.threshold = float(spike_threshold)
        self._spike_detector.record(self.spike_times)

    def set_gK(self, gK):
        """Set delayed-rectifier density (mS/cm2) -- the 4-AP knob."""
        self.gK = float(gK)
        self.soma(0.5).ipotassium.g = self.gK

    def set_iext(self, iext):
        """Set the external DC bias current (uA/cm2) on the soma (operating point)."""
        self.iext = float(iext)
        self.soma(0.5).iext.iext = self.iext

    # Back-compat shim so old callers that reach for set_gbar_kA still drive the knob.
    def set_gbar_kA(self, value):  # noqa: D401 (deprecated alias)
        self.set_gK(value)

    def get_spike_times(self):
        return list(self.spike_times)


def build_cell(gid, is_inhibitory=False, gK=None, iext=0.0, ikpumpmax=None, cluster_id=-1,
               spike_threshold=-15.0, **legacy_ignored):
    """Thin wrapper over :class:`Cell` (legacy gbar_kA/adapt/sahp kwargs ignored)."""
    return Cell(gid, is_inhibitory=is_inhibitory, gK=gK, iext=iext, ikpumpmax=ikpumpmax,
                cluster_id=cluster_id, spike_threshold=spike_threshold)


# --- state helpers: gK sweeps replace the deprecated gbar_kA / tau_k knobs --- #
def normal_gK():
    """Reference (non-epileptic) gK per population, mS/cm2."""
    return {"gK_exc": 15.0, "gK_inh": 10.0, "state_name": "normal"}


def four_ap_gK(py_gK=0.3, fs_gK=10.0):
    """gK downmodulation in the PY population (Fig 7/8 route).

    py_gK sweeps 15 -> 0.3; leave fs_gK at 10 to isolate the PY effect, or
    set fs_gK = 7 (moderate, protective) / 0.03 (strong, depol-block -> Fig 9).
    """
    return {"gK_exc": float(py_gK), "gK_inh": float(fs_gK),
            "state_name": f"gK_py{py_gK:g}_fs{fs_gK:g}"}
