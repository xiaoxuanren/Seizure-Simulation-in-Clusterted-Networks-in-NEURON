"""Poisson background input -- the network's SOLE drive.

Each neuron receives an independent Poisson spike train (a NEURON ``NetStim``
with ``noise = 1``) delivered through a dedicated excitatory ``ExpSyn``. There
is deliberately *no* current injection, no periodic stimulation, and no tonic
driver anywhere in this project: the background noise is the only thing that
injects energy into the network.

Independent, reproducible, RE-SEEDABLE streams
    Each generator draws its intervals from its own ``Random`` object seeded with
    ``Random123(base_seed, gid, recording_index)`` and bound to the NetStim via
    ``noiseFromRandom``. This is what makes (a) every neuron's noise independent
    and (b) every *recording* a genuinely different trial: :meth:`PoissonNoise.reseed`
    re-keys the stream on the recording index, so recording ``r`` and recording
    ``r+1`` are not duplicates. (The old ``NetStim.seed()`` path did not reliably
    isolate per-recording streams and produced byte-identical recordings.)

Documented role (a validated finding):

* The noise is an *ignition seed*, not a driver. On its own -- with recurrent
  synapses removed -- it makes each neuron fire only ~0.07 Hz.
* Below a threshold noise level the network is silent; above it, recurrent
  excitation ignites synchronized network bursts.

Tuned defaults (rate 5 Hz, weight 0.0008 uS) keep a *single* background event a
few mV subthreshold, so the network integrates rather than chain-reacting.
"""

from neuron import h


class PoissonNoise:
    """A per-neuron Poisson generator wired to an excitatory synapse.

    Args:
        cell: The :class:`neuron_simulation.neurons.Cell` to drive.
        rate_hz: Mean Poisson rate in Hz.
        weight: Peak conductance increment (uS) per background event. The tuned
            default 0.0008 uS keeps a single event a few mV subthreshold.
        tau: Excitatory conductance decay time constant (ms).
        e_rev: Excitatory reversal potential (mV).
        start: Onset time of the generator (ms).
        delay: Synaptic delay from generator to synapse (ms).
        base_seed: Base seed; combined with the neuron id and recording index to
            key an independent ``Random123`` stream.
        recording_index: Recording index used to key the stream, so different
            recordings get different noise realizations.

    Returns:
        An initialized ``PoissonNoise`` holding the ``NetStim``, ``ExpSyn``,
        ``Random``, and connecting ``NetCon`` (all kept alive as attributes).
    """

    def __init__(
        self,
        cell,
        rate_hz=2.0,
        weight=0.0008,
        tau=3.0,
        e_rev=0.0,
        start=0.0,
        delay=0.0,
        base_seed=1000,
        recording_index=0,
    ):
        self.cell = cell
        self.gid = int(getattr(cell, "gid", 0))
        self.base_seed = int(base_seed)
        self.rate_hz = float(rate_hz)

        self.syn = h.ExpSyn(cell.soma(0.5))
        self.syn.tau = float(tau)
        self.syn.e = float(e_rev)

        self.stim = h.NetStim()
        self.stim.interval = 1000.0 / self.rate_hz if self.rate_hz > 0 else 1e12
        self.stim.number = 1e12  # effectively unbounded
        self.stim.start = float(start)
        self.stim.noise = 1.0  # fully Poisson

        # Independent, reproducible interval stream via a counter-based RNG.
        self.rng = h.Random()
        self.rng.Random123(self.base_seed, self.gid, int(recording_index))
        self.rng.negexp(1)  # unit-mean exponential; NetStim scales by `interval`
        self.stim.noiseFromRandom(self.rng)

        self.netcon = h.NetCon(self.stim, self.syn)
        self.netcon.weight[0] = float(weight)
        self.netcon.delay = float(delay)

    def reseed(self, recording_index):
        """Re-key this generator's stream to a new recording index.

        Every neuron's stream is a function of ``(base_seed, gid,
        recording_index)``, so calling this with a fresh ``recording_index``
        before a run yields a genuinely different -- yet reproducible -- Poisson
        realization for that neuron.

        Args:
            recording_index: Integer recording index that selects the stream.

        Returns:
            None.
        """
        self.rng.Random123(self.base_seed, self.gid, int(recording_index))
        self.rng.negexp(1)

    def set_rate(self, rate_hz):
        """Update the mean Poisson rate in place.

        Args:
            rate_hz: New mean rate in Hz.

        Returns:
            None. The generator's ``interval`` is updated.
        """
        self.rate_hz = float(rate_hz)
        self.stim.interval = 1000.0 / self.rate_hz if self.rate_hz > 0 else 1e12

    def set_weight(self, weight):
        """Update the background synaptic weight (uS) in place.

        Args:
            weight: New peak conductance increment in uS.

        Returns:
            None.
        """
        self.netcon.weight[0] = float(weight)


def add_poisson_noise(cells, rate_hz=2.0, weight=0.0008, tau=3.0, base_seed=1000, recording_index=0):
    """Attach an independent Poisson background generator to every cell.

    Args:
        cells: Sequence of :class:`neuron_simulation.neurons.Cell` objects.
        rate_hz: Mean Poisson rate in Hz for every generator.
        weight: Peak conductance increment (uS) per background event.
        tau: Excitatory conductance decay time constant (ms).
        base_seed: Base seed; each generator keys ``Random123(base_seed, gid,
            recording_index)`` so streams are independent yet reproducible.
        recording_index: Initial recording index for every generator's stream.

    Returns:
        A list of :class:`PoissonNoise` objects, one per cell (also stored on
        each cell's ``noise`` attribute).
    """
    generators = []
    for cell in cells:
        gen = PoissonNoise(
            cell, rate_hz=rate_hz, weight=weight, tau=tau,
            base_seed=base_seed, recording_index=recording_index,
        )
        cell.noise = gen
        generators.append(gen)
    return generators


def reseed_noise(generators, recording_index):
    """Re-key every generator's stream for a new recording.

    Args:
        generators: Sequence of :class:`PoissonNoise` objects.
        recording_index: Recording index selecting the new streams.

    Returns:
        None.
    """
    for gen in generators:
        gen.reseed(recording_index)
