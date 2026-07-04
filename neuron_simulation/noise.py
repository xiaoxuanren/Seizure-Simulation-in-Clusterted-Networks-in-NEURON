"""Poisson background input -- the network's SOLE drive.

Each neuron receives an independent Poisson spike train (a NEURON ``NetStim``
with ``noise = 1``) delivered through a dedicated excitatory ``ExpSyn``. There
is deliberately *no* current injection, no periodic stimulation, and no tonic
driver anywhere in this project: the background noise is the only thing that
injects energy into the network.

Documented role (a validated finding):

* The noise is an *ignition seed*, not a driver. On its own -- with recurrent
  synapses removed -- it makes each neuron fire only ~0.07 Hz.
* Below a threshold noise level the network is silent; above it, recurrent
  excitation ignites synchronized network bursts.
* With static synapses (``DepSyn`` d = 0) the noise level sets the burst
  frequency.

Typical per-neuron rates are ~16-22 Hz.
"""

from neuron import h


class PoissonNoise:
    """A per-neuron Poisson generator wired to an excitatory synapse.

    Args:
        cell: The :class:`neuron_simulation.neurons.Cell` to drive.
        rate_hz: Mean Poisson rate in Hz.
        weight: Peak conductance increment (uS) per background event.
        tau: Excitatory conductance decay time constant (ms).
        e_rev: Excitatory reversal potential (mV).
        start: Onset time of the generator (ms).
        delay: Synaptic delay from generator to synapse (ms).
        seed: Integer seed for this generator's private random stream.

    Returns:
        An initialized ``PoissonNoise`` holding the ``NetStim``, ``ExpSyn``, and
        connecting ``NetCon`` (all kept alive as attributes).
    """

    def __init__(
        self,
        cell,
        rate_hz=18.0,
        weight=0.0016,
        tau=3.0,
        e_rev=0.0,
        start=0.0,
        delay=0.0,
        seed=0,
    ):
        self.cell = cell
        self.rate_hz = float(rate_hz)

        self.syn = h.ExpSyn(cell.soma(0.5))
        self.syn.tau = float(tau)
        self.syn.e = float(e_rev)

        self.stim = h.NetStim()
        self.stim.interval = 1000.0 / self.rate_hz if self.rate_hz > 0 else 1e12
        self.stim.number = 1e12  # effectively unbounded
        self.stim.start = float(start)
        self.stim.noise = 1.0  # fully Poisson
        self.stim.seed(int(seed))

        self.netcon = h.NetCon(self.stim, self.syn)
        self.netcon.weight[0] = float(weight)
        self.netcon.delay = float(delay)

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


def add_poisson_noise(cells, rate_hz=18.0, weight=0.0016, tau=3.0, base_seed=1000):
    """Attach an independent Poisson background generator to every cell.

    Args:
        cells: Sequence of :class:`neuron_simulation.neurons.Cell` objects.
        rate_hz: Mean Poisson rate in Hz for every generator.
        weight: Peak conductance increment (uS) per background event.
        tau: Excitatory conductance decay time constant (ms).
        base_seed: Base seed; generator ``i`` uses ``base_seed + i`` so streams
            are independent yet reproducible.

    Returns:
        A list of :class:`PoissonNoise` objects, one per cell (also stored on
        each cell's ``noise`` attribute).
    """
    generators = []
    for i, cell in enumerate(cells):
        gen = PoissonNoise(cell, rate_hz=rate_hz, weight=weight, tau=tau, seed=base_seed + i)
        cell.noise = gen
        generators.append(gen)
    return generators
