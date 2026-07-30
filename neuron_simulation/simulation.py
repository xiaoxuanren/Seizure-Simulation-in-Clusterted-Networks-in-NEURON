"""Run the NEURON network and collect spikes (and optional voltage).

This is the biophysical analogue of the LIF project's ``simulation.py``. It
sets the integration parameters, runs a fixed-step simulation with
``finitialize``/``continuerun``, and returns:

* ``spike_data`` -- a ``{neuron_id: np.ndarray of spike times in ms}`` mapping,
  read straight from each cell's built-in spike detector, and
* ``voltage_data`` -- an optional downsampled membrane-voltage bundle in the
  same shape the LIF ``save_recording_data`` expects (enabling the
  voltage-augmented inference mode).

The startup transient is handled here: the simulation runs for
``discard_transient_ms + duration``, and everything before the transient cutoff
is dropped and the clock re-zeroed, so saved spikes/voltage already exclude the
first ~1 s (matching "drop the first ~1 s before inference and all analysis").
"""

import numpy as np
from neuron import h


def run_simulation(
    network,
    duration=60000.0,
    dt=0.05,
    v_init=-65.0,
    discard_transient_ms=1000.0,
    record_voltage=False,
    voltage_dt=1.0,
    record_ko=False,
    ko_dt=5.0,
    progress_every_ms=None,
):
    """Run the network and return post-transient spikes, voltage, and [K+]o.

    Args:
        network: A built :class:`neuron_simulation.network_builder.Network`.
        duration: Length of the *kept* recording in milliseconds (after the
            discarded transient).
        dt: Fixed integration time step in milliseconds.
        v_init: Initial membrane voltage for ``finitialize`` (mV).
        discard_transient_ms: Startup transient discarded from the front of the
            run. Spike/voltage times are shifted so the kept window starts at 0.
        record_voltage: Whether to record downsampled somatic voltage per neuron.
        voltage_dt: Voltage sampling interval in milliseconds (>= ``dt``).
        record_ko: Whether to record the ``kdyn`` extracellular potassium
            concentration ``[K+]o`` per neuron (for the seizure analysis/plots).
        ko_dt: [K+]o sampling interval in milliseconds.
        progress_every_ms: If set, run in chunks of this many ms and print
            progress; if ``None`` the whole run happens in one ``continuerun``.

    Returns:
        A tuple ``(spike_data, voltage_data, ko_data)``:

        * ``spike_data``: ``{neuron_id: np.ndarray}`` of spike times in ms over
          ``[0, duration]`` (transient removed, clock re-zeroed).
        * ``voltage_data``: ``None`` unless ``record_voltage``; otherwise a dict
          with keys ``sample_rate`` (ms interval), ``times`` (ms), ``traces``
          (``float32`` ``[N, T]`` in mV), ``storage_backend='inline_npz'``, and
          ``units='mV'``.
        * ``ko_data``: ``None`` unless ``record_ko``; otherwise a dict with
          ``times`` (ms), ``mean_ko``/``min_ko``/``max_ko`` (``[T]`` in mM across
          neurons), ``sample_rate`` (ms), and ``units='mM'``.
    """
    total_ms = float(discard_transient_ms) + float(duration)
    h.dt = float(dt)
    h.celsius = float(network.config.get("celsius", h.celsius))

    voltage_vectors = None
    t_vector = None
    if record_voltage:
        if voltage_dt < dt:
            raise ValueError("voltage_dt must be >= dt.")
        voltage_vectors = [
            h.Vector().record(cell.soma(0.5)._ref_v, float(voltage_dt)) for cell in network.cells
        ]
        t_vector = h.Vector().record(h._ref_t, float(voltage_dt))

    ko_vectors = None
    ko_t_vector = None
    if record_ko:
        if ko_dt < dt:
            raise ValueError("ko_dt must be >= dt.")
        ko_vectors = [
            h.Vector().record(cell.soma(0.5).kdyn._ref_ko, float(ko_dt)) for cell in network.cells
        ]
        ko_t_vector = h.Vector().record(h._ref_t, float(ko_dt))

    h.finitialize(float(v_init))
    if progress_every_ms:
        next_stop = float(progress_every_ms)
        while h.t < total_ms - 1e-9:
            target = min(next_stop, total_ms)
            h.continuerun(target)
            print(f"  t = {h.t:.0f} / {total_ms:.0f} ms")
            # ``continuerun`` can land a hair below its target (fixed-step dt
            # does not divide total_ms exactly), which left the loop spinning
            # forever on a target it could no longer advance past.
            if target >= total_ms - 1e-9:
                break
            next_stop += float(progress_every_ms)
    else:
        h.continuerun(total_ms)

    # --- spikes: drop transient, re-zero the clock ---
    spike_data = {}
    for cell in network.cells:
        times = np.asarray(cell.get_spike_times(), dtype=float)
        kept = times[times >= discard_transient_ms] - discard_transient_ms
        spike_data[cell.gid] = kept

    # --- voltage: slice to the kept window, re-zero ---
    voltage_data = None
    if record_voltage:
        times = np.asarray(t_vector, dtype=float)
        keep = times >= discard_transient_ms
        kept_times = times[keep] - discard_transient_ms
        traces = np.array(
            [np.asarray(vec, dtype=np.float32)[keep] for vec in voltage_vectors],
            dtype=np.float32,
        )
        voltage_data = {
            "sample_rate": float(voltage_dt),
            "times": kept_times.astype(np.float64),
            "traces": traces,
            "storage_backend": "inline_npz",
            "units": "mV",
        }

    # --- [K+]o: slice to the kept window, average across neurons ---
    ko_data = None
    if record_ko:
        times = np.asarray(ko_t_vector, dtype=float)
        keep = times >= discard_transient_ms
        kept_times = times[keep] - discard_transient_ms
        ko_traces = np.array(
            [np.asarray(vec, dtype=np.float64)[keep] for vec in ko_vectors], dtype=np.float64
        )
        ko_data = {
            "sample_rate": float(ko_dt),
            "times": kept_times.astype(np.float64),
            "mean_ko": ko_traces.mean(axis=0),
            "min_ko": ko_traces.min(axis=0),
            "max_ko": ko_traces.max(axis=0),
            "units": "mM",
        }

    total_spikes = int(sum(len(v) for v in spike_data.values()))
    mean_rate = total_spikes / (network.n_neurons * duration / 1000.0) if network.n_neurons else 0.0
    ko_note = ""
    if ko_data is not None:
        ko_note = f", [K+]o {ko_data['mean_ko'].min():.1f}-{ko_data['mean_ko'].max():.1f} mM"
    print(
        f"Ran {total_ms:.0f} ms (kept {duration:.0f} ms), {total_spikes} spikes, "
        f"mean rate {mean_rate:.2f} Hz{ko_note}"
    )
    return spike_data, voltage_data, ko_data
