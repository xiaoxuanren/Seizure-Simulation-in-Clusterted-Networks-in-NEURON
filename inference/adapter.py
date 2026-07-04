"""Adapter: run the vendored LIF inference on NEURON output, report AUC/FDR.

The NEURON simulator writes sessions in the EXACT LIF layout (see
``neuron_simulation/io.py``), so the vendored ``lif_inference`` package
(``inference/lif_inference/``) runs against them unmodified. This adapter is the
thin glue that:

1. validates that a saved session matches the format the inference expects
   (fields, shapes, dtypes -- the required round-trip assertion),
2. runs the spike-only learned-LIF pipeline,
3. runs the training-free CCG baseline on the *same* surfaced inputs, and
4. reports AUC and FDR (false-discovery rate) for both against the ground truth,

exactly as for the LIF data. The learned pipeline requires ``torch`` and
``scikit-learn``; the CCG baseline is training-free but shares the same import
chain, so ``torch`` is required either way.

Note on the startup transient: ``run_simulation`` already discards the first
~1 s before saving, so on-disk recordings begin at steady state and inference
sees no startup transient.
"""

import argparse
import glob
import os
import sys

import numpy as np


def _ensure_paths():
    """Put the vendored ``lif_inference`` package and project root on ``sys.path``.

    Args:
        None.

    Returns:
        None. Side effect: ``inference/`` and the project root are importable.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    for path in (here, project_root):
        if path not in sys.path:
            sys.path.insert(0, path)


# --------------------------------------------------------------------------- #
# Format validation (the required round-trip assertion)
# --------------------------------------------------------------------------- #
def validate_session_format(session_dir, recording_idx=0):
    """Assert a saved session matches the inference input contract.

    Loads the network file and one recording and checks the fields, shapes, and
    dtypes the inference relies on. Raises ``AssertionError`` on the first
    mismatch. This is the "load it back and assert" gate that must pass before a
    dataset is handed to inference.

    Args:
        session_dir: Path to a saved session directory.
        recording_idx: Recording index to validate.

    Returns:
        A dict of the validated shapes/values (``n_neurons``, ``n_connections``,
        ``duration``, ``has_voltage``) for logging.
    """
    net_files = glob.glob(os.path.join(session_dir, "network_*.npz"))
    assert net_files, f"No network_*.npz in {session_dir}"
    net = np.load(net_files[0], allow_pickle=True)

    for key in ("connections", "neuron_positions", "cluster_assignments"):
        assert key in net.files, f"network file missing required field '{key}'"

    connections = net["connections"]
    positions = net["neuron_positions"]
    cluster_assignments = net["cluster_assignments"]
    n_neurons = len(positions)

    assert positions.ndim == 2 and positions.shape[1] == 2, "neuron_positions must be (N, 2)"
    assert cluster_assignments.shape == (n_neurons,), "cluster_assignments must be (N,)"
    assert len(connections) > 0, "connection table is empty"
    row = connections[0]
    assert len(row) == 4, "each connection row must be [pre, post, weight, type]"
    assert str(row[3]) in ("exc", "inh"), "connection type must be 'exc' or 'inh'"
    ids = np.array([[int(c[0]), int(c[1])] for c in connections])
    assert ids.min() >= 0 and ids.max() < n_neurons, "connection ids out of range 0..N-1"

    rec_path = os.path.join(session_dir, f"recording{recording_idx:03d}.npz")
    assert os.path.exists(rec_path), f"missing {rec_path}"
    rec = np.load(rec_path, allow_pickle=True)
    for key in ("spike_times", "duration", "resampled_spikes"):
        assert key in rec.files, f"recording missing required field '{key}'"
    spike_times = rec["spike_times"]
    assert len(spike_times) == n_neurons, "spike_times length must equal N"
    # Spike times must be a 1-D float array per neuron, in milliseconds.
    for i in range(min(n_neurons, 5)):
        arr = np.asarray(spike_times[i], dtype=float)
        assert arr.ndim == 1, f"spike_times[{i}] must be 1-D"
    duration = float(rec["duration"])
    assert duration > 0, "duration must be positive"

    has_voltage = "voltage_traces" in rec.files or "voltage_hdf5_file" in rec.files
    print(
        f"[validate] OK: N={n_neurons}, connections={len(connections)}, "
        f"duration={duration:.0f} ms, voltage={'yes' if has_voltage else 'no'}"
    )
    return {
        "n_neurons": n_neurons,
        "n_connections": int(len(connections)),
        "duration": duration,
        "has_voltage": bool(has_voltage),
    }


def _empirical_fdr(metrics):
    """Compute the empirical false-discovery rate ``fp / (tp + fp)``.

    Args:
        metrics: A metrics dict containing integer ``tp`` and ``fp`` counts.

    Returns:
        The empirical FDR as a float (0.0 when no positives were predicted).
    """
    tp = int(metrics.get("tp", 0))
    fp = int(metrics.get("fp", 0))
    return float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run_inference(
    session_dir,
    run_learned=True,
    run_ccg=True,
    learned_params=None,
    ccg_min_lag=1,
    ccg_max_lag=5,
    ccg_threshold_mode="oracle_f1",
):
    """Run learned-LIF + CCG inference on a NEURON session and report AUC/FDR.

    Args:
        session_dir: Path to a saved NEURON session directory.
        run_learned: Whether to run the spike-only learned-LIF pipeline.
        run_ccg: Whether to run the CCG baseline (reuses the learned pipeline's
            surfaced inputs, so it requires ``run_learned=True``).
        learned_params: Optional overrides for
            ``lif_inference.run_learned_lif_pipeline`` (e.g. ``K``, ``n_epochs``,
            ``max_delay``, ``connectivity_threshold_mode``).
        ccg_min_lag: Smallest presynaptic lead (bins) of the CCG synaptic window.
        ccg_max_lag: Largest presynaptic lead (bins) of the CCG synaptic window.
        ccg_threshold_mode: CCG thresholding mode (``'oracle_f1'`` or
            ``'surrogate_fdr'``).

    Returns:
        A dict summarizing ``learned`` and ``ccg`` results, each with ``auc`` and
        ``fdr`` (plus ``ap``/``f1`` where available).
    """
    _ensure_paths()
    from sklearn.metrics import roc_auc_score

    validate_session_format(session_dir)

    summary = {"session_dir": session_dir, "learned": None, "ccg": None}
    results = None

    if run_learned:
        from lif_inference import run_learned_lif_pipeline

        params = {
            "K": 100,
            "n_epochs": 40,
            "max_delay": ccg_max_lag,
            "use_all_recordings": True,
            "exclude_detected_bursts": True,
            "connectivity_threshold_mode": "oracle_f1",
        }
        params.update(learned_params or {})
        print("\n=== learned-LIF (spike-only) ===")
        results, _conn_matrix = run_learned_lif_pipeline(str(session_dir), **params)
        summary["learned"] = {
            "auc": results.get("auc"),
            "ap": results.get("ap"),
            "f1": results.get("f1"),
            "fdr": _empirical_fdr(results),
            "estimated_fdr": results.get("estimated_fdr"),
        }
        print(f"learned-LIF: AUC={summary['learned']['auc']}, FDR={summary['learned']['fdr']:.3f}")

    if run_ccg:
        if results is None:
            raise ValueError("run_ccg=True requires run_learned=True (it reuses surfaced inputs).")
        from lif_inference.ccg_baseline import run_ccg_baseline

        print("\n=== CCG baseline ===")
        ccg = run_ccg_baseline(
            results["spike_matrix"],
            results["neighbor_indices"],
            neuron_ids=results["neuron_ids"],
            true_binary=results["true_binary"],
            boundaries=results["boundaries"],
            excluded_bins=results.get("excluded_bins"),
            threshold_mode=ccg_threshold_mode,
            min_lag=ccg_min_lag,
            max_lag=ccg_max_lag,
        )
        ccg_auc = float(roc_auc_score(ccg["labels"], ccg["scores"])) if len(np.unique(ccg["labels"])) > 1 else float("nan")
        summary["ccg"] = {
            "auc": ccg_auc,
            "f1": ccg["metrics"].get("f1"),
            "fdr": _empirical_fdr(ccg["metrics"]),
        }
        print(f"CCG baseline: AUC={ccg_auc:.3f}, FDR={summary['ccg']['fdr']:.3f}")

    print("\n=== inference summary ===")
    print(summary)
    return summary


def _resolve_session(session_arg):
    """Resolve a session argument that may be a path or the literal 'latest'.

    Args:
        session_arg: A session directory path or ``"latest"``.

    Returns:
        A concrete session directory path.
    """
    if session_arg != "latest":
        return session_arg
    _ensure_paths()
    from neuron_simulation.io import latest_session

    session = latest_session()
    if session is None:
        raise FileNotFoundError("No sessions found under 'NEURON data'.")
    return session


def main(argv=None):
    """CLI entry point: ``python inference/adapter.py <session_dir|latest>``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Run LIF inference on a NEURON session.")
    parser.add_argument("session", nargs="?", default="latest",
                        help="Session directory, or 'latest' (default).")
    parser.add_argument("--no-learned", action="store_true", help="Skip the learned-LIF model.")
    parser.add_argument("--no-ccg", action="store_true", help="Skip the CCG baseline.")
    parser.add_argument("--epochs", type=int, default=40, help="Learned-LIF training epochs.")
    parser.add_argument("--k", type=int, default=100, help="Candidate presynaptic neighbours (K).")
    args = parser.parse_args(argv)

    session_dir = _resolve_session(args.session)
    run_inference(
        session_dir,
        run_learned=not args.no_learned,
        run_ccg=not args.no_ccg,
        learned_params={"n_epochs": args.epochs, "K": args.k},
    )


if __name__ == "__main__":
    main()
