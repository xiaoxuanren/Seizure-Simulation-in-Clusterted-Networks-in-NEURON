"""Micro-benchmark: is learned-LIF training faster on GPU, and how much does
num_workers matter?

Builds the REAL event-window training path (same dataset, model, and
train step as run_pipeline) on a small session, then times, per config:
  * load_only : iterate the DataLoader + move batch to device (data-loading term)
  * full_step : load + forward + backward + optimizer step (end-to-end term)

Per-batch time is essentially independent of N (compute is B*K*window_len; the
W[neuron_ids] gather is O(B*K)), so a small session is representative of the
flagship's per-batch cost. Also validates that num_workers>0 works on Windows.

Run: python -u _bench_gpu.py --session "NEURON data/20260704_133726"
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO, os.path.join(REPO, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def build_dataset(session, K, max_delay):
    from lif_inference.learned_lif_connectivity import (
        load_all_recordings, compute_neighbor_indices,
        build_train_val_event_datasets)
    data = load_all_recordings(session, dt=1.0)
    sm = data["spike_matrix"]
    bnd = data["boundaries"]
    pos = data["neuron_positions"]
    n = data["n_neurons"]
    print(f"  loaded N={n} recs={data['n_recordings']} "
          f"spike_matrix={sm.shape} dtype={sm.dtype}", flush=True)
    ni, K_actual, _ = compute_neighbor_indices(
        pos, K, spike_matrix=sm, mode="hybrid", boundaries=bnd,
        spatial_frac=0.8, excluded_bins=None,
        temporal_min_lag=1, temporal_max_lag=max_delay)
    train_ds, _val_ds, _strat = build_train_val_event_datasets(
        sm, ni, np.arange(n),
        pre_context=50, post_context=10, warmup=30,
        neg_ratio=1.0, neg_min_distance=100,
        boundaries=bnd, excluded_bins=None,
        event_anchor_mode="post", pre_event_min_lag=1,
        pre_event_max_lag=max_delay, pre_event_max_anchors=None,
        val_fraction=0.2, rng_seed=42)
    return n, K_actual, train_ds


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def time_config(device, num_workers, n, K_actual, max_delay, train_ds,
                batch_size, warmup_batches, timed_batches):
    from lif_inference.learned_lif_connectivity import PerNeuronLIF
    from lif_inference.event_training import compute_event_loss

    pin = torch.cuda.is_available()
    model = PerNeuronLIF(n_neurons=n, K=K_actual, max_delay=max_delay,
                         threshold_mode="adaptive").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def new_loader():
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers,
                          persistent_workers=num_workers > 0, pin_memory=pin)

    # ---- load_only ----
    loader = new_loader()
    it = iter(loader)
    for _ in range(warmup_batches):
        pre, post, nid, _ = next(it)
        pre.to(device); post.to(device); nid.to(device)
    _sync(device)
    t0 = time.time()
    c = 0
    for pre, post, nid, _ in it:
        pre = pre.to(device); post = post.to(device); nid = nid.to(device)
        c += 1
        if c >= timed_batches:
            break
    _sync(device)
    load_ms = 1000 * (time.time() - t0) / max(c, 1)
    win_shape = tuple(pre.shape)
    del loader, it

    # ---- full_step ----
    model.train()
    loader = new_loader()
    it = iter(loader)
    for _ in range(warmup_batches):
        pre, post, nid, _ = next(it)
        pre = pre.to(device); post = post.to(device); nid = nid.to(device)
        opt.zero_grad()
        sp, _v, w = model(pre, post, nid, tbptt_len=pre.shape[2])
        loss, _s, _l = compute_event_loss(sp, post, w, 30, 5.0, 0.01)
        loss.backward(); opt.step()
    _sync(device)
    t0 = time.time()
    c = 0
    for pre, post, nid, _ in it:
        pre = pre.to(device); post = post.to(device); nid = nid.to(device)
        opt.zero_grad()
        sp, _v, w = model(pre, post, nid, tbptt_len=pre.shape[2])
        loss, _s, _l = compute_event_loss(sp, post, w, 30, 5.0, 0.01)
        loss.backward(); opt.step()
        c += 1
        if c >= timed_batches:
            break
    _sync(device)
    step_ms = 1000 * (time.time() - t0) / max(c, 1)
    del loader, it, model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(device=str(device), num_workers=num_workers, batch=batch_size,
                load_ms=load_ms, step_ms=step_ms,
                compute_ms=max(step_ms - load_ms, 0.0), win=win_shape,
                windows_per_s=batch_size * 1000.0 / step_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="NEURON data/20260704_133726")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--max-delay", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup-batches", type=int, default=5)
    ap.add_argument("--timed-batches", type=int, default=60)
    args = ap.parse_args()

    print(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"threads={torch.get_num_threads()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"[bench] building dataset from {args.session} ...", flush=True)
    tb = time.time()
    n, K_actual, train_ds = build_dataset(args.session, args.K, args.max_delay)
    print(f"[bench] dataset ready: N={n} K={K_actual} windows={len(train_ds)} "
          f"(build {time.time()-tb:.1f}s)", flush=True)

    batch_sizes = [64, 256, 1024]
    configs = []
    for bs in batch_sizes:
        configs.append(("cpu", 0, bs))
        configs.append(("cuda", 0, bs))
    configs.append(("cpu", 4, 64))   # does parallel loading help CPU at all?
    rows = []
    for device, nw, bs in configs:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        print(f"\n[bench] timing device={device} nw={nw} batch={bs} ...", flush=True)
        try:
            r = time_config(device, nw, n, K_actual, args.max_delay, train_ds,
                            bs, args.warmup_batches, args.timed_batches)
            rows.append(r)
            print(f"  win={r['win']} step={r['step_ms']:.1f}ms/batch "
                  f"=> {r['windows_per_s']:.0f} windows/s", flush=True)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n========= SUMMARY (throughput = windows/sec, higher = faster) =========", flush=True)
    print(f"{'config':<16}{'batch':>7}{'step_ms':>10}{'load_ms':>10}{'windows/s':>12}", flush=True)
    for r in rows:
        tag = f"{r['device']}/nw={r['num_workers']}"
        print(f"{tag:<16}{r['batch']:>7}{r['step_ms']:>10.1f}{r['load_ms']:>10.1f}"
              f"{r['windows_per_s']:>12.0f}", flush=True)
    if rows:
        best = max(rows, key=lambda r: r["windows_per_s"])
        print(f"\nFASTEST: {best['device']}/nw={best['num_workers']} "
              f"batch={best['batch']} => {best['windows_per_s']:.0f} windows/s", flush=True)


if __name__ == "__main__":
    main()
