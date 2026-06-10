#!/usr/bin/env python3
"""Benchmark probe-training throughput: CPU (thread sweep) vs GPU vs vectorized GPU.

Standalone (torch only). Mirrors train_probe_with_direction in the experiment
scripts: nn.Linear(H, 1), full-batch Adam(lr=1e-2) for n_epochs, BCEWithLogits.
Reports ms per probe for each configuration and extrapolates to a full bootstrap
(n_bootstrap x n_layers x n_foundations probes). Writes a JSON baseline.

Why: the bootstrap trains tens of thousands of tiny probes. Whether that belongs
on CPU or GPU (and how many CPU threads) had been extrapolated from a laptop;
this measures it on the actual accelerator.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import torch
import torch.nn as nn


def make_data(n: int, h: int, device: torch.device, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, h, generator=g)
    y = (torch.rand(n, generator=g) > 0.5).float()
    return X.to(device), y.to(device)


def train_one(X, y, n_epochs: int, lr: float):
    probe = nn.Linear(X.shape[1], 1).to(X.device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(n_epochs):
        loss = loss_fn(probe(X).squeeze(-1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probe.weight.detach()


def time_sequential(device, n, h, n_epochs, lr, n_probes):
    X, y = make_data(n, h, device)
    train_one(X, y, 5, lr)  # warmup (kernels/caches)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_probes):
        train_one(X, y, n_epochs, lr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_probes * 1000.0  # ms/probe


def time_vectorized(device, n, h, n_epochs, lr, batch):
    """Train `batch` independent probes at once (the real speedup lever)."""
    g = torch.Generator().manual_seed(1)
    Xb = torch.randn(batch, n, h, generator=g).to(device)   # (B, N, H)
    yb = (torch.rand(batch, n, generator=g) > 0.5).float().to(device)
    W = torch.zeros(batch, h, 1, device=device, requires_grad=True)
    b = torch.zeros(batch, 1, device=device, requires_grad=True)
    with torch.no_grad():
        W.normal_(0, 0.02)
    opt = torch.optim.Adam([W, b], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def run():
        for _ in range(n_epochs):
            logits = torch.bmm(Xb, W).squeeze(-1) + b  # (B, N)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    run()  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / batch  # ms per probe-equivalent


def main() -> None:
    p = argparse.ArgumentParser(description="Probe-training device benchmark.")
    p.add_argument("--hidden", type=int, default=4096, help="hidden dim (7B=4096, 1B=2048)")
    p.add_argument("--n-samples", type=int, default=64, help="train rows (32 pairs x 2)")
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n-probes", type=int, default=60, help="sequential timing iterations")
    p.add_argument("--batch", type=int, default=200, help="vectorized batch (= n_bootstrap)")
    p.add_argument("--threads", default="1,2,4,8,16,32", help="CPU thread counts to sweep")
    p.add_argument("--full-probes", type=int, default=200 * 32 * 6,
                   help="probe count to extrapolate to (n_bootstrap*layers*foundations)")
    p.add_argument("--output",
                   default="papers/3_moral_geometry/outputs/benchmarks/probe_device_bench.json")
    args = p.parse_args()

    results = {
        "workload": {"hidden": args.hidden, "n_samples": args.n_samples,
                     "n_epochs": args.n_epochs, "lr": args.lr},
        "platform": {"machine": platform.machine(), "processor": platform.processor(),
                     "cpu_count": os.cpu_count(), "torch": torch.__version__,
                     "cuda": torch.cuda.is_available(),
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "full_probes": args.full_probes,
        "cpu_threads": {}, "gpu": {},
    }

    def extrap(ms):
        return round(ms * args.full_probes / 1000.0 / 60.0, 1)  # minutes

    print(f"Workload: Linear({args.hidden},1), N={args.n_samples}, "
          f"{args.n_epochs} epochs; extrapolating to {args.full_probes} probes")
    print(f"Platform: {results['platform']['gpu'] or 'CPU-only'}, "
          f"{os.cpu_count()} cores, torch {torch.__version__}\n")

    def seq(dev):
        return time_sequential(dev, args.n_samples, args.hidden, args.n_epochs, args.lr,
                               args.n_probes)

    def vec(dev):
        return time_vectorized(dev, args.n_samples, args.hidden, args.n_epochs, args.lr, args.batch)

    cpu = torch.device("cpu")
    orig_threads = torch.get_num_threads()
    print(f"{'config':28s} {'ms/probe':>10s} {'full (min)':>12s}")
    for nt in [int(x) for x in args.threads.split(",")]:
        if nt > (os.cpu_count() or 1):
            continue
        torch.set_num_threads(nt)
        ms = seq(cpu)
        results["cpu_threads"][str(nt)] = round(ms, 2)
        print(f"{'CPU seq, ' + str(nt) + ' threads':28s} {ms:>10.2f} {extrap(ms):>12.1f}")
    torch.set_num_threads(orig_threads)

    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        ms_seq = seq(gpu)
        results["gpu"]["sequential"] = round(ms_seq, 2)
        print(f"{'GPU seq (per-probe)':28s} {ms_seq:>10.2f} {extrap(ms_seq):>12.1f}")
        ms_vec = vec(gpu)
        results["gpu"]["vectorized"] = round(ms_vec, 4)
        label = f"GPU vec (B={args.batch})"
        print(f"{label:28s} {ms_vec:>10.3f} {extrap(ms_vec):>12.1f}")
    else:
        # Local (MPS) reference so the script is exercisable off-pod.
        if torch.backends.mps.is_available():
            mps = torch.device("mps")
            ms_seq = seq(mps)
            results["gpu"]["sequential_mps"] = round(ms_seq, 2)
            print(f"{'MPS seq (per-probe)':28s} {ms_seq:>10.2f} {extrap(ms_seq):>12.1f}")

    out = args.output
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
