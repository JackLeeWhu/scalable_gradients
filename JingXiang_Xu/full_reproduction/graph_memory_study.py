"""Measure autograd graph growth for direct and adjoint Milstein gradients.

This measures tensors retained by autograd, not peak process RAM. Brownian
storage, Python objects, and the returned trajectory are excluded from the
reported payload. Querying only t=0 and t=1 holds returned-state size fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde

from numerical_study import TestSDE, exact_grad


class _PackedTensor:
    def __init__(self, tensor: torch.Tensor, counter: "SavedTensorCounter"):
        self.tensor = tensor
        self.counter = counter
        self.nbytes = tensor.numel() * tensor.element_size()
        counter.live_bytes += self.nbytes
        counter.peak_bytes = max(counter.peak_bytes, counter.live_bytes)
        counter.live_slots += 1

    def __del__(self):
        self.counter.live_bytes -= self.nbytes
        self.counter.live_slots -= 1


class SavedTensorCounter:
    def __init__(self):
        self.live_bytes = 0
        self.peak_bytes = 0
        self.live_slots = 0

    def pack(self, tensor: torch.Tensor) -> _PackedTensor:
        return _PackedTensor(tensor, self)

    @staticmethod
    def unpack(packed: _PackedTensor) -> torch.Tensor:
        return packed.tensor


def graph_nodes(grad_fn) -> int:
    seen = set()
    todo = [grad_fn]
    while todo:
        fn = todo.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        todo.extend(next_fn for next_fn, _ in fn.next_functions)
    return len(seen)


def run_one(steps: int, seed: int, batch: int, method: str) -> dict:
    torch.manual_seed(seed)
    dtype = torch.float64
    dim = 10
    x0 = torch.randn(batch, dim, dtype=dtype)
    p0 = torch.randn(dim, dtype=dtype).sigmoid()
    ts = torch.tensor([0.0, 1.0], dtype=dtype)
    model = TestSDE(p0)
    bm = torchsde.BrownianInterval(
        t0=0.0, t1=1.0, size=(batch, dim), dtype=dtype,
        entropy=seed + 10000, levy_area_approximation="none",
    )
    target = exact_grad(p0, x0, bm(0.0, 1.0))
    solver = torchsde.sdeint_adjoint if method == "adjoint" else torchsde.sdeint
    kwargs = {"method": "milstein", "dt": 1.0 / steps, "bm": bm}
    if method == "adjoint":
        kwargs["adjoint_method"] = "milstein"
    counter = SavedTensorCounter()
    start = time.process_time()
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        ys = solver(model, x0, ts, **kwargs)
        retained = counter.live_bytes
        slots = counter.live_slots
        nodes = graph_nodes(ys.grad_fn)
        loss = ys[-1].sum(dim=1).mean()
        loss.backward()
    seconds = time.process_time() - start
    rmse = float((model.p.grad - target).square().mean().sqrt())
    return {"steps": steps, "seed": seed, "method": method, "batch": batch,
            "retained_forward_bytes": retained, "saved_slots": slots,
            "graph_nodes": nodes, "peak_hooked_bytes": counter.peak_bytes,
            "cpu_seconds": seconds, "gradient_rmse": rmse}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_one(8, 5599, args.batch, "backprop")
    run_one(8, 5599, args.batch, "adjoint")
    rows = []
    for steps in args.steps:
        for seed_index in range(args.seeds):
            for method in ("backprop", "adjoint"):
                row = run_one(steps, 5600 + seed_index, args.batch, method)
                rows.append(row)
                print(row, flush=True)
    with (args.output_dir / "graph_memory_study.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"settings": {"steps": args.steps, "seeds": args.seeds,
                             "batch": args.batch, "dim": 10, "dtype": "float64",
                             "solver": "Milstein", "output_times": [0.0, 1.0],
                             "torch": torch.__version__, "torchsde": torchsde.__version__,
                             "platform": platform.platform()},
               "metric_note": "Retained forward bytes sum logical sizes of tensors held for backward, "
                              "not peak RAM or Brownian cache size.", "results": {}}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for method, color in (("backprop", "tab:blue"), ("adjoint", "tab:orange")):
        group = [r for r in rows if r["method"] == method]
        med_bytes = [float(np.median([r["retained_forward_bytes"] for r in group
                                      if r["steps"] == n])) for n in args.steps]
        med_nodes = [float(np.median([r["graph_nodes"] for r in group
                                      if r["steps"] == n])) for n in args.steps]
        med_time = [float(np.median([r["cpu_seconds"] for r in group
                                     if r["steps"] == n])) for n in args.steps]
        summary["results"][method] = {"median_retained_forward_bytes": med_bytes,
                                      "median_graph_nodes": med_nodes,
                                      "median_cpu_seconds": med_time}
        axes[0].plot(args.steps, np.asarray(med_bytes) / (1024**2), "o-", color=color, label=method)
        axes[1].plot(args.steps, med_nodes, "o-", color=color, label=method)
    axes[0].set(xlabel="Milstein steps", ylabel="saved tensor payload after forward (MiB)",
                title="Autograd retention")
    axes[1].set(xlabel="Milstein steps", ylabel="autograd graph nodes after forward",
                title="Graph size")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.savefig(args.output_dir / "graph_memory_study.png", dpi=180)
    plt.close(fig)
    (args.output_dir / "graph_memory_study_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
