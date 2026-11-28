"""Figure 4(c)-style gradient error versus runtime comparison on Example 2."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde

from numerical_study import TestSDE, exact_grad


CONFIGS = (
    ("Euler backprop", "euler", False, 256),
    ("Milstein backprop", "milstein", False, 64),
    ("Milstein adjoint", "milstein", True, 64),
)


def one(seed: int, batch: int, dim: int):
    torch.manual_seed(seed)
    dtype = torch.float64
    x0 = torch.randn(batch, dim, dtype=dtype)
    p0 = torch.randn(dim, dtype=dtype).sigmoid()
    bm = torchsde.BrownianInterval(t0=0.0, t1=1.0, size=(batch, dim),
                                  dtype=dtype, entropy=seed + 10000)
    target = exact_grad(p0, x0, bm(0.0, 1.0))
    ts = torch.tensor([0.0, 1.0], dtype=dtype)
    rows = []
    for name, method, adjoint, n in CONFIGS:
        model = TestSDE(p0)
        solver = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        kwargs = {"method": method, "dt": 1.0 / n, "bm": bm}
        if adjoint:
            kwargs["adjoint_method"] = method
        # CPU process time is stable when other training jobs are scheduled.
        start = time.process_time()
        ys = solver(model, x0, ts, **kwargs)
        ys[-1].sum(dim=1).mean().backward()
        duration = time.process_time() - start
        rows.append({"seed": seed, "strategy": name, "steps": n, "seconds": duration,
                     "gradient_rmse": float((model.p.grad - target).square().mean().sqrt())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Warm up import, autograd and solver dispatch outside measured runs.
    one(5599, args.batch, args.dim)
    rows = [item for idx in range(args.seeds) for item in one(5600 + idx, args.batch, args.dim)]
    with (args.output_dir / "efficiency_study.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.5), layout="constrained")
    colors = ("tab:blue", "tab:orange", "tab:green")
    for (name, _, _, _), color in zip(CONFIGS, colors):
        group = [r for r in rows if r["strategy"] == name]
        ax.scatter([r["seconds"] for r in group], [r["gradient_rmse"] for r in group],
                   s=30, alpha=0.75, label=name, color=color)
    ax.set(xlabel="single-process CPU time, forward + backward (s)",
           ylabel="gradient RMSE versus analytic",
           title="Example 2 efficiency (independent run)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(args.output_dir / "efficiency_study.png", dpi=200)
    plt.close(fig)
    for name, _, _, _ in CONFIGS:
        group = [r for r in rows if r["strategy"] == name]
        print(name, "median seconds", np.median([r["seconds"] for r in group]),
              "median RMSE", np.median([r["gradient_rmse"] for r in group]))


if __name__ == "__main__":
    main()
