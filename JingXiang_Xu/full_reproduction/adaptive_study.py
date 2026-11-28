"""Figure 4(b)-style adaptive tolerance, NFE, and gradient-error study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde

from numerical_study import TestSDE, exact_grad


class CountingSDE(TestSDE):
    def __init__(self, p):
        super().__init__(p)
        self.f_calls = 0
        self.g_calls = 0

    def f(self, t, x):
        self.f_calls += 1
        return super().f(t, x)

    def g(self, t, x):
        self.g_calls += 1
        return super().g(t, x)


def run_one(seed: int, atol: float, batch: int, dim: int):
    torch.manual_seed(seed)
    dtype = torch.float64
    x0 = torch.randn(batch, dim, dtype=dtype)
    p0 = torch.randn(dim, dtype=dtype).sigmoid()
    bm = torchsde.BrownianInterval(
        t0=0.0, t1=1.0, size=(batch, dim), dtype=dtype,
        entropy=seed + 10000, levy_area_approximation="space-time",
    )
    model = CountingSDE(p0)
    ys = torchsde.sdeint_adjoint(
        model, x0, torch.tensor([0.0, 1.0], dtype=dtype),
        method="srk", adjoint_method="milstein", adaptive=True,
        adjoint_adaptive=True, atol=atol, adjoint_atol=atol,
        rtol=0.0, adjoint_rtol=0.0, dt=0.05, bm=bm,
    )
    forward_nfe = model.f_calls + model.g_calls
    ys[-1].sum(dim=1).mean().backward()
    target = exact_grad(p0, x0, bm(0.0, 1.0))
    return {"seed": seed, "atol": atol, "forward_nfe": forward_nfe,
            "total_nfe": model.f_calls + model.g_calls,
            "gradient_rmse": float((model.p.grad - target).square().mean().sqrt())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--atols", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for atol in args.atols:
        for idx in range(args.seeds):
            row = run_one(5600 + idx, atol, args.batch, args.dim)
            rows.append(row)
            print(row, flush=True)
    with (args.output_dir / "adaptive_study.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
    for atol, group in [(a, [r for r in rows if r["atol"] == a]) for a in args.atols]:
        ax.scatter([r["total_nfe"] for r in group], [r["gradient_rmse"] for r in group],
                   label=f"atol={atol:g}", s=28, alpha=0.7)
    ax.set(xlabel="total forward + backward drift/diffusion evaluations", ylabel="gradient RMSE versus analytic",
           title="Adaptive SRK forward + Milstein adjoint (independent run)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(args.output_dir / "adaptive_study.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
