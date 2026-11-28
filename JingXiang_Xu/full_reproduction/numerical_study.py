"""Section 7.1 / Appendix 9.11 Example 2, with an explicit scalar loss.

The paper omits the exact loss, random draws, and plotting data for Figure 4.
This script therefore reproduces its test SDE and analytic-gradient comparison,
but labels the resulting figure as an independent study rather than Figure 4.
Appendix 9.11 prints -(p^2)^2 for the drift; Itô's lemma applied to its stated
closed-form solution gives -p^2 instead. We use the mathematically consistent
coefficient and document the discrepancy.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchsde


class TestSDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, p: torch.Tensor):
        super().__init__()
        self.p = nn.Parameter(p.clone())

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -self.p.square() * x.sin() * x.cos().pow(3)

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.p * x.cos().square()


def exact_grad(p: torch.Tensor, x0: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    u = p[None, :] * w1 + x0.tan()
    return (w1 / (1 + u.square())).mean(dim=0)


def run_one(n: int, seed: int, batch: int, dim: int, method: str) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    dtype = torch.float64
    x0 = torch.randn(batch, dim, dtype=dtype)
    p0 = torch.randn(dim, dtype=dtype).sigmoid()
    ts = torch.tensor([0.0, 1.0], dtype=dtype)
    bm = torchsde.BrownianInterval(
        t0=0.0, t1=1.0, size=(batch, dim), dtype=dtype, entropy=seed + 10000,
        levy_area_approximation="none",
    )
    target = exact_grad(p0, x0, bm(0.0, 1.0))
    errors = {}
    durations = {}
    for label, solver in (("backprop", torchsde.sdeint), ("adjoint", torchsde.sdeint_adjoint)):
        model = TestSDE(p0)
        kwargs = {"dt": 1.0 / n, "method": method, "bm": bm}
        if label == "adjoint":
            kwargs["adjoint_method"] = method
        t0 = time.perf_counter()
        ys = solver(model, x0, ts, **kwargs)
        ys[-1].sum(dim=1).mean().backward()
        durations[label] = time.perf_counter() - t0
        errors[label] = float((model.p.grad - target).square().mean().sqrt())
    return {
        "steps": n, "seed": seed, "batch": batch, "dim": dim, "method": method,
        "backprop_rmse": errors["backprop"], "adjoint_rmse": errors["adjoint"],
        "backprop_seconds": durations["backprop"], "adjoint_seconds": durations["adjoint"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--method", choices=["euler", "milstein"], default="milstein")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.steps:
        for s in range(args.seeds):
            row = run_one(n, 5600 + s, args.batch, args.dim, args.method)
            rows.append(row)
            print(row, flush=True)
    with (args.output_dir / "numerical_study.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    steps = np.asarray(args.steps)
    summaries = {}
    for label in ("backprop", "adjoint"):
        errors = np.asarray([[r[f"{label}_rmse"] for r in rows if r["steps"] == n] for n in steps])
        summaries[label] = {"median_rmse": np.median(errors, axis=1).tolist(),
                            "median_mse": np.median(errors**2, axis=1).tolist()}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), layout="constrained")
    adjoint_errors = [[r["adjoint_rmse"]**2 for r in rows if r["steps"] == n] for n in steps]
    axes[0].boxplot(adjoint_errors, tick_labels=[str(n) for n in steps], showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="number of solver steps", ylabel="gradient MSE versus analytic",
                title=f"Adjoint errors over {args.seeds} Brownian seeds")
    axes[0].grid(True, axis="y", which="both", alpha=0.3)
    ax = axes[1]
    for label, style in (("backprop", "o-"), ("adjoint", "s--")):
        ax.loglog(1.0 / steps, summaries[label]["median_mse"], style, label=label)
    ax.set(xlabel="solver step size", ylabel="median gradient MSE",
           title=f"Example 2 ({args.method}, independent draws)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(args.output_dir / "numerical_study.png", dpi=200)
    plt.close(fig)
    (args.output_dir / "numerical_study_summary.json").write_text(
        json.dumps({"settings": vars(args) | {"output_dir": str(args.output_dir)},
                    "summaries": summaries}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
