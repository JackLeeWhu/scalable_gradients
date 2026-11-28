"""Independent Appendix Figure 7 study for analytic SDE Examples 1 and 3.

Appendix Example 1 swaps alpha and beta in its printed closed form. We use the
printed SDE and its mathematically consistent GBM solution. All random seeds,
loss choices, tolerances and hardware timing here are new, not author data.
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


class AnalyticSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, example: int, alpha: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        self.example = example
        self.alpha = nn.Parameter(alpha.clone())
        self.beta = nn.Parameter(beta.clone())
        self.f_calls = 0
        self.g_calls = 0

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.f_calls += 1
        if self.example == 1:
            return self.alpha * x
        return self.beta / (1 + t).sqrt() - x / (2 * (1 + t))

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.g_calls += 1
        if self.example == 1:
            return self.beta * x
        return (self.alpha * self.beta / (1 + t).sqrt()).expand_as(x)


def exact_gradient(example: int, x0: torch.Tensor, alpha: torch.Tensor,
                   beta: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    if example == 1:
        final = x0 * torch.exp(alpha - 0.5 * beta.square() + beta * w1)
        d_alpha = final.mean(dim=0)
        d_beta = (final * (w1 - beta)).mean(dim=0)
    else:
        d_alpha = (beta * w1 / np.sqrt(2.0)).mean(dim=0)
        d_beta = ((1.0 + alpha * w1) / np.sqrt(2.0)).mean(dim=0)
    return torch.cat((d_alpha, d_beta))


def setup(example: int, seed: int, batch: int, dim: int, adaptive: bool = False):
    torch.manual_seed(seed)
    dtype = torch.float64
    x0 = torch.randn(batch, dim, dtype=dtype)
    alpha = torch.randn(dim, dtype=dtype).sigmoid()
    beta = torch.randn(dim, dtype=dtype).sigmoid()
    bm = torchsde.BrownianInterval(
        t0=0.0, t1=1.0, size=(batch, dim), dtype=dtype,
        entropy=seed + example * 10000,
        levy_area_approximation="space-time" if adaptive else "none",
    )
    target = exact_gradient(example, x0, alpha, beta, bm(0.0, 1.0))
    ts = torch.tensor([0.0, 1.0], dtype=dtype)
    return x0, alpha, beta, bm, target, ts


def solve(example: int, seed: int, batch: int, dim: int, method: str,
          adjoint: bool, steps: int | None = None, atol: float | None = None) -> dict:
    x0, alpha, beta, bm, target, ts = setup(example, seed, batch, dim, atol is not None)
    model = AnalyticSDE(example, alpha, beta)
    solver = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
    kwargs = {"bm": bm, "method": method, "dt": 1.0 / steps if steps else 0.05}
    if adjoint:
        kwargs["adjoint_method"] = "milstein" if method == "srk" else method
    if atol is not None:
        kwargs.update(adaptive=True, adjoint_adaptive=True,
                      atol=atol, adjoint_atol=atol,
                      rtol=0.0, adjoint_rtol=0.0)
    start = time.process_time()
    ys = solver(model, x0, ts, **kwargs)
    forward_nfe = model.f_calls + model.g_calls
    ys[-1].sum(dim=1).mean().backward()
    elapsed = time.process_time() - start
    gradient = torch.cat((model.alpha.grad, model.beta.grad))
    rmse = float((gradient - target).square().mean().sqrt())
    return {"example": example, "seed": seed, "method": method, "adjoint": adjoint,
            "steps": steps, "atol": atol, "forward_nfe": forward_nfe,
            "total_nfe": model.f_calls + model.g_calls,
            "cpu_seconds": elapsed, "gradient_rmse": rmse}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-seeds", type=int, default=8)
    parser.add_argument("--adaptive-seeds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    steps_list = (16, 32, 64, 128)
    configs = (("Euler BP", "euler", False, 256),
               ("Milstein BP", "milstein", False, 64),
               ("Milstein adjoint", "milstein", True, 64))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout="constrained")
    summary = {}
    for row_idx, example in enumerate((1, 3)):
        atols = (1e-2, 1e-3, 1e-4) if example == 1 else (1e-4, 1e-5, 1e-6)
        for steps in steps_list:
            for idx in range(args.fixed_seeds):
                for adjoint in (False, True):
                    rows.append(solve(example, 5600 + idx, args.batch, args.dim,
                                      "milstein", adjoint, steps=steps) | {"study": "fixed"})
        fixed = [r for r in rows if r["example"] == example and r["study"] == "fixed"]
        for adjoint, label in ((False, "backprop"), (True, "adjoint")):
            medians = [float(np.median([r["gradient_rmse"] ** 2 for r in fixed
                                        if r["steps"] == s and r["adjoint"] == adjoint]))
                       for s in steps_list]
            axes[row_idx, 0].loglog(1 / np.array(steps_list), medians, "o-", label=label)
            summary[f"example{example}_{label}_median_mse"] = medians
        axes[row_idx, 0].set(xlabel="solver step", ylabel="median gradient MSE",
                             title=f"Example {example}: fixed step")
        axes[row_idx, 0].legend(fontsize=8)
        for atol in atols:
            for idx in range(args.adaptive_seeds):
                rows.append(solve(example, 6600 + idx, args.batch, args.dim,
                                  "srk", True, atol=atol) | {"study": "adaptive"})
        adaptive = [r for r in rows if r["example"] == example and r["study"] == "adaptive"]
        for atol in atols:
            subset = [r for r in adaptive if r["atol"] == atol]
            axes[row_idx, 1].scatter([r["total_nfe"] for r in subset],
                                      [r["gradient_rmse"] for r in subset], label=f"atol={atol:g}")
        axes[row_idx, 1].set(xlabel="total forward + backward NFE", ylabel="gradient RMSE",
                             title=f"Example {example}: adaptive SRK")
        axes[row_idx, 1].set_yscale("log")
        axes[row_idx, 1].legend(fontsize=8)
        # Warm up each solver before collecting process-time measurements.
        for _, method, adjoint, steps in configs:
            solve(example, 7599, args.batch, args.dim, method, adjoint, steps=steps)
        for label, method, adjoint, steps in configs:
            measured = [solve(example, 7600 + idx, args.batch, args.dim,
                              method, adjoint, steps=steps) | {"study": "efficiency", "strategy": label}
                        for idx in range(args.fixed_seeds)]
            rows.extend(measured)
            axes[row_idx, 2].scatter([r["cpu_seconds"] for r in measured],
                                      [r["gradient_rmse"] for r in measured], label=label)
            summary[f"example{example}_{label}_median_cpu_seconds"] = float(
                np.median([r["cpu_seconds"] for r in measured]))
        axes[row_idx, 2].set(xlabel="single-process CPU time (s)", ylabel="gradient RMSE",
                             title=f"Example {example}: efficiency")
        axes[row_idx, 2].set_yscale("log")
        axes[row_idx, 2].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.25, which="both")
    fig.suptitle("Appendix Figure 7 style: independent analytic-SDE study")
    fig.savefig(args.output_dir / "supplement_figure7_style.png", dpi=180)
    plt.close(fig)
    fields = ["study", "example", "seed", "method", "adjoint", "steps", "atol",
              "forward_nfe", "total_nfe", "cpu_seconds", "gradient_rmse", "strategy"]
    with (args.output_dir / "supplement_examples_1_3.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "supplement_examples_1_3_summary.json").write_text(
        json.dumps({"settings": vars(args) | {"output_dir": str(args.output_dir)},
                    "summary": summary,
                    "note": "Independent loss, seeds and solver settings; Example 1 uses the corrected exact solution for the printed SDE."},
                   indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
