"""TorchSDE stochastic-adjoint comparison for Team 7's GBM project.

This is self-contained: it does not import or write the team's existing files.
Run ``python3 adjoint_compare.py`` after installing the requirements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torchsde


X0 = 1.0
MU = 0.18
SIGMA = 0.35
TARGET = 1.30
T = 1.0
SEED = 5600
DTYPE = torch.float64


class GBM(torch.nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(MU, dtype=DTYPE, device=device))
        self.sigma = torch.nn.Parameter(torch.tensor(SIGMA, dtype=DTYPE, device=device))

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mu * y

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.sigma * y


def loss_from_terminal(terminal: torch.Tensor) -> torch.Tensor:
    return ((terminal - TARGET) ** 2).mean()


def solve_and_grad(
    model: GBM,
    y0: torch.Tensor,
    ts: torch.Tensor,
    bm: torchsde.BrownianInterval,
    n_steps: int,
    adjoint: bool,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    solve = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
    kwargs = {"adjoint_method": "euler"} if adjoint else {}
    terminal = solve(
        model, y0, ts, bm=bm, method="euler", dt=T / n_steps, **kwargs
    )[-1]
    loss = loss_from_terminal(terminal)
    dmu, dsigma = torch.autograd.grad(loss, (model.mu, model.sigma))
    return terminal.detach(), loss.item(), {"mu": dmu.item(), "sigma": dsigma.item()}


@torch.no_grad()
def euler_reference(
    bm: torchsde.BrownianInterval, n_steps: int, n_paths: int, device: torch.device
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Independent pathwise derivative of the *discrete Euler* GBM update."""
    dt = T / n_steps
    x = torch.full((n_paths, 1), X0, dtype=DTYPE, device=device)
    dmu = torch.zeros_like(x)
    dsigma = torch.zeros_like(x)
    for k in range(n_steps):
        dw = bm(k * dt, (k + 1) * dt)
        factor = 1.0 + MU * dt + SIGMA * dw
        dmu, dsigma = dmu * factor + x * dt, dsigma * factor + x * dw
        x = x * factor
    weight = 2.0 * (x - TARGET)
    return x, loss_from_terminal(x).item(), {
        "mu": (weight * dmu).mean().item(),
        "sigma": (weight * dsigma).mean().item(),
    }


@torch.no_grad()
def exact_reference(bm: torchsde.BrownianInterval) -> tuple[float, dict[str, float]]:
    """Pathwise exact GBM solution for the same Brownian endpoints."""
    wt = bm(0.0, T)
    xt = X0 * torch.exp((MU - 0.5 * SIGMA**2) * T + SIGMA * wt)
    weight = 2.0 * (xt - TARGET) * xt
    return loss_from_terminal(xt).item(), {
        "mu": (weight * T).mean().item(),
        "sigma": (weight * (wt - SIGMA * T)).mean().item(),
    }


@torch.no_grad()
def finite_difference_reference(
    model: GBM, y0: torch.Tensor, ts: torch.Tensor,
    bm: torchsde.BrownianInterval, n_steps: int, epsilon: float = 1e-5,
) -> dict[str, float]:
    """Central differences of TorchSDE's forward Euler solver at fixed noise."""
    result = {}
    for name in ("mu", "sigma"):
        parameter = getattr(model, name)
        original = parameter.detach().clone()
        try:
            parameter.copy_(original + epsilon)
            upper = loss_from_terminal(
                torchsde.sdeint(model, y0, ts, bm=bm, method="euler", dt=T / n_steps)[-1]
            ).item()
            parameter.copy_(original - epsilon)
            lower = loss_from_terminal(
                torchsde.sdeint(model, y0, ts, bm=bm, method="euler", dt=T / n_steps)[-1]
            ).item()
        finally:
            parameter.copy_(original)
        result[name] = (upper - lower) / (2.0 * epsilon)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--paths", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.paths < 1 or any(n < 1 or n & (n - 1) for n in args.steps):
        parser.error("--paths must be positive; --steps must be positive powers of two")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    device = torch.device(args.device)
    model = GBM(device)
    y0 = torch.full((args.paths, 1), X0, dtype=DTYPE, device=device)
    ts = torch.tensor([0.0, T], dtype=DTYPE, device=device)
    # A single Brownian path object is reused by every method and step size.
    # halfway_tree makes its values independent of the order of queries.
    bm = torchsde.BrownianInterval(
        t0=0.0, t1=T, size=(args.paths, 1), dtype=DTYPE,
        device=device, entropy=SEED, tol=1e-9, halfway_tree=True,
    )
    exact_loss, exact_grad = exact_reference(bm)
    rows = []
    for n_steps in args.steps:
        bp_x, bp_loss, bp_grad = solve_and_grad(model, y0, ts, bm, n_steps, False)
        adj_x, adj_loss, adj_grad = solve_and_grad(model, y0, ts, bm, n_steps, True)
        ref_x, ref_loss, ref_grad = euler_reference(bm, n_steps, args.paths, device)
        forward_gap = max(
            (bp_x - adj_x).abs().max().item(),
            (bp_x - ref_x).abs().max().item(),
        )
        if forward_gap > 1e-10 or abs(bp_loss - ref_loss) > 1e-10:
            raise AssertionError(f"Forward Euler paths disagree at {n_steps} steps")
        if abs(bp_loss - adj_loss) > 1e-10:
            raise AssertionError(f"Forward losses disagree at {n_steps} steps")
        if max(abs(bp_grad[p] - ref_grad[p]) for p in ("mu", "sigma")) > 1e-9:
            raise AssertionError(f"Autograd disagrees with discrete Euler derivative at {n_steps} steps")
        for method, loss, grad in (
            ("sdeint_backprop", bp_loss, bp_grad),
            ("sdeint_adjoint", adj_loss, adj_grad),
            ("discrete_euler_reference", ref_loss, ref_grad),
            ("exact_gbm_reference", exact_loss, exact_grad),
        ):
            rows.append({
                "steps": n_steps,
                "dt": T / n_steps,
                "method": method,
                "loss": loss,
                "grad_mu": grad["mu"],
                "grad_sigma": grad["sigma"],
                "abs_error_mu_vs_discrete": abs(grad["mu"] - ref_grad["mu"]),
                "abs_error_sigma_vs_discrete": abs(grad["sigma"] - ref_grad["sigma"]),
                "abs_error_mu_vs_exact": abs(grad["mu"] - exact_grad["mu"]),
                "abs_error_sigma_vs_exact": abs(grad["sigma"] - exact_grad["sigma"]),
                "max_forward_path_gap": forward_gap,
            })
        print(
            f"steps={n_steps:>3}: backprop=({bp_grad['mu']:+.8f}, {bp_grad['sigma']:+.8f}), "
            f"adjoint=({adj_grad['mu']:+.8f}, {adj_grad['sigma']:+.8f}), "
            f"max |adjoint-backprop|={max(abs(adj_grad[p] - bp_grad[p]) for p in ('mu', 'sigma')):.3e}"
        )

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "gradient_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finest = max(args.steps)
    fd_grad = finite_difference_reference(model, y0, ts, bm, finest)
    finest_bp = next(r for r in rows if r["steps"] == finest and r["method"] == "sdeint_backprop")
    if max(abs(fd_grad[p] - finest_bp[f"grad_{p}"]) for p in ("mu", "sigma")) > 1e-7:
        raise AssertionError("Finite differences disagree with standard backpropagation")
    summary = {
        "description": "TorchSDE Ito GBM; fixed Brownian path; Euler forward and Euler adjoint",
        "settings": {
            "seed": SEED, "x0": X0, "mu": MU, "sigma": SIGMA,
            "target": TARGET, "T": T, "paths": args.paths,
            "steps": args.steps, "device": str(device), "dtype": str(DTYPE),
            "torch_version": torch.__version__, "torchsde_version": torchsde.__version__,
        },
        "exact_loss": exact_loss,
        "exact_gradient": exact_grad,
        "finite_difference_gradient_at_finest_step": fd_grad,
        "rows": rows,
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"Saved results in {out_dir}")


if __name__ == "__main__":
    main()
