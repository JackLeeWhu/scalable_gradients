"""Appendix 9.13-style latent SDE training on generated GBM or Lorenz data.

Architecture follows Li et al. (2020), Appendix 9.13 and the authors'
Apache-2.0 TorchSDE Lorenz example (google-research/torchsde). A minibatch
setting is exposed because the paper does not state its training batch size.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torchsde


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_dim: int, latent_dim: int = 4, hidden: int = 100,
                 context_dim: int = 1):
        super().__init__()
        self.data_dim, self.latent_dim = data_dim, latent_dim
        # Appendix 9.13 states a 1-dimensional context for GBM and the same
        # architecture for Lorenz except for the diffusion networks.
        self.encoder = nn.GRU(data_dim, hidden)
        self.context_net = nn.Linear(hidden, context_dim)
        self.qz0_net = nn.Linear(context_dim, 2 * latent_dim)
        self.f_net = nn.Sequential(nn.Linear(latent_dim + context_dim, hidden), nn.Softplus(),
                                   nn.Linear(hidden, latent_dim))
        self.h_net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.Softplus(),
                                   nn.Linear(hidden, latent_dim))
        self.g_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.Softplus(), nn.Linear(hidden, 1), nn.Sigmoid())
            for _ in range(latent_dim)
        ])
        self.projector = nn.Linear(latent_dim, data_dim)
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_dim))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_dim))
        self._ts = None
        self._ctx = None

    def encode(self, xs: torch.Tensor, ts: torch.Tensor):
        reverse, _ = self.encoder(xs.flip(0))
        ctx = self.context_net(reverse.flip(0))
        qmean, qlogstd = self.qz0_net(ctx[0]).chunk(2, dim=-1)
        self._ts, self._ctx = ts, ctx
        return ctx, qmean, qlogstd.clamp(-5.0, 3.0)

    def f(self, t: torch.Tensor, y: torch.Tensor):
        index = min(int(torch.searchsorted(self._ts, t, right=True).item()), len(self._ts) - 1)
        return self.f_net(torch.cat([y, self._ctx[index]], dim=-1))

    def h(self, t: torch.Tensor, y: torch.Tensor):
        return self.h_net(y)

    def g(self, t: torch.Tensor, y: torch.Tensor):
        parts = y.split(1, dim=-1)
        return torch.cat([network(part) for network, part in zip(self.g_nets, parts)], dim=-1)

    def loss(self, xs: torch.Tensor, ts: torch.Tensor, kl_weight: float,
             obs_noise: float = 0.01, adjoint: bool = True, dt: float = 0.01):
        ctx, qmean, qlogstd = self.encode(xs, ts)
        z0 = qmean + qlogstd.exp() * torch.randn_like(qmean)
        kwargs = {"method": "euler", "dt": dt, "logqp": True}
        if adjoint:
            kwargs["adjoint_params"] = (ctx,) + tuple(self.f_net.parameters()) + \
                tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            zs, log_ratio = torchsde.sdeint_adjoint(self, z0, ts, **kwargs)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, **kwargs)
        decoded = self.projector(zs)
        log_like = Normal(decoded, obs_noise).log_prob(xs).sum(dim=(0, 2)).mean()
        qz0 = Normal(qmean, qlogstd.exp())
        pz0 = Normal(self.pz0_mean, self.pz0_logstd.exp())
        kl0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=-1).mean()
        klpath = log_ratio.sum(dim=0).mean()
        loss = -log_like + kl_weight * (kl0 + klpath)
        return loss, {"negative_log_likelihood": float(-log_like.detach()),
                      "initial_kl": float(kl0.detach()), "path_kl": float(klpath.detach())}

    @torch.no_grad()
    def posterior_samples(self, xs: torch.Tensor, ts: torch.Tensor, repeats: int = 32):
        inp = xs.expand(-1, repeats, -1).contiguous()
        _, qmean, qlogstd = self.encode(inp, ts)
        z0 = qmean + qlogstd.exp() * torch.randn_like(qmean)
        zs = torchsde.sdeint(self, z0, ts, dt=0.01, method="euler")
        return self.projector(zs)

    @torch.no_grad()
    def prior_samples(self, ts: torch.Tensor, samples: int = 8, same_initial: bool = False):
        z0 = self.pz0_mean + self.pz0_logstd.exp() * torch.randn(
            samples, self.latent_dim, device=ts.device)
        if same_initial:
            z0 = z0[0:1].expand_as(z0).contiguous()
        zs = torchsde.sdeint(self, z0, ts, names={"drift": "h"}, dt=0.01, method="euler")
        return self.projector(zs), zs


def plot_gbm(model: LatentSDE, data: torch.Tensor, ts: torch.Tensor, output: Path):
    times = ts.cpu().numpy()
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), layout="constrained")
    for i in range(8):
        axes[0, 0].plot(times, data[i, :, 0].cpu(), lw=1)
    axes[0, 0].set(title="Training paths (8 of 1024)", xlabel="time")
    one = data[0:1].permute(1, 0, 2).to(ts.device)
    # Appendix Figure 9 uses 512 posterior draws for its 95% contour.
    recon = model.posterior_samples(one, ts, repeats=512)[:, :, 0].cpu().numpy()
    axes[0, 1].plot(times, one[:, 0, 0].cpu(), "k.", ms=3, label="observed")
    axes[0, 1].plot(times, recon.mean(axis=1), color="tab:orange", label="posterior mean")
    axes[0, 1].fill_between(times, *np.quantile(recon, [0.025, 0.975], axis=1),
                            color="tab:orange", alpha=0.25, label="95% interval")
    axes[0, 1].set(title="Posterior reconstruction", xlabel="time")
    axes[0, 1].legend(fontsize=8)
    for row, same in ((1, False), (2, True)):
        obs, latent = model.prior_samples(ts, samples=8, same_initial=same)
        for i in range(8):
            axes[row, 0].plot(times, obs[:, i, 0].cpu(), lw=1)
            axes[row, 1].plot(times, latent[:, i, 0].cpu(), lw=1)
        label = "fixed z0" if same else "independent z0"
        axes[row, 0].set(title=f"Prior samples ({label})", xlabel="time")
        axes[row, 1].set(title=f"Latent coordinate 1 ({label})", xlabel="time")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_lorenz(model: LatentSDE, data: torch.Tensor, ts: torch.Tensor, output: Path):
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4)
    for i in range(8):
        x = data[i].cpu().numpy()
        ax1.plot(*x.T, lw=1)
    ax1.set_title("Training data")
    one = data[0:1].permute(1, 0, 2).to(ts.device)
    recon = model.posterior_samples(one, ts, repeats=32).mean(dim=1).cpu().numpy()
    ax2.plot(*recon.T, label="reconstruction")
    ax2.plot(*one[:, 0].cpu().numpy().T, "k--", alpha=0.6, label="data")
    ax2.set_title("Posterior reconstruction")
    prior, latent = model.prior_samples(ts, samples=8)
    for i in range(8):
        ax3.plot(*prior[:, i].cpu().numpy().T, lw=1)
        ax4.plot(ts.cpu(), latent[:, i, 0].cpu(), lw=1)
    ax3.set_title("Learned prior samples")
    ax4.set(title="Prior latent coordinate 1", xlabel="time")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["gbm", "lorenz"])
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--context-dim", type=int, default=1)
    ap.add_argument("--seed", type=int, default=5600)
    ap.add_argument("--adjoint", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.output_dir / f"{args.dataset}_1024.npz")
    data = torch.tensor(raw["x"], device=device)
    ts = torch.tensor(raw["ts"], dtype=torch.float32, device=device)
    model = LatentSDE(data_dim=data.shape[-1], context_dim=args.context_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    history = []
    for step in range(1, args.iterations + 1):
        indices = torch.randperm(data.shape[0], device=device)[:args.batch_size]
        xs = data[indices].permute(1, 0, 2).contiguous()
        optimizer.zero_grad()
        loss, metrics = model.loss(xs, ts, kl_weight=min(1.0, step / 50.0),
                                   adjoint=args.adjoint)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at iteration {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        entry = {"step": step, "loss": float(loss.detach()), **metrics}
        history.append(entry)
        if step == 1 or step % 20 == 0 or step == args.iterations:
            print(entry, flush=True)
    prefix = args.output_dir / f"latent_{args.dataset}"
    torch.save({"state_dict": model.state_dict(), "args": vars(args) | {"output_dir": str(args.output_dir)}},
               str(prefix) + "_checkpoint.pt")
    (Path(str(prefix) + "_history.json")).write_text(json.dumps(history, indent=2) + "\n")
    model.eval()
    with torch.no_grad():
        if args.dataset == "gbm":
            plot_gbm(model, data, ts, Path(str(prefix) + "_figure.png"))
        else:
            plot_lorenz(model, data, ts, Path(str(prefix) + "_figure.png"))


if __name__ == "__main__":
    main()
