"""Motion-capture latent SDE architecture from Appendix 9.15 / Figure 10.

This is a reproducible implementation attempt. The paper does not give every
training detail (e.g. exact observation likelihood parameterization or its
chosen KL coefficient), so this script reports its settings and must not be
presented as an exact reproduction of the paper's 4.03 test MSE.
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
from scipy.io import loadmat
import torch
from torch import nn
from torch.distributions import Normal
import torchsde


class MocapSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self):
        super().__init__()
        # 3 observed frames x 50 coordinates -> z0 mean/std (6 each), context (3).
        self.encoder = nn.Sequential(nn.Linear(150, 30), nn.Softplus(),
                                     nn.Linear(30, 30), nn.Softplus(), nn.Linear(30, 15))
        # Figure 10: 6 -> 30 -> 30 -> 100 -> observation mean/std (50 each).
        self.decoder = nn.Sequential(nn.Linear(6, 30), nn.Softplus(),
                                     nn.Linear(30, 30), nn.Softplus(), nn.Linear(30, 100))
        self.prior_net = nn.Sequential(nn.Linear(7, 30), nn.Softplus(), nn.Linear(30, 6))
        self.posterior_net = nn.Sequential(nn.Linear(10, 30), nn.Softplus(), nn.Linear(30, 6))
        self.g_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(2, 30), nn.Softplus(), nn.Linear(30, 1), nn.Sigmoid())
            for _ in range(6)
        ])
        self.pz0_mean = nn.Parameter(torch.zeros(1, 6))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, 6))
        self._ctx = None
        # A stable initial SDE has zero prior/posterior drift. Without this,
        # random drift mismatch over 300 frames can make the path KL explode.
        # This initialization is an engineering choice, not specified by Li et al.
        nn.init.zeros_(self.prior_net[-1].weight)
        nn.init.zeros_(self.prior_net[-1].bias)
        nn.init.zeros_(self.posterior_net[-1].weight)
        nn.init.zeros_(self.posterior_net[-1].bias)

    @staticmethod
    def t_feature(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ones(y.size(0), 1, device=y.device, dtype=y.dtype) * t

    def encode(self, xs: torch.Tensor):
        raw = self.encoder(xs[:3].permute(1, 0, 2).reshape(xs.size(1), 150))
        mean, logstd, ctx = raw[:, :6], raw[:, 6:12], raw[:, 12:15]
        self._ctx = ctx
        return mean, logstd.clamp(-5.0, 3.0), ctx

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.posterior_net(torch.cat((y, self.t_feature(t, y), self._ctx), dim=1))

    def h(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.prior_net(torch.cat((y, self.t_feature(t, y)), dim=1))

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tcol = self.t_feature(t, y)
        return torch.cat([net(torch.cat((yi, tcol), dim=1))
                          for net, yi in zip(self.g_nets, y.split(1, dim=1))], dim=1)

    def decode(self, zs: torch.Tensor):
        raw = self.decoder(zs)
        return raw[..., :50], raw[..., 50:].clamp(-4.0, 3.0).exp()

    def loss(self, xs: torch.Tensor, ts: torch.Tensor, kl_weight: float,
             adjoint: bool, dt: float):
        qmean, qlogstd, ctx = self.encode(xs)
        z0 = qmean + qlogstd.exp() * torch.randn_like(qmean)
        kwargs = {"dt": dt, "method": "euler", "logqp": True}
        if adjoint:
            kwargs["adjoint_params"] = (ctx,) + tuple(self.posterior_net.parameters()) + \
                tuple(self.prior_net.parameters()) + tuple(self.g_nets.parameters())
            zs, path_ratio = torchsde.sdeint_adjoint(self, z0, ts, **kwargs)
        else:
            zs, path_ratio = torchsde.sdeint(self, z0, ts, **kwargs)
        obs_mean, obs_std = self.decode(zs)
        loglike = Normal(obs_mean, obs_std).log_prob(xs).sum(dim=(0, 2)).mean()
        qz0 = Normal(qmean, qlogstd.exp())
        pz0 = Normal(self.pz0_mean, self.pz0_logstd.exp())
        kl0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean()
        klpath = path_ratio.sum(dim=0).mean()
        return -loglike + kl_weight * (kl0 + klpath)

    @torch.no_grad()
    def predict(self, xs: torch.Tensor, ts: torch.Tensor, samples: int, dt: float,
                drift: str = "posterior"):
        if drift not in {"posterior", "prior"}:
            raise ValueError("drift must be posterior or prior")
        qmean, qlogstd, ctx = self.encode(xs)
        batch = xs.size(1)
        z0 = qmean.repeat(samples, 1) + qlogstd.exp().repeat(samples, 1) * \
            torch.randn(samples * batch, 6, device=xs.device)
        self._ctx = ctx.repeat(samples, 1)
        names = {"drift": "h"} if drift == "prior" else None
        zs = torchsde.sdeint(self, z0, ts, dt=dt, method="euler", names=names)
        mean, _ = self.decode(zs)
        return mean.reshape(len(ts), samples, batch, 50)


def baseline_mse(xs: np.ndarray, frames: int) -> float:
    pred = np.broadcast_to(xs[:, 2:3], xs[:, 3:frames].shape)
    return float(np.mean((pred - xs[:, 3:frames]) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--observation-dt", type=float, default=0.1,
                    help="ODE2VAE mocap preprocessing uses 0.1 time units per frame")
    ap.add_argument("--dt", type=float, default=0.02,
                    help="SDE solver step, one fifth of the observation interval")
    ap.add_argument("--kl-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=5600)
    ap.add_argument("--adjoint", action="store_true")
    ap.add_argument("--eval-samples", type=int, default=50)
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = ap.parse_args()
    if not 4 <= args.frames <= 300:
        raise ValueError("frames must be between 4 and 300")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = loadmat(Path(__file__).parent / "data" / "mocap35.mat")
    arrays = {k: data[k][:, :args.frames].astype(np.float32) for k in ("Xtr", "Xval", "Xtest")}
    tensors = {k: torch.tensor(v, device=device).permute(1, 0, 2).contiguous()
               for k, v in arrays.items()}
    ts = torch.arange(args.frames, dtype=torch.float32, device=device) * args.observation_dt
    model = MocapSDE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    history = []
    best_val = float("inf")
    ckpt = args.output_dir / "mocap_best_checkpoint.pt"
    for step in range(1, args.iterations + 1):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(tensors["Xtr"], ts,
                          kl_weight=args.kl_weight * min(1.0, step / 200),
                          adjoint=args.adjoint, dt=args.dt)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 10 == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                pred = model.predict(tensors["Xval"], ts, samples=3, dt=args.dt)
                val = float((pred[3:] - tensors["Xval"][3:, None]).square().mean())
            entry = {"step": step, "train_elbo_loss": float(loss.detach()), "validation_mse": val}
            history.append(entry)
            print(entry, flush=True)
            if val < best_val:
                best_val = val
                torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        predicted = model.predict(tensors["Xtest"], ts, samples=args.eval_samples, dt=args.dt)
        mse = float((predicted[3:] - tensors["Xtest"][3:, None]).square().mean())
    report = {
        "settings": vars(args) | {"output_dir": str(args.output_dir), "device": device},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_validation_mse": best_val,
        "test_mse": mse,
        "constant_last_observation_test_mse": baseline_mse(arrays["Xtest"], args.frames),
        "paper_full_300_frame_mse": 4.03,
        "matches_paper_evaluation_length_and_samples": args.frames == 300 and args.eval_samples == 50,
        "strict_replication_of_paper_implementation": False,
        "known_differences": [
            "This model has 11845 parameters versus 11605 reported in Appendix 9.15.",
            "The exact observation variance parameterization and selected KL coefficient were not published.",
            "Zero drift-layer initialization is an added stability choice.",
        ],
        "history": history,
    }
    (args.output_dir / "mocap_report.json").write_text(json.dumps(report, indent=2) + "\n")
    times = np.arange(args.frames)
    mean = predicted[:, :, 0].mean(dim=1).cpu().numpy()
    truth = arrays["Xtest"][0]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
    for j, ax in enumerate(axes):
        ax.plot(times, truth[:, j], label="true", lw=1.5)
        ax.plot(times, mean[:, j], label="predicted mean", lw=1.2)
        ax.axvline(2, color="gray", ls=":")
        ax.set_ylabel(f"dimension {j+1}")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"CMU mocap pilot: {args.frames} frames, test MSE {mse:.3f}")
    fig.savefig(args.output_dir / "mocap_prediction.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
