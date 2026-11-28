"""Controlled mocap ablations addressing Appendix 9.15 implementation ambiguities.

Each run uses the same data split and first-three-frame forecast protocol.
This diagnostic script deliberately exposes changes relative to the first
implementation; its results must not be relabelled as the paper's result.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.distributions import Normal
import torchsde

from train_mocap_latent_sde import MocapSDE


class DiagnosticMocapSDE(MocapSDE):
    def __init__(self, q_logvar: bool, observation_logvar: bool,
                 fixed_observation_std: float | None, time_input: bool,
                 max_observation_std: float | None = None,
                 likelihood_time_average: bool = False):
        super().__init__()
        self.q_logvar = q_logvar
        self.observation_logvar = observation_logvar
        self.fixed_observation_std = fixed_observation_std
        self.max_observation_std = max_observation_std
        self.likelihood_time_average = likelihood_time_average
        self.time_input = time_input
        if not time_input:
            self.prior_net[0] = nn.Linear(6, 30)
            self.posterior_net[0] = nn.Linear(9, 30)
            for net in self.g_nets:
                net[0] = nn.Linear(1, 30)

    def encode(self, xs: torch.Tensor):
        raw = self.encoder(xs[:3].permute(1, 0, 2).reshape(xs.size(1), 150))
        mean, variance_raw, ctx = raw[:, :6], raw[:, 6:12], raw[:, 12:15]
        self._ctx = ctx
        if self.q_logvar:
            logstd = 0.5 * variance_raw.clamp(-10.0, 6.0)
        else:
            logstd = variance_raw.clamp(-5.0, 3.0)
        return mean, logstd, ctx

    def f(self, t: torch.Tensor, y: torch.Tensor):
        parts = (y, self._ctx) if not self.time_input else (y, self.t_feature(t, y), self._ctx)
        return self.posterior_net(torch.cat(parts, dim=1))

    def h(self, t: torch.Tensor, y: torch.Tensor):
        parts = y if not self.time_input else torch.cat((y, self.t_feature(t, y)), dim=1)
        return self.prior_net(parts)

    def g(self, t: torch.Tensor, y: torch.Tensor):
        if self.time_input:
            tcol = self.t_feature(t, y)
            return torch.cat([net(torch.cat((yi, tcol), dim=1))
                              for net, yi in zip(self.g_nets, y.split(1, dim=1))], dim=1)
        return torch.cat([net(yi) for net, yi in zip(self.g_nets, y.split(1, dim=1))], dim=1)

    def decode(self, zs: torch.Tensor):
        raw = self.decoder(zs)
        mean = raw[..., :50]
        if self.fixed_observation_std is not None:
            std = torch.ones_like(mean) * self.fixed_observation_std
        elif self.observation_logvar:
            std = (0.5 * raw[..., 50:].clamp(-8.0, 6.0)).exp()
        else:
            std = raw[..., 50:].clamp(-4.0, 3.0).exp()
        if self.max_observation_std is not None:
            std = std.clamp(max=self.max_observation_std)
        return mean, std

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
        if self.likelihood_time_average:
            loglike = loglike / xs.size(0)
        qz0 = Normal(qmean, qlogstd.exp())
        pz0 = Normal(self.pz0_mean, self.pz0_logstd.exp())
        kl0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean()
        klpath = path_ratio.sum(dim=0).mean()
        return -loglike + kl_weight * (kl0 + klpath)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--q-logvar", action="store_true")
    parser.add_argument("--observation-logvar", action="store_true")
    parser.add_argument("--fixed-observation-std", type=float)
    parser.add_argument("--max-observation-std", type=float,
                        help="Cap learned observation std to diagnose variance inflation")
    parser.add_argument("--variance-warmup-iterations", type=int, default=0,
                        help="Temporarily use unit observation std, then train learned log-variance")
    parser.add_argument("--no-time-input", action="store_true")
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--likelihood-time-average", action="store_true",
                        help="Diagnostic alternative: divide summed log likelihood by frames")
    parser.add_argument("--anneal-iterations", type=int, default=200)
    parser.add_argument("--curriculum-start-frames", type=int)
    parser.add_argument("--curriculum-full-at", type=int, default=200)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--observation-dt", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--standardize", action="store_true",
                        help="Fit on per-coordinate train-set z scores; report MSE in original units")
    parser.add_argument("--seed", type=int, default=5600)
    parser.add_argument("--validation-samples", type=int, default=3)
    parser.add_argument("--evaluation-samples", type=int, default=20)
    parser.add_argument("--adjoint", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "mocap_diagnostics")
    args = parser.parse_args()
    if not 4 <= args.frames <= 300:
        raise ValueError("frames must be in [4, 300]")
    if args.fixed_observation_std is not None and args.fixed_observation_std <= 0:
        raise ValueError("fixed observation standard deviation must be positive")
    if args.max_observation_std is not None and args.max_observation_std <= 0:
        raise ValueError("maximum observation standard deviation must be positive")
    if args.variance_warmup_iterations:
        if (not args.observation_logvar or args.fixed_observation_std is not None
                or args.variance_warmup_iterations >= args.iterations):
            raise ValueError("variance warmup requires learned observation log-variance and ends before training")
    if args.curriculum_start_frames is not None and not 4 <= args.curriculum_start_frames <= args.frames:
        raise ValueError("curriculum start frames must be in [4, frames]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = args.output_dir / args.name
    output.mkdir(parents=True, exist_ok=True)
    mat = loadmat(Path(__file__).parent / "data" / "mocap35.mat")
    arrays_raw = {key: mat[key][:, :args.frames].astype(np.float32)
                  for key in ("Xtr", "Xval", "Xtest")}
    if args.standardize:
        center = arrays_raw["Xtr"].mean(axis=(0, 1), keepdims=True)
        scale = arrays_raw["Xtr"].std(axis=(0, 1), keepdims=True).clip(0.1)
        arrays = {key: (value - center) / scale for key, value in arrays_raw.items()}
        np.savez(output / "normalization.npz", center=center, scale=scale)
    else:
        center = np.zeros((1, 1, 50), dtype=np.float32)
        scale = np.ones((1, 1, 50), dtype=np.float32)
        arrays = arrays_raw
    data = {key: torch.tensor(value).permute(1, 0, 2).contiguous()
            for key, value in arrays.items()}
    ts = torch.arange(args.frames, dtype=torch.float32) * args.observation_dt
    initial_std = 1.0 if args.variance_warmup_iterations else args.fixed_observation_std
    model = DiagnosticMocapSDE(args.q_logvar, args.observation_logvar,
                               initial_std, not args.no_time_input,
                               args.max_observation_std,
                               args.likelihood_time_average)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    best_validation = float("inf")
    best_step = None
    history = []
    checkpoint = output / "mocap_diagnostic_checkpoint.pt"
    for step in range(1, args.iterations + 1):
        if args.variance_warmup_iterations and step == args.variance_warmup_iterations + 1:
            model.fixed_observation_std = None
        if args.curriculum_start_frames is None:
            length = args.frames
        else:
            fraction = min(1.0, step / args.curriculum_full_at)
            length = round(args.curriculum_start_frames +
                           (args.frames - args.curriculum_start_frames) * fraction)
        weight = args.kl_weight * (min(1.0, step / args.anneal_iterations)
                                   if args.anneal_iterations > 0 else 1.0)
        model.train()
        optimizer.zero_grad()
        loss = model.loss(data["Xtr"][:length], ts[:length], weight, args.adjoint, args.dt)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)) \
            if args.clip_norm > 0 else float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 10 == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                predicted = model.predict(data["Xval"], ts,
                                          args.validation_samples, args.dt)
                val_scale = torch.tensor(scale.reshape(1, 1, 1, 50))
                val = float(((predicted[3:] - data["Xval"][3:, None]) *
                             val_scale).square().mean())
            entry = {"step": step, "train_frames": length, "loss": float(loss.detach()),
                     "gradient_norm_before_clip": gradient_norm, "validation_mse": val}
            history.append(entry)
            print(entry, flush=True)
            if val < best_validation and step > args.variance_warmup_iterations:
                best_validation, best_step = val, step
                torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    random.seed(args.seed + 1000)
    np.random.seed(args.seed + 1000)
    torch.manual_seed(args.seed + 1000)
    with torch.no_grad():
        metrics = {}
        for key in ("Xtr", "Xval", "Xtest"):
            preds = model.predict(data[key], ts, args.evaluation_samples, args.dt)
            raw = torch.tensor(arrays_raw[key]).permute(1, 0, 2)
            center_t = torch.tensor(center.reshape(1, 1, 1, 50))
            scale_t = torch.tensor(scale.reshape(1, 1, 1, 50))
            preds_raw = preds * scale_t + center_t
            metrics[key] = {
                "sample_mse": float((preds_raw[3:] - raw[3:, None]).square().mean()),
                "mean_prediction_mse": float((preds_raw[3:].mean(dim=1) - raw[3:]).square().mean()),
                "zero_baseline_mse": float(raw[3:].square().mean()),
            }
    result = {"settings": vars(args) | {"output_dir": str(output)},
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "best_step": best_step, "best_validation_mse": best_validation,
              "metrics": metrics, "history": history}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
