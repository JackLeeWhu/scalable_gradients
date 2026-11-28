"""Inspect decoder uncertainty and reconstruction at a saved mocap checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch.distributions import Normal, kl_divergence
import torchsde

from diagnose_mocap import DiagnosticMocapSDE
from train_mocap_latent_sde import MocapSDE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--seed", type=int, default=25600)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    settings = json.loads((run / ("report.json" if args.diagnostic else
                                   "mocap_report.json")).read_text())["settings"]
    frames = settings["frames"]
    raw = loadmat(Path(__file__).parent / "data" / "mocap35.mat")["Xtr"]
    raw = raw[:, :frames].astype(np.float32)
    if settings.get("standardize", False):
        norm = np.load(run / "normalization.npz")
        raw = (raw - norm["center"]) / norm["scale"]
    xs = torch.tensor(raw).permute(1, 0, 2).contiguous()
    ts = torch.arange(frames, dtype=torch.float32) * settings.get("observation_dt", 0.1)
    if args.diagnostic:
        model = DiagnosticMocapSDE(
            settings["q_logvar"], settings["observation_logvar"],
            settings["fixed_observation_std"], not settings["no_time_input"],
            settings.get("max_observation_std"))
        checkpoint = run / "mocap_diagnostic_checkpoint.pt"
    else:
        model = MocapSDE()
        checkpoint = run / "mocap_best_checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with torch.no_grad():
        qmean, qlogstd, _ = model.encode(xs)
        z0 = qmean + qlogstd.exp() * torch.randn_like(qmean)
        zs, path_ratio = torchsde.sdeint(model, z0, ts, dt=settings["dt"],
                                         method="euler", logqp=True)
        obs_mean, obs_std = model.decode(zs)
        nll = -Normal(obs_mean, obs_std).log_prob(xs).sum(dim=(0, 2)).mean()
        kl0 = kl_divergence(Normal(qmean, qlogstd.exp()),
                            Normal(model.pz0_mean, model.pz0_logstd.exp())
                           ).sum(dim=1).mean()
        klpath = path_ratio.sum(dim=0).mean()
        result = {
            "run": str(run), "frames": frames, "seed": args.seed,
            "negative_log_likelihood": float(nll),
            "initial_state_kl": float(kl0), "path_kl": float(klpath),
            "train_reconstruction_mse": float(((obs_mean - xs) ** 2).mean()),
            "mean_observation_std": float(obs_std.mean()),
            "maximum_observation_std": float(obs_std.max()),
            "mean_initial_state_std": float(qlogstd.exp().mean()),
        }
    (run / "likelihood_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
