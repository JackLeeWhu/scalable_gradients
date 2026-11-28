"""Quantify reconstruction and stochastic-prior diagnostics for a trained toy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_synthetic_latent_sde import LatentSDE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["gbm", "lorenz"])
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--posterior-samples", type=int, default=16)
    ap.add_argument("--prior-samples", type=int, default=256)
    args = ap.parse_args()
    raw = np.load(args.output_dir / f"{args.dataset}_1024.npz")
    data = torch.tensor(raw["x"])
    ts = torch.tensor(raw["ts"], dtype=torch.float32)
    checkpoint = torch.load(args.output_dir / f"latent_{args.dataset}_checkpoint.pt",
                            map_location="cpu", weights_only=False)
    context_dim = checkpoint["state_dict"]["context_net.weight"].shape[0]
    model = LatentSDE(data.shape[-1], context_dim=context_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.manual_seed(6600)
    errors = []
    with torch.no_grad():
        for i in range(min(args.paths, len(data))):
            xs = data[i:i+1].permute(1, 0, 2)
            samples = model.posterior_samples(xs, ts, repeats=args.posterior_samples)
            mean = samples.mean(dim=1)
            errors.append(float((mean - xs[:, 0]).square().mean()))
        independent, _ = model.prior_samples(ts, samples=args.prior_samples)
        fixed, _ = model.prior_samples(ts, samples=args.prior_samples, same_initial=True)
    report = {
        "dataset": args.dataset, "context_dim": context_dim,
        "training_paths_checked": len(errors),
        "posterior_mean_train_mse": float(np.mean(errors)),
        "empirical_data_terminal_variance": data[:, -1].var(dim=0).tolist(),
        "learned_prior_terminal_variance_independent_z0": independent[-1].var(dim=0).tolist(),
        "learned_prior_terminal_variance_fixed_z0": fixed[-1].var(dim=0).tolist(),
        "note": "These are descriptive diagnostics on training paths, not held-out generalization metrics.",
    }
    (args.output_dir / f"latent_{args.dataset}_diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
