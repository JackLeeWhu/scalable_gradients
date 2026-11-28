"""Check whether the learned Lorenz prior covers both terminal x modes."""

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

from train_synthetic_latent_sde import LatentSDE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "outputs" / "lorenz_context1")
    parser.add_argument("--prior-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=16561)
    args = parser.parse_args()
    raw = np.load(args.output_dir / "lorenz_1024.npz")
    data_final = raw["x"][:, -1, 0]
    ts = torch.tensor(raw["ts"], dtype=torch.float32)
    checkpoint = torch.load(args.output_dir / "latent_lorenz_checkpoint.pt",
                            map_location="cpu", weights_only=False)
    context_dim = checkpoint["state_dict"]["context_net.weight"].shape[0]
    model = LatentSDE(3, context_dim=context_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    with torch.no_grad():
        prior, _ = model.prior_samples(ts, samples=args.prior_samples)
    prior_final = prior[-1, :, 0].cpu().numpy()
    result = {
        "seed": args.seed,
        "prior_samples": args.prior_samples,
        "data_samples": len(data_final),
        "coordinate": "normalized Lorenz x at t=1",
        "data_negative_fraction": float(np.mean(data_final < 0)),
        "prior_negative_fraction": float(np.mean(prior_final < 0)),
        "data_terminal_quantiles": np.quantile(data_final, [.1, .25, .5, .75, .9]).tolist(),
        "prior_terminal_quantiles": np.quantile(prior_final, [.1, .25, .5, .75, .9]).tolist(),
        "note": "A fixed-checkpoint, finite-sample prior diagnostic; sign is a proxy for the two Lorenz lobes.",
    }
    (args.output_dir / "lorenz_mode_check.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7.1, 4.4), layout="constrained")
    bins = np.linspace(-2.5, 2.5, 51)
    ax.hist(data_final, bins=bins, density=True, alpha=.45,
            label=f"generated data (n={len(data_final)})")
    ax.hist(prior_final, bins=bins, density=True, alpha=.45,
            label=f"learned prior (n={len(prior_final)})")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set(xlabel="normalized Lorenz x at t=1", ylabel="density",
           title="Terminal-mode coverage of the learned Lorenz prior")
    ax.legend()
    fig.savefig(args.output_dir / "lorenz_mode_check.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
