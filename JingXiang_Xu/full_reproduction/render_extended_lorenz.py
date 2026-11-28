"""Render a Figure 8-style extended diagnostic from a trained Lorenz checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_synthetic_latent_sde import LatentSDE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = ap.parse_args()
    out = args.output_dir
    raw = np.load(out / "lorenz_1024.npz")
    times = raw["ts"]
    data = torch.tensor(raw["x"])
    ts = torch.tensor(times, dtype=torch.float32)
    checkpoint = torch.load(out / "latent_lorenz_checkpoint.pt", map_location="cpu", weights_only=False)
    context_dim = checkpoint["state_dict"]["context_net.weight"].shape[0]
    model = LatentSDE(data_dim=3, context_dim=context_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.manual_seed(5600)
    fig, axes = plt.subplots(3, 5, figsize=(15, 8.3), layout="constrained")

    for d in range(3):
        axes[0, 0].plot(times, data[0, :, d], lw=1.3, label=f"coord {d+1}")
    axes[0, 0].set_title("Training data")
    axes[0, 0].legend(fontsize=7)
    with torch.no_grad():
        for col in range(1, 5):
            one = data[col - 1:col].permute(1, 0, 2)
            sample = model.posterior_samples(one, ts, repeats=1)[:, 0]
            for d in range(3):
                axes[0, col].plot(times, one[:, 0, d], "k:", alpha=0.45, lw=0.8)
                axes[0, col].plot(times, sample[:, d], lw=1)
            axes[0, col].set_title(f"Posterior sample {col}")

        for row, same in ((1, False), (2, True)):
            obs, latent = model.prior_samples(ts, samples=8, same_initial=same)
            for i in range(8):
                for d in range(3):
                    axes[row, 0].plot(times, obs[:, i, d], lw=0.7, alpha=0.75)
                for d in range(4):
                    axes[row, d+1].plot(times, latent[:, i, d], lw=0.8, alpha=0.85)
            mode = "same z0" if same else "independent z0"
            axes[row, 0].set_title(f"Prior data ({mode})")
            for d in range(4):
                axes[row, d+1].set_title(f"Latent dimension {d+1}")

    for ax in axes.flat:
        ax.set_xlabel("time", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Lorenz latent SDE: Figure 8-style diagnostics (independent run)")
    fig.savefig(out / "latent_lorenz_extended_figure.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
