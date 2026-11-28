"""Render the Appendix Figure 9 style plot from a trained GBM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_synthetic_latent_sde import LatentSDE, plot_gbm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    raw = np.load(args.output_dir / "gbm_1024.npz")
    data = torch.tensor(raw["x"])
    ts = torch.tensor(raw["ts"], dtype=torch.float32)
    state = torch.load(args.output_dir / "latent_gbm_checkpoint.pt",
                       map_location="cpu", weights_only=False)["state_dict"]
    model = LatentSDE(data_dim=1, context_dim=state["context_net.weight"].shape[0])
    model.load_state_dict(state)
    model.eval()
    torch.manual_seed(6600)
    with torch.no_grad():
        plot_gbm(model, data, ts, args.output_dir / "latent_gbm_figure.png")


if __name__ == "__main__":
    main()
