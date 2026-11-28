"""Evaluate the saved mocap model on all 297 future frames with 50 draws."""

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
from scipy.stats import t as student_t
import torch

from train_mocap_latent_sde import MocapSDE, baseline_mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5600)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--observation-dt", type=float, default=0.1)
    parser.add_argument("--drift", choices=["posterior", "prior"], default="posterior",
                        help="Use the context-conditioned drift or the learned generative prior drift")
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("At least two Monte Carlo samples are needed for an interval")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mat = loadmat(Path(__file__).parent / "data" / "mocap35.mat")
    test = mat["Xtest"].astype(np.float32)
    xs = torch.tensor(test).permute(1, 0, 2).contiguous()
    ts = torch.arange(xs.size(0), dtype=xs.dtype) * args.observation_dt
    model = MocapSDE()
    ckpt = args.output_dir / "mocap_best_checkpoint.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        predictions = model.predict(xs, ts, samples=args.samples, dt=args.dt,
                                    drift=args.drift)
        # Each draw is averaged across all four test sequences, 297 forecast
        # frames, and 50 coordinates. This isolates Monte Carlo variability.
        squared = (predictions[3:] - xs[3:, None]).square()
        draw_mse = squared.mean(dim=(0, 2, 3)).cpu().numpy()
    mean = float(draw_mse.mean())
    sem = float(draw_mse.std(ddof=1) / np.sqrt(args.samples))
    half_width = float(student_t.ppf(0.975, args.samples - 1) * sem)
    result = {
        "test_mse": mean,
        "mc_95_percent_t_interval": [mean - half_width, mean + half_width],
        "mc_interval_half_width": half_width,
        "samples": args.samples,
        "drift_used_for_forecast": args.drift,
        "test_sequences": int(test.shape[0]),
        "forecast_frames": int(test.shape[1] - 3),
        "coordinates": int(test.shape[2]),
        "constant_last_observation_mse": baseline_mse(test, test.shape[1]),
        "zero_prediction_mse": float(np.mean(test[:, 3:] ** 2)),
        "paper_latent_sde_mse": 4.03,
        "paper_reported_95_percent_half_width": 0.20,
        "interval_note": "This t interval quantifies only Monte Carlo prediction-draw uncertainty at a fixed checkpoint and test set. The paper does not fully specify its interval unit; do not claim identical confidence-interval construction.",
        "implementation_note": "Approximate architecture (11845 vs paper 11605 parameters), zero-drift initialization, chosen KL weight 0.1 and posterior-validation checkpoint selection. The paper does not release evaluation code clarifying drift choice; this is not an exact numerical replication.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"test_ci_{args.drift}.json").write_text(json.dumps(result, indent=2) + "\n")
    means = predictions[:, :, 0].mean(dim=1).cpu().numpy()
    times = np.arange(test.shape[1])
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
    # Show coordinates with the largest test-set variation so the plot makes
    # the model's inability to track walking dynamics visible.
    display_dims = np.argsort(test[:, 3:].var(axis=(0, 1)))[-3:][::-1]
    for j, ax in zip(display_dims, axes):
        ax.plot(times, test[0, :, j], label="test sequence", lw=1.5)
        ax.plot(times, means[:, j], label="50-sample mean prediction", lw=1.2)
        ax.axvline(2, color="gray", ls=":")
        ax.set_ylabel(f"coordinate {j + 1}")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"CMU subject 35, {args.drift} drift: MSE {mean:.3f} ± {half_width:.3f} (MC 95% t)")
    fig.savefig(args.output_dir / f"mocap_prediction_{args.drift}.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
