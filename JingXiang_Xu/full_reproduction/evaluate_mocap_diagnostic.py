"""Evaluate a saved diagnostic model on the paper's 297-frame, 50-draw protocol."""

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

from diagnose_mocap import DiagnosticMocapSDE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=15600)
    ap.add_argument("--drift", choices=["posterior", "prior"], default="posterior")
    args = ap.parse_args()
    run = args.run_dir.resolve()
    report = json.loads((run / "report.json").read_text())
    settings = report["settings"]
    frames = settings["frames"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mat = loadmat(Path(__file__).parent / "data" / "mocap35.mat")
    train = mat["Xtr"][:, :frames].astype(np.float32)
    test = mat["Xtest"][:, :frames].astype(np.float32)
    if settings.get("standardize", False):
        norm = np.load(run / "normalization.npz")
        center, scale = norm["center"], norm["scale"]
    else:
        center = np.zeros((1, 1, 50), dtype=np.float32)
        scale = np.ones((1, 1, 50), dtype=np.float32)
    inp = torch.tensor((test - center) / scale).permute(1, 0, 2).contiguous()
    times = torch.arange(frames, dtype=torch.float32) * settings.get("observation_dt", 0.1)
    model = DiagnosticMocapSDE(
        settings["q_logvar"], settings["observation_logvar"],
        settings["fixed_observation_std"], not settings["no_time_input"],
        settings.get("max_observation_std")
    )
    model.load_state_dict(torch.load(run / "mocap_diagnostic_checkpoint.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        pred_norm = model.predict(inp, times, args.samples, settings["dt"],
                                  drift=args.drift)
    pred = pred_norm.numpy() * scale.reshape(1, 1, 1, 50) + center.reshape(1, 1, 1, 50)
    truth = np.transpose(test, (1, 0, 2))
    squared = (pred[3:] - truth[3:, None]) ** 2
    mse_by_draw = squared.mean(axis=(0, 2, 3))
    mse_by_frame = squared.mean(axis=(1, 2, 3))
    blocks = {}
    for start, stop in ((3, 50), (50, 100), (100, 200), (200, 300)):
        stop = min(stop, frames)
        if start < stop:
            blocks[f"frames_{start}_to_{stop - 1}"] = float(
                mse_by_frame[start - 3:stop - 3].mean())
    mean = float(mse_by_draw.mean())
    halfwidth = float(student_t.ppf(.975, args.samples - 1) *
                      mse_by_draw.std(ddof=1) / np.sqrt(args.samples))
    result = {
        "frames": frames,
        "predicted_frames": frames - 3,
        "test_sequences": int(test.shape[0]),
        "coordinates": int(test.shape[2]),
        "samples": args.samples,
        "seed": args.seed,
        "drift": args.drift,
        "test_mse": mean,
        "mc_95_percent_t_halfwidth": halfwidth,
        "mean_prediction_mse": float(((pred[3:].mean(axis=1) - truth[3:]) ** 2).mean()),
        "zero_prediction_mse": float((truth[3:] ** 2).mean()),
        "per_sequence_mse": [float(x) for x in squared.mean(axis=(0, 1, 3))],
        "horizon_blocks_mse": blocks,
    }
    (run / f"test_ci_{args.drift}.json").write_text(json.dumps(result, indent=2) + "\n")
    chosen = np.argsort(train.std(axis=(0, 1)))[-3:][::-1]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for ax, coord in zip(axes, chosen):
        ax.plot(truth[:, 0, coord], color="#243746", lw=1.3, label="observed")
        ax.plot(pred[:, :, 0, coord].mean(axis=1), color="#dc6b30", lw=1.2,
                label="mean forecast")
        ax.axvline(2, color="#777777", ls=":", lw=1)
        ax.set_ylabel(f"coordinate {coord + 1}")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"CMU subject 35: {frames - 3} future frames, test MSE {mean:.3f}")
    fig.savefig(run / f"mocap_prediction_{args.drift}.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(np.arange(3, frames), mse_by_frame, lw=1.5, label="model")
    ax.plot(np.arange(3, frames), (truth[3:] ** 2).mean(axis=(1, 2)),
            lw=1, ls="--", label="zero forecast")
    ax.set(xlabel="future frame", ylabel="MSE (original units)",
           title=f"Prediction error by horizon: {args.drift} drift")
    ax.legend()
    fig.savefig(run / f"mocap_error_by_frame_{args.drift}.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
