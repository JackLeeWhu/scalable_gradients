"""Locate long-horizon decoder saturation in a saved mocap diagnostic run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
import torchsde

from diagnose_mocap import DiagnosticMocapSDE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--seed", type=int, default=15600)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    settings = json.loads((run / "report.json").read_text())["settings"]
    frames = settings["frames"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw = loadmat(Path(__file__).parent / "data" / "mocap35.mat")["Xtest"]
    raw = raw[:, :frames].astype(np.float32)
    if settings.get("standardize", False):
        norm = np.load(run / "normalization.npz")
        center, scale = norm["center"], norm["scale"]
    else:
        center = np.zeros((1, 1, 50), dtype=np.float32)
        scale = np.ones((1, 1, 50), dtype=np.float32)
    xs = torch.tensor((raw - center) / scale).permute(1, 0, 2).contiguous()
    ts = torch.arange(frames, dtype=torch.float32) * settings["observation_dt"]
    model = DiagnosticMocapSDE(
        settings["q_logvar"], settings["observation_logvar"],
        settings["fixed_observation_std"], not settings["no_time_input"],
        settings.get("max_observation_std"),
        settings.get("likelihood_time_average", False))
    model.load_state_dict(torch.load(run / "mocap_diagnostic_checkpoint.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        qmean, qlogstd, _ = model.encode(xs)
        z0 = qmean + qlogstd.exp() * torch.randn_like(qmean)
        zs = torchsde.sdeint(model, z0, ts, dt=settings["dt"], method="euler")
        mean, _ = model.decode(zs)
        rows = []
        for frame in (2, 49, 99, 149, 199, 249, 299):
            if frame >= frames:
                continue
            hidden1 = model.decoder[1](model.decoder[0](zs[frame]))
            hidden2 = model.decoder[3](model.decoder[2](hidden1))
            rows.append({
                "frame": frame,
                "latent_mean_absolute_value": float(zs[frame].abs().mean()),
                "decoder_second_hidden_near_zero_fraction":
                    float((hidden2 < 0.001).float().mean()),
                "decoder_output_mean_absolute_value": float(mean[frame].abs().mean()),
                "observation_mean_absolute_value": float(xs[frame].abs().mean()),
            })
        late = min(200, frames - 2)
        result = {
            "seed": args.seed,
            "frames": frames,
            "rows": rows,
            "latent_step_change_after_frame_200":
                float((zs[late + 1:] - zs[late:-1]).abs().mean()),
            "decoded_step_change_after_frame_200":
                float((mean[late + 1:] - mean[late:-1]).abs().mean()),
        }
    (run / "dynamics_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
