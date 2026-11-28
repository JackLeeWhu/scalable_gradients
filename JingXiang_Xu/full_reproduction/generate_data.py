"""Recreate the Section 7.2 datasets and inspect the Section 7.3 dataset.

The paper does not publish its original random draws. These files use a fixed
new seed and match the distributions and train/validation/test shapes stated
in Li et al. (2020), Appendix 9.13 and Section 7.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def gbm(seed: int, n_paths: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, 1.0, 51, dtype=np.float64)
    dt = float(ts[1] - ts[0])
    x0 = 0.1 + rng.normal(0.0, 0.03, (n_paths, 1))
    dw = rng.normal(0.0, np.sqrt(dt), (n_paths, len(ts) - 1))
    w = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dw, axis=1)], axis=1)
    clean = x0 * np.exp((1.0 - 0.5 * 0.5**2) * ts[None, :] + 0.5 * w)
    noisy = clean + rng.normal(0.0, 0.01, clean.shape)
    return ts, noisy[..., None].astype(np.float32)


def lorenz(seed: int, n_paths: int = 1024, fine_dt: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Euler-Maruyama for the appendix's additive-noise stochastic Lorenz SDE."""
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, 1.0, 41, dtype=np.float64)
    ratio = (ts[1] - ts[0]) / fine_dt
    if abs(ratio - round(ratio)) > 1e-10:
        raise ValueError("fine_dt must divide the 0.025 observation interval")
    substeps = round(ratio)
    x = rng.normal(0.0, 1.0, (n_paths, 3))
    out = np.empty((n_paths, len(ts), 3), dtype=np.float64)
    out[:, 0] = x
    for j in range(1, len(ts)):
        for _ in range(substeps):
            drift = np.column_stack(
                (10.0 * (x[:, 1] - x[:, 0]),
                 x[:, 0] * (28.0 - x[:, 2]) - x[:, 1],
                 x[:, 0] * x[:, 1] - (8.0 / 3.0) * x[:, 2])
            )
            x = x + drift * fine_dt + 0.15 * rng.normal(0.0, np.sqrt(fine_dt), x.shape)
        out[:, j] = x
    mean = out.mean(axis=(0, 1), keepdims=True)
    std = out.std(axis=(0, 1), keepdims=True)
    noisy = (out - mean) / std + rng.normal(0.0, 0.01, out.shape)
    return ts, noisy.astype(np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5600)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    gbm_ts, gbm_x = gbm(args.seed)
    lorenz_ts, lorenz_x = lorenz(args.seed + 1)
    np.savez_compressed(out / "gbm_1024.npz", ts=gbm_ts, x=gbm_x)
    np.savez_compressed(out / "lorenz_1024.npz", ts=lorenz_ts, x=lorenz_x)

    mocap_path = Path(__file__).parent / "data" / "mocap35.mat"
    if not mocap_path.exists():
        raise FileNotFoundError(
            f"Download mocap35.mat from the ODE2VAE data link in README_CN.md "
            f"and place it at {mocap_path}"
        )
    mocap = loadmat(mocap_path)
    sizes = {k: list(mocap[k].shape) for k in ("Xtr", "Xval", "Xtest")}
    assert sizes == {"Xtr": [16, 300, 50], "Xval": [3, 300, 50], "Xtest": [4, 300, 50]}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    for idx in range(8):
        axes[0].plot(gbm_ts, gbm_x[idx, :, 0], alpha=0.8)
    axes[0].set(title="GBM: 8 of 1024 generated paths", xlabel="time", ylabel="observation")
    for idx in range(5):
        axes[1].plot(lorenz_ts, lorenz_x[idx, :, 0], alpha=0.8)
    axes[1].set(title="Lorenz: normalized x coordinate", xlabel="time")
    for idx in range(3):
        axes[2].plot(mocap["Xtr"][0, :, idx], label=f"dim {idx+1}", lw=1)
    axes[2].set(title="CMU mocap: one walking sequence", xlabel="frame")
    axes[2].legend(fontsize=8)
    fig.savefig(out / "dataset_preview.png", dpi=180)
    plt.close(fig)

    manifest = {
        "seed": args.seed,
        "gbm": {"shape": list(gbm_x.shape), "observations_dt": 0.02,
                "mu": 1.0, "sigma": 0.5, "x0_mean": 0.1, "x0_std": 0.03,
                "observation_noise_std": 0.01, "solver": "exact GBM"},
        "lorenz": {"shape": list(lorenz_x.shape), "observations_dt": 0.025,
                   "sigma_rho_beta": [10, 28, 8 / 3], "diffusion": [0.15] * 3,
                   "observation_noise_std": 0.01, "solver": "Euler-Maruyama",
                   "internal_dt": 0.005},
        "mocap": {"file": "data/mocap35.mat", "sha256": sha256(mocap_path),
                  "source": "ODE2VAE authors' public Google Drive folder",
                  "source_file_id": "1adqzHEXaGDVObSKbJgT5HBdoZkeMZ8S3",
                  "shapes": sizes},
    }
    (out / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
