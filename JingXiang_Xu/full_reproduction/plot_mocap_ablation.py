"""Plot the controlled 47-future-frame mocap optimization diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent / "outputs" / "mocap_diagnostics"
RUNS = [
    ("logvar_both_50", "learned variance, raw data"),
    ("logvar_no_time_50", "learned variance, no time input"),
    ("learned_std_cap2_50", "learned variance, std ≤ 2"),
    ("fixedstd1_50", "fixed std = 1"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    rows = []
    for run_name, label in RUNS:
        report = json.loads((ROOT / run_name / "report.json").read_text())
        if report["settings"]["frames"] != 50:
            raise ValueError(f"Unexpected horizon in {run_name}")
        history = report["history"]
        ax.plot([h["step"] for h in history],
                [h["validation_mse"] for h in history],
                marker="o", ms=3, lw=1.6, label=label)
        for h in history:
            rows.append({"run": run_name, "step": h["step"],
                         "validation_mse": h["validation_mse"]})
    zero_mse = json.loads((ROOT / "fixedstd1_50" / "report.json").read_text())[
        "metrics"]["Xval"]["zero_baseline_mse"]
    ax.axhline(zero_mse, color="black", ls="--", lw=1, label="zero forecast")
    ax.set(xlabel="training iteration", ylabel="validation MSE (original units)",
           title="CMU mocap diagnostic: predict 47 future frames")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.savefig(ROOT / "mocap_ablation_50.png", dpi=180)
    plt.close(fig)
    with (ROOT / "mocap_ablation_50.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
