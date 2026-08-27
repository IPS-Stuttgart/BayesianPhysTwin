"""Render the sealed opened-data state-restart result without refitting anything."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesian_phystwin_experiments.deform_state_restart import file_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--noise-run", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    barrier = json.loads((args.run / "prediction_barrier.json").read_text())
    if file_digest(args.run / "predictions.npz") != barrier["predictions_sha256"]:
        raise ValueError("sealed predictions changed")
    result = json.loads((args.run / "result.json").read_text())["summaries"]
    noise = json.loads((args.noise_run / "result.json").read_text())["conditions"]
    with np.load(args.archive, allow_pickle=False) as source:
        keep = source["names"] != "103.pkl"
        truth = source["targets"][keep, 50:170][:, :, (3, 5, 7, 9)].astype(float)
    with np.load(args.run / "predictions.npz", allow_pickle=False) as source:
        predictions = {}
        for key in (
            "incumbent",
            "readout_sparse_pose",
            "incumbent_propagated_pose_velocity",
            "incumbent_propagated_pose_velocity_quarter",
        ):
            predictions[key] = source[key][keep][:, :, (3, 5, 7, 9)].astype(float)
    styles = (
        ("incumbent", "Unchanged incumbent", "#333333", "-"),
        ("readout_sparse_pose", "Readout correction", "#b4453f", "--"),
        (
            "incumbent_propagated_pose_velocity",
            "Propagated state, gain 1",
            "#1870b8",
            "-",
        ),
        (
            "incumbent_propagated_pose_velocity_quarter",
            "Propagated state, gain 0.25",
            "#258064",
            "-",
        ),
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    bins = np.array_split(np.arange(120), 6)
    times = [(indices.mean() + 1) * 0.01 for indices in bins]
    for key, label, color, style in styles:
        squared = np.sum((predictions[key] - truth) ** 2, axis=-1)
        values = [
            1000 * np.sqrt(squared[:, indices].mean(axis=(1, 2))).mean()
            for indices in bins
        ]
        axes[0].plot(
            times,
            values,
            marker="o",
            color=color,
            linestyle=style,
            label=label,
            linewidth=2,
        )
    axes[0].set(
        xlabel="Seconds after the prefix update",
        ylabel="Hidden-point RMSE (mm)",
        title="Most of the gain is early",
    )
    axes[0].grid(axis="y", alpha=0.2)
    baseline = result["incumbent"]["point_rmse_mm"]
    for index, (native_key, noise_key, label, color) in enumerate(
        (
            (
                "incumbent_propagated_pose_velocity",
                "propagated_pose_velocity_full",
                "State, gain 1",
                "#1870b8",
            ),
            (
                "incumbent_propagated_pose_velocity_quarter",
                "propagated_pose_velocity_quarter",
                "State, gain 0.25",
                "#258064",
            ),
            ("readout_sparse_pose", "readout_sparse_pose", "Readout", "#b4453f"),
        )
    ):
        rows = [
            result[native_key],
            noise["independent_1mm"]["summaries"][noise_key],
            noise["independent_1mm_shared_5mm"]["summaries"][noise_key],
        ]
        delta = np.array([row["point_rmse_mm"] - baseline for row in rows])
        intervals = np.array([row["point_rmse_mm_delta_ci95"] for row in rows])
        axes[1].bar(
            np.arange(3) + (index - 1) * 0.24,
            delta,
            0.22,
            color=color,
            label=label,
            yerr=np.stack((delta - intervals[:, 0], intervals[:, 1] - delta)),
            capsize=3,
            error_kw={"linewidth": 1},
        )
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set(
        xticks=np.arange(3),
        xticklabels=[
            "Native\nannotations",
            "1 mm\nnoise",
            "1 mm noise +\n5 mm shared bias",
        ],
        ylabel="RMSE change versus incumbent (mm)",
        title="Simulated measurement robustness",
    )
    axes[1].grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="outside lower center", ncol=2)
    fig.suptitle(
        "DEFORM: eight sparse prefix observations, disjoint hidden identities\n"
        "13 already-open trajectories of one object; exploratory, not official SOTA",
        fontsize=12,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "state-restart-results.png", dpi=170)
    fig.savefig(
        args.output / "state-restart-results.pdf", metadata={"CreationDate": None}
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
