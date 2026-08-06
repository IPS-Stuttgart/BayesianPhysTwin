#!/usr/bin/env python3
"""Build the PokeFlex same-object calibration and result figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_same_object_reporting import (  # noqa: E402
    build_candidate_diagnostics,
    load_json_object,
    sha256_file,
    write_json,
)


def _as_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("diagnostic rows are missing")
    return [row for row in rows if isinstance(row, dict)]


def render_figure(diagnostic: dict[str, Any], png_path: Path, pdf_path: Path) -> None:
    """Render one paper-ready calibration/result figure."""

    import matplotlib.pyplot as plt

    rows = [row for row in _as_rows(diagnostic) if row["candidate_supported"]]
    if not rows:
        raise ValueError("no candidate-supported rows are available")
    result = diagnostic["bounded_result"]
    calibration = diagnostic["candidate_diagnostic"]

    figure = plt.figure(figsize=(12.0, 5.35), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.16, 0.84))
    axis_cal = figure.add_subplot(grid[0, 0])
    axis_take = figure.add_subplot(grid[0, 1])

    categories = (
        (
            "Accepted, improved",
            lambda row: row["accepted"] and row["candidate_regret_mm"] < 0.0,
            "o",
            "#087e8b",
        ),
        (
            "Accepted, harmful",
            lambda row: row["accepted"] and row["candidate_regret_mm"] > 0.0,
            "X",
            "#c0362c",
        ),
        (
            "Fallback, candidate improved",
            lambda row: not row["accepted"] and row["candidate_regret_mm"] < 0.0,
            "^",
            "#4466aa",
        ),
        (
            "Fallback, candidate harmful",
            lambda row: not row["accepted"] and row["candidate_regret_mm"] > 0.0,
            "v",
            "#999999",
        ),
    )
    all_x = np.asarray(
        [row["selector_adjusted_upper_regret_mm"] for row in rows],
        dtype=np.float64,
    )
    all_y = np.asarray([row["candidate_regret_mm"] for row in rows], dtype=np.float64)
    for label, selector, marker, color in categories:
        current = [row for row in rows if selector(row)]
        if not current:
            continue
        x = [row["selector_adjusted_upper_regret_mm"] for row in current]
        y = [row["candidate_regret_mm"] for row in current]
        axis_cal.scatter(
            x,
            y,
            s=31,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.86,
            label=f"{label} (n={len(current)})",
            zorder=3,
        )

    lower = float(min(np.min(all_x), np.min(all_y), -0.1))
    upper = float(max(np.max(all_x), np.max(all_y), 0.1))
    pad = 0.07 * max(upper - lower, 1.0)
    lower -= pad
    upper += pad
    axis_cal.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.0,
        color="#555555",
        label="realized regret = upper bound",
        zorder=1,
    )
    axis_cal.axvline(0.0, linewidth=1.15, color="#222222", zorder=2)
    axis_cal.axhline(0.0, linewidth=1.15, color="#222222", zorder=2)
    axis_cal.set_xlim(lower, upper)
    axis_cal.set_ylim(lower, upper)
    axis_cal.set_xlabel("Frozen selector-adjusted regret upper bound [mm]")
    axis_cal.set_ylabel("Post-outcome selected-candidate regret [mm]")
    axis_cal.set_title("(a) Frozen guard calibration diagnostic")
    axis_cal.grid(alpha=0.18, linewidth=0.6)
    axis_cal.legend(loc="upper left", fontsize=7.8, frameon=True)
    coverage = calibration["adjusted_upper_bound_coverage"]
    false_safe = calibration["accepted_harmful_fraction"]
    axis_cal.text(
        0.985,
        0.02,
        (
            f"Upper-bound coverage: {100.0 * coverage:.1f}%\n"
            f"Harmful among accepted: {100.0 * false_safe:.1f}%\n"
            "Vertical line: frozen accept threshold"
        ),
        transform=axis_cal.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.1,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    takes = result["takes"]
    labels = [str(value["take_id"]).replace("_", "\n") for value in takes]
    baseline = np.asarray(
        [value["baseline_mean_CD_UL1_mm"] for value in takes],
        dtype=np.float64,
    )
    guarded = np.asarray(
        [value["selected_mean_CD_UL1_mm"] for value in takes],
        dtype=np.float64,
    )
    positions = np.arange(len(takes), dtype=np.float64)
    width = 0.36
    baseline_bars = axis_take.bar(
        positions - width / 2.0,
        baseline,
        width,
        color="#b8bdc7",
        label="Released checkpoint",
    )
    guarded_bars = axis_take.bar(
        positions + width / 2.0,
        guarded,
        width,
        color="#087e8b",
        label="Regret guarded",
    )
    for index, take in enumerate(takes):
        improvement = 100.0 * float(take["relative_improvement"])
        height = max(baseline[index], guarded[index])
        axis_take.text(
            positions[index],
            height + 0.16,
            f"−{improvement:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
        )
    axis_take.bar_label(baseline_bars, fmt="%.2f", padding=2, fontsize=7.6)
    axis_take.bar_label(guarded_bars, fmt="%.2f", padding=2, fontsize=7.6)
    axis_take.set_xticks(positions, labels)
    axis_take.set_ylabel("Mean CD$_{UL1}$ [mm]")
    axis_take.set_title("(b) Prospective new-take result")
    axis_take.grid(axis="y", alpha=0.18, linewidth=0.6)
    axis_take.legend(loc="upper left", fontsize=8.0)
    aggregate = 100.0 * result["object_balanced_relative_improvement"]
    axis_take.text(
        0.98,
        0.03,
        (
            f"Object-balanced: {result['baseline_object_mean_CD_UL1_mm']:.3f} → "
            f"{result['guarded_object_mean_CD_UL1_mm']:.3f} mm\n"
            f"Relative reduction: {aggregate:.2f}%\n"
            f"Exact fallback: {result['exact_fallback_frame_count']}/"
            f"{result['frame_count']} frames"
        ),
        transform=axis_take.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "white", "alpha": 0.92},
    )
    axis_take.set_ylim(0.0, float(max(baseline) * 1.26))

    figure.suptitle(
        "PokeFlex: source-calibrated guard transfers to new takes of seen objects",
        fontsize=13.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        (
            "Candidate outcomes in (a) are opened only for post-hoc visualization; "
            "the estimator, certificate, and zero threshold were frozen beforehand. "
            "This does not establish independent-object transfer."
        ),
        ha="center",
        va="top",
        fontsize=8.1,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-result", type=Path, required=True)
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        action="append",
        required=True,
        dest="candidate_artifacts",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    result_path = args.prospective_result.resolve()
    candidate_paths = [path.resolve() for path in args.candidate_artifacts]
    result = load_json_object(result_path)
    candidates = [load_json_object(path) for path in candidate_paths]
    diagnostic = build_candidate_diagnostics(candidates, result)

    output_root = args.output_root.resolve()
    diagnostic_path = output_root / "pokeflex_same_object_diagnostic.json"
    png_path = output_root / "pokeflex_same_object_calibration.png"
    pdf_path = output_root / "pokeflex_same_object_calibration.pdf"
    write_json(diagnostic_path, diagnostic)
    render_figure(diagnostic, png_path, pdf_path)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexSameObjectPaperFigureManifestV1",
        "prospective_result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "candidate_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in candidate_paths
        ],
        "outputs": {
            path.name: sha256_file(path)
            for path in (diagnostic_path, png_path, pdf_path)
        },
        "claim": diagnostic["bounded_result"]["claim"],
        "excluded_claims": diagnostic["bounded_result"]["excluded_claims"],
    }
    manifest_path = output_root / "pokeflex_same_object_figure_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
