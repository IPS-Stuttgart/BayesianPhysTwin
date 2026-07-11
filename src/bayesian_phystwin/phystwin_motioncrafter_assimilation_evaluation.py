"""Locked-cohort evaluation for MotionCrafter graph assimilation."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_motioncrafter_association import _load_pickle, _sha256


def _bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) < 1 or not np.all(np.isfinite(data)):
        raise ValueError("bootstrap values must be a finite nonempty vector")
    if samples < 100 or seed < 0:
        raise ValueError("bootstrap settings are invalid")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = np.mean(data[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_test(difference: np.ndarray) -> dict[str, float | int]:
    delta = np.asarray(difference, dtype=float)
    nonzero = delta[np.abs(delta) > 1e-12]
    wins = int(np.sum(nonzero < 0.0))
    losses = int(np.sum(nonzero > 0.0))
    count = wins + losses
    if count == 0:
        probability = 1.0
    else:
        tail = min(wins, losses)
        probability = min(
            1.0,
            2.0
            * sum(math.comb(count, index) for index in range(tail + 1))
            / (2.0**count),
        )
    return {
        "wins": wins,
        "losses": losses,
        "ties": int(len(delta) - count),
        "two_sided_p": float(probability),
    }


def _paired_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    baseline = np.asarray(reference, dtype=float)
    method = np.asarray(candidate, dtype=float)
    if baseline.shape != method.shape or baseline.ndim != 1:
        raise ValueError("paired metrics must have matching vector shapes")
    finite = np.isfinite(baseline) & np.isfinite(method)
    baseline = baseline[finite]
    method = method[finite]
    if len(baseline) == 0:
        raise ValueError("paired metrics contain no finite cases")
    difference = method - baseline
    low, high = _bootstrap_interval(
        difference,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    rng = np.random.default_rng(bootstrap_seed + 1)
    indices = rng.integers(0, len(baseline), size=(bootstrap_samples, len(baseline)))
    sampled_reference = np.mean(baseline[indices], axis=1)
    sampled_candidate = np.mean(method[indices], axis=1)
    relative = 100.0 * (sampled_candidate / sampled_reference - 1.0)
    relative_low, relative_high = np.quantile(relative, [0.025, 0.975])
    return {
        "case_count": int(len(baseline)),
        "reference_equal_case_mean_m": float(np.mean(baseline)),
        "candidate_equal_case_mean_m": float(np.mean(method)),
        "candidate_minus_reference_mean_m": float(np.mean(difference)),
        "candidate_minus_reference_median_m": float(np.median(difference)),
        "candidate_minus_reference_bootstrap_95_m": [low, high],
        "relative_change_percent": float(
            100.0 * (np.mean(method) / np.mean(baseline) - 1.0)
        ),
        "relative_change_bootstrap_95_percent": [
            float(relative_low),
            float(relative_high),
        ],
        "sign_test": _exact_sign_test(difference),
    }


def _training_track_fit(
    report: dict[str, Any],
    archive_path: str | Path,
) -> tuple[float, float]:
    data = _load_pickle(report["inputs"]["final_data"]["path"])
    baseline = np.asarray(
        _load_pickle(report["inputs"]["baseline"]["path"]), dtype=float
    )
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(
        data.get(
            "object_visibilities",
            np.ones(observed.shape[:2], dtype=bool),
        ),
        dtype=bool,
    )
    motion_valid = np.asarray(
        data.get(
            "object_motions_valid",
            np.ones(observed.shape[:2], dtype=bool),
        ),
        dtype=bool,
    )
    with np.load(archive_path) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        candidate = np.asarray(archive["position_flow_graph_positions"], dtype=float)[
            :, : observed.shape[1]
        ]
    training = frame_indices < int(report["train_end_frame"])
    target = observed[frame_indices]
    reference = baseline[frame_indices, : observed.shape[1]]
    usable = (
        visible[frame_indices]
        & motion_valid[frame_indices]
        & training[:, None]
        & np.all(np.isfinite(target), axis=2)
        & np.all(np.isfinite(reference), axis=2)
        & np.all(np.isfinite(candidate), axis=2)
    )
    if not np.any(usable):
        raise ValueError("no common training-prefix automatic tracks")
    reference_error = np.linalg.norm(reference - target, axis=2)
    candidate_error = np.linalg.norm(candidate - target, axis=2)
    return float(np.mean(reference_error[usable])), float(
        np.mean(candidate_error[usable])
    )


def _manual_error_vector(
    report: dict[str, Any],
    variant: str,
) -> np.ndarray:
    audit = report["variants"][variant]["manual_identity_audit"]
    if not audit.get("available", False):
        raise ValueError(f"manual audit unavailable for {variant}")
    return np.asarray(audit["error_by_sampled_frame_m"], dtype=float)


def evaluate_motioncrafter_assimilation(
    summary_paths: list[str | Path],
    output_dir: str | Path,
    *,
    bootstrap_samples: int = 20000,
    bootstrap_seed: int = 20260711,
) -> dict[str, object]:
    """Aggregate a frozen non-development cohort with equal case weighting."""

    paths = [Path(path) for path in summary_paths]
    if not paths:
        raise ValueError("at least one summary is required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    cases = [str(report["case"]) for report in reports]
    if len(set(cases)) != len(cases):
        raise ValueError("case summaries must be unique")
    order = np.argsort(cases)
    paths = [paths[index] for index in order]
    reports = [reports[index] for index in order]
    cases = [cases[index] for index in order]

    rows: list[dict[str, object]] = []
    horizon_values: dict[str, list[list[float]]] = {
        "released_phystwin": [[], [], []],
        "position_only_graph": [[], [], []],
        "position_flow_graph": [[], [], []],
    }
    for path, report in zip(paths, reports, strict=True):
        frame_indices = np.asarray(report["frame_indices"], dtype=np.int64)
        future = frame_indices >= int(report["train_end_frame"])
        future_indices = np.flatnonzero(future)
        thirds = np.array_split(future_indices, 3)
        for variant in horizon_values:
            errors = _manual_error_vector(report, variant)
            for horizon, indices in enumerate(thirds):
                selected = errors[indices]
                selected = selected[np.isfinite(selected)]
                horizon_values[variant][horizon].append(
                    float(np.mean(selected)) if len(selected) else math.nan
                )
        archive_path = report["outputs"]["assimilation_npz"]
        training_reference, training_candidate = _training_track_fit(
            report, archive_path
        )
        variants = report["variants"]
        baseline = float(
            variants["released_phystwin"]["manual_identity_audit"]["future_mean_m"]
        )
        position = float(
            variants["position_only_graph"]["manual_identity_audit"]["future_mean_m"]
        )
        flow = float(
            variants["position_flow_graph"]["manual_identity_audit"]["future_mean_m"]
        )
        apply_training_gate = training_candidate < training_reference
        rows.append(
            {
                "case": report["case"],
                "released_phystwin_future_manual_m": baseline,
                "position_only_graph_future_manual_m": position,
                "position_flow_graph_future_manual_m": flow,
                "position_flow_minus_released_m": flow - baseline,
                "position_flow_direct_future_vertex_fraction": float(
                    variants["position_flow_direct"]["future_vertex_fraction"]
                ),
                "training_track_released_m": training_reference,
                "training_track_position_flow_m": training_candidate,
                "training_gate_applies_assimilation": apply_training_gate,
                "training_gated_future_manual_m": (
                    flow if apply_training_gate else baseline
                ),
                "summary_path": str(path.resolve()),
                "summary_sha256": _sha256(path),
            }
        )

    baseline = np.asarray(
        [row["released_phystwin_future_manual_m"] for row in rows], dtype=float
    )
    position = np.asarray(
        [row["position_only_graph_future_manual_m"] for row in rows],
        dtype=float,
    )
    flow = np.asarray(
        [row["position_flow_graph_future_manual_m"] for row in rows],
        dtype=float,
    )
    gated = np.asarray(
        [row["training_gated_future_manual_m"] for row in rows], dtype=float
    )
    coverage = np.asarray(
        [row["position_flow_direct_future_vertex_fraction"] for row in rows],
        dtype=float,
    )
    horizon_summary: dict[str, object] = {}
    for horizon, label in enumerate(("early", "middle", "late")):
        horizon_reference = np.asarray(
            horizon_values["released_phystwin"][horizon], dtype=float
        )
        horizon_candidate = np.asarray(
            horizon_values["position_flow_graph"][horizon], dtype=float
        )
        usable = np.isfinite(horizon_reference) & np.isfinite(horizon_candidate)
        horizon_summary[label] = _paired_summary(
            horizon_reference[usable],
            horizon_candidate[usable],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 100 + horizon,
        )

    result: dict[str, object] = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "case_weighting": "equal case",
            "future_manual_tracks": "evaluation only",
            "training_gate": "apply only when graph assimilation lowers available automatic training-track error",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "case_count": len(rows),
        "cases": rows,
        "comparisons": {
            "position_only_graph_vs_released": _paired_summary(
                baseline,
                position,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "position_flow_graph_vs_released": _paired_summary(
                baseline,
                flow,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 10,
            ),
            "position_flow_graph_vs_position_only_graph": _paired_summary(
                position,
                flow,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 20,
            ),
            "training_gated_vs_released": _paired_summary(
                baseline,
                gated,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 30,
            ),
        },
        "direct_future_vertex_fraction": {
            "equal_case_mean": float(np.mean(coverage)),
            "median": float(np.median(coverage)),
            "bootstrap_95": list(
                _bootstrap_interval(
                    coverage,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + 40,
                )
            ),
        },
        "training_gate": {
            "applied_case_count": int(
                sum(bool(row["training_gate_applies_assimilation"]) for row in rows)
            ),
            "rejected_case_count": int(
                sum(not bool(row["training_gate_applies_assimilation"]) for row in rows)
            ),
        },
        "horizon": horizon_summary,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases_path = output / "cases.csv"
    with cases_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result["outputs"] = {
        "cases_csv": str(cases_path.resolve()),
        "cases_csv_sha256": _sha256(cases_path),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(summary_path.resolve())
    return result
