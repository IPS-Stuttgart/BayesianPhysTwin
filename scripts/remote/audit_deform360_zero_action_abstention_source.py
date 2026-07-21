#!/usr/bin/env python3
"""Test zero-action physical abstention on already-open Deform360 source data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_bias_aware_belief_development import (
    _load_source_target_pickle,
    _scored_frames,
)
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.deform360_online_belief_evaluation import (
    EXPECTED_SOURCE_EPISODES,
    PRIMARY_METRICS,
    UPDATE_FRAMES,
    _resolve_prediction_archive,
    score_deform360_hidden_trajectory,
)


P99_LIMIT_M = 0.25
MAXIMUM_LIMIT_M = 0.50


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _expected_cases() -> tuple[str, ...]:
    return tuple(
        f"{object_id}-ep{episode_id:04d}"
        for object_id, episode_ids in EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    )


def _object_balanced_mean(
    rows: list[dict[str, Any]], arm: str, metric: str
) -> float:
    by_object: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_object[row["object_id"]].append(row["scores"][arm][metric])
    return float(np.mean([np.mean(values) for values in by_object.values()]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--selected-baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_root = args.source_root.resolve()
    measurement_root = args.measurement_root.resolve()
    baseline_root = args.selected_baseline_root.resolve()
    output = args.output.resolve()
    rows: list[dict[str, Any]] = []
    for case in _expected_cases():
        case_dir = source_root / case
        seal_path = case_dir / "prediction_seal.json"
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        prediction_path = _resolve_prediction_archive(case_dir, seal)
        with np.load(prediction_path, allow_pickle=False) as stored:
            physical = np.asarray(stored["prediction_m"])
            persistence = np.asarray(stored["persistence_m"])
            zero = np.asarray(stored["zero_action_readout_m"], dtype=np.float64)
        with np.load(baseline_root / f"{case}.npz", allow_pickle=False) as stored:
            selected = np.asarray(stored["selected_raw_backbone"])
        measurement_path = measurement_root / case / "measurement.npz"
        with np.load(measurement_path, allow_pickle=False) as stored:
            center_ids = np.asarray(stored["center_ids"], dtype=np.int64)
        target_path = case_dir / "target_data.pkl"
        target_data = _load_source_target_pickle(target_path)
        target = np.asarray(target_data["object_points"])
        visibility = np.asarray(target_data["object_visibilities"], dtype=bool)
        validity = np.asarray(target_data["object_motions_valid"], dtype=bool)
        _require(target.shape == selected.shape, f"target shape changed: {case}")
        displacement = np.linalg.norm(zero - zero[:1], axis=2)
        p99 = float(np.quantile(displacement, 0.99))
        maximum = float(np.max(displacement))
        admitted = bool(
            np.all(np.isfinite(zero))
            and p99 <= P99_LIMIT_M
            and maximum <= MAXIMUM_LIMIT_M
        )
        guarded = selected if admitted else persistence
        frames = _scored_frames(len(target), UPDATE_FRAMES)
        arrays = {
            "physical": physical,
            "persistence": persistence,
            "selected_raw_baseline": selected,
            "zero_action_abstention": guarded,
        }
        scores = {
            arm: score_deform360_hidden_trajectory(
                value,
                target,
                visibility,
                validity,
                center_ids=center_ids,
                scored_frames=frames,
            )
            for arm, value in arrays.items()
        }
        rows.append(
            {
                "case": case,
                "object_id": str(seal["object_id"]),
                "zero_action_p99_displacement_m": p99,
                "zero_action_maximum_displacement_m": maximum,
                "physical_admitted": admitted,
                "scores": scores,
                "input_sha256": {
                    "prediction_seal": file_sha256(seal_path),
                    "prediction_archive": file_sha256(prediction_path),
                    "measurement": file_sha256(measurement_path),
                    "selected_baseline": file_sha256(
                        baseline_root / f"{case}.npz"
                    ),
                    "source_target": file_sha256(target_path),
                },
            }
        )

    aggregate: dict[str, dict[str, float]] = {}
    for arm in (
        "physical",
        "persistence",
        "selected_raw_baseline",
        "zero_action_abstention",
    ):
        aggregate[arm] = {
            metric: _object_balanced_mean(rows, arm, metric)
            for metric in PRIMARY_METRICS
        }
    comparisons = {}
    for metric in PRIMARY_METRICS:
        baseline = aggregate["selected_raw_baseline"][metric]
        guarded = aggregate["zero_action_abstention"][metric]
        comparisons[metric] = {
            "difference_m": guarded - baseline,
            "relative_change": (guarded - baseline) / baseline,
        }
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ZeroActionAbstentionOpenSourceAudit",
        "status": "post-open-negative-diagnostic",
        "case_count": len(rows),
        "object_count": len({row["object_id"] for row in rows}),
        "thresholds": {
            "p99_displacement_limit_m": P99_LIMIT_M,
            "maximum_displacement_limit_m": MAXIMUM_LIMIT_M,
            "origin": "Reused without outcome tuning from the frozen frame-zero physical diagnostic.",
        },
        "admitted_case_count": int(sum(row["physical_admitted"] for row in rows)),
        "rejected_case_count": int(sum(not row["physical_admitted"] for row in rows)),
        "aggregate": aggregate,
        "comparison_to_selected_raw_baseline": comparisons,
        "promoted": bool(
            all(value["difference_m"] <= 0.0 for value in comparisons.values())
        ),
        "decision": "reject-zero-action-displacement-as-physical-selector",
        "cases": rows,
        "information_boundary": {
            "selector_uses_target": False,
            "already_open_target_used_for_diagnostic_scoring": True,
            "fresh_or_reserved_object_read": False,
            "threshold_tuned_on_source_outcome": False,
        },
        "input_roots": {
            "source": str(source_root),
            "measurement": str(measurement_root),
            "selected_baseline": str(baseline_root),
        },
        "script_sha256": file_sha256(Path(__file__).resolve()),
    }
    result["result_sha256"] = canonical_sha256(result, digest_key="result_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
