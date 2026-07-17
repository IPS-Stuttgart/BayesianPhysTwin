#!/usr/bin/env python3
"""Aggregate the five frozen Deform360 residual source folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_OBJECTS = {
    "002-rope-silk",
    "083-blanket-cloth",
    "085-scarf-cloth",
    "092-squirrel",
    "170-spider",
}
ARMS = (
    "persistence",
    "physics",
    "deterministic_residual",
    "gated_residual",
)
METRICS = (
    "future_track_error_m",
    "future_chamfer_m",
    "late_track_error_m",
    "late_chamfer_m",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    _require(len(args.input) == len(EXPECTED_OBJECTS), "expected five source folds")
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    for fold in folds:
        _require(
            fold.get("artifact_kind") == "Deform360BayesianResidualSourceSmoke",
            "unexpected source-fold artifact kind",
        )
        _require(fold.get("result_sha256") == _sha256(fold), "fold checksum differs")
    _require(
        {fold["held_object_id"] for fold in folds} == EXPECTED_OBJECTS,
        "held-object folds differ",
    )
    rows = [row for fold in folds for row in fold["episodes"]]
    _require(len(rows) == 27, "source folds do not contain 27 held episodes")
    _require(
        len({row["episode_key"] for row in rows}) == len(rows),
        "held source episodes are duplicated",
    )

    aggregate = {
        arm: {
            metric: sum(row["arms"][arm][metric] for row in rows) / len(rows)
            for metric in METRICS
        }
        for arm in ARMS
    }
    improvements = {
        baseline: {
            arm: {
                metric: 1.0
                - aggregate[arm][metric] / aggregate[baseline][metric]
                for metric in METRICS
            }
            for arm in ARMS
        }
        for baseline in ("persistence", "physics")
    }
    episode_diagnostics = {}
    for arm in ARMS[1:]:
        joint_wins = sum(
            row["arms"][arm]["future_track_error_m"]
            < row["arms"]["persistence"]["future_track_error_m"]
            and row["arms"][arm]["future_chamfer_m"]
            < row["arms"]["persistence"]["future_chamfer_m"]
            for row in rows
        )
        maximum_degradation = max(
            max(
                row["arms"][arm]["future_track_error_m"]
                / row["arms"]["persistence"]["future_track_error_m"],
                row["arms"][arm]["future_chamfer_m"]
                / row["arms"]["persistence"]["future_chamfer_m"],
            )
            - 1.0
            for row in rows
        )
        episode_diagnostics[arm] = {
            "joint_win_count_vs_persistence": joint_wins,
            "maximum_future_degradation_fraction_vs_persistence": maximum_degradation,
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BayesianResidualTrustedSourceAggregate",
        "protocol_id": "deform360-bayesian-residual-source-v1",
        "fold_count": len(folds),
        "episode_count": len(rows),
        "held_object_ids": sorted(EXPECTED_OBJECTS),
        "aggregate": aggregate,
        "improvement_fraction": improvements,
        "episode_diagnostics": episode_diagnostics,
        "selected_utility_threshold_by_object": {
            fold["held_object_id"]: fold["utility_selection"][
                "selected_utility_threshold"
            ]
            for fold in folds
        },
        "residual_gate_admitted_any_outer_fold": any(
            fold["utility_selection"]["selected_utility_threshold"] <= 1.0
            for fold in folds
        ),
        "claim_boundary": (
            "Already-open 27-episode source development result on a deterministic "
            "256-node subset; not an official Deform360 or state-of-the-art claim."
        ),
        "input_result_sha256": {
            fold["held_object_id"]: fold["result_sha256"] for fold in folds
        },
    }
    payload["result_sha256"] = _sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(payload["result_sha256"])


if __name__ == "__main__":
    main()
