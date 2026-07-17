"""Prospective evaluation for the fresh reusable-twin admission panel."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_independent_source import sha256_file
from .deform360_phystwin_trust import (
    CausalTrustEpisode,
    score_causal_trust_interval,
)
from .deform360_reusable_trust_protocol import (
    EXPECTED_SPLITS,
    authorize_reusable_trust_held_outcome,
    validate_reusable_trust_prediction,
    validate_reusable_trust_prediction_cohort_seal,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    _require(isinstance(value, Mapping), f"{path.name} must contain a mapping")
    return value


def evaluate_reusable_trust_held_prediction(
    prediction: Mapping[str, Any],
    *,
    target_data_path: str | Path,
    outcome: Mapping[str, Any],
    cohort_seal: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one held prediction after the complete cohort seal is verified."""

    prediction_record = validate_reusable_trust_prediction(
        prediction, protocol=protocol, verify_archive=True
    )
    validate_reusable_trust_prediction_cohort_seal(
        cohort_seal, protocol=protocol, verify_predictions=True
    )
    object_id = prediction_record["object_id"]
    episode_id = int(prediction_record["episode_id"])
    authorize_reusable_trust_held_outcome(
        protocol,
        cohort_seal,
        object_id=object_id,
        episode_id=episode_id,
    )
    _require(
        outcome.get("artifact_kind") == "Deform360ReusableTwinFreshOutcome"
        and outcome.get("result_sha256") == _result_sha256(outcome),
        "fresh outcome is invalid",
    )
    _require(
        outcome.get("role") == "held-out-evaluation"
        and outcome.get("object_id") == object_id
        and int(outcome.get("episode_id", -1)) == episode_id,
        "fresh outcome belongs to another held episode",
    )
    _require(
        outcome.get("future_access_seal_result_sha256")
        == cohort_seal["result_sha256"],
        "held outcome was not opened after this cohort seal",
    )
    target_path = Path(target_data_path).resolve()
    _require(
        sha256_file(target_path) == outcome.get("output_sha256", {}).get("target_data"),
        "held target checksum differs from its outcome artifact",
    )
    target = _load_pickle(target_path)
    points = np.asarray(target["object_points"], dtype=np.float64)
    visibility = np.asarray(target["object_visibilities"], dtype=bool)
    validity = np.asarray(target["object_motions_valid"], dtype=bool)
    with np.load(prediction_record["prediction_path"], allow_pickle=False) as stored:
        predicted = np.asarray(stored["prediction_m"], dtype=np.float64)
        persistence = np.asarray(stored["persistence_m"], dtype=np.float64)
        frame_zero = np.asarray(stored["frame_zero_points_m"], dtype=np.float64)
    _require(
        points.shape == predicted.shape == persistence.shape,
        "held target and prediction shapes differ",
    )
    _require(len(points) == 76, "fresh admission evaluation requires 76 frames")
    _require(
        np.array_equal(points[0].astype(np.float32), frame_zero.astype(np.float32)),
        "held target frame zero differs from the sealed prediction",
    )
    _require(
        np.array_equal(
            persistence,
            np.repeat(frame_zero[None], len(points), axis=0),
        ),
        "held persistence baseline changed",
    )
    episode = CausalTrustEpisode(
        episode_id=f"{object_id}/{episode_id}",
        target_m=points,
        visibility=visibility,
        validity=validity,
        driven_m=persistence,
        zero_action_m=persistence,
        train_stop_frame=60,
        source_data_sha256=sha256_file(target_path),
        driven_trajectory_sha256=prediction_record["prediction_file_sha256"],
        zero_action_trajectory_sha256=prediction_record["prediction_file_sha256"],
    )
    intervals = {
        "future": (1, 76),
        "early": (1, 26),
        "middle": (26, 51),
        "late": (51, 76),
    }
    metrics: dict[str, Any] = {}
    for name, (start, stop) in intervals.items():
        scored = score_causal_trust_interval(episode, predicted, start, stop)
        scored["track_improvement_fraction"] = 1.0 - (
            float(scored["track_rmse_m"])
            / float(scored["persistence_track_rmse_m"])
        )
        scored["chamfer_improvement_fraction"] = 1.0 - (
            float(scored["chamfer_m"])
            / float(scored["persistence_chamfer_m"])
        )
        metrics[name] = scored
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTwinFreshEvaluation",
        "protocol_id": protocol["parent"]["protocol_id"],
        "physics_addendum_id": protocol["addendum"]["protocol_id"],
        "object_id": object_id,
        "episode_id": episode_id,
        "episode_key": f"{object_id}/{episode_id}",
        "prediction_result_sha256": prediction_record["prediction_result_sha256"],
        "cohort_seal_result_sha256": cohort_seal["result_sha256"],
        "target_data_sha256": sha256_file(target_path),
        "outcome_result_sha256": outcome["result_sha256"],
        "metrics": metrics,
        "joint_future_win": bool(
            metrics["future"]["track_improvement_fraction"] > 0.0
            and metrics["future"]["chamfer_improvement_fraction"] > 0.0
        ),
        "information_boundary": {
            "complete_prediction_cohort_previously_sealed": True,
            "held_outcome_opened_for_scoring": True,
            "method_or_hyperparameter_changed_after_seal": False,
        },
        "claim_boundary": (
            "short-window fresh admission evaluation; not an official Deform360 "
            "state-of-the-art result"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def _balanced_improvement(
    rows: Sequence[Mapping[str, Any]], interval: str, metric: str
) -> float:
    predicted = np.asarray([float(row["metrics"][interval][metric]) for row in rows])
    baseline = np.asarray(
        [float(row["metrics"][interval][f"persistence_{metric}"]) for row in rows]
    )
    return float(1.0 - np.mean(predicted) / np.mean(baseline))


def aggregate_reusable_trust_fresh_gate(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    cohort_seal: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the locked fresh-panel gates to exactly twelve held executions."""

    validate_reusable_trust_prediction_cohort_seal(
        cohort_seal, protocol=protocol, verify_predictions=True
    )
    rows = tuple(evaluations)
    expected_keys = {
        f"{object_id}/{episode_id}"
        for object_id, split in EXPECTED_SPLITS.items()
        for episode_id in split["held_out_episode_ids"]
    }
    observed_keys = {str(row.get("episode_key")) for row in rows}
    _require(
        len(rows) == len(observed_keys) and observed_keys == expected_keys,
        "fresh gate requires exactly the locked twelve held episodes",
    )
    for row in rows:
        _require(
            row.get("artifact_kind") == "Deform360ReusableTwinFreshEvaluation"
            and row.get("result_sha256") == _result_sha256(row)
            and row.get("cohort_seal_result_sha256")
            == cohort_seal["result_sha256"],
            "fresh episode evaluation is invalid",
        )
    track = _balanced_improvement(rows, "future", "track_rmse_m")
    chamfer = _balanced_improvement(rows, "future", "chamfer_m")
    late_track = _balanced_improvement(rows, "late", "track_rmse_m")
    late_chamfer = _balanced_improvement(rows, "late", "chamfer_m")
    maximum_degradation = {"track": -np.inf, "chamfer": -np.inf}
    by_object: dict[str, Any] = {}
    no_object_median_degradation = True
    for object_id in EXPECTED_SPLITS:
        selected = [row for row in rows if row["object_id"] == object_id]
        track_changes = np.asarray(
            [
                -float(row["metrics"]["future"]["track_improvement_fraction"])
                for row in selected
            ]
        )
        chamfer_changes = np.asarray(
            [
                -float(row["metrics"]["future"]["chamfer_improvement_fraction"])
                for row in selected
            ]
        )
        non_degrading = bool(
            np.median(track_changes) <= 0.0 and np.median(chamfer_changes) <= 0.0
        )
        no_object_median_degradation &= non_degrading
        maximum_degradation["track"] = max(
            maximum_degradation["track"], float(np.max(track_changes))
        )
        maximum_degradation["chamfer"] = max(
            maximum_degradation["chamfer"], float(np.max(chamfer_changes))
        )
        by_object[object_id] = {
            "episode_count": len(selected),
            "joint_win_count": sum(bool(row["joint_future_win"]) for row in selected),
            "median_track_change_fraction": float(np.median(track_changes)),
            "median_chamfer_change_fraction": float(np.median(chamfer_changes)),
            "median_non_degradation_passed": non_degrading,
        }
    gate = protocol["addendum"]["admission_gate"]
    maximum = float(gate["maximum_per_episode_degradation_fraction_per_metric"])
    gates = {
        "future_track": track
        >= float(gate["minimum_future_track_improvement_fraction"]),
        "future_chamfer": chamfer
        >= float(gate["minimum_future_chamfer_improvement_fraction"]),
        "late_track": late_track
        >= float(gate["minimum_late_track_improvement_fraction"]),
        "late_chamfer": late_chamfer
        >= float(gate["minimum_late_chamfer_improvement_fraction"]),
        "maximum_episode_track_degradation": maximum_degradation["track"] <= maximum,
        "maximum_episode_chamfer_degradation": maximum_degradation["chamfer"]
        <= maximum,
        "no_object_median_degradation": no_object_median_degradation,
    }
    passed = all(gates.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTwinFreshAdmissionGate",
        "protocol_id": protocol["parent"]["protocol_id"],
        "physics_addendum_id": protocol["addendum"]["protocol_id"],
        "cohort_seal_result_sha256": cohort_seal["result_sha256"],
        "episode_count": len(rows),
        "metrics": {
            "future_track_improvement_fraction": track,
            "future_chamfer_improvement_fraction": chamfer,
            "late_track_improvement_fraction": late_track,
            "late_chamfer_improvement_fraction": late_chamfer,
            "maximum_per_episode_track_degradation_fraction": maximum_degradation[
                "track"
            ],
            "maximum_per_episode_chamfer_degradation_fraction": maximum_degradation[
                "chamfer"
            ],
            "by_object": by_object,
        },
        "gates": gates,
        "passed": passed,
        "next_step": (
            "run the exact official full-horizon Deform360 replication"
            if passed
            else "freeze the reusable-trust route as a prospective negative result"
        ),
        "evaluation_result_sha256": {
            str(row["episode_key"]): str(row["result_sha256"])
            for row in sorted(rows, key=lambda item: str(item["episode_key"]))
        },
        "information_boundary": {
            "all_twelve_held_outcomes_read": True,
            "admission_method_frozen_before_outcomes": True,
            "official_full_horizon_targets_read": False,
        },
        "claim_boundary": (
            "fresh short-window admission gate only; passing authorizes but does "
            "not establish an official state-of-the-art claim"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


__all__ = [
    "aggregate_reusable_trust_fresh_gate",
    "evaluate_reusable_trust_held_prediction",
]
