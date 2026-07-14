"""Source-only forward fitting for the Deform360 rope pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360 import Deform360ProtocolConfig
from .deform360_rope_dynamics import (
    RopeDynamicsObservation,
    SharedRopeDynamicsParameters,
    rollout_rope_dynamics,
)
from .deform360_rope_observations import (
    DEFORM360_ROPE_OBSERVATION_SCHEMA_VERSION,
    load_source_rope_dynamics_observation,
    rope_source_observation_artifact_sha256,
)


DEFORM360_ROPE_FORWARD_FIT_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def rope_forward_fit_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


@dataclass(frozen=True)
class RopeForwardFitConfig:
    """Choices frozen before any held-out future geometry is read."""

    prefix_frame_count: int = 6
    bending_acceleration_grid: tuple[float, ...] = (0.0, 0.5, 2.0, 8.0)
    contact_acceleration_grid: tuple[float, ...] = (0.0, 1.0, 3.0, 10.0, 30.0)
    contact_damping_grid: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 10.0)
    drag_grid: tuple[float, ...] = (0.0, 0.2)
    substeps: int = 4
    constraint_iterations: int = 16
    minimum_pooled_chamfer_improvement_fraction: float = 0.05
    minimum_loo_better_episode_fraction: float = 0.6

    def __post_init__(self) -> None:
        _require(self.prefix_frame_count >= 2, "fit prefix must contain two frames")
        for name in (
            "bending_acceleration_grid",
            "contact_acceleration_grid",
            "contact_damping_grid",
            "drag_grid",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            _require(len(values) >= 1, f"{name} must be nonempty")
            _require(
                np.all(np.isfinite(values)) and np.all(values >= 0.0),
                f"{name} must contain finite nonnegative values",
            )
        _require(self.substeps >= 1, "fit substeps must be positive")
        _require(
            self.constraint_iterations >= 1,
            "forward fit requires inextensibility projection",
        )
        _require(
            0.0 <= self.minimum_pooled_chamfer_improvement_fraction < 1.0,
            "invalid pooled source gate",
        )
        _require(
            0.0 <= self.minimum_loo_better_episode_fraction <= 1.0,
            "invalid leave-one-out source gate",
        )


def _candidate_parameters(
    config: RopeForwardFitConfig,
) -> tuple[SharedRopeDynamicsParameters, ...]:
    candidates = []
    for bending, contact, contact_damping, drag in product(
        config.bending_acceleration_grid,
        config.contact_acceleration_grid,
        config.contact_damping_grid,
        config.drag_grid,
    ):
        candidates.append(
            SharedRopeDynamicsParameters(
                spring_acceleration_per_m_s2=0.0,
                edge_damping_per_s=0.0,
                bending_acceleration_per_m_s2=float(bending),
                bending_damping_per_s=0.0,
                contact_acceleration_per_m_s2=float(contact),
                contact_damping_per_s=float(contact_damping),
                drag_per_s=float(drag),
            )
        )
    return tuple(candidates)


def _mean_chamfer_m(reference: np.ndarray, prediction: np.ndarray) -> float:
    _require(reference.shape == prediction.shape, "trajectory shapes disagree")
    difference = reference[:, :, None, :] - prediction[:, None, :, :]
    distances = np.linalg.norm(difference, axis=3)
    per_frame = 0.5 * (
        np.mean(np.min(distances, axis=1), axis=1)
        + np.mean(np.min(distances, axis=2), axis=1)
    )
    return float(np.mean(per_frame))


def _source_forecast_case(
    observation: RopeDynamicsObservation, config: RopeForwardFitConfig
) -> dict[str, Any]:
    contact_frames = np.flatnonzero(np.any(observation.contact_active, axis=1))
    _require(len(contact_frames) > 0, "source observation contains no active contact")
    prefix_start = int(contact_frames[0])
    prefix_end = prefix_start + config.prefix_frame_count
    _require(prefix_end < len(observation.positions_m), "source prefix has no future")
    initial = observation.positions_m[prefix_end - 1]
    reference = observation.positions_m[prefix_end - 1 :]
    return {
        "episode_id": observation.episode_id,
        "prefix_start_index": prefix_start,
        "prefix_end_index_exclusive": prefix_end,
        "initial_positions_m": initial,
        "initial_velocities_m_s": np.zeros_like(initial),
        "controller_positions_m": observation.controller_positions_m[prefix_end - 1 :],
        "contact_active": observation.contact_active[prefix_end - 1 :],
        "contact_node_indices": observation.contact_node_indices,
        "contact_offsets_m": observation.contact_offsets_m,
        "rest_lengths_m": np.linalg.norm(np.diff(initial, axis=0), axis=1),
        "reference_positions_m": reference,
        "dt_seconds": observation.dt_seconds,
    }


def _score_candidate(
    parameters: SharedRopeDynamicsParameters,
    cases: Sequence[Mapping[str, Any]],
    config: RopeForwardFitConfig,
) -> tuple[dict[str, float], ...]:
    scores = []
    for case in cases:
        prediction = rollout_rope_dynamics(
            case["initial_positions_m"],
            case["initial_velocities_m_s"],
            case["controller_positions_m"],
            case["contact_active"],
            case["contact_node_indices"],
            case["contact_offsets_m"],
            case["rest_lengths_m"],
            parameters,
            dt_seconds=float(case["dt_seconds"]),
            gravity_m_s2=np.zeros(3),
            substeps=config.substeps,
            constraint_iterations=config.constraint_iterations,
        )
        reference = case["reference_positions_m"][1:]
        prediction = prediction[1:]
        scores.append(
            {
                "chamfer_distance_m": _mean_chamfer_m(reference, prediction),
                "track_error_m": float(
                    np.mean(np.linalg.norm(reference - prediction, axis=2))
                ),
            }
        )
    return tuple(scores)


def fit_forward_rope_dynamics(
    observations: Sequence[RopeDynamicsObservation],
    *,
    config: RopeForwardFitConfig = RopeForwardFitConfig(),
) -> dict[str, Any]:
    """Select shared parameters by source-only forward prediction."""

    _require(len(observations) >= 2, "forward fit needs at least two source episodes")
    episode_ids = [observation.episode_id for observation in observations]
    _require(len(episode_ids) == len(set(episode_ids)), "source episodes repeat")
    cases = tuple(
        _source_forecast_case(observation, config) for observation in observations
    )
    persistence_scores = []
    for case in cases:
        reference = case["reference_positions_m"][1:]
        persistence = np.repeat(
            case["initial_positions_m"][None], len(reference), axis=0
        )
        persistence_scores.append(
            {
                "chamfer_distance_m": _mean_chamfer_m(reference, persistence),
                "track_error_m": float(
                    np.mean(np.linalg.norm(reference - persistence, axis=2))
                ),
            }
        )
    candidate_rows = []
    for index, parameters in enumerate(_candidate_parameters(config)):
        scores = _score_candidate(parameters, cases, config)
        candidate_rows.append(
            {
                "candidate_index": index,
                "parameters": parameters.as_dict(),
                "per_episode": [
                    {"episode_id": episode_id, **score}
                    for episode_id, score in zip(episode_ids, scores, strict=True)
                ],
                "mean_chamfer_distance_m": float(
                    np.mean([score["chamfer_distance_m"] for score in scores])
                ),
                "mean_track_error_m": float(
                    np.mean([score["track_error_m"] for score in scores])
                ),
            }
        )
    selected = min(
        candidate_rows,
        key=lambda row: (row["mean_chamfer_distance_m"], row["candidate_index"]),
    )
    leave_one_out = []
    for held_index, held_episode_id in enumerate(episode_ids):
        training_indices = [
            index for index in range(len(episode_ids)) if index != held_index
        ]
        fold_selected = min(
            candidate_rows,
            key=lambda row: (
                float(
                    np.mean(
                        [
                            row["per_episode"][index]["chamfer_distance_m"]
                            for index in training_indices
                        ]
                    )
                ),
                row["candidate_index"],
            ),
        )
        held_score = fold_selected["per_episode"][held_index]
        baseline = persistence_scores[held_index]
        leave_one_out.append(
            {
                "held_out_episode_id": held_episode_id,
                "selected_candidate_index": fold_selected["candidate_index"],
                "held_out_chamfer_distance_m": held_score["chamfer_distance_m"],
                "persistence_chamfer_distance_m": baseline["chamfer_distance_m"],
                "held_out_track_error_m": held_score["track_error_m"],
                "persistence_track_error_m": baseline["track_error_m"],
                "chamfer_better_than_persistence": bool(
                    held_score["chamfer_distance_m"] < baseline["chamfer_distance_m"]
                ),
            }
        )
    persistence_mean = float(
        np.mean([row["chamfer_distance_m"] for row in persistence_scores])
    )
    selected_mean = float(selected["mean_chamfer_distance_m"])
    pooled_improvement = (persistence_mean - selected_mean) / persistence_mean
    loo_better_fraction = float(
        np.mean([row["chamfer_better_than_persistence"] for row in leave_one_out])
    )
    loo_mean = float(
        np.mean([row["held_out_chamfer_distance_m"] for row in leave_one_out])
    )
    gate = {
        "minimum_pooled_chamfer_improvement_fraction": (
            config.minimum_pooled_chamfer_improvement_fraction
        ),
        "observed_pooled_chamfer_improvement_fraction": pooled_improvement,
        "minimum_loo_better_episode_fraction": (
            config.minimum_loo_better_episode_fraction
        ),
        "observed_loo_better_episode_fraction": loo_better_fraction,
        "loo_mean_chamfer_distance_m": loo_mean,
        "persistence_mean_chamfer_distance_m": persistence_mean,
    }
    gate["passed"] = bool(
        pooled_improvement >= config.minimum_pooled_chamfer_improvement_fraction
        and loo_better_fraction >= config.minimum_loo_better_episode_fraction
        and loo_mean < persistence_mean
    )
    return {
        "config": asdict(config),
        "fit_method": "finite source-only forward-rollout grid",
        "initial_velocity_policy": "zero for silhouette pseudo-correspondences",
        "effective_gravity_m_s2": [0.0, 0.0, 0.0],
        "candidate_count": len(candidate_rows),
        "selected_candidate_index": selected["candidate_index"],
        "selected_parameters": selected["parameters"],
        "selected_source_metrics": {
            "mean_chamfer_distance_m": selected_mean,
            "mean_track_error_m": selected["mean_track_error_m"],
        },
        "persistence_per_episode": [
            {"episode_id": episode_id, **score}
            for episode_id, score in zip(episode_ids, persistence_scores, strict=True)
        ],
        "candidate_scores": candidate_rows,
        "leave_one_episode_out": leave_one_out,
        "source_competence_gate": gate,
    }


def build_forward_rope_fit_artifact(
    protocol: Deform360ProtocolConfig,
    observation_payloads: Sequence[Mapping[str, Any]],
    *,
    config: RopeForwardFitConfig = RopeForwardFitConfig(),
) -> dict[str, Any]:
    """Bind a source forward fit to all locked source observation artifacts."""

    _require(
        config.prefix_frame_count == protocol.prefix_frame_count,
        "fit prefix differs from the locked protocol",
    )
    expected = set(protocol.source_episode_ids)
    observed = {int(payload["episode_index"]) for payload in observation_payloads}
    _require(observed == expected, "fit inputs do not cover the locked source split")
    accepted = []
    excluded = []
    inputs = []
    for payload in sorted(observation_payloads, key=lambda row: row["episode_index"]):
        _require(
            payload.get("schema_version") == DEFORM360_ROPE_OBSERVATION_SCHEMA_VERSION,
            "source observation schema mismatch",
        )
        _require(
            payload.get("result_sha256")
            == rope_source_observation_artifact_sha256(payload),
            "source observation checksum mismatch",
        )
        _require(payload.get("split") == "source", "fit input is not source-only")
        _require(
            payload.get("information_boundary", {}).get("target_files_read") is False,
            "fit input read target files",
        )
        inputs.append(
            {
                "episode_index": int(payload["episode_index"]),
                "episode_id": payload["episode_id"],
                "result_sha256": payload["result_sha256"],
                "quality_passed": bool(payload["quality"]["passed"]),
            }
        )
        if payload["quality"]["passed"]:
            accepted.append(load_source_rope_dynamics_observation(payload))
        else:
            excluded.append(
                {
                    "episode_index": int(payload["episode_index"]),
                    "episode_id": payload["episode_id"],
                    "reason": "predeclared source contact-registration gates failed",
                    "quality": payload["quality"],
                }
            )
    _require(len(accepted) >= 3, "too few quality-passing source observations")
    fit = fit_forward_rope_dynamics(accepted, config=config)
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_ROPE_FORWARD_FIT_SCHEMA_VERSION,
        "artifact_kind": "Deform360SharedRopeForwardDynamicsFit",
        "protocol_id": protocol.protocol_id,
        "source_inputs": inputs,
        "accepted_source_episode_ids": [row.episode_id for row in accepted],
        "excluded_source_episodes": excluded,
        **fit,
        "information_boundary": {
            "source_episodes_read": True,
            "calibration_episodes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_tactile_oracle_read": False,
        },
        "claim_boundary": (
            "Exploratory public-data source fit; source competence is assessed by "
            "leave-one-action-out prediction before target-future access."
        ),
    }
    payload["result_sha256"] = rope_forward_fit_artifact_sha256(payload)
    return payload


def validate_forward_rope_fit_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_ROPE_FORWARD_FIT_SCHEMA_VERSION,
        "unsupported rope forward-fit schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360SharedRopeForwardDynamicsFit",
        "unexpected rope forward-fit artifact kind",
    )
    _require(
        payload.get("result_sha256") == rope_forward_fit_artifact_sha256(payload),
        "rope forward-fit checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(boundary.get("target_prefix_read") is False, "fit used target prefix")
    _require(
        boundary.get("target_future_geometry_read") is False,
        "fit used target future geometry",
    )
    _require(
        boundary.get("target_tactile_oracle_read") is False,
        "fit used target tactile oracle",
    )
    parameters = SharedRopeDynamicsParameters.from_array(
        np.asarray(
            [
                payload["selected_parameters"][name]
                for name in SharedRopeDynamicsParameters.__dataclass_fields__
            ],
            dtype=np.float64,
        )
    )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "source_competence_passed": bool(payload["source_competence_gate"]["passed"]),
        "selected_parameters": parameters.as_dict(),
    }


def load_forward_rope_fit_parameters(
    payload: Mapping[str, Any],
) -> SharedRopeDynamicsParameters:
    validate_forward_rope_fit_artifact(payload)
    values = [
        payload["selected_parameters"][name]
        for name in SharedRopeDynamicsParameters.__dataclass_fields__
    ]
    return SharedRopeDynamicsParameters.from_array(np.asarray(values, dtype=np.float64))


def write_forward_rope_fit_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFORM360_ROPE_FORWARD_FIT_SCHEMA_VERSION",
    "RopeForwardFitConfig",
    "build_forward_rope_fit_artifact",
    "fit_forward_rope_dynamics",
    "load_forward_rope_fit_parameters",
    "rope_forward_fit_artifact_sha256",
    "validate_forward_rope_fit_artifact",
    "write_forward_rope_fit_artifact",
]
