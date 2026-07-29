"""Prediction-runtime custody for prospective V14 source executions.

This module does not change the frozen V14 estimator. It reconstructs the
already-sealed frame-zero carrier, converts released prefix streams into the
typed V12 prefix contract, and validates a child runtime lock before a source
prediction can be sealed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_causal_response_adaptive_query import (
    AdaptiveCameraPanels,
    AdaptiveCausalResponseQueryConfig,
    AdaptiveCausalResponseQuerySchedule,
    validate_adaptive_causal_response_query_artifacts,
)
from .deform360_causal_response_direct_depth_admission_v14 import (
    CARRIER_DIRECTORY,
    aggregate_source_sha256,
    validate_v14_admission_report,
)
from .deform360_causal_response_direct_depth_source_lock import (
    AdaptiveDirectDepthSourceCaseV14,
    AdaptiveDirectDepthSourceLockV14,
    validate_adaptive_direct_depth_source_lock_v14,
)
from .deform360_causal_response_prefix import (
    CausalResponsePrefixConfig,
    CausalResponsePrefixInputs,
)
from .deform360_causal_response_query import (
    CausalResponseQueryConfig,
    CausalResponseQuerySchedule,
)
from .deform360_object_exclusion import file_sha256
from .deform360_selective_virtual_sensing_staging import (
    end_effector_origins,
)

RUNTIME_KIND = "Deform360CausalResponseDirectDepthPredictionRuntimeV14"
RUNTIME_CONTRACT = (
    "deform360-causal-response-direct-depth-prediction-runtime-v14"
)
RUNTIME_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-prediction-runtime"
)
PREFIX_FRAME_COUNT = 58
PREDICTION_FRAME_COUNT = 76
TACTILE_AGGREGATION = "framewise-max-across-released-sensors-and-taxels"
ACTUATOR_POSITION_FIELD = "robot.actions[...,0,:]"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-prediction-runtime-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_v14_prediction_runtime(
    path: str | Path,
    *,
    method_protocol_path: str | Path,
    source_lock_path: str | Path,
) -> dict[str, Any]:
    """Validate the post-source-lock runtime without opening an outcome."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RUNTIME_KIND
        and payload.get("contract") == RUNTIME_CONTRACT
        and payload.get("protocol_id") == RUNTIME_PROTOCOL_ID
        and payload.get("status") == "locked_after_source_selection_before_prefix_scan"
        and payload.get("config_sha256") == _canonical_sha256(payload),
        "V14 prediction runtime identity or checksum changed",
    )
    method_path = Path(method_protocol_path)
    method = json.loads(method_path.read_text(encoding="utf-8"))
    lock_path = Path(source_lock_path)
    source_lock = validate_adaptive_direct_depth_source_lock_v14(lock_path)
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and parents.get("method_protocol", {}).get("semantic_sha256")
        == method.get("config_sha256")
        and parents["method_protocol"].get("file_sha256")
        == file_sha256(method_path)
        and parents.get("source_lock", {}).get("semantic_sha256")
        == source_lock.artifact_sha256
        and parents["source_lock"].get("file_sha256") == file_sha256(lock_path),
        "V14 prediction runtime parent changed",
    )
    implementation = payload.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and isinstance(implementation.get("file_sha256"), Mapping)
        and set(implementation["file_sha256"])
        == {"prediction_module", "prediction_runner", "preflight_module"}
        and all(
            _valid_digest(value)
            for value in implementation["file_sha256"].values()
        ),
        "V14 prediction runtime implementation binding changed",
    )
    numerical = payload.get("numerical_contract")
    _require(
        isinstance(numerical, Mapping)
        and numerical.get("prefix_frame_count") == PREFIX_FRAME_COUNT
        and numerical.get("prediction_frame_count") == PREDICTION_FRAME_COUNT
        and numerical.get("depth_scale_to_m") == 0.001
        and numerical.get("tactile_aggregation") == TACTILE_AGGREGATION
        and numerical.get("actuator_position_field") == ACTUATOR_POSITION_FIELD
        and numerical.get("tactile_values_are_calibrated_probabilities") is False,
        "V14 prediction runtime numerical contract changed",
    )
    cases = payload.get("cases")
    _require(
        isinstance(cases, list)
        and len(cases) == 12
        and {record.get("case_hash") for record in cases}
        == {case.case_hash for case in source_lock.cases}
        and len({int(record.get("queue_rank", 0)) for record in cases}) == 12
        and all(
            int(record.get("queue_rank", 0)) >= 1
            and all(
                _valid_digest(record.get(key))
                for key in (
                    "case_hash",
                    "object_hash",
                    "admission_artifact_sha256",
                    "admission_file_sha256",
                    "physical_artifact_sha256",
                    "physical_manifest_file_sha256",
                    "physical_archive_file_sha256",
                )
            )
            for record in cases
        ),
        "V14 prediction runtime case ledger changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("maximum_object_observation_frame")
        == PREFIX_FRAME_COUNT - 1
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 prediction runtime crossed its information boundary",
    )
    return payload


def v14_prediction_case_record(
    runtime: Mapping[str, Any],
    source_lock: AdaptiveDirectDepthSourceLockV14,
    *,
    queue_rank: int,
) -> tuple[dict[str, Any], AdaptiveDirectDepthSourceCaseV14]:
    """Return one runtime/source-lock case pair."""

    matches = [
        dict(record)
        for record in runtime["cases"]
        if int(record["queue_rank"]) == int(queue_rank)
    ]
    _require(len(matches) == 1, "V14 prediction queue rank is not unique")
    record = matches[0]
    locked = [
        case for case in source_lock.cases if case.case_hash == record["case_hash"]
    ]
    _require(
        len(locked) == 1
        and locked[0].object_hash == record["object_hash"],
        "V14 runtime case differs from the source lock",
    )
    return record, locked[0]


def load_v14_admitted_carrier(
    admission_dir: str | Path,
) -> tuple[
    dict[str, Any],
    AdaptiveCausalResponseQuerySchedule,
]:
    """Reconstruct and verify the immutable carrier from its typed artifacts."""

    root = Path(admission_dir)
    admission = validate_v14_admission_report(root)
    _require(admission["status"] == "admitted", "V14 carrier was not admitted")
    report, arrays = validate_adaptive_causal_response_query_artifacts(
        root / CARRIER_DIRECTORY
    )
    descriptor = report["schedule"]
    query_descriptor = descriptor["query_schedule"]
    query = CausalResponseQuerySchedule(
        config=CausalResponseQueryConfig(**query_descriptor["config"]),
        camera_ids=tuple(query_descriptor["camera_ids"]),
        proposal_camera_indices=np.asarray(
            query_descriptor["proposal_camera_indices"],
            dtype=np.int64,
        ),
        validation_camera_indices=np.asarray(
            query_descriptor["validation_camera_indices"],
            dtype=np.int64,
        ),
        entity_ids=arrays["entity_ids"],
        query_points_world_m=arrays["query_points_world_m"],
        association_query_points_xy=arrays["association_query_points_xy"],
        association_valid=arrays["association_valid"],
        association_probability=arrays["association_probability"],
        association_entropy=arrays["association_entropy"],
        association_candidate_count=arrays["association_candidate_count"],
        association_covariance_px2=arrays["association_covariance_px2"],
        selected_action_support=arrays["selected_action_support"],
        selected_total_score=arrays["selected_total_score"],
        eligible_entity_count=int(query_descriptor["eligible_entity_count"]),
        input_array_sha256=dict(query_descriptor["input_array_sha256"]),
        artifact_sha256=query_descriptor["artifact_sha256"],
    )
    panel_score = descriptor["panel_score"]
    panels = AdaptiveCameraPanels(
        proposal_indices=arrays["proposal_complete_camera_indices"],
        validation_indices=arrays["validation_complete_camera_indices"],
        strict_eligible_count=int(panel_score["strict_eligible_count"]),
        fallback_eligible_count=int(panel_score["fallback_eligible_count"]),
        supported_incidence_count=int(panel_score["supported_incidence_count"]),
        association_probability_mass=float(
            panel_score["association_probability_mass"]
        ),
    )
    carrier = AdaptiveCausalResponseQuerySchedule(
        config=AdaptiveCausalResponseQueryConfig(**descriptor["config"]),
        available_camera_ids=tuple(descriptor["available_camera_ids"]),
        panels=panels,
        arm=str(descriptor["arm"]),
        covariance_inflation=float(descriptor["covariance_inflation"]),
        query_schedule=query,
        input_array_sha256=dict(descriptor["input_array_sha256"]),
        artifact_sha256=descriptor["artifact_sha256"],
    )
    _require(
        carrier.descriptor() == descriptor
        and carrier.artifact_sha256 == admission["carrier_artifact_sha256"]
        and report["result_sha256"] == admission["carrier_result_sha256"],
        "V14 reconstructed carrier differs from the admission",
    )
    return admission, carrier


def aggregate_tactile_contact_confidence(
    sensor_arrays: Iterable[np.ndarray],
    *,
    prefix_frame_count: int = PREFIX_FRAME_COUNT,
) -> np.ndarray:
    """Aggregate correlated tactile sensors without precision multiplication.

    Released tactile values are thresholded, unitless, episode-peak-relative
    responses rather than calibrated probabilities. The framewise maximum is
    therefore retained as a contact confidence and duplicated sensors cannot
    increase it.
    """

    arrays = tuple(np.asarray(values, dtype=np.float64) for values in sensor_arrays)
    _require(arrays, "V14 tactile panel is empty")
    framewise: list[np.ndarray] = []
    for values in arrays:
        _require(
            values.ndim == 3
            and values.shape[0] >= prefix_frame_count
            and values.shape[1:] == (16, 32)
            and np.all(np.isfinite(values[:prefix_frame_count]))
            and np.all(
                (values[:prefix_frame_count] >= 0.0)
                & (values[:prefix_frame_count] <= 1.0)
            ),
            "V14 tactile array violates the released normalized contract",
        )
        framewise.append(
            np.max(values[:prefix_frame_count], axis=(1, 2))
        )
    confidence = np.max(np.stack(framewise), axis=0)
    return np.ascontiguousarray(confidence, dtype=np.float64)


def measured_actuator_origins(
    robot_actions: np.ndarray,
    *,
    prefix_frame_count: int = PREFIX_FRAME_COUNT,
) -> np.ndarray:
    """Extract measured end-effector translations through the causal prefix."""

    actions = np.asarray(robot_actions, dtype=np.float64)
    _require(
        len(actions) >= prefix_frame_count,
        "V14 robot action stream is shorter than the prefix",
    )
    origins = end_effector_origins(actions[:prefix_frame_count])
    _require(
        origins.ndim == 3
        and origins.shape[0] == prefix_frame_count
        and origins.shape[2] == 3
        and np.all(np.isfinite(origins)),
        "V14 measured actuator origins are invalid",
    )
    return np.ascontiguousarray(origins, dtype=np.float64)


def build_v14_prefix_inputs(
    *,
    camera_ids: tuple[str, ...],
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    depths_m: np.ndarray,
    object_masks: np.ndarray,
    tactile_sensor_arrays: Iterable[np.ndarray],
    robot_actions: np.ndarray,
) -> CausalResponsePrefixInputs:
    """Build the typed prefix using the frozen V14 aggregation rules."""

    return CausalResponsePrefixInputs(
        config=CausalResponsePrefixConfig(
            prefix_frame_count=PREFIX_FRAME_COUNT,
            minimum_camera_count=8,
        ),
        camera_ids=camera_ids,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        depths_m=depths_m,
        object_masks=object_masks,
        tactile_contact_probability=aggregate_tactile_contact_confidence(
            tactile_sensor_arrays
        ),
        measured_actuator_positions_m=measured_actuator_origins(robot_actions),
    )


def tactile_source_sha256(
    sensor_paths: Mapping[str, str | Path],
) -> str:
    """Reproduce the preflight's correlation-safe tactile source digest."""

    _require(sensor_paths, "V14 tactile source set is empty")
    return aggregate_source_sha256(
        "tactile",
        {
            str(sensor): file_sha256(path)
            for sensor, path in sorted(sensor_paths.items())
        },
    )


__all__ = [
    "ACTUATOR_POSITION_FIELD",
    "PREFIX_FRAME_COUNT",
    "PREDICTION_FRAME_COUNT",
    "RUNTIME_CONTRACT",
    "RUNTIME_KIND",
    "RUNTIME_PROTOCOL_ID",
    "TACTILE_AGGREGATION",
    "aggregate_tactile_contact_confidence",
    "build_v14_prefix_inputs",
    "load_v14_admitted_carrier",
    "load_v14_prediction_runtime",
    "measured_actuator_origins",
    "tactile_source_sha256",
    "v14_prediction_case_record",
]
