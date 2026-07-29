"""Pre-lock physical-backbone custody for V14 causal direct depth.

The physical carrier is deliberately constructed before the twelve-case
source lock. It consumes only frame-zero object geometry and the released
robot action, and it seals hash-only case provenance. Future object
observations, tactile measurements, identities, and evaluation outcomes are
outside this module's contract.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from .deform360_causal_response_direct_depth_preflight import (
    deform360_v14_case_hash,
)
from .deform360_causal_response_preflight import deform360_object_hash
from .deform360_dynamic_tapnextpp_physical import build_readout_graph_basis
from .deform360_fresh_pairwise_physical import (
    CANONICAL_NODE_COUNT,
    load_controller_trajectory,
    load_frame_zero_ply,
)
from .deform360_object_exclusion import file_sha256
from .observation_belief import array_sha256

METHOD_PROTOCOL_ID = "deform360-causal-response-direct-depth-v14-source"
PRELOCK_PROTOCOL_ID = "deform360-causal-response-direct-depth-v14-physical-prelock"
PRELOCK_PROTOCOL_KIND = "Deform360CausalResponseDirectDepthPhysicalPrelockProtocolV14"
PRELOCK_PROTOCOL_CONTRACT = (
    "deform360-causal-response-direct-depth-physical-prelock-v14"
)
PHYSICAL_ARTIFACT_KIND = "Deform360CausalResponseDirectDepthPhysicalBackboneV14"
PHYSICAL_CONTRACT = "deform360-causal-response-direct-depth-physical-backbone-v14"
PHYSICAL_ARCHIVE_FILENAME = "causal_response_direct_depth_physical_v14.npz"
PHYSICAL_MANIFEST_FILENAME = "causal_response_direct_depth_physical_v14.json"
PHYSICAL_FRAME_COUNT = 76
GRAPH_BASIS_RANK = 8

_ARCHIVE_FIELDS = frozenset(
    {
        "action_support",
        "driven_readout_m",
        "frame_zero_points_m",
        "graph_basis",
        "persistence_prediction_m",
        "physical_prediction_m",
        "zero_action_readout_m",
    }
)
_CASE_DIGEST_FIELDS = (
    "case_hash",
    "object_hash",
    "metadata_sha256",
    "geometry_manifest_artifact_sha256",
    "geometry_manifest_file_sha256",
    "geometry_result_artifact_sha256",
    "geometry_result_file_sha256",
    "runtime_application_artifact_sha256",
    "runtime_application_file_sha256",
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    namespace: bytes,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        namespace
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {source}") from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def _geometry_ledger_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-physical-geometry-v14\0"
        + json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_v14_physical_prelock_protocol(path: str | Path) -> dict[str, Any]:
    """Validate the exact geometry ledger and pre-lock physical contract."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == PRELOCK_PROTOCOL_KIND
        and payload.get("contract") == PRELOCK_PROTOCOL_CONTRACT
        and payload.get("protocol_id") == PRELOCK_PROTOCOL_ID
        and payload.get("method_protocol_id") == METHOD_PROTOCOL_ID
        and payload.get("status")
        == "locked_after_geometry_before_physical_carrier_execution",
        "V14 physical pre-lock protocol identity changed",
    )
    _require(
        payload.get("config_sha256")
        == _canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-physical-prelock-v14\0"
            ),
            digest_key="config_sha256",
        ),
        "V14 physical pre-lock protocol checksum changed",
    )
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and all(
            _valid_digest(parents.get(key))
            for key in (
                "method_protocol_config_sha256",
                "method_protocol_file_sha256",
                "staging_queue_sha256",
                "staging_queue_file_sha256",
                "geometry_protocol_config_sha256",
                "geometry_protocol_file_sha256",
                "runtime_v1_config_sha256",
                "runtime_v1_file_sha256",
                "validation_v1_config_sha256",
                "validation_v1_file_sha256",
                "runtime_v2_config_sha256",
                "runtime_v2_file_sha256",
                "validation_v2_config_sha256",
                "validation_v2_file_sha256",
            )
        ),
        "V14 physical pre-lock parent binding changed",
    )
    numerical = payload.get("numerical_contract")
    _require(
        isinstance(numerical, Mapping)
        and numerical.get("canonical_node_count") == CANONICAL_NODE_COUNT
        and numerical.get("graph_basis_rank") == GRAPH_BASIS_RANK
        and numerical.get("prediction_frame_count") == PHYSICAL_FRAME_COUNT
        and numerical.get("automatic_twin_source") == "frame_zero_geometry_only"
        and numerical.get("future_robot_action_known") is True
        and numerical.get("automatic_twin_inadmissible_fallback")
        == "bit_exact_persistence",
        "V14 physical numerical contract changed",
    )
    cases = payload.get("geometry_cases")
    _require(
        isinstance(cases, list)
        and len(cases) == 12
        and [record.get("queue_rank") for record in cases] == list(range(3, 15)),
        "V14 physical geometry ledger ranks changed",
    )
    for record in cases:
        _require(
            record.get("runtime_contract_version") in {1, 2}
            and (record["runtime_contract_version"] == 1) == (record["queue_rank"] == 3)
            and 128 <= int(record.get("physical_node_count", 0)) <= 10_000
            and 8 <= int(record.get("successful_camera_count", 0)) <= 12
            and all(_valid_digest(record.get(key)) for key in _CASE_DIGEST_FIELDS),
            "V14 physical geometry ledger record changed",
        )
    _require(
        payload.get("geometry_ledger_sha256") == _geometry_ledger_sha256(cases),
        "V14 physical geometry ledger checksum changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("object_observation_frames_used") == [0]
        and boundary.get("known_robot_action_frames_used")
        == list(range(PHYSICAL_FRAME_COUNT))
        and boundary.get("future_object_observation_read") is False
        and boundary.get("prefix_tactile_read") is False
        and boundary.get("identity_or_metric_outcome_read") is False
        and boundary.get("source_lock_required_before_execution") is False
        and boundary.get("source_lock_construction_uses_output_hashes_only") is True
        and boundary.get("plaintext_identity_retained_in_sealed_output") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 physical pre-lock protocol crossed its information boundary",
    )
    return payload


def v14_physical_case_record(
    protocol: Mapping[str, Any],
    queue: Mapping[str, Any] | str | Path,
    *,
    queue_rank: int,
) -> dict[str, Any]:
    """Return one hash-only pre-lock case binding."""

    normalized_queue = validate_v14_staging_queue(queue)
    parents = protocol["parent_artifacts"]
    _require(
        normalized_queue["queue_sha256"] == parents["staging_queue_sha256"],
        "V14 physical pre-lock queue semantic checksum changed",
    )
    _require(
        1 <= queue_rank <= len(normalized_queue["candidates"]),
        "V14 physical queue rank is invalid",
    )
    candidate = normalized_queue["candidates"][queue_rank - 1]
    geometry = next(
        (
            record
            for record in protocol["geometry_cases"]
            if int(record["queue_rank"]) == int(queue_rank)
        ),
        None,
    )
    _require(geometry is not None, "V14 physical rank is not geometry-bound")
    object_id = str(candidate["object_id"])
    episode_id = int(candidate["episode_id"])
    expected_object_hash = deform360_object_hash(object_id)
    expected_case_hash = deform360_v14_case_hash(object_id, episode_id)
    _require(
        geometry["object_hash"] == expected_object_hash
        and geometry["case_hash"] == expected_case_hash
        and geometry["metadata_sha256"] == candidate["metadata_sha256"],
        "V14 physical geometry differs from its queue identity",
    )
    return {
        "queue_rank": int(queue_rank),
        "object_hash": expected_object_hash,
        "case_hash": expected_case_hash,
        "category": str(candidate["category"]),
        "bimanual_value": str(candidate["bimanual"]),
        "metadata_sha256": str(candidate["metadata_sha256"]),
        "physical_node_count": int(geometry["physical_node_count"]),
        "successful_camera_count": int(geometry["successful_camera_count"]),
        "runtime_contract_version": int(geometry["runtime_contract_version"]),
        **{
            key: str(geometry[key])
            for key in _CASE_DIGEST_FIELDS
            if key not in {"case_hash", "object_hash", "metadata_sha256"}
        },
    }


def build_v14_prediction_only_bundle(
    frame_zero_ply: str | Path,
    known_action_archive: str | Path,
    output_path: str | Path,
    *,
    case_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a constant-object, hash-only input for the automatic twin."""

    points, colors = load_frame_zero_ply(frame_zero_ply)
    controllers, action = load_controller_trajectory(known_action_archive)
    _require(
        len(controllers) == PHYSICAL_FRAME_COUNT,
        "V14 controller frame count changed",
    )
    object_points = np.repeat(points[None], PHYSICAL_FRAME_COUNT, axis=0)
    object_colors = np.repeat(colors[None], PHYSICAL_FRAME_COUNT, axis=0)
    observed = np.ones(object_points.shape[:2], dtype=bool)
    marker = {
        "schema_version": 1,
        "protocol_id": PRELOCK_PROTOCOL_ID,
        "queue_rank": int(case_record["queue_rank"]),
        "object_hash": str(case_record["object_hash"]),
        "case_hash": str(case_record["case_hash"]),
        "object_observation_frames_used": [0],
        "known_future_robot_trajectory_used": True,
        "future_object_observations_present": False,
        "future_tactile_used": False,
        "plaintext_object_or_episode_identity_present": False,
        "frame_zero_ply_sha256": file_sha256(frame_zero_ply),
        "known_action_sha256": file_sha256(known_action_archive),
        "action_window": action,
    }
    payload = {
        "object_points": object_points,
        "object_colors": object_colors,
        "object_visibilities": observed,
        "object_motions_valid": observed.copy(),
        "controller_points": controllers,
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
        "prediction_only_input": marker,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "queue_rank": int(case_record["queue_rank"]),
        "object_hash": str(case_record["object_hash"]),
        "case_hash": str(case_record["case_hash"]),
        "frame_count": PHYSICAL_FRAME_COUNT,
        "point_count": int(len(points)),
        "controller_point_count": int(controllers.shape[1]),
        "frame_zero_points_sha256": array_sha256(points),
        "frame_zero_colors_sha256": array_sha256(colors),
        "controller_trajectory_sha256": array_sha256(controllers),
        "output_sha256": file_sha256(destination),
        "action_window": action,
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_observations_present": False,
            "future_tactile_used": False,
            "plaintext_object_or_episode_identity_present": False,
        },
    }


def build_v14_physical_arrays(
    base: Mapping[str, np.ndarray],
    *,
    vertices: np.ndarray,
    springs: np.ndarray,
    readout_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Attach the frozen rank-eight material readout basis."""

    return {
        "action_support": np.asarray(base["action_support"]),
        "driven_readout_m": np.asarray(base["driven_readout_m"]),
        "frame_zero_points_m": np.asarray(base["frame_zero_points_m"]),
        "graph_basis": build_readout_graph_basis(
            vertices,
            springs,
            readout_weights,
            rank=GRAPH_BASIS_RANK,
        ).astype(np.float32),
        "persistence_prediction_m": np.asarray(base["persistence_m"]),
        "physical_prediction_m": np.asarray(base["prediction_m"]),
        "zero_action_readout_m": np.asarray(base["zero_action_readout_m"]),
    }


def _normalized_archive(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    _require(set(arrays) == _ARCHIVE_FIELDS, "V14 physical archive fields changed")
    result = {
        name: np.ascontiguousarray(np.asarray(value)) for name, value in arrays.items()
    }
    physical = np.asarray(result["physical_prediction_m"], dtype=np.float64)
    persistence = np.asarray(
        result["persistence_prediction_m"],
        dtype=np.float64,
    )
    frame_zero = np.asarray(result["frame_zero_points_m"], dtype=np.float64)
    basis = np.asarray(result["graph_basis"], dtype=np.float64)
    _require(
        physical.ndim == 3
        and physical.shape[0] == PHYSICAL_FRAME_COUNT
        and physical.shape[2] == 3
        and persistence.shape == physical.shape
        and frame_zero.shape == physical.shape[1:]
        and np.all(np.isfinite(physical))
        and np.all(np.isfinite(persistence))
        and np.all(np.isfinite(frame_zero)),
        "V14 physical prediction arrays are invalid",
    )
    _require(
        basis.shape == (len(frame_zero), 3, GRAPH_BASIS_RANK)
        and np.all(np.isfinite(basis)),
        "V14 physical graph basis is invalid",
    )
    for name in ("driven_readout_m", "zero_action_readout_m"):
        values = np.asarray(result[name], dtype=np.float64)
        _require(
            values.shape == physical.shape and np.all(np.isfinite(values)),
            f"V14 {name} is invalid",
        )
        _require(
            np.array_equal(values[0], frame_zero),
            f"V14 {name} changed frame-zero material identities",
        )
    support = np.asarray(result["action_support"], dtype=np.float64)
    _require(
        support.shape == (len(frame_zero),)
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "V14 physical action support is invalid",
    )
    _require(
        np.array_equal(physical[0], frame_zero)
        and np.array_equal(persistence[0], frame_zero),
        "V14 physical backbone changed frame-zero material identities",
    )
    flat_basis = basis.reshape(-1, GRAPH_BASIS_RANK)
    _require(
        np.allclose(
            flat_basis.T @ flat_basis,
            np.eye(GRAPH_BASIS_RANK),
            atol=1e-6,
        ),
        "V14 physical graph basis is not orthonormal",
    )
    return result


def write_v14_physical_artifacts(
    output_dir: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    prelock_protocol_path: str | Path,
    case_record: Mapping[str, Any],
    physical_mode: str,
    code_revision: str,
    input_files: Mapping[str, str | Path],
    runtime_provenance: Mapping[str, Any],
    fallback_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal one pre-lock physical carrier without plaintext identity."""

    protocol = load_v14_physical_prelock_protocol(prelock_protocol_path)
    _require(
        len(code_revision) == 40
        and all(character in "0123456789abcdef" for character in code_revision),
        "V14 physical code revision is invalid",
    )
    _require(
        physical_mode in {"warp_twin", "automatic_twin_persistence_fallback"},
        "V14 physical mode changed",
    )
    _require(
        (physical_mode == "automatic_twin_persistence_fallback")
        == (fallback_diagnostics is not None),
        "V14 physical fallback diagnostics disagree with mode",
    )
    _require(
        all(
            _valid_digest(case_record.get(key))
            for key in ("object_hash", "case_hash", "metadata_sha256")
        ),
        "V14 physical case record is not hash-only",
    )
    normalized = _normalized_archive(arrays)
    output = Path(output_dir).resolve()
    _require(not output.exists(), "V14 physical output directory already exists")
    output.mkdir(parents=True)
    archive = output / PHYSICAL_ARCHIVE_FILENAME
    temporary = output / (PHYSICAL_ARCHIVE_FILENAME + ".tmp.npz")
    np.savez_compressed(temporary, **normalized)
    temporary.replace(archive)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PHYSICAL_ARTIFACT_KIND,
        "contract": PHYSICAL_CONTRACT,
        "protocol_id": PRELOCK_PROTOCOL_ID,
        "method_protocol_id": METHOD_PROTOCOL_ID,
        "physical_prelock_config_sha256": protocol["config_sha256"],
        "queue_rank": int(case_record["queue_rank"]),
        "object_hash": str(case_record["object_hash"]),
        "case_hash": str(case_record["case_hash"]),
        "category": str(case_record["category"]),
        "bimanual_value": str(case_record["bimanual_value"]),
        "metadata_sha256": str(case_record["metadata_sha256"]),
        "physical_node_count": int(case_record["physical_node_count"]),
        "successful_camera_count": int(case_record["successful_camera_count"]),
        "physical_mode": physical_mode,
        "physical_admitted": physical_mode == "warp_twin",
        "code_revision": code_revision,
        "fallback_diagnostics": (
            None if fallback_diagnostics is None else dict(fallback_diagnostics)
        ),
        "physical_archive": {
            "filename": PHYSICAL_ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive),
            "array_sha256": {
                name: array_sha256(value) for name, value in sorted(normalized.items())
            },
        },
        "inputs_sha256": {
            "physical_prelock_protocol": file_sha256(prelock_protocol_path),
            **{name: file_sha256(path) for name, path in sorted(input_files.items())},
        },
        "runtime_provenance": dict(runtime_provenance),
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "prefix_or_future_object_rgb_read": False,
            "prefix_or_future_object_geometry_read": False,
            "prefix_or_future_object_track_read": False,
            "prefix_or_future_tactile_read": False,
            "identity_or_metric_outcome_read": False,
            "source_lock_read": False,
            "plaintext_object_or_episode_identity_retained": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    manifest["artifact_sha256"] = _canonical_sha256(
        manifest,
        namespace=b"deform360-causal-response-direct-depth-physical-v14\0",
        digest_key="artifact_sha256",
    )
    (output / PHYSICAL_MANIFEST_FILENAME).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_v14_physical_artifacts(
        output,
        prelock_protocol_path=prelock_protocol_path,
    )
    return manifest


def validate_v14_physical_artifacts(
    output_dir: str | Path,
    *,
    prelock_protocol_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate one pre-lock physical carrier and its numeric archive."""

    output = Path(output_dir).resolve()
    manifest = _read_json(output / PHYSICAL_MANIFEST_FILENAME)
    _require(
        manifest.get("artifact_kind") == PHYSICAL_ARTIFACT_KIND
        and manifest.get("contract") == PHYSICAL_CONTRACT
        and manifest.get("protocol_id") == PRELOCK_PROTOCOL_ID
        and manifest.get("method_protocol_id") == METHOD_PROTOCOL_ID,
        "V14 physical artifact identity changed",
    )
    _require(
        manifest.get("artifact_sha256")
        == _canonical_sha256(
            manifest,
            namespace=b"deform360-causal-response-direct-depth-physical-v14\0",
            digest_key="artifact_sha256",
        ),
        "V14 physical manifest checksum changed",
    )
    if prelock_protocol_path is not None:
        protocol = load_v14_physical_prelock_protocol(prelock_protocol_path)
        _require(
            manifest.get("physical_prelock_config_sha256") == protocol["config_sha256"]
            and manifest.get("inputs_sha256", {}).get("physical_prelock_protocol")
            == file_sha256(prelock_protocol_path),
            "V14 physical artifact uses another pre-lock protocol",
        )
    mode = manifest.get("physical_mode")
    _require(
        mode in {"warp_twin", "automatic_twin_persistence_fallback"}
        and manifest.get("physical_admitted") is (mode == "warp_twin")
        and (manifest.get("fallback_diagnostics") is not None)
        == (mode == "automatic_twin_persistence_fallback"),
        "V14 physical artifact status is invalid",
    )
    _require(
        all(
            _valid_digest(manifest.get(key))
            for key in ("object_hash", "case_hash", "metadata_sha256")
        ),
        "V14 physical artifact retained invalid identity provenance",
    )
    archive = output / PHYSICAL_ARCHIVE_FILENAME
    _require(
        manifest.get("physical_archive", {}).get("file_sha256") == file_sha256(archive),
        "V14 physical archive checksum changed",
    )
    with np.load(archive, allow_pickle=False) as stored:
        arrays = _normalized_archive(
            {name: np.asarray(stored[name]) for name in stored.files}
        )
    _require(
        manifest["physical_archive"]["array_sha256"]
        == {name: array_sha256(value) for name, value in sorted(arrays.items())},
        "V14 physical archive array checksum changed",
    )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("prefix_or_future_object_rgb_read") is False
        and boundary.get("prefix_or_future_object_geometry_read") is False
        and boundary.get("prefix_or_future_object_track_read") is False
        and boundary.get("prefix_or_future_tactile_read") is False
        and boundary.get("identity_or_metric_outcome_read") is False
        and boundary.get("source_lock_read") is False
        and boundary.get("plaintext_object_or_episode_identity_retained") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 physical artifact crossed its information boundary",
    )
    return manifest, arrays


__all__ = [
    "GRAPH_BASIS_RANK",
    "METHOD_PROTOCOL_ID",
    "PHYSICAL_ARCHIVE_FILENAME",
    "PHYSICAL_ARTIFACT_KIND",
    "PHYSICAL_CONTRACT",
    "PHYSICAL_FRAME_COUNT",
    "PHYSICAL_MANIFEST_FILENAME",
    "PRELOCK_PROTOCOL_CONTRACT",
    "PRELOCK_PROTOCOL_ID",
    "PRELOCK_PROTOCOL_KIND",
    "build_v14_physical_arrays",
    "build_v14_prediction_only_bundle",
    "load_v14_physical_prelock_protocol",
    "v14_physical_case_record",
    "validate_v14_physical_artifacts",
    "write_v14_physical_artifacts",
]
