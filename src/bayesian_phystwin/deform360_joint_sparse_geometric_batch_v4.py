"""Geometric factor construction for Deform360 joint-sparse v4."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from .deform360_joint_sparse_geometric_candidates_v4 import _Candidate
from .deform360_joint_sparse_geometric_common_v4 import (
    EXPECTED_DEVELOPMENT_BOUNDARY,
    _array_digest,
    _canonical_bytes,
    _require,
    _sha256_file,
)
from .deform360_joint_sparse_observability_v4 import (
    Deform360JointSparseFactorBatchV4,
)


def _mode_matrices() -> np.ndarray:
    root2 = math.sqrt(2.0)
    root6 = math.sqrt(6.0)
    return np.asarray(
        [
            [[1.0 / root2, 0.0, 0.0], [0.0, -1.0 / root2, 0.0], [0.0, 0.0, 0.0]],
            [[1.0 / root6, 0.0, 0.0], [0.0, 1.0 / root6, 0.0], [0.0, 0.0, -2.0 / root6]],
            [[0.0, 1.0 / root2, 0.0], [1.0 / root2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0 / root2], [0.0, 0.0, 0.0], [1.0 / root2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / root2], [0.0, 1.0 / root2, 0.0]],
        ],
        dtype=np.float64,
    )


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = map(float, value)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _covariance(point: np.ndarray, camera_center: np.ndarray, *, lateral: float, axial: float) -> np.ndarray:
    ray = np.asarray(point, dtype=np.float64) - np.asarray(camera_center, dtype=np.float64)
    norm = float(np.linalg.norm(ray))
    _require(np.isfinite(norm) and norm > 1e-9, "metric point coincides with camera center")
    direction = ray / norm
    return lateral**2 * np.eye(3) + (axial**2 - lateral**2) * np.outer(direction, direction)


def _hash_record(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _build_object_batch(
    *,
    candidates: Sequence[_Candidate],
    selection_artifact_sha256: str,
    visual_provider_lock_id: str,
    implementation_revision: str,
    object_id: str,
    episode_id: int,
    stratum: str,
    excluded_factor_count: int,
    source_artifacts: Mapping[str, str],
    policy: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Deform360JointSparseFactorBatchV4:
    _require(bool(candidates), f"object {object_id!r} has no active partial factors")
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.camera_id,
            item.window_id,
            item.frame,
            item.spatial_cluster_id,
            item.row,
            item.column,
            item.job_id,
        ),
    )
    unique_points: dict[tuple[int, str], np.ndarray] = {}
    for item in ordered:
        unique_points.setdefault(
            (item.frame, item.spatial_cluster_id), item.point_world_m
        )
    geometry = np.asarray(list(unique_points.values()), dtype=np.float64)
    centroid = np.mean(geometry, axis=0)
    rms = float(np.sqrt(np.mean(np.sum(np.square(geometry - centroid), axis=1))))
    rms = max(rms, cast(float, policy["minimum_object_rms_radius_m"]))
    modes = _mode_matrices()
    count = len(ordered)
    state = np.zeros((count, 3, 5), dtype=np.float64)
    local_gauge = np.zeros((count, 3, 7), dtype=np.float64)
    covariance = np.zeros((count, 3, 3), dtype=np.float64)
    cameras = sorted({item.camera_id for item in ordered})
    camera_index = {camera: index + 1 for index, camera in enumerate(cameras)}
    gauge_ids = ("global-similarity-root-v1",) + tuple(
        f"camera-similarity-v1:{camera}" for camera in cameras
    )
    gauge_indices = np.empty(count, dtype=np.int64)
    factor_ids: list[str] = []
    camera_ids: list[str] = []
    window_ids: list[str] = []
    cluster_ids: list[str] = []
    group_ids: list[str] = []
    for index, item in enumerate(ordered):
        normalized = (item.point_world_m - centroid) / rms
        for mode_index, mode in enumerate(modes):
            state[index, :, mode_index] = mode @ normalized
        local_gauge[index, :, :3] = np.eye(3)
        local_gauge[index, :, 3:6] = -_skew(normalized)
        local_gauge[index, :, 6] = normalized
        covariance[index] = _covariance(
            item.point_world_m,
            item.camera_center_world_m,
            lateral=cast(float, policy["lateral_observation_std_m"]),
            axial=cast(float, policy["axial_observation_std_m"]),
        )
        gauge_indices[index] = camera_index[item.camera_id]
        point_digest = _array_digest(
            np.asarray(item.point_world_m, dtype=np.dtype("<f8"))
        )
        factor_ids.append(
            _hash_record(
                {
                    "schema": "bayesian-phystwin.deform360-joint-sparse-geometric-factor",
                    "schema_version": 1,
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "job_id": item.job_id,
                    "camera_id": item.camera_id,
                    "window_id": item.window_id,
                    "frame": item.frame,
                    "row": item.row,
                    "column": item.column,
                    "spatial_cluster_id": item.spatial_cluster_id,
                    "correlation_group_id": item.correlation_group_id,
                    "point_world_m_sha256": point_digest,
                    "support_digest": item.support_digest,
                }
            )
        )
        camera_ids.append(item.camera_id)
        window_ids.append(item.window_id)
        cluster_ids.append(item.spatial_cluster_id)
        group_ids.append(item.correlation_group_id)
    group_count = Counter(group_ids)
    composite = np.asarray([1.0 / group_count[group] for group in group_ids])
    gauge_count = len(gauge_ids)
    parents = np.zeros(gauge_count, dtype=np.int64)
    parents[0] = -1
    transitions = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    transitions[1:] = np.eye(7)
    scales = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    scales[0] = np.eye(7) * cast(float, policy["root_gauge_prior_std_m"])
    scales[1:] = np.eye(7) * cast(float, policy["camera_gauge_innovation_std_m"])
    query = np.eye(5, dtype=np.float64)
    gauge_prior_id = _hash_record(
        {
            "schema": "bayesian-phystwin.deform360-joint-sparse-camera-gauge-tree",
            "schema_version": 1,
            "semantics": policy["gauge_semantics"],
            "gauge_ids": list(gauge_ids),
            "parent_indices_sha256": _array_digest(parents),
            "transition_matrices_sha256": _array_digest(transitions),
            "innovation_scale_tril_sha256": _array_digest(scales),
            "materializer_policy_id": policy["artifact_id"],
        }
    )
    observation_artifact_id = _hash_record(
        {
            "schema": "bayesian-phystwin.deform360-joint-sparse-geometric-observation",
            "schema_version": 1,
            "object_id": object_id,
            "episode_id": episode_id,
            "factor_ids": factor_ids,
            "camera_ids": camera_ids,
            "window_ids": window_ids,
            "spatial_cluster_ids": cluster_ids,
            "correlation_group_ids": group_ids,
            "observation_covariance_m2_sha256": _array_digest(covariance),
            "source_artifacts": dict(sorted(source_artifacts.items())),
            "materializer_policy_id": policy["artifact_id"],
        }
    )
    linearization_artifact_id = _hash_record(
        {
            "schema": "bayesian-phystwin.deform360-joint-sparse-geometric-linearization",
            "schema_version": 1,
            "basis_semantics": policy["basis_semantics"],
            "gauge_semantics": policy["gauge_semantics"],
            "query_semantics": policy["query_semantics"],
            "object_id": object_id,
            "episode_id": episode_id,
            "centroid_world_m": centroid.tolist(),
            "rms_radius_m": rms,
            "state_jacobian_sha256": _array_digest(state),
            "local_gauge_jacobian_sha256": _array_digest(local_gauge),
            "gauge_indices_sha256": _array_digest(gauge_indices),
            "query_jacobian_sha256": _array_digest(query),
            "gauge_prior_id": gauge_prior_id,
            "materializer_policy_id": policy["artifact_id"],
        }
    )
    shared = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    view = np.zeros((count, 3, 0), dtype=np.float64)
    object_metadata = {
        **dict(metadata),
        "materializer_schema": MATERIALIZER_SCHEMA,
        "materializer_schema_version": MATERIALIZER_VERSION,
        "materializer_policy_id": policy["artifact_id"],
        "basis_semantics": policy["basis_semantics"],
        "gauge_semantics": policy["gauge_semantics"],
        "query_semantics": policy["query_semantics"],
        "centroid_world_m": centroid.tolist(),
        "rms_radius_m": rms,
        "unique_frame_world_cluster_count": len(unique_points),
        "prediction_support_masks_used": True,
        "prediction_point_values_used": False,
        "prediction_residuals_used": False,
        "released_robot_metric_points_used": True,
        "development_cohort_only": True,
    }
    return Deform360JointSparseFactorBatchV4(
        selection_artifact_sha256=selection_artifact_sha256,
        visual_provider_lock_id=visual_provider_lock_id,
        observation_artifact_id=observation_artifact_id,
        linearization_artifact_id=linearization_artifact_id,
        implementation_revision=implementation_revision,
        object_id=object_id,
        episode_id=episode_id,
        stratum=cast(Any, stratum),
        factor_ids=tuple(factor_ids),
        camera_ids=tuple(camera_ids),
        window_ids=tuple(window_ids),
        spatial_cluster_ids=tuple(cluster_ids),
        correlation_group_ids=tuple(group_ids),
        gauge_ids=gauge_ids,
        gauge_prior_id=gauge_prior_id,
        observation_covariance_m2=covariance,
        state_jacobian=state,
        local_gauge_jacobian=local_gauge,
        gauge_indices=gauge_indices,
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        query_jacobian=query,
        prior_reliability=np.ones(count),
        association_probability=np.ones(count),
        composite_weight=composite,
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        excluded_factor_count=excluded_factor_count,
        source_artifacts=source_artifacts,
        information_boundary=EXPECTED_DEVELOPMENT_BOUNDARY,
        metadata=object_metadata,
    )


def _npz_payload(batch: Deform360JointSparseFactorBatchV4) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in batch.arrays().items()}


def _record_for_file(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _write_checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text(
        "".join(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="ascii",
    )
