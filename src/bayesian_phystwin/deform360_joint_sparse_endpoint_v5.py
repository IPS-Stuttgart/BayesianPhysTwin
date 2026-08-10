"""Exact held-out-view geometry endpoint for the public Deform360 v5 study."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    immutable_array,
    integer_array,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
)
from .deform360_joint_sparse_prediction_v5 import RAW_METHOD_IDS

ENDPOINT_SCHEMA: Final = "bayesian-phystwin.deform360-joint-sparse-endpoint"
ENDPOINT_VERSION: Final = 1
ENDPOINT_SEMANTICS: Final = (
    "reserved-view-masked-depth-surface-symmetric-chamfer-v1"
)
RESERVED_VIEW_RANKING_DOMAIN: Final = b"v5-endpoint-view-v1"
TARGET_PIXEL_RANKING_DOMAIN: Final = b"v5-endpoint-target-pixel-v1"
EvaluationRoleV5 = Literal["development_source", "independent_confirmation"]
_ENDPOINT_REPORT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "object_id",
        "episode_id",
        "stratum",
        "prediction_seal_id",
        "evaluation_role",
        "opening_authorization_id",
        "endpoint_config_id",
        "reserved_camera_ids",
        "frame_indices",
        "cell_count_per_method",
        "method_loss_mm",
        "method_cell_loss_mm",
        "prediction_support_failure_count",
        "technical_failure",
        "missing_target_cells",
        "source_artifact_ids",
        "information_boundary",
        "endpoint_config",
        "endpoint_report_id",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    return result


def _camera_id(value: object) -> str:
    result = nonempty_string(value, name="camera_id")
    _require(result.strip() == result and "\x00" not in result, "camera_id is not canonical")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be literal strings")
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    _require(
        np.isfinite(result) and result >= 0.0,
        f"{name} must be a finite nonnegative number",
    )
    return result


@dataclass(frozen=True, slots=True)
class Deform360JointSparseEndpointConfigV5:
    """Frozen numerical choices for one object-level geometry score."""

    evaluation_frame_range_half_open: tuple[int, int] = (58, 76)
    reserved_view_count: int = 2
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 2.5
    maximum_target_points_per_frame_view: int = 4096
    minimum_target_points_per_frame_view: int = 32
    prediction_occlusion_tolerance_m: float = 0.020
    technical_failure_penalty_mm: float = 1000.0
    distance_chunk_size: int = 512

    def __post_init__(self) -> None:
        _require(
            type(self.evaluation_frame_range_half_open) is tuple
            and len(self.evaluation_frame_range_half_open) == 2
            and all(
                type(value) is int
                for value in self.evaluation_frame_range_half_open
            ),
            "evaluation frame range must contain two exact integers",
        )
        start, stop = self.evaluation_frame_range_half_open
        _require(0 <= start < stop, "evaluation frame range is invalid")
        for name, value in (
            ("reserved_view_count", self.reserved_view_count),
            (
                "maximum_target_points_per_frame_view",
                self.maximum_target_points_per_frame_view,
            ),
            (
                "minimum_target_points_per_frame_view",
                self.minimum_target_points_per_frame_view,
            ),
            ("distance_chunk_size", self.distance_chunk_size),
        ):
            _require(
                type(value) is int and value >= 1,
                f"{name} must be a positive integer",
            )
        _require(
            self.minimum_target_points_per_frame_view
            <= self.maximum_target_points_per_frame_view,
            "minimum target count exceeds its maximum",
        )
        positive = (
            self.minimum_depth_m,
            self.maximum_depth_m,
            self.prediction_occlusion_tolerance_m,
            self.technical_failure_penalty_mm,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "endpoint scales must be finite and positive",
        )
        _require(
            self.minimum_depth_m < self.maximum_depth_m,
            "endpoint depth interval is empty",
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "evaluation_frame_range_half_open": list(
                self.evaluation_frame_range_half_open
            ),
            "reserved_view_count": self.reserved_view_count,
            "minimum_depth_m": self.minimum_depth_m,
            "maximum_depth_m": self.maximum_depth_m,
            "maximum_target_points_per_frame_view": (
                self.maximum_target_points_per_frame_view
            ),
            "minimum_target_points_per_frame_view": (
                self.minimum_target_points_per_frame_view
            ),
            "prediction_occlusion_tolerance_m": (
                self.prediction_occlusion_tolerance_m
            ),
            "technical_failure_penalty_mm": self.technical_failure_penalty_mm,
            "distance_chunk_size": self.distance_chunk_size,
            "reserved_view_ranking": (
                "ascending-sha256(object_id\\0camera_id\\0v5-endpoint-view-v1)"
            ),
            "target_pixel_subsampling": (
                "ascending-sha256(object_id\\0camera_id\\0frame\\0row\\0column"
                "\\0v5-endpoint-target-pixel-v1)"
            ),
            "projection_pixel_rounding": "floor-coordinate-plus-one-half-v1",
            "target_geometry": (
                "official-expected-depth-intersected-with-object-mask-and-valid-depth"
            ),
            "prediction_visibility": (
                "projected-inside-target-mask-and-not-behind-target-depth-plus-tolerance"
            ),
            "cell_loss": "half-sum-of-two-directed-mean-euclidean-distances-mm",
            "object_aggregation": "arithmetic-mean-over-all-frame-view-cells",
            "missing_cell_action": "retained-object-fixed-technical-failure-penalty",
        }

    @property
    def config_id(self) -> str:
        return content_id(self.descriptor())


@dataclass(frozen=True, slots=True)
class Deform360ReservedViewGeometryV5:
    """One reserved camera's released future depth geometry."""

    object_id: str
    episode_id: int
    camera_id: str
    frame_indices: np.ndarray
    depth_m: np.ndarray
    object_mask: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    source_artifact_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        object_id = nonempty_string(self.object_id, name="object_id")
        _require(object_id.strip() == object_id, "object_id is not canonical")
        _require(
            type(self.episode_id) is int and self.episode_id >= 0,
            "episode_id must be a nonnegative integer",
        )
        camera_id = _camera_id(self.camera_id)
        frames = integer_array(self.frame_indices, name="frame_indices")
        _require(frames.ndim == 1 and len(frames) > 0, "frame_indices must be 1-D")
        _require(
            np.array_equal(frames, np.unique(frames)),
            "frame_indices must be strictly increasing",
        )
        depth = np.asarray(self.depth_m)
        _require(
            depth.dtype.kind == "f" and depth.ndim == 3 and len(depth) == len(frames),
            "depth_m must have shape (F,H,W) and floating dtype",
        )
        mask = np.asarray(self.object_mask)
        _require(
            mask.dtype.kind == "b" and mask.shape == depth.shape,
            "object_mask must be Boolean and match depth_m",
        )
        intrinsics = _finite_array(self.intrinsics, name="intrinsics", ndim=2)
        transform = _finite_array(
            self.camera_to_world,
            name="camera_to_world",
            ndim=2,
        )
        _require(intrinsics.shape == (3, 3), "intrinsics must have shape (3,3)")
        _require(transform.shape == (4, 4), "camera_to_world must have shape (4,4)")
        _require(
            np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-10, rtol=0.0)
            and intrinsics[0, 0] > 0.0
            and intrinsics[1, 1] > 0.0,
            "intrinsics changed convention",
        )
        _require(
            np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0.0),
            "camera_to_world is not homogeneous",
        )
        rotation = transform[:3, :3]
        _require(
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0)
            and np.linalg.det(rotation) > 0.0,
            "camera_to_world rotation is invalid",
        )
        sources = source_artifact_mapping(
            self.source_artifact_ids,
            name="source_artifact_ids",
        )
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "frame_indices", immutable_array(frames, dtype=np.int64))
        object.__setattr__(self, "depth_m", immutable_array(depth, dtype=depth.dtype))
        object.__setattr__(self, "object_mask", immutable_array(mask, dtype=bool))
        object.__setattr__(self, "intrinsics", immutable_array(intrinsics, dtype=np.float64))
        object.__setattr__(
            self,
            "camera_to_world",
            immutable_array(transform, dtype=np.float64),
        )
        object.__setattr__(self, "source_artifact_ids", sources)


def select_reserved_endpoint_views_v5(
    object_id: str,
    camera_ids: Sequence[str],
    *,
    count: int = 2,
) -> tuple[str, ...]:
    """Select endpoint cameras using identities only."""

    canonical_object = nonempty_string(object_id, name="object_id")
    _require(canonical_object.strip() == canonical_object, "object_id is not canonical")
    _require(type(count) is int and count >= 1, "count must be a positive integer")
    cameras = tuple(_camera_id(camera) for camera in camera_ids)
    _require(len(cameras) == len(set(cameras)), "camera IDs repeat")
    _require(len(cameras) >= count, "too few cameras for endpoint reservation")
    ranked = sorted(
        cameras,
        key=lambda camera: (
            hashlib.sha256(
                canonical_object.encode("utf-8")
                + b"\0"
                + camera.encode("utf-8")
                + b"\0"
                + RESERVED_VIEW_RANKING_DOMAIN
            ).digest(),
            camera,
        ),
    )
    return tuple(ranked[:count])


def _target_pixel_indices(
    valid: np.ndarray,
    *,
    object_id: str,
    camera_id: str,
    frame: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(valid)
    if len(rows) <= maximum:
        return rows, columns
    prefix = (
        object_id.encode("utf-8")
        + b"\0"
        + camera_id.encode("utf-8")
        + b"\0"
        + str(frame).encode("ascii")
        + b"\0"
    )
    order = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            prefix
            + str(int(rows[index])).encode("ascii")
            + b"\0"
            + str(int(columns[index])).encode("ascii")
            + b"\0"
            + TARGET_PIXEL_RANKING_DOMAIN
        ).digest(),
    )[:maximum]
    selected = np.asarray(sorted(order), dtype=np.int64)
    return rows[selected], columns[selected]


def _backproject(
    depth: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    z = np.asarray(depth[rows, columns], dtype=np.float64)
    x = (columns.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    camera_points = np.column_stack((x, y, z))
    return camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def _visible_prediction(
    points_world_m: np.ndarray,
    *,
    depth: np.ndarray,
    valid_target: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    config: Deform360JointSparseEndpointConfigV5,
) -> np.ndarray:
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    camera_points = (points_world_m - translation) @ rotation
    z = camera_points[:, 2]
    finite = np.all(np.isfinite(camera_points), axis=1)
    depth_valid = (
        finite & (z >= config.minimum_depth_m) & (z <= config.maximum_depth_m)
    )
    safe_z = np.where(depth_valid, z, 1.0)
    u = intrinsics[0, 0] * camera_points[:, 0] / safe_z + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera_points[:, 1] / safe_z + intrinsics[1, 2]
    columns = np.floor(u + 0.5).astype(np.int64)
    rows = np.floor(v + 0.5).astype(np.int64)
    height, width = depth.shape
    inside = (
        depth_valid
        & (rows >= 0)
        & (rows < height)
        & (columns >= 0)
        & (columns < width)
    )
    indices = np.nonzero(inside)[0]
    if not len(indices):
        return np.zeros((0, 3), dtype=np.float64)
    rows_inside = rows[indices]
    columns_inside = columns[indices]
    supported = valid_target[rows_inside, columns_inside]
    target_depth = depth[rows_inside, columns_inside]
    supported &= z[indices] <= target_depth + config.prediction_occlusion_tolerance_m
    return np.asarray(points_world_m[indices[supported]], dtype=np.float64)


def _directed_mean_distance(
    source: np.ndarray,
    target: np.ndarray,
    *,
    chunk_size: int,
) -> float:
    minima: list[np.ndarray] = []
    for start in range(0, len(source), chunk_size):
        block = source[start : start + chunk_size]
        squared = np.sum(np.square(block[:, None, :] - target[None, :, :]), axis=2)
        minima.append(np.sqrt(np.min(squared, axis=1)))
    return float(np.mean(np.concatenate(minima)))


def _symmetric_chamfer_mm(
    first: np.ndarray,
    second: np.ndarray,
    *,
    chunk_size: int,
) -> float:
    return 500.0 * (
        _directed_mean_distance(first, second, chunk_size=chunk_size)
        + _directed_mean_distance(second, first, chunk_size=chunk_size)
    )


def score_deform360_joint_sparse_endpoint_v5(
    *,
    object_id: str,
    episode_id: int,
    stratum: str,
    prediction_seal_id: str,
    trajectories_m: Mapping[str, np.ndarray],
    reserved_views: Sequence[Deform360ReservedViewGeometryV5],
    all_camera_ids: Sequence[str],
    evaluation_role: EvaluationRoleV5,
    opening_authorization_id: str | None = None,
    config: Deform360JointSparseEndpointConfigV5 | None = None,
) -> dict[str, Any]:
    """Score all raw methods after their prediction seal exists."""

    cfg = config or Deform360JointSparseEndpointConfigV5()
    canonical_object = nonempty_string(object_id, name="object_id")
    _require(type(episode_id) is int and episode_id >= 0, "episode_id is invalid")
    _require(stratum in {"sheet", "volumetric"}, "stratum changed")
    seal_id = sha256_digest(prediction_seal_id, name="prediction_seal_id")
    _require(
        evaluation_role in {"development_source", "independent_confirmation"},
        "evaluation_role changed",
    )
    authorization: str | None = None
    if evaluation_role == "development_source":
        _require(
            opening_authorization_id is None,
            "development scoring must not carry confirmation authorization",
        )
    else:
        authorization = sha256_digest(
            opening_authorization_id,
            name="opening_authorization_id",
        )
    _require(set(trajectories_m) == set(RAW_METHOD_IDS), "method roster changed")
    selected = select_reserved_endpoint_views_v5(
        canonical_object,
        all_camera_ids,
        count=cfg.reserved_view_count,
    )
    by_camera = {view.camera_id: view for view in reserved_views}
    _require(
        len(by_camera) == len(reserved_views) and tuple(sorted(by_camera)) == tuple(sorted(selected)),
        "reserved endpoint view roster changed",
    )
    start, stop = cfg.evaluation_frame_range_half_open
    expected_frames: np.ndarray = np.arange(start, stop, dtype=np.int64)
    source_artifacts: dict[str, str] = {}
    for camera_id in selected:
        view = by_camera[camera_id]
        _require(
            view.object_id == canonical_object and view.episode_id == episode_id,
            "reserved endpoint identity changed",
        )
        _require(
            np.array_equal(view.frame_indices, expected_frames),
            "reserved endpoint frame roster changed",
        )
        for path, digest in view.source_artifact_ids.items():
            key = f"{camera_id}/{path}"
            _require(key not in source_artifacts, "endpoint source path repeats")
            source_artifacts[key] = digest

    prepared: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    missing_cells: list[dict[str, Any]] = []
    for camera_id in selected:
        view = by_camera[camera_id]
        for local_index, frame in enumerate(expected_frames):
            depth = np.asarray(view.depth_m[local_index], dtype=np.float64)
            valid = (
                np.asarray(view.object_mask[local_index], dtype=bool)
                & np.isfinite(depth)
                & (depth >= cfg.minimum_depth_m)
                & (depth <= cfg.maximum_depth_m)
            )
            rows, columns = _target_pixel_indices(
                valid,
                object_id=canonical_object,
                camera_id=camera_id,
                frame=int(frame),
                maximum=cfg.maximum_target_points_per_frame_view,
            )
            if len(rows) < cfg.minimum_target_points_per_frame_view:
                missing_cells.append(
                    {
                        "camera_id": camera_id,
                        "frame": int(frame),
                        "valid_target_point_count": int(len(rows)),
                    }
                )
                continue
            target = _backproject(
                depth,
                rows,
                columns,
                view.intrinsics,
                view.camera_to_world,
            )
            prepared[(camera_id, int(frame))] = (depth, valid, target)

    technical_failure = bool(missing_cells)
    per_method_cells: dict[str, list[float]] = {method_id: [] for method_id in RAW_METHOD_IDS}
    prediction_support_failures: dict[str, int] = {
        method_id: 0 for method_id in RAW_METHOD_IDS
    }
    if not technical_failure:
        for method_id in RAW_METHOD_IDS:
            trajectory = np.asarray(trajectories_m[method_id])
            _require(
                trajectory.dtype.kind == "f"
                and trajectory.ndim == 3
                and trajectory.shape[0] >= stop
                and trajectory.shape[2] == 3
                and np.all(np.isfinite(trajectory)),
                f"trajectory {method_id} changed shape or values",
            )
            for camera_id in selected:
                view = by_camera[camera_id]
                for frame in expected_frames:
                    depth, valid, target = prepared[(camera_id, int(frame))]
                    predicted = _visible_prediction(
                        np.asarray(trajectory[int(frame)], dtype=np.float64),
                        depth=depth,
                        valid_target=valid,
                        intrinsics=view.intrinsics,
                        camera_to_world=view.camera_to_world,
                        config=cfg,
                    )
                    if not len(predicted):
                        prediction_support_failures[method_id] += 1
                        per_method_cells[method_id].append(
                            cfg.technical_failure_penalty_mm
                        )
                    else:
                        per_method_cells[method_id].append(
                            _symmetric_chamfer_mm(
                                predicted,
                                target,
                                chunk_size=cfg.distance_chunk_size,
                            )
                        )
    else:
        expected_count = cfg.reserved_view_count * (stop - start)
        for method_id in RAW_METHOD_IDS:
            per_method_cells[method_id] = [
                cfg.technical_failure_penalty_mm
            ] * expected_count

    method_losses = {
        method_id: float(np.mean(per_method_cells[method_id]))
        for method_id in RAW_METHOD_IDS
    }
    report: dict[str, Any] = {
        "schema": ENDPOINT_SCHEMA,
        "schema_version": ENDPOINT_VERSION,
        "semantics": ENDPOINT_SEMANTICS,
        "object_id": canonical_object,
        "episode_id": episode_id,
        "stratum": stratum,
        "prediction_seal_id": seal_id,
        "evaluation_role": evaluation_role,
        "opening_authorization_id": authorization,
        "endpoint_config_id": cfg.config_id,
        "reserved_camera_ids": list(selected),
        "frame_indices": expected_frames.tolist(),
        "cell_count_per_method": len(per_method_cells[RAW_METHOD_IDS[0]]),
        "method_loss_mm": method_losses,
        "method_cell_loss_mm": per_method_cells,
        "prediction_support_failure_count": prediction_support_failures,
        "technical_failure": technical_failure,
        "missing_target_cells": missing_cells,
        "source_artifact_ids": dict(sorted(source_artifacts.items())),
        "information_boundary": {
            "prediction_seal_verified_before_target_open": True,
            "reserved_views_contributed_likelihood": False,
            "future_geometry_used_for_prediction": False,
            "future_geometry_used_for_scoring_after_seal": True,
            "tactile_used_to_define_target": False,
            "development_suffix_used_for_scoring": (
                evaluation_role == "development_source"
            ),
            "confirmation_payloads_opened_for_authorized_scoring": (
                evaluation_role == "independent_confirmation"
            ),
            "confirmation_target_outcomes_used_for_scoring": (
                evaluation_role == "independent_confirmation"
            ),
            "confirmation_outcomes_used_before_authorization": False,
            "human_approval_required": False,
            "new_measurements_required": False,
        },
        "endpoint_config": cfg.descriptor(),
    }
    report["endpoint_report_id"] = content_id(report)
    return plain_json(frozen_finite_json_mapping(report, name="endpoint report"))


def validate_deform360_joint_sparse_endpoint_report_v5(
    report: Mapping[str, Any],
    *,
    expected_evaluation_role: EvaluationRoleV5 | None = None,
    expected_opening_authorization_id: str | None = None,
) -> Mapping[str, Any]:
    """Validate one content-addressed endpoint report without coercion."""

    payload = _mapping(report, name="endpoint report")
    require_exact_fields(
        payload,
        expected=_ENDPOINT_REPORT_FIELDS,
        name="endpoint report",
    )
    _require(payload.get("schema") == ENDPOINT_SCHEMA, "endpoint schema changed")
    _require(
        payload.get("schema_version") == ENDPOINT_VERSION,
        "endpoint version changed",
    )
    _require(
        payload.get("semantics") == ENDPOINT_SEMANTICS,
        "endpoint semantics changed",
    )
    report_id = sha256_digest(
        payload.get("endpoint_report_id"),
        name="endpoint_report_id",
    )
    identity = {
        key: value for key, value in payload.items() if key != "endpoint_report_id"
    }
    _require(report_id == content_id(identity), "endpoint_report_id changed")

    object_id = nonempty_string(payload.get("object_id"), name="object_id")
    _require(
        object_id.strip() == object_id and "\x00" not in object_id,
        "object_id is not canonical",
    )
    episode_id = payload.get("episode_id")
    _require(
        type(episode_id) is int and episode_id >= 0,
        "episode_id must be a nonnegative integer",
    )
    _require(payload.get("stratum") in {"sheet", "volumetric"}, "stratum changed")
    sha256_digest(payload.get("prediction_seal_id"), name="prediction_seal_id")

    role = payload.get("evaluation_role")
    _require(
        type(role) is str
        and role in {"development_source", "independent_confirmation"},
        "evaluation_role changed",
    )
    if expected_evaluation_role is not None:
        _require(role == expected_evaluation_role, "endpoint evaluation role changed")
    authorization = payload.get("opening_authorization_id")
    if role == "development_source":
        _require(authorization is None, "development report carries authorization")
        _require(
            expected_opening_authorization_id is None,
            "development validation requested confirmation authorization",
        )
    else:
        authorization = sha256_digest(
            authorization,
            name="opening_authorization_id",
        )
        if expected_opening_authorization_id is not None:
            _require(
                authorization
                == sha256_digest(
                    expected_opening_authorization_id,
                    name="expected_opening_authorization_id",
                ),
                "endpoint opening authorization changed",
            )

    config = _mapping(payload.get("endpoint_config"), name="endpoint_config")
    _require(
        sha256_digest(payload.get("endpoint_config_id"), name="endpoint_config_id")
        == content_id(config),
        "endpoint config ID changed",
    )
    config_range = _sequence(
        config.get("evaluation_frame_range_half_open"),
        name="endpoint config evaluation range",
    )
    _require(
        len(config_range) == 2
        and all(type(value) is int for value in config_range)
        and config_range[0] < config_range[1],
        "endpoint config evaluation range changed",
    )
    reserved_count = config.get("reserved_view_count")
    _require(
        type(reserved_count) is int and reserved_count >= 1,
        "endpoint reserved view count changed",
    )
    frame_indices = _sequence(payload.get("frame_indices"), name="frame_indices")
    _require(
        list(frame_indices) == list(range(config_range[0], config_range[1])),
        "endpoint frame roster changed",
    )
    camera_ids = tuple(
        _camera_id(value)
        for value in _sequence(
            payload.get("reserved_camera_ids"), name="reserved_camera_ids"
        )
    )
    _require(
        len(camera_ids) == reserved_count
        and len(set(camera_ids)) == len(camera_ids),
        "reserved camera roster changed",
    )
    raw_cell_count = payload.get("cell_count_per_method")
    _require(
        type(raw_cell_count) is int
        and raw_cell_count == len(frame_indices) * len(camera_ids),
        "endpoint cell count changed",
    )
    cell_count = cast(int, raw_cell_count)

    losses = _mapping(payload.get("method_loss_mm"), name="method_loss_mm")
    cells = _mapping(
        payload.get("method_cell_loss_mm"),
        name="method_cell_loss_mm",
    )
    support_failures = _mapping(
        payload.get("prediction_support_failure_count"),
        name="prediction_support_failure_count",
    )
    for name, values in (
        ("method_loss_mm", losses),
        ("method_cell_loss_mm", cells),
        ("prediction_support_failure_count", support_failures),
    ):
        _require(set(values) == set(RAW_METHOD_IDS), f"{name} roster changed")
    for method_id in RAW_METHOD_IDS:
        method_cells = tuple(
            _finite_nonnegative(value, name=f"{method_id} cell loss")
            for value in _sequence(cells[method_id], name=f"{method_id} cell losses")
        )
        _require(len(method_cells) == cell_count, f"{method_id} cell count changed")
        method_loss = _finite_nonnegative(
            losses[method_id], name=f"{method_id} loss"
        )
        _require(
            np.isclose(
                method_loss,
                float(np.mean(method_cells)),
                rtol=1e-12,
                atol=1e-12,
            ),
            f"{method_id} aggregate loss changed",
        )
        failure_count = support_failures[method_id]
        _require(
            type(failure_count) is int and 0 <= failure_count <= cell_count,
            f"{method_id} support failure count changed",
        )

    technical_failure = payload.get("technical_failure")
    _require(type(technical_failure) is bool, "technical_failure must be Boolean")
    missing = _sequence(
        payload.get("missing_target_cells"), name="missing_target_cells"
    )
    _require(
        technical_failure == bool(missing),
        "technical failure and missing target cells disagree",
    )
    source_artifact_mapping(
        _mapping(payload.get("source_artifact_ids"), name="source_artifact_ids"),
        name="endpoint source_artifact_ids",
    )
    expected_boundary = {
        "prediction_seal_verified_before_target_open": True,
        "reserved_views_contributed_likelihood": False,
        "future_geometry_used_for_prediction": False,
        "future_geometry_used_for_scoring_after_seal": True,
        "tactile_used_to_define_target": False,
        "development_suffix_used_for_scoring": role == "development_source",
        "confirmation_payloads_opened_for_authorized_scoring": (
            role == "independent_confirmation"
        ),
        "confirmation_target_outcomes_used_for_scoring": (
            role == "independent_confirmation"
        ),
        "confirmation_outcomes_used_before_authorization": False,
        "human_approval_required": False,
        "new_measurements_required": False,
    }
    _require(
        plain_json(_mapping(payload.get("information_boundary"), name="boundary"))
        == expected_boundary,
        "endpoint information boundary changed",
    )
    return frozen_finite_json_mapping(payload, name="endpoint report")


def load_deform360_joint_sparse_endpoint_report_v5(
    path: str | Path,
    *,
    expected_evaluation_role: EvaluationRoleV5 | None = None,
    expected_opening_authorization_id: str | None = None,
) -> Mapping[str, Any]:
    """Load and validate one endpoint report from disk."""

    report = load_strict_json_object(path, label="v5 endpoint report")
    return validate_deform360_joint_sparse_endpoint_report_v5(
        report,
        expected_evaluation_role=expected_evaluation_role,
        expected_opening_authorization_id=expected_opening_authorization_id,
    )


__all__ = [
    "Deform360JointSparseEndpointConfigV5",
    "Deform360ReservedViewGeometryV5",
    "EvaluationRoleV5",
    "load_deform360_joint_sparse_endpoint_report_v5",
    "score_deform360_joint_sparse_endpoint_v5",
    "select_reserved_endpoint_views_v5",
    "validate_deform360_joint_sparse_endpoint_report_v5",
]
