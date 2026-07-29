"""Frame-local, rigid-invariant RGB-D shape signatures for Deform360 V15."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .deform360_event_conditioned_window_v15 import EventPanelEvidence
from .observation_belief import array_sha256

CONTRACT = "deform360-event-shape-signature-v15"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-event-shape-signature-v15\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class EventShapeSignatureConfig:
    """Frozen per-frame metric shape-summary construction."""

    pixel_stride: int = 8
    maximum_points_per_camera: int = 128
    minimum_points_per_camera: int = 32
    component_quantiles: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    minimum_camera_support: int = 2
    gripper_exclusion_dilation_px: int = 8
    maximum_gripper_overlap_fraction: float = 0.15
    depth_standard_deviation_m: float = 0.005
    variance_floor_m2: float = 1e-10

    def __post_init__(self) -> None:
        _require(self.pixel_stride >= 1, "pixel stride must be positive")
        _require(
            self.maximum_points_per_camera >= self.minimum_points_per_camera >= 4,
            "camera point-count bounds are invalid",
        )
        _require(
            len(self.component_quantiles) >= 3
            and tuple(sorted(set(self.component_quantiles)))
            == self.component_quantiles
            and all(0.0 < value < 1.0 for value in self.component_quantiles),
            "shape-signature quantiles are invalid",
        )
        _require(
            self.minimum_camera_support >= 2,
            "shape evidence requires at least two cameras",
        )
        _require(
            self.gripper_exclusion_dilation_px >= 0,
            "gripper dilation is negative",
        )
        _require(
            np.isfinite(self.maximum_gripper_overlap_fraction)
            and 0.0 <= self.maximum_gripper_overlap_fraction < 1.0,
            "gripper-overlap limit is invalid",
        )
        _require(
            np.isfinite(self.depth_standard_deviation_m)
            and self.depth_standard_deviation_m > 0.0
            and np.isfinite(self.variance_floor_m2)
            and self.variance_floor_m2 > 0.0,
            "shape-signature uncertainty is invalid",
        )


@dataclass(frozen=True)
class EventPanelShapeSignature:
    """One panel's frame-local signatures and camera-level diagnostics."""

    config: EventShapeSignatureConfig
    camera_ids: tuple[str, ...]
    evidence: EventPanelEvidence
    per_camera_signature_m: np.ndarray
    per_camera_point_count: np.ndarray
    per_camera_gripper_overlap_fraction: np.ndarray
    per_camera_gripper_clear: np.ndarray
    artifact_sha256: str

    def __post_init__(self) -> None:
        signature = _readonly(self.per_camera_signature_m, dtype=np.float64)
        point_count = _readonly(self.per_camera_point_count, dtype=np.int64)
        overlap = _readonly(
            self.per_camera_gripper_overlap_fraction,
            dtype=np.float64,
        )
        clear = _readonly(self.per_camera_gripper_clear, dtype=bool)
        camera_count = len(self.camera_ids)
        frame_count = self.evidence.frame_count
        component_count = self.evidence.component_count
        _require(
            camera_count >= self.config.minimum_camera_support
            and len(set(self.camera_ids)) == camera_count
            and all(camera_id.strip() for camera_id in self.camera_ids),
            "shape-signature camera panel is invalid",
        )
        _require(
            signature.shape == (camera_count, frame_count, component_count)
            and point_count.shape == (camera_count, frame_count)
            and overlap.shape == (camera_count, frame_count)
            and clear.shape == (camera_count, frame_count),
            "shape-signature diagnostics have invalid shapes",
        )
        _require(
            np.all(point_count >= 0)
            and np.all(np.isfinite(overlap))
            and np.all((overlap >= 0.0) & (overlap <= 1.0)),
            "shape-signature diagnostics are outside their domains",
        )
        object.__setattr__(self, "per_camera_signature_m", signature)
        object.__setattr__(self, "per_camera_point_count", point_count)
        object.__setattr__(
            self,
            "per_camera_gripper_overlap_fraction",
            overlap,
        )
        object.__setattr__(self, "per_camera_gripper_clear", clear)
        _require(
            _canonical_sha256(self.descriptor()) == self.artifact_sha256,
            "shape-signature artifact digest changed",
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360EventPanelShapeSignatureV15",
            "contract": CONTRACT,
            "config": asdict(self.config),
            "camera_ids": list(self.camera_ids),
            "frame_count": self.evidence.frame_count,
            "component_count": self.evidence.component_count,
            "component_ids": self.evidence.component_ids.tolist(),
            "array_sha256": {
                "component_signature_m": array_sha256(
                    self.evidence.component_signature_m
                ),
                "variance_m2": array_sha256(self.evidence.variance_m2),
                "available": array_sha256(self.evidence.available),
                "camera_support": array_sha256(self.evidence.camera_support),
                "gripper_clear": array_sha256(self.evidence.gripper_clear),
                "per_camera_signature_m": array_sha256(
                    self.per_camera_signature_m
                ),
                "per_camera_point_count": array_sha256(
                    self.per_camera_point_count
                ),
                "per_camera_gripper_overlap_fraction": array_sha256(
                    self.per_camera_gripper_overlap_fraction
                ),
                "per_camera_gripper_clear": array_sha256(
                    self.per_camera_gripper_clear
                ),
            },
            "information_boundary": {
                "frame_local_processing_only": True,
                "future_frame_used_for_current_signature": False,
                "tracker_or_material_identity_used": False,
                "physical_prediction_used": False,
                "pairwise_distance_signature_is_rigid_invariant": True,
                "gripper_neighborhood_excluded_before_signature": True,
                "camera_scatter_not_divided_by_camera_count": True,
                "camera_support_is_not_independent_pixel_evidence": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def pairwise_shape_signature(
    points_m: np.ndarray,
    *,
    quantiles: tuple[float, ...],
    maximum_points: int,
) -> np.ndarray:
    """Return deterministic pairwise-distance quantiles for one point cloud."""

    points = np.asarray(points_m, dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) >= 4
        and np.all(np.isfinite(points)),
        "shape-signature points must have shape (N, 3)",
    )
    _require(
        maximum_points >= 4
        and len(quantiles) >= 1
        and all(0.0 < value < 1.0 for value in quantiles),
        "shape-signature sampling settings are invalid",
    )
    if len(points) > maximum_points:
        selected = np.linspace(
            0,
            len(points) - 1,
            maximum_points,
            dtype=np.int64,
        )
        points = points[selected]
    left, right = np.triu_indices(len(points), k=1)
    distances = np.linalg.norm(points[left] - points[right], axis=1)
    _require(
        len(distances) > 0 and np.all(np.isfinite(distances)),
        "pairwise shape distances are invalid",
    )
    return np.asarray(np.quantile(distances, quantiles), dtype=np.float64)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if radius == 0:
        return values.copy()
    padded = np.pad(values, radius, mode="constant", constant_values=False)
    output = np.zeros_like(values)
    height, width = values.shape
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            output |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return output


def _camera_points_m(
    depth_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    *,
    pixel_stride: int,
    maximum_points: int,
) -> np.ndarray:
    rows, columns = np.nonzero(valid)
    if len(rows) > maximum_points:
        selected = np.linspace(
            0,
            len(rows) - 1,
            maximum_points,
            dtype=np.int64,
        )
        rows = rows[selected]
        columns = columns[selected]
    z = depth_m[rows, columns]
    u = columns.astype(np.float64) * pixel_stride
    v = rows.astype(np.float64) * pixel_stride
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    return np.column_stack(
        (
            (u - cx) * z / fx,
            (v - cy) * z / fy,
            z,
        )
    )


def build_event_panel_shape_signature(
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    gripper_mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    camera_ids: tuple[str, ...],
    config: EventShapeSignatureConfig | None = None,
) -> EventPanelShapeSignature:
    """Build frame-independent metric shape signatures for one camera panel."""

    cfg = config or EventShapeSignatureConfig()
    depth = np.asarray(depth_m, dtype=np.float64)
    objects = np.asarray(object_mask, dtype=bool)
    grippers = np.asarray(gripper_mask, dtype=bool)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    _require(
        depth.ndim == 4
        and objects.shape == depth.shape
        and grippers.shape == depth.shape,
        "RGB-D shape inputs must share shape (V, T, H, W)",
    )
    camera_count, frame_count, _, _ = depth.shape
    _require(
        len(camera_ids) == camera_count
        and len(set(camera_ids)) == camera_count
        and matrices.shape == (camera_count, 3, 3)
        and np.all(np.isfinite(matrices))
        and np.all(matrices[:, 0, 0] > 0.0)
        and np.all(matrices[:, 1, 1] > 0.0),
        "shape-signature camera calibration is invalid",
    )
    component_count = len(cfg.component_quantiles)
    per_camera_signature = np.full(
        (camera_count, frame_count, component_count),
        np.nan,
        dtype=np.float64,
    )
    point_count = np.zeros((camera_count, frame_count), dtype=np.int64)
    overlap_fraction = np.zeros((camera_count, frame_count), dtype=np.float64)
    camera_clear = np.zeros((camera_count, frame_count), dtype=bool)
    sampled_dilation = math.ceil(
        cfg.gripper_exclusion_dilation_px / cfg.pixel_stride
    )
    for camera in range(camera_count):
        matrix = matrices[camera]
        for frame in range(frame_count):
            sampled_depth = depth[
                camera,
                frame,
                :: cfg.pixel_stride,
                :: cfg.pixel_stride,
            ]
            sampled_object = objects[
                camera,
                frame,
                :: cfg.pixel_stride,
                :: cfg.pixel_stride,
            ]
            sampled_gripper = grippers[
                camera,
                frame,
                :: cfg.pixel_stride,
                :: cfg.pixel_stride,
            ]
            dilated_gripper = _dilate_mask(sampled_gripper, sampled_dilation)
            object_count = int(np.sum(sampled_object))
            overlap = (
                0.0
                if object_count == 0
                else float(np.sum(sampled_object & dilated_gripper) / object_count)
            )
            overlap_fraction[camera, frame] = overlap
            clear = overlap <= cfg.maximum_gripper_overlap_fraction
            camera_clear[camera, frame] = clear
            valid = (
                sampled_object
                & ~dilated_gripper
                & np.isfinite(sampled_depth)
                & (sampled_depth > 0.0)
            )
            point_count[camera, frame] = int(np.sum(valid))
            if point_count[camera, frame] < cfg.minimum_points_per_camera:
                continue
            points = _camera_points_m(
                sampled_depth,
                valid,
                matrix,
                pixel_stride=cfg.pixel_stride,
                maximum_points=cfg.maximum_points_per_camera,
            )
            per_camera_signature[camera, frame] = pairwise_shape_signature(
                points,
                quantiles=cfg.component_quantiles,
                maximum_points=cfg.maximum_points_per_camera,
            )

    signature = np.full((frame_count, component_count), np.nan, dtype=np.float64)
    variance = np.full_like(signature, np.nan)
    camera_support = np.zeros_like(signature, dtype=np.int64)
    gripper_clear = np.zeros_like(signature, dtype=bool)
    base_pair_variance = 2.0 * cfg.depth_standard_deviation_m**2
    for frame in range(frame_count):
        for component in range(component_count):
            values = per_camera_signature[:, frame, component]
            finite = np.isfinite(values)
            support = int(np.sum(finite))
            camera_support[frame, component] = support
            if support == 0:
                continue
            local = values[finite]
            median = float(np.median(local))
            robust_scatter = float(
                1.4826 * np.median(np.abs(local - median))
            )
            signature[frame, component] = median
            variance[frame, component] = max(
                base_pair_variance + robust_scatter**2,
                cfg.variance_floor_m2,
            )
            clear_support = int(np.sum(finite & camera_clear[:, frame]))
            gripper_clear[frame, component] = (
                clear_support >= cfg.minimum_camera_support
            )
    available = camera_support >= cfg.minimum_camera_support
    evidence = EventPanelEvidence(
        component_signature_m=signature,
        variance_m2=variance,
        available=available,
        camera_support=camera_support,
        gripper_clear=gripper_clear,
        component_ids=np.arange(component_count, dtype=np.int64),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360EventPanelShapeSignatureV15",
        "contract": CONTRACT,
        "config": asdict(cfg),
        "camera_ids": list(camera_ids),
        "frame_count": frame_count,
        "component_count": component_count,
        "component_ids": evidence.component_ids.tolist(),
        "array_sha256": {
            "component_signature_m": array_sha256(
                evidence.component_signature_m
            ),
            "variance_m2": array_sha256(evidence.variance_m2),
            "available": array_sha256(evidence.available),
            "camera_support": array_sha256(evidence.camera_support),
            "gripper_clear": array_sha256(evidence.gripper_clear),
            "per_camera_signature_m": array_sha256(per_camera_signature),
            "per_camera_point_count": array_sha256(point_count),
            "per_camera_gripper_overlap_fraction": array_sha256(
                overlap_fraction
            ),
            "per_camera_gripper_clear": array_sha256(camera_clear),
        },
        "information_boundary": {
            "frame_local_processing_only": True,
            "future_frame_used_for_current_signature": False,
            "tracker_or_material_identity_used": False,
            "physical_prediction_used": False,
            "pairwise_distance_signature_is_rigid_invariant": True,
            "gripper_neighborhood_excluded_before_signature": True,
            "camera_scatter_not_divided_by_camera_count": True,
            "camera_support_is_not_independent_pixel_evidence": True,
        },
    }
    digest = _canonical_sha256(payload)
    return EventPanelShapeSignature(
        config=cfg,
        camera_ids=tuple(camera_ids),
        evidence=evidence,
        per_camera_signature_m=per_camera_signature,
        per_camera_point_count=point_count,
        per_camera_gripper_overlap_fraction=overlap_fraction,
        per_camera_gripper_clear=camera_clear,
        artifact_sha256=digest,
    )


__all__ = [
    "CONTRACT",
    "EventPanelShapeSignature",
    "EventShapeSignatureConfig",
    "build_event_panel_shape_signature",
    "pairwise_shape_signature",
]
