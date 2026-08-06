"""Causal single-view RGB-D completion for a strict multiview track carrier.

The strict carrier remains authoritative wherever it has support.  Its accepted
rows are also used as target-free calibration evidence for selecting one camera
whose frame-zero-anchored RGB-D tracks can fill carrier abstentions.  Camera
selection never uses a physical-state innovation or withheld material tracks.

Only one camera contributes to any completed row.  Consequently, duplicated or
correlated camera streams cannot manufacture precision through repeated
independent fusion.  Camera disagreement is retained as observation covariance
for the later Bayesian state update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _immutable(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TAPNextPPDepthCompletionConfig:
    """Frozen target-free lifting and camera-selection settings."""

    visibility_threshold: float = 0.50
    mask_patch_radius_px: int = 2
    minimum_object_mask_fraction: float = 0.20
    maximum_local_depth_mad_m: float = 0.020
    depth_sensor_std_m: float = 0.003
    pixel_noise_std_px: float = 1.5
    shared_bias_std_m: float = 0.005
    minimum_carrier_overlap_rows: int = 16
    minimum_carrier_overlap_fraction: float = 0.25
    maximum_penalized_agreement_m: float = 0.005

    def __post_init__(self) -> None:
        _require(
            0.0 < self.visibility_threshold < 1.0,
            "visibility threshold must lie in (0, 1)",
        )
        _require(
            self.mask_patch_radius_px >= 0,
            "mask patch radius must be nonnegative",
        )
        _require(
            0.0 <= self.minimum_object_mask_fraction <= 1.0,
            "minimum mask fraction must lie in [0, 1]",
        )
        for name, value in (
            ("maximum depth MAD", self.maximum_local_depth_mad_m),
            ("depth sensor standard deviation", self.depth_sensor_std_m),
            ("pixel noise standard deviation", self.pixel_noise_std_px),
            ("shared bias standard deviation", self.shared_bias_std_m),
            ("maximum penalized agreement", self.maximum_penalized_agreement_m),
        ):
            _require(np.isfinite(value) and value > 0.0, f"{name} must be positive")
        _require(
            self.minimum_carrier_overlap_rows >= 2,
            "at least two carrier-overlap rows are required",
        )
        _require(
            0.0 < self.minimum_carrier_overlap_fraction <= 1.0,
            "minimum overlap fraction must lie in (0, 1]",
        )


@dataclass(frozen=True)
class PerCameraMetricTracks:
    """Frame-zero-anchored metric tracks retained separately by camera."""

    points_world_m: np.ndarray
    valid: np.ndarray
    prior_reliability: np.ndarray
    covariance_m2: np.ndarray
    local_depth_mad_m: np.ndarray
    object_mask_fraction: np.ndarray

    def __post_init__(self) -> None:
        points = _immutable(self.points_world_m, np.float64)
        valid = _immutable(self.valid, bool)
        reliability = _immutable(self.prior_reliability, np.float64)
        covariance = _immutable(self.covariance_m2, np.float64)
        depth_mad = _immutable(self.local_depth_mad_m, np.float64)
        mask_fraction = _immutable(self.object_mask_fraction, np.float64)
        _require(
            points.ndim == 4 and points.shape[-1] == 3,
            "points must have shape (V, T, N, 3)",
        )
        rows = points.shape[:3]
        _require(valid.shape == rows, "validity shape changed")
        _require(reliability.shape == rows, "reliability shape changed")
        _require(
            covariance.shape == (*rows, 3, 3),
            "covariance must have shape (V, T, N, 3, 3)",
        )
        _require(depth_mad.shape == rows, "depth-MAD shape changed")
        _require(mask_fraction.shape == rows, "mask-fraction shape changed")
        _require(np.all(np.isfinite(points[valid])), "valid points are not finite")
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior reliability must lie in [0, 1]",
        )
        _require(
            np.all(reliability[~valid] == 0.0),
            "invalid rows must have zero reliability",
        )
        _require(
            np.all(np.isfinite(covariance)),
            "observation covariance is not finite",
        )
        eigenvalues = np.linalg.eigvalsh(covariance.reshape(-1, 3, 3))
        _require(np.all(eigenvalues >= -1e-12), "observation covariance is not PSD")
        _require(
            np.all(np.isfinite(depth_mad)) and np.all(depth_mad >= 0.0),
            "depth MAD must be finite and nonnegative",
        )
        _require(
            np.all(np.isfinite(mask_fraction))
            and np.all((mask_fraction >= 0.0) & (mask_fraction <= 1.0)),
            "mask fractions must lie in [0, 1]",
        )
        for name, value in (
            ("points_world_m", points),
            ("valid", valid),
            ("prior_reliability", reliability),
            ("covariance_m2", covariance),
            ("local_depth_mad_m", depth_mad),
            ("object_mask_fraction", mask_fraction),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CameraCompetence:
    """Target-free agreement of one camera with accepted carrier rows."""

    camera_index: int
    accepted: bool
    reason: str
    overlap_rows: int
    overlap_fraction: float
    centered_median_m: float
    centered_p90_m: float
    penalized_agreement_m: float
    carrier_offset_m: np.ndarray
    residual_covariance_m2: np.ndarray

    def __post_init__(self) -> None:
        offset = _immutable(self.carrier_offset_m, np.float64)
        covariance = _immutable(self.residual_covariance_m2, np.float64)
        _require(offset.shape == (3,), "camera offset must have shape (3,)")
        _require(covariance.shape == (3, 3), "residual covariance must be 3x3")
        _require(np.all(np.isfinite(offset)), "camera offset is not finite")
        _require(np.all(np.isfinite(covariance)), "residual covariance is not finite")
        object.__setattr__(self, "carrier_offset_m", offset)
        object.__setattr__(self, "residual_covariance_m2", covariance)


@dataclass(frozen=True)
class DepthCompletionResult:
    """Strict carrier plus target-free single-camera completion rows."""

    points_world_m: np.ndarray
    support: np.ndarray
    prior_reliability: np.ndarray
    covariance_m2: np.ndarray
    source_camera: np.ndarray
    selected_camera: int | None
    accepted: bool
    reason: str
    camera_competence: tuple[CameraCompetence, ...]

    def __post_init__(self) -> None:
        points = _immutable(self.points_world_m, np.float64)
        support = _immutable(self.support, bool)
        reliability = _immutable(self.prior_reliability, np.float64)
        covariance = _immutable(self.covariance_m2, np.float64)
        source = _immutable(self.source_camera, np.int16)
        _require(
            points.ndim == 3 and points.shape[-1] == 3,
            "completed points must have shape (T, N, 3)",
        )
        rows = points.shape[:2]
        _require(support.shape == rows, "completed support shape changed")
        _require(reliability.shape == rows, "completed reliability shape changed")
        _require(covariance.shape == (*rows, 3, 3), "completed covariance changed")
        _require(source.shape == rows, "source-camera shape changed")
        _require(np.all(np.isfinite(points)), "completed points are not finite")
        _require(np.all(np.isfinite(covariance)), "completed covariance is not finite")
        object.__setattr__(self, "points_world_m", points)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "prior_reliability", reliability)
        object.__setattr__(self, "covariance_m2", covariance)
        object.__setattr__(self, "source_camera", source)
        object.__setattr__(self, "camera_competence", tuple(self.camera_competence))


def _sample_depth_patch(
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    point_xy: np.ndarray,
    radius_px: int,
) -> tuple[float, float, float]:
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(object_mask, dtype=bool)
    point = np.asarray(point_xy, dtype=np.float64)
    _require(depth.ndim == 2, "depth frame must be a matrix")
    _require(mask.shape == depth.shape, "depth and object-mask shapes differ")
    _require(point.shape == (2,), "image point must have shape (2,)")
    if not np.all(np.isfinite(point)):
        return float("nan"), 0.0, 0.0
    center_x, center_y = np.rint(point).astype(np.int64)
    if not (0 <= center_x < depth.shape[1] and 0 <= center_y < depth.shape[0]):
        return float("nan"), 0.0, 0.0
    x0 = max(0, int(center_x) - radius_px)
    x1 = min(depth.shape[1], int(center_x) + radius_px + 1)
    y0 = max(0, int(center_y) - radius_px)
    y1 = min(depth.shape[0], int(center_y) + radius_px + 1)
    mask_patch = mask[y0:y1, x0:x1]
    depth_patch = depth[y0:y1, x0:x1]
    mask_fraction = float(np.mean(mask_patch)) if mask_patch.size else 0.0
    selected = depth_patch[
        mask_patch & np.isfinite(depth_patch) & (depth_patch > 0.0)
    ]
    if not len(selected):
        return float("nan"), 0.0, mask_fraction
    median = float(np.median(selected))
    mad = float(np.median(np.abs(selected - median)))
    return median, mad, mask_fraction


def lift_per_camera_rgbd_tracks(
    tracks_xy: np.ndarray,
    visibility_probability: np.ndarray,
    depth_m: np.ndarray,
    object_masks: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    frame_zero_points_world_m: np.ndarray,
    *,
    config: TAPNextPPDepthCompletionConfig | None = None,
) -> PerCameraMetricTracks:
    """Lift every camera separately and anchor each material identity at frame zero."""

    cfg = config or TAPNextPPDepthCompletionConfig()
    tracks = np.asarray(tracks_xy, dtype=np.float64)
    visibility = np.asarray(visibility_probability, dtype=np.float64)
    depths = np.asarray(depth_m, dtype=np.float64)
    masks = np.asarray(object_masks, dtype=bool)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    frame_zero = np.asarray(frame_zero_points_world_m, dtype=np.float64)
    _require(
        tracks.ndim == 4 and tracks.shape[-1] == 2,
        "tracks must have shape (V, T, N, 2)",
    )
    view_count, frame_count, identity_count, _ = tracks.shape
    _require(visibility.shape == tracks.shape[:3], "visibility shape changed")
    _require(depths.ndim == 4, "depth must have shape (V, T, H, W)")
    _require(depths.shape[:2] == (view_count, frame_count), "depth shape changed")
    _require(masks.shape == depths.shape, "object-mask shape changed")
    _require(matrices.shape == (view_count, 3, 3), "intrinsics shape changed")
    _require(poses.shape == (view_count, 4, 4), "camera poses shape changed")
    _require(frame_zero.shape == (identity_count, 3), "frame-zero points changed")
    _require(np.all(np.isfinite(frame_zero)), "frame-zero points are not finite")

    raw_world = np.full((view_count, frame_count, identity_count, 3), np.nan)
    valid = np.zeros((view_count, frame_count, identity_count), dtype=bool)
    reliability = np.zeros_like(valid, dtype=np.float64)
    depth_mad = np.zeros_like(valid, dtype=np.float64)
    mask_fraction = np.zeros_like(valid, dtype=np.float64)
    row_variance = np.zeros_like(valid, dtype=np.float64)
    for view in range(view_count):
        inverse_intrinsic = np.linalg.inv(matrices[view])
        focal = max(
            1e-12,
            float(min(abs(matrices[view, 0, 0]), abs(matrices[view, 1, 1]))),
        )
        for frame in range(frame_count):
            for identity in range(identity_count):
                probability = visibility[view, frame, identity]
                if not np.isfinite(probability) or probability < cfg.visibility_threshold:
                    continue
                depth, mad, fraction = _sample_depth_patch(
                    depths[view, frame],
                    masks[view, frame],
                    tracks[view, frame, identity],
                    cfg.mask_patch_radius_px,
                )
                depth_mad[view, frame, identity] = mad
                mask_fraction[view, frame, identity] = fraction
                if (
                    not np.isfinite(depth)
                    or fraction < cfg.minimum_object_mask_fraction
                    or mad > cfg.maximum_local_depth_mad_m
                ):
                    continue
                pixel = np.append(tracks[view, frame, identity], 1.0)
                camera_point = (inverse_intrinsic @ pixel) * depth
                world_point = poses[view] @ np.append(camera_point, 1.0)
                raw_world[view, frame, identity] = world_point[:3]
                valid[view, frame, identity] = True
                mad_weight = np.exp(
                    -0.5 * (mad / cfg.maximum_local_depth_mad_m) ** 2
                )
                reliability[view, frame, identity] = float(
                    np.clip(probability * fraction * mad_weight, 0.0, 1.0)
                )
                lateral_std = cfg.pixel_noise_std_px * depth / focal
                row_variance[view, frame, identity] = (
                    cfg.depth_sensor_std_m**2 + mad**2 + lateral_std**2
                )

    anchored = np.full_like(raw_world, np.nan)
    anchored_covariance = np.zeros((*valid.shape, 3, 3), dtype=np.float64)
    for view in range(view_count):
        for identity in range(identity_count):
            if not valid[view, 0, identity]:
                valid[view, :, identity] = False
                reliability[view, :, identity] = 0.0
                continue
            anchored[view, :, identity] = (
                frame_zero[identity]
                + raw_world[view, :, identity]
                - raw_world[view, 0, identity]
            )
            variance = (
                row_variance[view, :, identity]
                + row_variance[view, 0, identity]
            )
            anchored_covariance[view, :, identity] = (
                variance[:, None, None] * np.eye(3)[None]
            )
    valid &= np.all(np.isfinite(anchored), axis=-1)
    reliability[~valid] = 0.0
    anchored[~valid] = 0.0
    return PerCameraMetricTracks(
        points_world_m=anchored,
        valid=valid,
        prior_reliability=reliability,
        covariance_m2=anchored_covariance,
        local_depth_mad_m=depth_mad,
        object_mask_fraction=mask_fraction,
    )


def _psd_covariance(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if len(values) < 2:
        return np.zeros((3, 3), dtype=np.float64)
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def assess_camera_competence(
    strict_points_world_m: np.ndarray,
    strict_support: np.ndarray,
    observations: PerCameraMetricTracks,
    *,
    config: TAPNextPPDepthCompletionConfig | None = None,
) -> tuple[CameraCompetence, ...]:
    """Score cameras using accepted carrier rows only, never withheld targets."""

    cfg = config or TAPNextPPDepthCompletionConfig()
    carrier = np.asarray(strict_points_world_m, dtype=np.float64)
    support = np.asarray(strict_support, dtype=bool)
    _require(
        carrier.ndim == 3 and carrier.shape[-1] == 3,
        "strict carrier must have shape (T, N, 3)",
    )
    _require(support.shape == carrier.shape[:2], "strict support shape changed")
    _require(
        observations.points_world_m.shape[1:] == carrier.shape,
        "camera observations and strict carrier differ",
    )
    carrier_valid = support & np.all(np.isfinite(carrier), axis=-1)
    total_rows = int(np.sum(carrier_valid))
    results = []
    for camera in range(observations.points_world_m.shape[0]):
        overlap = carrier_valid & observations.valid[camera]
        count = int(np.sum(overlap))
        fraction = count / max(total_rows, 1)
        if count:
            residual = carrier[overlap] - observations.points_world_m[camera][overlap]
            offset = np.median(residual, axis=0)
            centered = residual - offset
            norms = np.linalg.norm(centered, axis=1)
            median = float(np.median(norms))
            p90 = float(np.quantile(norms, 0.90))
            covariance = _psd_covariance(centered)
            penalized = median / np.sqrt(max(fraction, np.finfo(np.float64).tiny))
        else:
            offset = np.zeros(3, dtype=np.float64)
            median = float("inf")
            p90 = float("inf")
            penalized = float("inf")
            covariance = np.zeros((3, 3), dtype=np.float64)
        accepted = True
        reason = "accepted"
        if total_rows == 0:
            accepted = False
            reason = "carrier-has-no-support"
        elif count < cfg.minimum_carrier_overlap_rows:
            accepted = False
            reason = "insufficient-carrier-overlap-rows"
        elif fraction < cfg.minimum_carrier_overlap_fraction:
            accepted = False
            reason = "insufficient-carrier-overlap-fraction"
        elif penalized > cfg.maximum_penalized_agreement_m:
            accepted = False
            reason = "carrier-agreement-failed"
        results.append(
            CameraCompetence(
                camera_index=camera,
                accepted=accepted,
                reason=reason,
                overlap_rows=count,
                overlap_fraction=fraction,
                centered_median_m=median,
                centered_p90_m=p90,
                penalized_agreement_m=penalized,
                carrier_offset_m=offset,
                residual_covariance_m2=covariance,
            )
        )
    return tuple(results)


def complete_strict_multiview_carrier(
    strict_points_world_m: np.ndarray,
    strict_support: np.ndarray,
    strict_prior_reliability: np.ndarray,
    strict_covariance_m2: np.ndarray,
    observations: PerCameraMetricTracks,
    *,
    config: TAPNextPPDepthCompletionConfig | None = None,
) -> DepthCompletionResult:
    """Fill strict-carrier abstentions from one target-free selected camera."""

    cfg = config or TAPNextPPDepthCompletionConfig()
    points = np.asarray(strict_points_world_m, dtype=np.float64).copy()
    support = np.asarray(strict_support, dtype=bool).copy()
    reliability = np.asarray(strict_prior_reliability, dtype=np.float64).copy()
    covariance = np.asarray(strict_covariance_m2, dtype=np.float64).copy()
    _require(
        points.ndim == 3 and points.shape[-1] == 3,
        "strict points must have shape (T, N, 3)",
    )
    rows = points.shape[:2]
    _require(support.shape == rows, "strict support shape changed")
    _require(reliability.shape == rows, "strict reliability shape changed")
    _require(covariance.shape == (*rows, 3, 3), "strict covariance shape changed")
    _require(np.all(np.isfinite(points)), "strict points are not finite")
    _require(np.all(np.isfinite(reliability)), "strict reliability is not finite")
    _require(np.all(np.isfinite(covariance)), "strict covariance is not finite")
    source_camera = np.full(rows, -2, dtype=np.int16)
    source_camera[support] = -1
    competence = assess_camera_competence(
        points,
        support,
        observations,
        config=cfg,
    )
    accepted = [item for item in competence if item.accepted]
    if not accepted:
        return DepthCompletionResult(
            points_world_m=points,
            support=support,
            prior_reliability=reliability,
            covariance_m2=covariance,
            source_camera=source_camera,
            selected_camera=None,
            accepted=False,
            reason="no-camera-passed-target-free-competence",
            camera_competence=competence,
        )
    selected = min(
        accepted,
        key=lambda item: (
            item.penalized_agreement_m,
            -item.overlap_rows,
            item.camera_index,
        ),
    )
    camera = selected.camera_index
    fill = (~support) & observations.valid[camera]
    points[fill] = observations.points_world_m[camera][fill]
    support[fill] = True
    competence_weight = float(
        np.exp(
            -0.5
            * (selected.penalized_agreement_m / cfg.maximum_penalized_agreement_m)
            ** 2
        )
    )
    reliability[fill] = (
        observations.prior_reliability[camera][fill] * competence_weight
    )
    bias_floor = np.eye(3) * (cfg.shared_bias_std_m**2)
    covariance[fill] = (
        observations.covariance_m2[camera][fill]
        + selected.residual_covariance_m2[None]
        + bias_floor[None]
    )
    source_camera[fill] = camera
    return DepthCompletionResult(
        points_world_m=points,
        support=support,
        prior_reliability=reliability,
        covariance_m2=covariance,
        source_camera=source_camera,
        selected_camera=camera,
        accepted=bool(np.any(fill)),
        reason="completed" if np.any(fill) else "selected-camera-added-no-support",
        camera_competence=competence,
    )
