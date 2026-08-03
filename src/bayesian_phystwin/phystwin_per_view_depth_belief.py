"""Bias-aware PhysTwin updates from causal per-view RGB-D tracks.

The observation path deliberately keeps camera rows separate.  It never turns
multiview agreement into independent precision and never uses the innovation
against the physical prediction as prior perception reliability.  Static
per-camera depth offsets are removed by frame-zero anchoring; remaining shared
and camera-specific bias is retained as an explicit nuisance in the Bayesian
state update.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bias_aware_belief import (
    BiasAwareStateUpdateConfig,
    build_physical_response_basis,
    decode_bias_aware_state,
    restrict_state_basis_to_identifiable_subspace,
    update_bias_aware_state,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _immutable(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PerViewDepthObservations:
    """Metric, frame-zero-anchored observations kept separate by camera."""

    points_world_m: np.ndarray
    valid: np.ndarray
    prior_reliability: np.ndarray
    variance_m2: np.ndarray
    local_depth_mad_m: np.ndarray

    def __post_init__(self) -> None:
        points = _immutable(self.points_world_m, np.float64)
        valid = _immutable(self.valid, bool)
        reliability = _immutable(self.prior_reliability, np.float64)
        variance = _immutable(self.variance_m2, np.float64)
        depth_mad = _immutable(self.local_depth_mad_m, np.float64)
        _require(
            points.ndim == 4 and points.shape[3] == 3,
            "per-view points must have shape (V, T, N, 3)",
        )
        expected = points.shape[:3]
        _require(valid.shape == expected, "per-view validity shape changed")
        _require(reliability.shape == expected, "per-view reliability shape changed")
        _require(variance.shape == expected, "per-view variance shape changed")
        _require(depth_mad.shape == expected, "per-view depth MAD shape changed")
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior reliability must lie in [0, 1]",
        )
        _require(
            np.all(np.isfinite(variance)) and np.all(variance > 0.0),
            "observation variance must be finite and positive",
        )
        _require(
            np.all(np.isfinite(depth_mad)) and np.all(depth_mad >= 0.0),
            "local depth MAD must be finite and nonnegative",
        )
        _require(
            np.all(np.isfinite(points[valid])),
            "valid per-view points must be finite",
        )
        _require(
            np.all(reliability[~valid] == 0.0),
            "invalid rows must have zero prior reliability",
        )
        for name, value in (
            ("points_world_m", points),
            ("valid", valid),
            ("prior_reliability", reliability),
            ("variance_m2", variance),
            ("local_depth_mad_m", depth_mad),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PerViewDepthLoaderConfig:
    """Residual-independent RGB-D lifting and reliability settings."""

    minimum_view_quality: float = 0.50
    maximum_cycle_error_px: float = 5.0
    depth_patch_radius_px: int = 1
    depth_sensor_std_m: float = 0.003
    pixel_noise_std_px: float = 1.0
    maximum_local_depth_mad_m: float = 0.020

    def __post_init__(self) -> None:
        _require(
            0.0 <= self.minimum_view_quality <= 1.0,
            "minimum view quality must lie in [0, 1]",
        )
        _require(
            np.isfinite(self.maximum_cycle_error_px)
            and self.maximum_cycle_error_px > 0.0,
            "maximum cycle error must be positive",
        )
        _require(
            self.depth_patch_radius_px >= 0,
            "depth patch radius must be nonnegative",
        )
        _require(
            np.isfinite(self.depth_sensor_std_m)
            and self.depth_sensor_std_m > 0.0,
            "depth sensor standard deviation must be positive",
        )
        _require(
            np.isfinite(self.pixel_noise_std_px)
            and self.pixel_noise_std_px > 0.0,
            "pixel noise standard deviation must be positive",
        )
        _require(
            np.isfinite(self.maximum_local_depth_mad_m)
            and self.maximum_local_depth_mad_m > 0.0,
            "maximum local depth MAD must be positive",
        )


@dataclass(frozen=True)
class PerViewDepthStateConfig:
    """Configuration for one causal endpoint state/discrepancy update."""

    window_frames: int = 8
    physical_response_rank: int = 4
    minimum_physical_response_m: float = 0.0005
    minimum_active_views: int = 2
    minimum_unique_identities: int = 16
    minimum_identifiable_fraction: float = 0.10
    maximum_correction_m: float = 0.020
    maximum_correction_to_response_ratio: float = 2.0
    update: BiasAwareStateUpdateConfig = BiasAwareStateUpdateConfig(
        observation_std_m=0.005,
        state_prior_std_m=0.020,
        shared_bias_prior_std_m=0.020,
        camera_bias_prior_std_m=0.010,
        effective_samples_per_view=64.0,
        maximum_state_update_m=0.020,
        reject_unanchored_ambiguity=True,
    )

    def __post_init__(self) -> None:
        _require(self.window_frames >= 1, "window length must be positive")
        _require(
            self.physical_response_rank >= 1,
            "physical response rank must be positive",
        )
        _require(
            np.isfinite(self.minimum_physical_response_m)
            and self.minimum_physical_response_m > 0.0,
            "minimum physical response must be positive",
        )
        _require(self.minimum_active_views >= 1, "active-view gate must be positive")
        _require(
            self.minimum_unique_identities >= 1,
            "identity-support gate must be positive",
        )
        _require(
            0.0 < self.minimum_identifiable_fraction <= 1.0,
            "minimum identifiable fraction must lie in (0, 1]",
        )
        _require(
            np.isfinite(self.maximum_correction_m)
            and self.maximum_correction_m > 0.0,
            "maximum correction must be positive",
        )
        _require(
            np.isfinite(self.maximum_correction_to_response_ratio)
            and self.maximum_correction_to_response_ratio > 0.0,
            "correction-to-response ratio must be positive",
        )


@dataclass(frozen=True)
class PerViewDepthStateResult:
    """Decoded graph correction and auditable fallback diagnostics."""

    accepted: bool
    reason: str
    correction_m: np.ndarray
    coefficient_covariance_m2: np.ndarray
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        correction = _immutable(self.correction_m, np.float64)
        covariance = _immutable(self.coefficient_covariance_m2, np.float64)
        _require(
            correction.ndim == 2 and correction.shape[1] == 3,
            "correction must have shape (N, 3)",
        )
        _require(
            covariance.ndim == 2 and covariance.shape[0] == covariance.shape[1],
            "coefficient covariance must be square",
        )
        _require(np.all(np.isfinite(correction)), "correction is not finite")
        _require(np.all(np.isfinite(covariance)), "covariance is not finite")
        if not self.accepted:
            _require(
                np.array_equal(correction, np.zeros_like(correction)),
                "a rejected update must return exact zero correction",
            )
        object.__setattr__(self, "correction_m", correction)
        object.__setattr__(self, "coefficient_covariance_m2", covariance)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _sample_depth_patch_m(
    depth_mm: np.ndarray,
    tracks_xy: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth_mm, dtype=np.float64)
    tracks = np.asarray(tracks_xy, dtype=np.float64)
    _require(depth.ndim == 2, "depth frame must be a matrix")
    _require(
        tracks.ndim == 2 and tracks.shape[1] == 2,
        "tracks must have shape (N, 2)",
    )
    finite = np.all(np.isfinite(tracks), axis=1)
    rounded = np.rint(np.where(finite[:, None], tracks, 0.0)).astype(np.int64)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    samples = np.full((len(tracks), len(offsets) ** 2), np.nan, dtype=np.float64)
    column = 0
    for dy in offsets:
        for dx in offsets:
            x = rounded[:, 0] + dx
            y = rounded[:, 1] + dy
            inside = (
                finite
                & (x >= 0)
                & (x < depth.shape[1])
                & (y >= 0)
                & (y < depth.shape[0])
            )
            selected = np.flatnonzero(inside)
            values = depth[y[selected], x[selected]] / 1000.0
            positive = np.isfinite(values) & (values > 0.0)
            samples[selected[positive], column] = values[positive]
            column += 1
    count = np.sum(np.isfinite(samples), axis=1)
    with np.errstate(all="ignore"):
        median = np.nanmedian(samples, axis=1)
        mad = np.nanmedian(np.abs(samples - median[:, None]), axis=1)
    valid = (count > 0) & np.isfinite(median) & (median > 0.0)
    median[~valid] = np.nan
    mad[~valid] = 0.0
    return median, mad, valid


def load_cotracker3_per_view_depth_observations(
    cues_path: str | Path,
    raw_case_dir: str | Path,
    initial_world_points_m: np.ndarray,
    *,
    train_end_frame: int,
    config: PerViewDepthLoaderConfig | None = None,
) -> PerViewDepthObservations:
    """Lift every causal CoTracker3 view through its own metric depth map."""

    cfg = config or PerViewDepthLoaderConfig()
    _require(train_end_frame >= 1, "training endpoint must be positive")
    required = {
        "multiview_tracks_xy_prefix",
        "multiview_quality_probability_prefix",
        "multiview_view_valid_prefix",
        "multiview_intrinsics",
        "multiview_camera_to_world",
        "forward_backward_error_px",
        "forward_backward_valid",
        "boundary_distance",
        "cue_available",
    }
    with np.load(cues_path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                "CoTracker3 archive lacks per-view depth fields: "
                + ", ".join(sorted(missing))
            )
        tracks = np.asarray(
            archive["multiview_tracks_xy_prefix"][:, :train_end_frame],
            dtype=np.float64,
        )
        quality = np.asarray(
            archive["multiview_quality_probability_prefix"][
                :, :train_end_frame
            ],
            dtype=np.float64,
        )
        archived_valid = np.asarray(
            archive["multiview_view_valid_prefix"][:, :train_end_frame],
            dtype=bool,
        )
        intrinsics = np.asarray(archive["multiview_intrinsics"], dtype=np.float64)
        camera_to_world = np.asarray(
            archive["multiview_camera_to_world"], dtype=np.float64
        )
        cycle_error = np.asarray(
            archive["forward_backward_error_px"][:train_end_frame],
            dtype=np.float64,
        )
        cycle_valid = np.asarray(
            archive["forward_backward_valid"][:train_end_frame], dtype=bool
        )
        boundary = np.asarray(
            archive["boundary_distance"][:train_end_frame], dtype=np.float64
        )
        cue_available = np.asarray(
            archive["cue_available"][:train_end_frame], dtype=bool
        )

    _require(tracks.ndim == 4 and tracks.shape[3] == 2, "track shape changed")
    view_count, frame_count, identity_count, _ = tracks.shape
    _require(frame_count == train_end_frame, "cue archive is shorter than requested")
    _require(quality.shape == tracks.shape[:3], "quality shape changed")
    _require(archived_valid.shape == quality.shape, "view-validity shape changed")
    _require(
        intrinsics.shape == (view_count, 3, 3), "intrinsics shape changed"
    )
    _require(
        camera_to_world.shape == (view_count, 4, 4),
        "camera pose shape changed",
    )
    _require(
        cycle_error.shape == (frame_count, identity_count),
        "cycle-error shape changed",
    )
    _require(cycle_valid.shape == cycle_error.shape, "cycle-validity shape changed")
    _require(boundary.shape == cycle_error.shape, "boundary shape changed")
    _require(cue_available.shape == cycle_error.shape, "cue availability changed")
    initial = np.asarray(initial_world_points_m, dtype=np.float64)
    _require(
        initial.shape == (identity_count, 3),
        "initial world points must have shape (N, 3)",
    )
    _require(np.all(np.isfinite(initial)), "initial world points are not finite")

    world = np.full((view_count, frame_count, identity_count, 3), np.nan)
    depth_valid = np.zeros((view_count, frame_count, identity_count), dtype=bool)
    local_mad = np.zeros((view_count, frame_count, identity_count), dtype=np.float64)
    raw_root = Path(raw_case_dir)
    range_variance = np.full(
        (view_count, frame_count, identity_count),
        cfg.depth_sensor_std_m**2,
        dtype=np.float64,
    )
    for view in range(view_count):
        inverse_intrinsic = np.linalg.inv(intrinsics[view])
        focal = max(
            1e-9,
            float(min(abs(intrinsics[view, 0, 0]), abs(intrinsics[view, 1, 1]))),
        )
        for frame in range(frame_count):
            depth_path = raw_root / "depth" / str(view) / f"{frame}.npy"
            if not depth_path.is_file():
                raise FileNotFoundError(f"missing raw depth frame: {depth_path}")
            depth_m, mad_m, valid_depth = _sample_depth_patch_m(
                np.load(depth_path),
                tracks[view, frame],
                cfg.depth_patch_radius_px,
            )
            local_mad[view, frame] = mad_m
            selected = np.flatnonzero(valid_depth)
            if not len(selected):
                continue
            homogeneous_pixels = np.column_stack(
                (tracks[view, frame, selected], np.ones(len(selected)))
            )
            camera_points = (
                homogeneous_pixels @ inverse_intrinsic.T
            ) * depth_m[selected, None]
            homogeneous_camera = np.column_stack(
                (camera_points, np.ones(len(camera_points)))
            )
            world_points = homogeneous_camera @ camera_to_world[view].T
            world[view, frame, selected] = world_points[:, :3]
            depth_valid[view, frame, selected] = True
            lateral_std = cfg.pixel_noise_std_px * depth_m[selected] / focal
            range_variance[view, frame, selected] = (
                cfg.depth_sensor_std_m**2
                + np.square(mad_m[selected])
                + np.square(lateral_std)
            )

    initial_valid = depth_valid[:, 0] & np.all(np.isfinite(world[:, 0]), axis=2)
    anchored = np.full_like(world, np.nan)
    for view in range(view_count):
        selected = np.flatnonzero(initial_valid[view])
        anchored[view][:, selected] = (
            initial[selected][None]
            + world[view][:, selected]
            - world[view, 0, selected][None]
        )
    cycle_support = (
        cycle_valid
        & np.isfinite(cycle_error)
        & (cycle_error <= cfg.maximum_cycle_error_px)
        & np.isfinite(boundary)
        & (boundary > 0.0)
        & cue_available
    )
    valid = (
        archived_valid
        & depth_valid
        & initial_valid[:, None]
        & np.isfinite(quality)
        & (quality >= cfg.minimum_view_quality)
        & cycle_support[None]
        & (local_mad <= cfg.maximum_local_depth_mad_m)
        & np.all(np.isfinite(anchored), axis=3)
    )
    cycle_weight = np.exp(
        -0.5 * np.square(cycle_error / cfg.maximum_cycle_error_px)
    )
    reliability = np.clip(quality * cycle_weight[None], 0.0, 1.0)
    reliability[~valid] = 0.0
    initial_variance = range_variance[:, 0]
    variance = range_variance + initial_variance[:, None]
    variance = np.maximum(variance, np.finfo(np.float64).tiny)
    return PerViewDepthObservations(
        points_world_m=anchored,
        valid=valid,
        prior_reliability=reliability,
        variance_m2=variance,
        local_depth_mad_m=local_mad,
    )


def _spatial_bias_basis(points_m: np.ndarray) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return np.zeros((len(points), 0), dtype=np.float64)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular_values[0]
    count = int(np.sum(singular_values > tolerance))
    basis = left[:, :count].copy()
    for mode in range(count):
        pivot = int(np.argmax(np.abs(basis[:, mode])))
        if basis[pivot, mode] < 0.0:
            basis[:, mode] *= -1.0
        basis[:, mode] /= np.max(np.abs(basis[:, mode]))
    return basis


def _fallback(
    node_count: int,
    reason: str,
    diagnostics: dict[str, object],
) -> PerViewDepthStateResult:
    return PerViewDepthStateResult(
        accepted=False,
        reason=reason,
        correction_m=np.zeros((node_count, 3), dtype=np.float64),
        coefficient_covariance_m2=np.zeros((0, 0), dtype=np.float64),
        diagnostics=diagnostics,
    )


def infer_per_view_depth_state_correction(
    observations: PerViewDepthObservations,
    baseline_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    *,
    end_frame: int,
    config: PerViewDepthStateConfig | None = None,
) -> PerViewDepthStateResult:
    """Infer a persistent, action-supported graph correction from RGB-D prefix rows."""

    cfg = config or PerViewDepthStateConfig()
    baseline = np.asarray(baseline_m, dtype=np.float64)
    frame_zero = np.asarray(frame_zero_points_m, dtype=np.float64)
    _require(
        baseline.ndim == 3 and baseline.shape[2] == 3,
        "baseline must have shape (T, N, 3)",
    )
    _require(frame_zero.shape == baseline.shape[1:], "frame-zero shape changed")
    _require(np.all(np.isfinite(baseline)), "baseline is not finite")
    _require(np.all(np.isfinite(frame_zero)), "frame-zero points are not finite")
    view_count, observation_frames, observed_count = observations.valid.shape
    _require(observed_count <= baseline.shape[1], "too many observed identities")
    _require(
        1 < end_frame <= min(len(baseline), observation_frames),
        "endpoint lies outside causal observations",
    )
    start = max(0, end_frame - cfg.window_frames)
    valid_window = observations.valid[:, start:end_frame]
    active_views = np.flatnonzero(np.any(valid_window, axis=(1, 2)))
    unique_identities = int(np.sum(np.any(valid_window, axis=(0, 1))))
    diagnostics: dict[str, object] = {
        "start_frame": start,
        "end_frame_exclusive": end_frame,
        "active_view_count": int(len(active_views)),
        "unique_identity_count": unique_identities,
        "prior_reliability_uses_innovation": False,
        "correlation_treatment": (
            "effective-sample cap within each view and equal-weight "
            "covariance intersection across views"
        ),
        "forecast_rule": "persistent endpoint graph correction",
    }
    if len(active_views) < cfg.minimum_active_views:
        return _fallback(len(frame_zero), "insufficient-active-views", diagnostics)
    if unique_identities < cfg.minimum_unique_identities:
        return _fallback(len(frame_zero), "insufficient-identity-support", diagnostics)

    causal_response = baseline[:end_frame] - baseline[0]
    response_magnitude = np.max(np.linalg.norm(causal_response, axis=2), axis=0)
    maximum_response = float(np.max(response_magnitude))
    diagnostics["maximum_physical_response_m"] = maximum_response
    if maximum_response < cfg.minimum_physical_response_m:
        return _fallback(len(frame_zero), "insufficient-physical-response", diagnostics)
    action_support = np.clip(
        response_magnitude / max(maximum_response, np.finfo(np.float64).tiny),
        0.0,
        1.0,
    )
    try:
        physical = build_physical_response_basis(
            causal_response,
            action_support=action_support,
            rank=cfg.physical_response_rank,
            minimum_response_m=cfg.minimum_physical_response_m,
        )
        spatial_bias = _spatial_bias_basis(frame_zero[:observed_count])
        observation_state = physical.basis[:observed_count]
        observation_bias = np.column_stack(
            (spatial_bias, np.ones(observed_count, dtype=np.float64))
        )
        identifiable = restrict_state_basis_to_identifiable_subspace(
            physical.basis,
            observation_state,
            observation_bias,
            minimum_identifiable_fraction=cfg.minimum_identifiable_fraction,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        diagnostics["basis_error"] = f"{type(error).__name__}: {error}"
        return _fallback(len(frame_zero), "unidentifiable-physical-response", diagnostics)

    frame_count = end_frame - start
    point_rows = frame_count * observed_count
    state_rows = np.tile(identifiable.observation_basis, (frame_count, 1))
    shared_bias_rows = np.tile(spatial_bias, (frame_count, 1))
    innovation = (
        observations.points_world_m[:, start:end_frame]
        - baseline[None, start:end_frame, :observed_count]
    ).reshape(view_count, point_rows, 3)
    available = valid_window.reshape(view_count, point_rows)
    reliability = observations.prior_reliability[:, start:end_frame].reshape(
        view_count, point_rows
    )
    variance = observations.variance_m2[:, start:end_frame].reshape(
        view_count, point_rows
    )
    try:
        update = update_bias_aware_state(
            innovation,
            available,
            state_rows,
            shared_bias_rows,
            prior_reliability=reliability,
            observation_variance_m2=variance,
            config=cfg.update,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        diagnostics["update_error"] = f"{type(error).__name__}: {error}"
        return _fallback(len(frame_zero), "state-update-failed", diagnostics)
    diagnostics.update(
        {
            "physical_basis_rank": int(physical.basis.shape[1]),
            "identifiable_basis_rank": int(identifiable.query_basis.shape[1]),
            "minimum_identifiable_fraction": float(
                np.min(identifiable.identifiable_fractions)
            ),
            "state_update": update.diagnostics,
        }
    )
    if not update.accepted:
        return _fallback(len(frame_zero), update.reason, diagnostics)
    correction = decode_bias_aware_state(update, identifiable.query_basis)
    maximum_correction = float(np.max(np.linalg.norm(correction, axis=1)))
    response_ratio = maximum_correction / maximum_response
    diagnostics["maximum_correction_m"] = maximum_correction
    diagnostics["maximum_correction_to_response_ratio"] = response_ratio
    if maximum_correction > cfg.maximum_correction_m:
        return _fallback(len(frame_zero), "absolute-correction-cap", diagnostics)
    if response_ratio > cfg.maximum_correction_to_response_ratio:
        return _fallback(len(frame_zero), "physical-response-relative-cap", diagnostics)
    state_count = len(update.state_coefficients_m)
    coefficient_covariance = update.posterior_covariance_m2[
        :state_count, :state_count
    ]
    return PerViewDepthStateResult(
        accepted=True,
        reason="accepted",
        correction_m=correction,
        coefficient_covariance_m2=coefficient_covariance,
        diagnostics=diagnostics,
    )
