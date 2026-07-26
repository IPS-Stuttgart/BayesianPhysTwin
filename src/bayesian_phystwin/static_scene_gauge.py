"""Static-scene nuisance estimation for causal material-point tracking.

The deformable object may move, but calibrated background structure should not.
This module uses tracker motion on that static structure to estimate a local
image-space nuisance field.  The field is learned without a PhysTwin state
innovation and is admitted only when it improves spatially held-out background
tracks.  Rejection leaves the object tracks byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

STATIC_SCENE_GAUGE_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class StaticSceneGaugeConfig:
    """Frozen selection, interpolation, and admission settings."""

    query_stride_px: int = 20
    dynamic_margin_px: int = 0
    maximum_query_count: int = 768
    correlation_cell_size_px: int = 20
    cross_validation_cell_size_px: int = 40
    neighbor_count: int = 16
    rbf_bandwidth_px: float = 24.0
    rbf_cutoff_multiplier: float = 3.0
    maximum_query_distance_px: float = 48.0
    minimum_quality: float = 0.5
    maximum_background_displacement_px: float = 10.0
    minimum_effective_support: float = 3.0
    minimum_variance_px2: float = 0.25**2
    minimum_cross_validation_count: int = 128
    minimum_cross_validation_raw_error_px: float = 0.05
    minimum_cross_validation_gain: float = 0.10

    def __post_init__(self) -> None:
        _require(self.query_stride_px >= 2, "query stride is too small")
        _require(self.dynamic_margin_px >= 0, "dynamic margin is negative")
        _require(self.maximum_query_count >= 8, "query cap is too small")
        _require(
            self.correlation_cell_size_px >= 1,
            "correlation cell size must be positive",
        )
        _require(
            self.cross_validation_cell_size_px
            >= self.correlation_cell_size_px,
            "cross-validation cells must not be smaller than correlation cells",
        )
        _require(self.neighbor_count >= 1, "neighbor count must be positive")
        _require(self.rbf_bandwidth_px > 0.0, "RBF bandwidth must be positive")
        _require(
            self.rbf_cutoff_multiplier >= 1.0,
            "RBF cutoff multiplier is too small",
        )
        _require(
            self.maximum_query_distance_px > 0.0,
            "maximum query distance must be positive",
        )
        _require(
            0.0 <= self.minimum_quality <= 1.0,
            "minimum quality must lie in [0, 1]",
        )
        _require(
            self.maximum_background_displacement_px > 0.0,
            "maximum background displacement must be positive",
        )
        _require(
            self.minimum_effective_support >= 1.0,
            "minimum effective support must be at least one",
        )
        _require(
            self.minimum_variance_px2 > 0.0,
            "minimum variance must be positive",
        )
        _require(
            self.minimum_cross_validation_count >= 1,
            "cross-validation support must be positive",
        )
        _require(
            self.minimum_cross_validation_raw_error_px >= 0.0,
            "cross-validation raw-error floor is negative",
        )
        _require(
            0.0 <= self.minimum_cross_validation_gain < 1.0,
            "cross-validation gain must lie in [0, 1)",
        )


@dataclass(frozen=True)
class StaticSceneGaugeEstimate:
    """A target-free local tracker-nuisance estimate at requested pixels."""

    correction_px: np.ndarray
    variance_px2: np.ndarray
    supported: np.ndarray
    effective_support: np.ndarray
    accepted: bool
    reason: str
    background_cluster_count: int
    cross_validation_count: int
    cross_validation_raw_error_px: float | None
    cross_validation_corrected_error_px: float | None
    cross_validation_relative_gain: float | None
    config: StaticSceneGaugeConfig

    def __post_init__(self) -> None:
        correction = np.asarray(self.correction_px, dtype=np.float64).copy()
        variance = np.asarray(self.variance_px2, dtype=np.float64).copy()
        supported = np.asarray(self.supported, dtype=bool).copy()
        effective = np.asarray(self.effective_support, dtype=np.float64).copy()
        _require(
            correction.ndim == 3 and correction.shape[2] == 2,
            "correction must have shape (T, N, 2)",
        )
        _require(
            variance.shape == correction.shape[:2],
            "variance must match correction rows",
        )
        _require(
            supported.shape == variance.shape,
            "support must match correction rows",
        )
        _require(
            effective.shape == variance.shape,
            "effective support must match correction rows",
        )
        _require(
            np.all(np.isfinite(correction)),
            "correction contains non-finite values",
        )
        _require(
            np.all(np.isfinite(variance)) and np.all(variance > 0.0),
            "variance must be finite and positive",
        )
        _require(
            np.all(np.isfinite(effective)) and np.all(effective >= 0.0),
            "effective support is invalid",
        )
        _require(
            self.background_cluster_count >= 1,
            "background cluster count must be positive",
        )
        _require(
            self.cross_validation_count >= 0,
            "cross-validation count is negative",
        )
        if not self.accepted:
            _require(
                not np.any(supported),
                "a rejected estimate cannot mark rows supported",
            )
            _require(
                np.array_equal(correction, np.zeros_like(correction)),
                "a rejected estimate must contain a zero correction",
            )
        correction.setflags(write=False)
        variance.setflags(write=False)
        supported.setflags(write=False)
        effective.setflags(write=False)
        object.__setattr__(self, "correction_px", correction)
        object.__setattr__(self, "variance_px2", variance)
        object.__setattr__(self, "supported", supported)
        object.__setattr__(self, "effective_support", effective)

    @property
    def content_sha256(self) -> str:
        """Hash the complete decision and numerical output."""

        digest = hashlib.sha256()
        metadata = {
            "schema_version": STATIC_SCENE_GAUGE_SCHEMA_VERSION,
            "accepted": self.accepted,
            "reason": self.reason,
            "background_cluster_count": self.background_cluster_count,
            "cross_validation_count": self.cross_validation_count,
            "cross_validation_raw_error_px": (
                self.cross_validation_raw_error_px
            ),
            "cross_validation_corrected_error_px": (
                self.cross_validation_corrected_error_px
            ),
            "cross_validation_relative_gain": (
                self.cross_validation_relative_gain
            ),
            "config": asdict(self.config),
        }
        digest.update(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for array in (
            self.correction_px,
            self.variance_px2,
            self.supported,
            self.effective_support,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    _require(result.ndim == 2, "mask must be two-dimensional")
    if radius == 0:
        return result.copy()
    padded = np.pad(result, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(result)
    height, width = result.shape
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            dilated |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return dilated


def select_static_scene_queries(
    dynamic_masks: np.ndarray,
    valid_initial_depth: np.ndarray,
    *,
    config: StaticSceneGaugeConfig | None = None,
) -> np.ndarray:
    """Select deterministic pixels static throughout the allowed prefix."""

    cfg = config or StaticSceneGaugeConfig()
    masks = np.asarray(dynamic_masks, dtype=bool)
    depth_valid = np.asarray(valid_initial_depth, dtype=bool)
    _require(
        masks.ndim == 3,
        "dynamic masks must have shape (T, H, W)",
    )
    _require(
        depth_valid.shape == masks.shape[1:],
        "depth validity must match the image shape",
    )
    dynamic = _dilate(np.any(masks, axis=0), cfg.dynamic_margin_px)
    eligible = depth_valid & ~dynamic
    offset = cfg.query_stride_px // 2
    rows = np.arange(offset, eligible.shape[0], cfg.query_stride_px)
    columns = np.arange(offset, eligible.shape[1], cfg.query_stride_px)
    queries = np.asarray(
        [
            (column, row)
            for row in rows
            for column in columns
            if eligible[row, column]
        ],
        dtype=np.float64,
    )
    _require(len(queries) >= 2, "static-scene query support is empty")
    if len(queries) > cfg.maximum_query_count:
        selected = np.linspace(
            0,
            len(queries) - 1,
            cfg.maximum_query_count,
            dtype=np.int64,
        )
        queries = queries[selected]
    return queries


def _cluster_background_tracks(
    query_pixels_xy: np.ndarray,
    tracks_xy: np.ndarray,
    quality_probability: np.ndarray,
    *,
    cell_size_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(query_pixels_xy, dtype=np.float64)
    tracks = np.asarray(tracks_xy, dtype=np.float64)
    quality = np.asarray(quality_probability, dtype=np.float64)
    _require(
        queries.ndim == 2 and queries.shape[1] == 2,
        "background queries must have shape (M, 2)",
    )
    _require(
        tracks.ndim == 3
        and tracks.shape[1:] == (len(queries), 2),
        "background tracks must have shape (T, M, 2)",
    )
    _require(
        quality.shape == tracks.shape[:2],
        "background quality must have shape (T, M)",
    )
    _require(
        np.all(np.isfinite(queries)),
        "background queries contain non-finite values",
    )
    _require(
        np.all(np.isfinite(quality))
        and np.all((quality >= 0.0) & (quality <= 1.0)),
        "background quality must lie in [0, 1]",
    )
    cell = np.floor(queries / cell_size_px).astype(np.int64)
    _, inverse = np.unique(cell, axis=0, return_inverse=True)
    cluster_count = int(np.max(inverse)) + 1
    displacement = tracks - tracks[0:1]
    clustered_queries = np.empty((cluster_count, 2), dtype=np.float64)
    clustered_displacement = np.full(
        (len(tracks), cluster_count, 2),
        np.nan,
        dtype=np.float64,
    )
    clustered_quality = np.zeros(
        (len(tracks), cluster_count),
        dtype=np.float64,
    )
    for cluster in range(cluster_count):
        selected = inverse == cluster
        clustered_queries[cluster] = np.mean(queries[selected], axis=0)
        for frame in range(len(tracks)):
            valid = selected & np.all(np.isfinite(displacement[frame]), axis=1)
            if not np.any(valid):
                continue
            clustered_displacement[frame, cluster] = np.median(
                displacement[frame, valid],
                axis=0,
            )
            clustered_quality[frame, cluster] = float(
                np.median(quality[frame, valid])
            )
    return (
        clustered_queries,
        clustered_displacement,
        clustered_quality,
    )


def _nearest_neighbors(
    reference_xy: np.ndarray,
    query_xy: np.ndarray,
    neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(reference_xy, dtype=np.float64)
    query = np.asarray(query_xy, dtype=np.float64)
    _require(
        reference.ndim == 2 and reference.shape[1] == 2 and len(reference) > 0,
        "reference pixels must have nonempty shape (M, 2)",
    )
    _require(
        query.ndim == 2 and query.shape[1] == 2,
        "query pixels must have shape (N, 2)",
    )
    finite = np.all(np.isfinite(query), axis=1)
    count = min(neighbor_count, len(reference))
    distances = np.full((len(query), count), np.inf, dtype=np.float64)
    indices = np.zeros((len(query), count), dtype=np.int64)
    selected = np.flatnonzero(finite)
    for start in range(0, len(selected), 1024):
        rows = selected[start : start + 1024]
        squared = np.sum(
            (query[rows, None, :] - reference[None, :, :]) ** 2,
            axis=2,
        )
        if count == len(reference):
            local = np.argsort(squared, axis=1)[:, :count]
        else:
            local = np.argpartition(squared, count - 1, axis=1)[:, :count]
            local_squared = np.take_along_axis(squared, local, axis=1)
            order = np.argsort(local_squared, axis=1)
            local = np.take_along_axis(local, order, axis=1)
        indices[rows] = local
        distances[rows] = np.sqrt(
            np.take_along_axis(squared, local, axis=1)
        )
    return distances, indices, finite


def _predict_local_drift(
    background_xy: np.ndarray,
    background_displacement_px: np.ndarray,
    background_quality: np.ndarray,
    query_xy: np.ndarray,
    config: StaticSceneGaugeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances, indices, finite_query = _nearest_neighbors(
        background_xy,
        query_xy,
        config.neighbor_count,
    )
    frame_count = background_displacement_px.shape[0]
    correction = np.zeros((frame_count, len(query_xy), 2), dtype=np.float64)
    variance = np.full(
        (frame_count, len(query_xy)),
        config.minimum_variance_px2,
        dtype=np.float64,
    )
    supported = np.zeros((frame_count, len(query_xy)), dtype=bool)
    effective = np.zeros((frame_count, len(query_xy)), dtype=np.float64)
    cutoff = config.rbf_cutoff_multiplier * config.rbf_bandwidth_px
    spatial_weight = np.exp(
        -0.5 * (distances / config.rbf_bandwidth_px) ** 2
    )
    spatial_weight[distances > cutoff] = 0.0
    for frame in range(frame_count):
        values = background_displacement_px[frame, indices]
        quality = background_quality[frame, indices]
        valid = (
            np.all(np.isfinite(values), axis=2)
            & np.isfinite(quality)
            & (quality >= config.minimum_quality)
            & (
                np.linalg.norm(values, axis=2)
                <= config.maximum_background_displacement_px
            )
        )
        weights = spatial_weight * quality * valid
        weight_sum = np.sum(weights, axis=1)
        squared_weight_sum = np.sum(weights**2, axis=1)
        frame_effective = weight_sum**2 / np.maximum(
            squared_weight_sum,
            1e-12,
        )
        frame_correction = np.sum(
            weights[:, :, None] * np.where(valid[:, :, None], values, 0.0),
            axis=1,
        ) / np.maximum(weight_sum[:, None], 1e-12)
        residual = np.where(
            valid[:, :, None],
            values - frame_correction[:, None, :],
            0.0,
        )
        frame_variance = (
            np.sum(weights * np.sum(residual**2, axis=2), axis=1)
            / np.maximum(weight_sum, 1e-12)
            / 2.0
        )
        frame_supported = (
            finite_query
            & (distances[:, 0] <= config.maximum_query_distance_px)
            & (frame_effective >= config.minimum_effective_support)
        )
        correction[frame, frame_supported] = frame_correction[
            frame_supported
        ]
        variance[frame, frame_supported] = np.maximum(
            frame_variance[frame_supported],
            config.minimum_variance_px2,
        )
        supported[frame] = frame_supported
        effective[frame] = np.where(
            frame_supported,
            frame_effective,
            0.0,
        )
    return correction, variance, supported, effective


def _cross_validate(
    background_xy: np.ndarray,
    background_displacement_px: np.ndarray,
    background_quality: np.ndarray,
    config: StaticSceneGaugeConfig,
) -> tuple[int, float | None, float | None, float | None]:
    cells = np.floor(
        background_xy / config.cross_validation_cell_size_px
    ).astype(np.int64)
    parity = (cells[:, 0] + cells[:, 1]) % 2
    raw_error: list[np.ndarray] = []
    corrected_error: list[np.ndarray] = []
    for fold in (0, 1):
        fit = parity != fold
        score = parity == fold
        if (
            np.sum(fit) < config.minimum_effective_support
            or not np.any(score)
        ):
            continue
        correction, _, supported, _ = _predict_local_drift(
            background_xy[fit],
            background_displacement_px[:, fit],
            background_quality[:, fit],
            background_xy[score],
            config,
        )
        target = background_displacement_px[:, score]
        target_valid = (
            np.all(np.isfinite(target), axis=2)
            & (background_quality[:, score] >= config.minimum_quality)
        )
        valid = supported & target_valid
        valid[0] = False
        if not np.any(valid):
            continue
        raw_error.append(np.linalg.norm(target[valid], axis=1))
        corrected_error.append(
            np.linalg.norm((target - correction)[valid], axis=1)
        )
    count = int(sum(len(values) for values in raw_error))
    if count == 0:
        return 0, None, None, None
    raw = float(np.mean(np.concatenate(raw_error)))
    corrected = float(np.mean(np.concatenate(corrected_error)))
    gain = None if raw <= 0.0 else float(1.0 - corrected / raw)
    return count, raw, corrected, gain


def estimate_static_scene_gauge(
    background_query_pixels_xy: np.ndarray,
    background_tracks_xy: np.ndarray,
    background_quality_probability: np.ndarray,
    requested_query_pixels_xy: np.ndarray,
    *,
    config: StaticSceneGaugeConfig | None = None,
) -> StaticSceneGaugeEstimate:
    """Estimate a locally varying tracker nuisance without state residuals."""

    cfg = config or StaticSceneGaugeConfig()
    (
        background_xy,
        background_displacement,
        background_quality,
    ) = _cluster_background_tracks(
        background_query_pixels_xy,
        background_tracks_xy,
        background_quality_probability,
        cell_size_px=cfg.correlation_cell_size_px,
    )
    correction, variance, supported, effective = _predict_local_drift(
        background_xy,
        background_displacement,
        background_quality,
        np.asarray(requested_query_pixels_xy, dtype=np.float64),
        cfg,
    )
    count, raw, corrected, gain = _cross_validate(
        background_xy,
        background_displacement,
        background_quality,
        cfg,
    )
    accepted = (
        count >= cfg.minimum_cross_validation_count
        and raw is not None
        and raw >= cfg.minimum_cross_validation_raw_error_px
        and gain is not None
        and gain >= cfg.minimum_cross_validation_gain
    )
    if accepted:
        reason = "held-out-static-scene-gain"
    elif count < cfg.minimum_cross_validation_count:
        reason = "insufficient-held-out-static-scene-support"
    elif raw is None or raw < cfg.minimum_cross_validation_raw_error_px:
        reason = "static-scene-drift-below-actionable-floor"
    else:
        reason = "held-out-static-scene-gain-failed"
    if not accepted:
        correction = np.zeros_like(correction)
        supported = np.zeros_like(supported)
        effective = np.zeros_like(effective)
    return StaticSceneGaugeEstimate(
        correction_px=correction,
        variance_px2=variance,
        supported=supported,
        effective_support=effective,
        accepted=accepted,
        reason=reason,
        background_cluster_count=len(background_xy),
        cross_validation_count=count,
        cross_validation_raw_error_px=raw,
        cross_validation_corrected_error_px=corrected,
        cross_validation_relative_gain=gain,
        config=cfg,
    )


def apply_static_scene_gauge(
    tracks_xy: np.ndarray,
    estimate: StaticSceneGaugeEstimate,
) -> np.ndarray:
    """Subtract an admitted nuisance field while preserving exact fallback."""

    tracks = np.asarray(tracks_xy)
    _require(
        tracks.ndim == 3 and tracks.shape[2] == 2,
        "tracks must have shape (T, N, 2)",
    )
    _require(
        tracks.shape == estimate.correction_px.shape,
        "track and gauge shapes differ",
    )
    corrected = tracks.copy()
    if not estimate.accepted:
        return corrected
    selected = estimate.supported
    corrected[selected] = (
        corrected[selected] - estimate.correction_px[selected]
    )
    return corrected


__all__ = [
    "STATIC_SCENE_GAUGE_SCHEMA_VERSION",
    "StaticSceneGaugeConfig",
    "StaticSceneGaugeEstimate",
    "apply_static_scene_gauge",
    "estimate_static_scene_gauge",
    "select_static_scene_queries",
]
