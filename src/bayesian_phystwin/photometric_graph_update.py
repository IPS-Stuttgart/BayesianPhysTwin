"""Correlation-aware photometric evidence for low-rank graph updates.

The module intentionally knows nothing about Gaussian splatting or Warp. A
remote adapter renders the nominal trajectory and finite-difference graph
responses, while this module performs the source-causal inference and prefix
validation. Dense pixels are grouped into spatial blocks, and every
frame-camera receives fixed total information regardless of pixel count.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: object = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    _require(np.all(np.isfinite(result)), "array contains non-finite values")
    result.setflags(write=False)
    return result


def _json_data(value: Mapping[str, Any]) -> dict[str, Any]:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(entry) for entry in item]
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float):
            _require(np.isfinite(item), "diagnostics contain non-finite values")
        return item

    converted = convert(dict(value))
    _require(isinstance(converted, dict), "diagnostics must be a mapping")
    return converted


@dataclass(frozen=True)
class PhotometricGraphConfig:
    """Frozen settings for one prefix-only graph-update decision."""

    fit_frame_count: int = 4
    correlation_block_size: int = 16
    state_ridge: float = 0.05
    nuisance_ridge: float = 1e-6
    huber_threshold: float = 2.5
    maximum_iterations: int = 8
    convergence_tolerance: float = 1e-7
    maximum_weight_norm: float = 2.0
    minimum_fit_groups: int = 12
    minimum_validation_groups: int = 6
    minimum_validation_improvement_fraction: float = 0.02
    minimum_validation_improvement_absolute: float = 5e-4

    def __post_init__(self) -> None:
        _require(self.fit_frame_count >= 2, "fit_frame_count must be at least two")
        _require(
            self.correlation_block_size >= 1,
            "correlation_block_size must be positive",
        )
        _require(self.state_ridge > 0.0, "state_ridge must be positive")
        _require(self.nuisance_ridge > 0.0, "nuisance_ridge must be positive")
        _require(self.huber_threshold > 0.0, "huber_threshold must be positive")
        _require(self.maximum_iterations >= 1, "maximum_iterations must be positive")
        _require(
            self.convergence_tolerance > 0.0,
            "convergence_tolerance must be positive",
        )
        _require(
            self.maximum_weight_norm > 0.0,
            "maximum_weight_norm must be positive",
        )
        _require(self.minimum_fit_groups >= 1, "minimum_fit_groups must be positive")
        _require(
            self.minimum_validation_groups >= 1,
            "minimum_validation_groups must be positive",
        )
        _require(
            0.0 <= self.minimum_validation_improvement_fraction < 1.0,
            "minimum validation improvement must lie in [0, 1)",
        )
        _require(
            self.minimum_validation_improvement_absolute >= 0.0,
            "minimum absolute validation improvement must be nonnegative",
        )


@dataclass(frozen=True)
class PhotometricGraphSelection:
    """Result of the held-out prefix decision.

    Rejected selections carry an exact all-zero graph vector. The caller can
    therefore route the original physical belief without reconstructing it.
    """

    accepted: bool
    reason: str
    state_weights: np.ndarray
    posterior_covariance: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = _readonly(self.state_weights)
        covariance = _readonly(self.posterior_covariance)
        _require(weights.ndim == 1 and len(weights) >= 1, "weights must be a vector")
        _require(
            covariance.shape == (len(weights), len(weights)),
            "posterior covariance shape changed",
        )
        _require(
            np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0),
            "posterior covariance must be symmetric",
        )
        if not self.accepted:
            _require(
                np.array_equal(weights, np.zeros_like(weights)),
                "rejected graph weights must be exact zeros",
            )
        object.__setattr__(self, "state_weights", weights)
        object.__setattr__(self, "posterior_covariance", covariance)
        object.__setattr__(self, "diagnostics", _json_data(self.diagnostics))


@dataclass(frozen=True)
class _Rows:
    observed: np.ndarray
    baseline: np.ndarray
    jacobian: np.ndarray
    frame: np.ndarray
    camera: np.ndarray
    channel: np.ndarray
    group: np.ndarray
    base_weight: np.ndarray


def _validate_inputs(
    observed_rgb: np.ndarray,
    baseline_rgb: np.ndarray,
    state_jacobian_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    fit_frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observed_rgb, dtype=np.float64)
    baseline = np.asarray(baseline_rgb, dtype=np.float64)
    jacobian = np.asarray(state_jacobian_rgb, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    _require(
        observed.ndim == 5 and observed.shape[-1] == 3,
        "observed_rgb must have shape (frame, camera, height, width, 3)",
    )
    _require(baseline.shape == observed.shape, "baseline_rgb shape changed")
    _require(
        jacobian.shape[:5] == observed.shape and jacobian.ndim == 6,
        "state_jacobian_rgb must append one parameter dimension",
    )
    _require(mask.shape == observed.shape[:-1], "valid_mask shape changed")
    _require(
        fit_frame_count < observed.shape[0],
        "selection requires held-out prefix frames",
    )
    _require(jacobian.shape[-1] >= 1, "state Jacobian has no parameters")
    finite = (
        np.all(np.isfinite(observed), axis=-1)
        & np.all(np.isfinite(baseline), axis=-1)
        & np.all(np.isfinite(jacobian), axis=(-1, -2))
    )
    mask = mask & finite
    return observed, baseline, jacobian, mask


def _build_rows(
    observed: np.ndarray,
    baseline: np.ndarray,
    jacobian: np.ndarray,
    mask: np.ndarray,
    *,
    frame_start: int,
    frame_stop: int,
    block_size: int,
) -> _Rows:
    entries: list[tuple[Any, ...]] = []
    group_position = 0
    for frame in range(frame_start, frame_stop):
        usable_cameras = [
            camera
            for camera in range(observed.shape[1])
            if np.any(mask[frame, camera])
        ]
        if not usable_cameras:
            continue
        camera_mass = 1.0 / len(usable_cameras)
        for camera in usable_cameras:
            camera_mask = mask[frame, camera]
            blocks: list[tuple[np.ndarray, np.ndarray]] = []
            for row in range(0, camera_mask.shape[0], block_size):
                for column in range(0, camera_mask.shape[1], block_size):
                    local = camera_mask[
                        row : row + block_size,
                        column : column + block_size,
                    ]
                    coordinates = np.argwhere(local)
                    if len(coordinates):
                        coordinates[:, 0] += row
                        coordinates[:, 1] += column
                        blocks.append((coordinates[:, 0], coordinates[:, 1]))
            if not blocks:
                continue
            block_mass = camera_mass / len(blocks)
            for rows, columns in blocks:
                scalar_count = 3 * len(rows)
                scalar_weight = block_mass / scalar_count
                for channel in range(3):
                    entries.append(
                        (
                            observed[frame, camera, rows, columns, channel],
                            baseline[frame, camera, rows, columns, channel],
                            jacobian[
                                frame,
                                camera,
                                rows,
                                columns,
                                channel,
                                :,
                            ],
                            np.full(len(rows), frame, dtype=np.int64),
                            np.full(len(rows), camera, dtype=np.int64),
                            np.full(len(rows), channel, dtype=np.int64),
                            np.full(len(rows), group_position, dtype=np.int64),
                            np.full(len(rows), scalar_weight, dtype=np.float64),
                        )
                    )
                group_position += 1
    _require(entries, "prefix interval has no valid photometric support")
    columns = list(zip(*entries, strict=True))
    return _Rows(
        observed=np.concatenate(columns[0]),
        baseline=np.concatenate(columns[1]),
        jacobian=np.concatenate(columns[2], axis=0),
        frame=np.concatenate(columns[3]),
        camera=np.concatenate(columns[4]),
        channel=np.concatenate(columns[5]),
        group=np.concatenate(columns[6]),
        base_weight=np.concatenate(columns[7]),
    )


def _profile_affine_nuisance(
    observed: np.ndarray,
    predicted: np.ndarray,
    jacobian: np.ndarray | None,
    frame: np.ndarray,
    camera: np.ndarray,
    channel: np.ndarray,
    weight: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    residual = np.empty_like(observed)
    profiled_jacobian = None if jacobian is None else np.empty_like(jacobian)
    keys = np.stack((frame, camera, channel), axis=1)
    for key in np.unique(keys, axis=0):
        selected = np.all(keys == key, axis=1)
        design = np.column_stack((np.ones(np.sum(selected)), predicted[selected]))
        local_weight = weight[selected]
        normal = design.T @ (local_weight[:, None] * design)
        normal += ridge * np.eye(2)
        solve = np.linalg.solve(normal, design.T * local_weight[None])
        fitted_observed = design @ (solve @ observed[selected])
        residual[selected] = observed[selected] - fitted_observed
        if jacobian is not None and profiled_jacobian is not None:
            profiled_jacobian[selected] = (
                jacobian[selected] - design @ (solve @ jacobian[selected])
            )
    return residual, profiled_jacobian


def _group_robust_weights(
    residual: np.ndarray,
    group: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, dict[str, float]]:
    group_ids = np.unique(group)
    rms = np.asarray(
        [
            np.sqrt(np.mean(np.square(residual[group == group_id])))
            for group_id in group_ids
        ],
        dtype=np.float64,
    )
    median = float(np.median(rms))
    scale = 1.4826 * float(np.median(np.abs(rms - median)))
    scale = max(scale, 1e-6)
    cutoff = median + threshold * scale
    robust = np.ones_like(rms)
    large = rms > cutoff
    robust[large] = cutoff / np.maximum(rms[large], 1e-12)
    row_weight = np.empty(len(group), dtype=np.float64)
    for position, group_id in enumerate(group_ids):
        row_weight[group == group_id] = robust[position]
    return row_weight, {
        "group_count": float(len(group_ids)),
        "group_rms_median": median,
        "group_rms_robust_scale": scale,
        "minimum_group_weight": float(np.min(robust)),
        "mean_group_weight": float(np.mean(robust)),
    }


def _limit_weights(weights: np.ndarray, maximum_norm: float) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(weights))
    scale = min(1.0, maximum_norm / max(norm, 1e-12))
    return weights * scale, scale


def _fit_weights(
    rows: _Rows,
    *,
    config: PhotometricGraphConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    parameter_count = rows.jacobian.shape[1]
    robust_weight = np.ones(len(rows.observed), dtype=np.float64)
    weights = np.zeros(parameter_count, dtype=np.float64)
    fit_diagnostics: dict[str, Any] = {}
    profiled_jacobian = rows.jacobian
    for iteration in range(config.maximum_iterations):
        scalar_weight = rows.base_weight * robust_weight
        residual, profiled = _profile_affine_nuisance(
            rows.observed,
            rows.baseline,
            rows.jacobian,
            rows.frame,
            rows.camera,
            rows.channel,
            scalar_weight,
            ridge=config.nuisance_ridge,
        )
        _require(profiled is not None, "profiled Jacobian is missing")
        profiled_jacobian = profiled
        normal = profiled.T @ (scalar_weight[:, None] * profiled)
        normal += config.state_ridge * np.eye(parameter_count)
        right = profiled.T @ (scalar_weight * residual)
        candidate = np.linalg.solve(normal, right)
        candidate, limit_scale = _limit_weights(
            candidate,
            config.maximum_weight_norm,
        )
        model_residual = residual - profiled @ candidate
        new_robust, robust_diagnostics = _group_robust_weights(
            model_residual,
            rows.group,
            threshold=config.huber_threshold,
        )
        change = float(np.linalg.norm(candidate - weights))
        weights = candidate
        robust_weight = new_robust
        fit_diagnostics = {
            "iterations": iteration + 1,
            "last_weight_change": change,
            "weight_limit_scale": limit_scale,
            "weight_norm": float(np.linalg.norm(weights)),
            **robust_diagnostics,
        }
        if change <= config.convergence_tolerance:
            break
    scalar_weight = rows.base_weight * robust_weight
    information = profiled_jacobian.T @ (
        scalar_weight[:, None] * profiled_jacobian
    )
    covariance = np.linalg.inv(information + config.state_ridge * np.eye(parameter_count))
    covariance = 0.5 * (covariance + covariance.T)
    fit_diagnostics["information_trace"] = float(np.trace(information))
    fit_diagnostics["posterior_trace"] = float(np.trace(covariance))
    return weights, covariance, fit_diagnostics


def _profiled_score(
    rows: _Rows,
    weights: np.ndarray,
    *,
    config: PhotometricGraphConfig,
) -> tuple[float, dict[str, Any]]:
    predicted = rows.baseline + rows.jacobian @ weights
    robust_weight = np.ones(len(rows.observed), dtype=np.float64)
    diagnostics: dict[str, float] = {}
    for _ in range(config.maximum_iterations):
        residual, _ = _profile_affine_nuisance(
            rows.observed,
            predicted,
            None,
            rows.frame,
            rows.camera,
            rows.channel,
            rows.base_weight * robust_weight,
            ridge=config.nuisance_ridge,
        )
        updated, diagnostics = _group_robust_weights(
            residual,
            rows.group,
            threshold=config.huber_threshold,
        )
        if np.max(np.abs(updated - robust_weight)) <= config.convergence_tolerance:
            robust_weight = updated
            break
        robust_weight = updated
    weight = rows.base_weight * robust_weight
    denominator = float(np.sum(weight))
    _require(denominator > 0.0, "score has no effective support")
    return (
        float(np.sqrt(np.sum(weight * np.square(residual)) / denominator)),
        diagnostics,
    )


def select_photometric_graph_update(
    observed_rgb: np.ndarray,
    baseline_rgb: np.ndarray,
    state_jacobian_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    config: PhotometricGraphConfig | None = None,
) -> PhotometricGraphSelection:
    """Fit on early prefix images and gate on later prefix images.

    The valid mask is the only prior reliability input. It must be created
    without inspecting the PhysTwin innovation. Image residuals enter once,
    through the robust grouped fit and held-out score.
    """

    cfg = config or PhotometricGraphConfig()
    observed, baseline, jacobian, mask = _validate_inputs(
        observed_rgb,
        baseline_rgb,
        state_jacobian_rgb,
        valid_mask,
        fit_frame_count=cfg.fit_frame_count,
    )
    fit_rows = _build_rows(
        observed,
        baseline,
        jacobian,
        mask,
        frame_start=0,
        frame_stop=cfg.fit_frame_count,
        block_size=cfg.correlation_block_size,
    )
    validation_rows = _build_rows(
        observed,
        baseline,
        jacobian,
        mask,
        frame_start=cfg.fit_frame_count,
        frame_stop=len(observed),
        block_size=cfg.correlation_block_size,
    )
    fit_group_count = len(np.unique(fit_rows.group))
    validation_group_count = len(np.unique(validation_rows.group))
    parameter_count = jacobian.shape[-1]
    zero = np.zeros(parameter_count, dtype=np.float64)
    if (
        fit_group_count < cfg.minimum_fit_groups
        or validation_group_count < cfg.minimum_validation_groups
    ):
        return PhotometricGraphSelection(
            accepted=False,
            reason="insufficient-correlation-cluster-support",
            state_weights=zero,
            posterior_covariance=np.eye(parameter_count) / cfg.state_ridge,
            diagnostics={
                "fit_group_count": fit_group_count,
                "validation_group_count": validation_group_count,
                "selection_uses_future_frames": False,
            },
        )

    fit_weights, fit_covariance, fit_diagnostics = _fit_weights(
        fit_rows,
        config=cfg,
    )
    baseline_score, baseline_diagnostics = _profiled_score(
        validation_rows,
        zero,
        config=cfg,
    )
    candidate_score, candidate_diagnostics = _profiled_score(
        validation_rows,
        fit_weights,
        config=cfg,
    )
    improvement_absolute = baseline_score - candidate_score
    improvement_fraction = (
        improvement_absolute / baseline_score if baseline_score > 0.0 else 0.0
    )
    diagnostics: dict[str, Any] = {
        "fit_group_count": fit_group_count,
        "validation_group_count": validation_group_count,
        "fit": fit_diagnostics,
        "baseline_validation_score": baseline_score,
        "candidate_validation_score": candidate_score,
        "validation_improvement_absolute": improvement_absolute,
        "validation_improvement_fraction": improvement_fraction,
        "minimum_validation_improvement_absolute": (
            cfg.minimum_validation_improvement_absolute
        ),
        "minimum_validation_improvement_fraction": (
            cfg.minimum_validation_improvement_fraction
        ),
        "baseline_validation": baseline_diagnostics,
        "candidate_validation": candidate_diagnostics,
        "selection_uses_future_frames": False,
        "pixel_count_does_not_define_information_mass": True,
        "camera_information_mass_is_normalized": True,
        "per_camera_affine_color_is_profiled": True,
    }
    if (
        improvement_absolute < cfg.minimum_validation_improvement_absolute
        or improvement_fraction < cfg.minimum_validation_improvement_fraction
    ):
        return PhotometricGraphSelection(
            accepted=False,
            reason="held-out-prefix-regret-guard",
            state_weights=zero,
            posterior_covariance=fit_covariance,
            diagnostics=diagnostics,
        )

    full_rows = _build_rows(
        observed,
        baseline,
        jacobian,
        mask,
        frame_start=0,
        frame_stop=len(observed),
        block_size=cfg.correlation_block_size,
    )
    full_weights, full_covariance, full_diagnostics = _fit_weights(
        full_rows,
        config=cfg,
    )
    diagnostics["full_prefix_refit"] = full_diagnostics
    return PhotometricGraphSelection(
        accepted=True,
        reason="accepted",
        state_weights=full_weights,
        posterior_covariance=full_covariance,
        diagnostics=diagnostics,
    )


__all__ = [
    "PhotometricGraphConfig",
    "PhotometricGraphSelection",
    "select_photometric_graph_update",
]
