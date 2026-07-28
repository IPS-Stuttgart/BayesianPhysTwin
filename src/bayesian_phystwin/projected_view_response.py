"""Residual-independent camera-space evidence for physical response admission."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class ProjectedViewResponseConfig:
    """Pixel-to-metric uncertainty and cycle-association settings."""

    pixel_standard_deviation: float = 2.0
    covariance_inflation: float = 4.0
    covariance_floor_m2: float = 1e-8
    cycle_error_scale_px: float = 2.0
    minimum_initial_depth_m: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.pixel_standard_deviation,
            self.covariance_inflation,
            self.covariance_floor_m2,
            self.cycle_error_scale_px,
            self.minimum_initial_depth_m,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in values),
            "projected-view scales must be finite and positive",
        )


@dataclass(frozen=True)
class ProjectedViewResponse:
    """Camera-tangent observations ready for action-response admission."""

    physical_positions_m: np.ndarray
    observed_positions_m: np.ndarray
    observation_validity: np.ndarray
    observation_covariance_m2: np.ndarray
    prior_reliability: np.ndarray
    association_probability: np.ndarray
    cycle_error_px: np.ndarray

    def __post_init__(self) -> None:
        physical = np.asarray(
            self.physical_positions_m,
            dtype=np.float64,
        ).copy()
        observed = np.asarray(
            self.observed_positions_m,
            dtype=np.float64,
        ).copy()
        valid = np.asarray(self.observation_validity, dtype=bool).copy()
        covariance = np.asarray(
            self.observation_covariance_m2,
            dtype=np.float64,
        ).copy()
        reliability = np.asarray(
            self.prior_reliability,
            dtype=np.float64,
        ).copy()
        association = np.asarray(
            self.association_probability,
            dtype=np.float64,
        ).copy()
        cycle = np.asarray(self.cycle_error_px, dtype=np.float64).copy()
        _require(
            physical.ndim == 4
            and physical.shape[3] == 3
            and observed.shape == physical.shape,
            "projected positions must have shape (S, T, N, 3)",
        )
        expected = physical.shape[:3]
        _require(valid.shape == expected, "observation validity shape changed")
        _require(
            covariance.shape == (*expected, 3, 3),
            "observation covariance shape changed",
        )
        _require(reliability.shape == expected, "prior reliability shape changed")
        _require(association.shape == expected, "association shape changed")
        _require(cycle.shape == expected, "cycle error shape changed")
        _require(
            np.all(np.isfinite(physical)),
            "physical camera-tangent positions are not finite",
        )
        _require(
            np.all(np.isfinite(observed[valid])),
            "valid observed camera-tangent position is not finite",
        )
        _require(
            np.all(np.isfinite(covariance[valid])),
            "valid projected covariance is not finite",
        )
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior reliability must lie in [0, 1]",
        )
        _require(
            np.all(np.isfinite(association))
            and np.all((association >= 0.0) & (association <= 1.0)),
            "association probability must lie in [0, 1]",
        )
        _require(
            np.all(reliability[~valid] == 0.0)
            and np.all(association[~valid] == 0.0),
            "invalid projected rows must carry zero support",
        )
        for name, value in (
            ("physical_positions_m", physical),
            ("observed_positions_m", observed),
            ("observation_validity", valid),
            ("observation_covariance_m2", covariance),
            ("prior_reliability", reliability),
            ("association_probability", association),
            ("cycle_error_px", cycle),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def build_projected_view_response(
    physical_pixels_px: np.ndarray,
    observed_pixels_px: np.ndarray,
    observation_validity: np.ndarray,
    initial_depth_m: np.ndarray,
    focal_lengths_px: np.ndarray,
    cycle_error_px: np.ndarray,
    source_confidence: np.ndarray,
    *,
    config: ProjectedViewResponseConfig | None = None,
) -> ProjectedViewResponse:
    """Convert per-view point flow into metric camera-tangent response.

    Reliability and association use only source confidence and forward/backward
    cycle error. The residual against the projected physical response is not
    used here and is processed once by the downstream robust admission model.
    """

    cfg = config or ProjectedViewResponseConfig()
    physical_pixels = np.asarray(physical_pixels_px, dtype=np.float64)
    observed_pixels = np.asarray(observed_pixels_px, dtype=np.float64)
    supplied_valid = np.asarray(observation_validity, dtype=bool)
    depth = np.asarray(initial_depth_m, dtype=np.float64)
    focal = np.asarray(focal_lengths_px, dtype=np.float64)
    cycle = np.asarray(cycle_error_px, dtype=np.float64)
    confidence = np.asarray(source_confidence, dtype=np.float64)
    _require(
        physical_pixels.ndim == 4
        and physical_pixels.shape[3] == 2
        and observed_pixels.shape == physical_pixels.shape,
        "pixel positions must have shape (S, T, N, 2)",
    )
    sensor_count, frame_count, point_count, _ = physical_pixels.shape
    expected = (sensor_count, frame_count, point_count)
    _require(supplied_valid.shape == expected, "validity shape changed")
    _require(depth.shape == (sensor_count, point_count), "depth shape changed")
    _require(focal.shape == (sensor_count, 2), "focal length shape changed")
    _require(cycle.shape == expected, "cycle error shape changed")
    _require(confidence.shape == expected, "source confidence shape changed")
    _require(
        np.all(np.isfinite(depth))
        and np.all(depth >= cfg.minimum_initial_depth_m),
        "initial depth is invalid",
    )
    _require(
        np.all(np.isfinite(focal)) and np.all(focal > 0.0),
        "focal length is invalid",
    )
    _require(
        np.all(np.isfinite(confidence))
        and np.all((confidence >= 0.0) & (confidence <= 1.0)),
        "source confidence must lie in [0, 1]",
    )
    finite_observed = np.all(np.isfinite(observed_pixels), axis=3)
    finite_observed_response = finite_observed & finite_observed[:, :1]
    finite_physical = np.all(np.isfinite(physical_pixels), axis=3)
    finite_physical_response = finite_physical & finite_physical[:, :1]
    finite_cycle = np.isfinite(cycle) & (cycle >= 0.0)
    valid = (
        supplied_valid
        & finite_observed_response
        & finite_physical_response
        & finite_cycle
    )
    scale_m_per_px = depth[:, :, None] / focal[:, None, :]
    physical_delta_px = physical_pixels - physical_pixels[:, :1]
    observed_delta_px = observed_pixels - observed_pixels[:, :1]
    physical_xy_m = physical_delta_px * scale_m_per_px[:, None]
    observed_xy_m = observed_delta_px * scale_m_per_px[:, None]
    physical_positions = np.zeros(
        (sensor_count, frame_count, point_count, 3),
        dtype=np.float64,
    )
    observed_positions = np.full_like(physical_positions, np.nan)
    physical_positions[..., :2] = np.where(
        finite_physical_response[..., None],
        physical_xy_m,
        0.0,
    )
    observed_positions[..., :2] = observed_xy_m
    observed_positions[..., 2] = 0.0
    sigma_xy_m = cfg.pixel_standard_deviation * scale_m_per_px
    covariance = np.zeros(
        (sensor_count, frame_count, point_count, 3, 3),
        dtype=np.float64,
    )
    covariance[..., 0, 0] = (
        cfg.covariance_inflation
        * np.square(sigma_xy_m[:, None, :, 0])
        + cfg.covariance_floor_m2
    )
    covariance[..., 1, 1] = (
        cfg.covariance_inflation
        * np.square(sigma_xy_m[:, None, :, 1])
        + cfg.covariance_floor_m2
    )
    covariance[..., 2, 2] = cfg.covariance_floor_m2
    association = np.exp(
        -0.5 * np.square(cycle / cfg.cycle_error_scale_px)
    )
    reliability = confidence.copy()
    reliability[~valid] = 0.0
    association[~valid] = 0.0
    return ProjectedViewResponse(
        physical_positions_m=physical_positions,
        observed_positions_m=observed_positions,
        observation_validity=valid,
        observation_covariance_m2=covariance,
        prior_reliability=reliability,
        association_probability=association,
        cycle_error_px=cycle,
    )


__all__ = [
    "ProjectedViewResponse",
    "ProjectedViewResponseConfig",
    "build_projected_view_response",
]
