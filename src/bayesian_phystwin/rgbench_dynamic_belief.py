"""Action-cross-fitted temporal readout discrepancy for RGBench.

The module extends an admitted static graph correction with a prefix-estimated
velocity field.  The temporal shrinkage is deliberately external: it must be
chosen on other source garments, never from the future of the case being
predicted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .cloth_sim2real_belief import (
    ClothReadoutBeliefConfig,
    GuardedReadoutCorrection,
    associate_dense_cloud,
    mesh_edges_from_faces,
)
from .phystwin_graph_discrepancy import (
    graph_smoothed_discrepancy_posterior,
    normalized_spring_laplacian,
)
from .pseudo_measurements import PseudoMeasurementBatch
from .robust_likelihood import RobustLikelihoodConfig, robust_mixture_likelihood


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(
    value: np.ndarray,
    *,
    dtype: np.dtype[np.generic] | type[np.generic] = np.float64,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _finite_trajectory(value: np.ndarray, name: str) -> np.ndarray:
    trajectory = np.asarray(value, dtype=np.float64)
    _require(
        trajectory.ndim == 3 and trajectory.shape[2] == 3,
        f"{name} must have shape (T, N, 3)",
    )
    _require(
        len(trajectory) >= 2
        and trajectory.shape[1] >= 1
        and np.all(np.isfinite(trajectory)),
        f"{name} must be finite and nonempty",
    )
    return trajectory


def _parse_graph_strength(selected_name: str) -> float | None:
    if selected_name == "global":
        return None
    if not selected_name.startswith("graph_l"):
        raise ValueError(f"unsupported static correction {selected_name}")
    strength_text = selected_name.split("_", maxsplit=2)[1][1:]
    strength = float(strength_text)
    _require(
        np.isfinite(strength) and strength > 0.0,
        "graph prior strength must be positive",
    )
    return strength


@dataclass(frozen=True)
class RGBenchDynamicSlope:
    """Robust graph-smoothed prefix slope and its metric covariance."""

    slope_m_per_s: np.ndarray
    variance_m2_per_s2: np.ndarray
    spatial_model: str
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        slope = _readonly(self.slope_m_per_s)
        variance = _readonly(self.variance_m2_per_s2)
        _require(
            slope.ndim == 2 and slope.shape[1] == 3,
            "slope_m_per_s must have shape (N, 3)",
        )
        _require(
            variance.shape == slope.shape
            and np.all(np.isfinite(variance))
            and np.all(variance > 0.0),
            "slope variance must be finite, positive, and match the slope",
        )
        _require(bool(self.spatial_model), "spatial_model is empty")
        object.__setattr__(self, "slope_m_per_s", slope)
        object.__setattr__(self, "variance_m2_per_s2", variance)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class RGBenchDynamicCandidate:
    """One sealed temporal-shrinkage trajectory."""

    shrinkage: float
    trajectory_m: np.ndarray
    variance_m2: np.ndarray
    exact_physical_fallback: bool

    def __post_init__(self) -> None:
        _require(
            np.isfinite(self.shrinkage) and 0.0 <= self.shrinkage <= 1.0,
            "shrinkage must lie in [0, 1]",
        )
        trajectory = _readonly(self.trajectory_m)
        variance = _readonly(self.variance_m2)
        _require(
            trajectory.ndim == 3 and trajectory.shape[2] == 3,
            "candidate trajectory must have shape (T, N, 3)",
        )
        _require(
            variance.shape == trajectory.shape
            and np.all(np.isfinite(variance))
            and np.all(variance > 0.0),
            "candidate variance must be finite, positive, and match trajectory",
        )
        object.__setattr__(self, "trajectory_m", trajectory)
        object.__setattr__(self, "variance_m2", variance)


def fit_rgbbench_dynamic_slope(
    physical_prefix_m: np.ndarray,
    observed_prefix_clouds_m: Sequence[np.ndarray],
    target_times_s: np.ndarray,
    faces: np.ndarray,
    static_belief: GuardedReadoutCorrection,
    *,
    config: ClothReadoutBeliefConfig | None = None,
) -> RGBenchDynamicSlope:
    """Fit a robust temporal residual slope from the complete allowed prefix.

    Association geometry is used only to form pseudo-measurements.  Prior
    reliability remains residual-independent, and the innovation enters once
    through the robust mixture likelihood.
    """

    cfg = config or ClothReadoutBeliefConfig()
    physical = _finite_trajectory(physical_prefix_m, "physical_prefix_m")
    times = np.asarray(target_times_s, dtype=np.float64)
    _require(
        times.shape == (len(physical),)
        and np.all(np.isfinite(times))
        and np.all(np.diff(times) > 0.0),
        "target_times_s must be finite and strictly increasing",
    )
    _require(
        len(observed_prefix_clouds_m) == len(physical),
        "observed prefix count must match physical prefix",
    )
    _require(static_belief.accepted, "dynamic slope requires an admitted static belief")
    _require(
        static_belief.correction_m.shape == physical.shape[1:],
        "static belief node count differs from physical prefix",
    )

    residuals: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    reliabilities: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    for state, cloud in zip(physical, observed_prefix_clouds_m, strict=True):
        association = associate_dense_cloud(
            state,
            cloud,
            candidate_count=cfg.candidate_count,
            sensor_std_m=cfg.sensor_std_m,
        )
        residuals.append(association.observed_points_m - state)
        variances.append(association.variance_m2)
        reliabilities.append(association.prior_reliability)
        entropies.append(association.assignment_entropy)
    residual = np.stack(residuals)
    variance = np.stack(variances)
    prior_reliability = np.stack(reliabilities)
    temporal_center = np.median(residual, axis=0)
    batch = PseudoMeasurementBatch(
        observed=residual.reshape(-1, 3),
        predicted=np.broadcast_to(temporal_center, residual.shape).reshape(-1, 3),
        variance=variance.reshape(-1),
    )
    robust = robust_mixture_likelihood(
        batch,
        prior_reliability=prior_reliability.reshape(-1),
        config=RobustLikelihoodConfig(
            outlier_variance_multiplier=cfg.outlier_variance_multiplier,
            model_discrepancy_variance=cfg.model_discrepancy_std_m**2,
        ),
    ).posterior_inlier_probability.reshape(residual.shape[:2])
    weight = robust * prior_reliability / variance
    weight_sum = np.sum(weight, axis=0)
    weighted_time = np.sum(weight * times[:, None], axis=0) / np.maximum(
        weight_sum,
        1e-15,
    )
    centered_time = times[:, None] - weighted_time[None]
    denominator = np.sum(weight * np.square(centered_time), axis=0)
    weighted_residual_mean = np.sum(
        weight[:, :, None] * residual,
        axis=0,
    ) / np.maximum(weight_sum[:, None], 1e-15)
    numerator = np.sum(
        weight[:, :, None]
        * centered_time[:, :, None]
        * (residual - weighted_residual_mean[None]),
        axis=0,
    )
    raw_slope = numerator / np.maximum(denominator[:, None], 1e-15)
    time_span = float(times[-1] - times[0])
    slope_floor = (cfg.shared_bias_std_m / time_span) ** 2
    raw_variance = 1.0 / np.maximum(denominator, 1e-15) + slope_floor

    node_count = physical.shape[1]
    edges = mesh_edges_from_faces(faces, node_count)
    laplacian = normalized_spring_laplacian(node_count, edges)
    graph_strength = _parse_graph_strength(static_belief.selected_name)
    if graph_strength is None:
        slope = np.broadcast_to(
            np.median(raw_slope, axis=0),
            raw_slope.shape,
        ).copy()
        slope_variance = np.full(
            node_count,
            max(float(np.median(raw_variance)), slope_floor),
            dtype=np.float64,
        )
        solve_methods: tuple[str, ...] = ()
        covariance_negative_fraction = None
    else:
        posterior = graph_smoothed_discrepancy_posterior(
            raw_slope,
            raw_variance,
            np.ones(node_count, dtype=bool),
            laplacian,
            prior_strength=graph_strength,
            covariance_probes=cfg.covariance_probes,
            covariance_seed=cfg.covariance_seed + 101,
        )
        slope = posterior.mean
        slope_variance = (
            np.full(node_count, posterior.reference_variance, dtype=np.float64)
            if posterior.marginal_variance is None
            else np.maximum(posterior.marginal_variance, slope_floor)
        )
        solve_methods = posterior.solve_methods
        covariance_negative_fraction = posterior.covariance_negative_fraction

    return RGBenchDynamicSlope(
        slope_m_per_s=slope,
        variance_m2_per_s2=np.repeat(slope_variance[:, None], 3, axis=1),
        spatial_model=static_belief.selected_name,
        diagnostics={
            "prefix_frame_count": len(physical),
            "node_count": node_count,
            "time_span_s": time_span,
            "mean_assignment_entropy": float(np.mean(entropies)),
            "mean_posterior_inlier_probability": float(np.mean(robust)),
            "minimum_posterior_inlier_probability": float(np.min(robust)),
            "median_raw_slope_m_per_s": float(
                np.median(np.linalg.norm(raw_slope, axis=1))
            ),
            "median_smoothed_slope_m_per_s": float(
                np.median(np.linalg.norm(slope, axis=1))
            ),
            "maximum_smoothed_slope_m_per_s": float(
                np.max(np.linalg.norm(slope, axis=1))
            ),
            "prior_reliability_uses_state_innovation": False,
            "innovation_processed_once": True,
            "graph_solve_methods": list(solve_methods),
            "covariance_negative_fraction": covariance_negative_fraction,
        },
    )


def _cap_vector_field(field_m: np.ndarray, maximum_m: float) -> np.ndarray:
    field = np.asarray(field_m, dtype=np.float64).copy()
    norms = np.linalg.norm(field, axis=2, keepdims=True)
    field *= np.minimum(1.0, maximum_m / np.maximum(norms, 1e-15))
    return field


def build_rgbbench_dynamic_candidates(
    physical_m: np.ndarray,
    target_times_s: np.ndarray,
    branch_index: int,
    static_belief: GuardedReadoutCorrection,
    slope: RGBenchDynamicSlope | None,
    *,
    shrinkages: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    maximum_correction_m: float = 0.10,
) -> dict[float, RGBenchDynamicCandidate]:
    """Build a frozen shrinkage bank while leaving every prefix frame untouched."""

    physical = _finite_trajectory(physical_m, "physical_m")
    times = np.asarray(target_times_s, dtype=np.float64)
    _require(
        times.shape == (len(physical),)
        and np.all(np.isfinite(times))
        and np.all(np.diff(times) > 0.0),
        "target_times_s must be finite and strictly increasing",
    )
    _require(
        1 <= branch_index < len(physical) - 1,
        "branch_index must leave a nonempty prefix and future",
    )
    _require(
        np.isfinite(maximum_correction_m) and maximum_correction_m > 0.0,
        "maximum_correction_m must be positive",
    )
    values = tuple(float(value) for value in shrinkages)
    _require(
        values
        and len(values) == len(set(values))
        and all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values),
        "shrinkages must be unique and lie in [0, 1]",
    )
    if static_belief.accepted:
        _require(slope is not None, "accepted belief requires a dynamic slope")
        _require(
            slope.slope_m_per_s.shape == physical.shape[1:],
            "dynamic slope node count differs from physical rollout",
        )
    else:
        _require(slope is None, "rejected belief must not carry a dynamic slope")

    candidates: dict[float, RGBenchDynamicCandidate] = {}
    future = slice(branch_index + 1, len(physical))
    future_times = times[future] - times[branch_index]
    for shrinkage in values:
        trajectory = physical.copy()
        variance = np.broadcast_to(
            static_belief.variance_m2[None],
            physical.shape,
        ).copy()
        if not static_belief.accepted:
            exact_fallback = True
        else:
            assert slope is not None
            correction = (
                static_belief.correction_m[None]
                + shrinkage
                * future_times[:, None, None]
                * slope.slope_m_per_s[None]
            )
            correction = _cap_vector_field(correction, maximum_correction_m)
            trajectory[future] += correction
            variance[future] = (
                static_belief.variance_m2[None]
                + shrinkage**2
                * np.square(future_times[:, None, None])
                * slope.variance_m2_per_s2[None]
            )
            exact_fallback = False
        candidates[shrinkage] = RGBenchDynamicCandidate(
            shrinkage=shrinkage,
            trajectory_m=trajectory,
            variance_m2=variance,
            exact_physical_fallback=exact_fallback,
        )
    return candidates


def select_leave_one_garment_out_shrinkages(
    case_scores_m: Sequence[Mapping[str, object]],
    *,
    garments: Sequence[str],
    actions: Sequence[str],
    shrinkages: Sequence[float],
) -> dict[tuple[str, str], float]:
    """Select one action-specific shrinkage on the other source garments."""

    garment_names = tuple(garments)
    action_names = tuple(actions)
    values = tuple(float(value) for value in shrinkages)
    _require(
        len(garment_names) >= 3 and len(set(garment_names)) == len(garment_names),
        "at least three unique garments are required",
    )
    _require(
        action_names and len(set(action_names)) == len(action_names),
        "actions must be nonempty and unique",
    )
    _require(
        values and len(set(values)) == len(values),
        "shrinkages must be nonempty and unique",
    )
    rows = list(case_scores_m)
    selections: dict[tuple[str, str], float] = {}
    for held_out in garment_names:
        training_garments = set(garment_names) - {held_out}
        for action in action_names:
            means: list[tuple[float, float]] = []
            for shrinkage in values:
                scores = [
                    float(row["candidate_score_m"])
                    for row in rows
                    if row["garment"] in training_garments
                    and row["action"] == action
                    and float(row["shrinkage"]) == shrinkage
                ]
                _require(
                    len(scores) >= len(training_garments),
                    f"missing training scores for {held_out}/{action}/{shrinkage:g}",
                )
                means.append((float(np.mean(scores)), shrinkage))
            selections[(held_out, action)] = min(means)[1]
    return selections


__all__ = [
    "RGBenchDynamicCandidate",
    "RGBenchDynamicSlope",
    "build_rgbbench_dynamic_candidates",
    "fit_rgbbench_dynamic_slope",
    "select_leave_one_garment_out_shrinkages",
]
