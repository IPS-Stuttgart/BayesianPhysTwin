"""Motion-stratified graph queries and conservative shared-bias estimation.

Moving graph identities are useful for observing action response, but a query
set containing only predicted-moving identities cannot distinguish local object
motion from a coherent observation bias.  This module reserves part of a fixed
query budget for physically near-static sentinel identities.  Sentinels provide
an explicit nuisance channel while the existing physics-guided planner remains
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .phystwin_active_queries import (
    PhysicsGuidedQueryConfig,
    PhysicsGuidedQueryPlan,
    plan_physics_guided_queries,
)

ACTIVE_QUERY_ROLE = "active"
SENTINEL_QUERY_ROLE = "sentinel"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


def _solve_spd(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    cholesky = np.linalg.cholesky(matrix)
    return np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, right_hand_side))


@dataclass(frozen=True)
class MotionStratifiedQueryConfig:
    """Fixed-budget split between response queries and static sentinels."""

    total_query_count: int = 8
    sentinel_query_count: int = 2
    sentinel_maximum_motion_m: float = 0.0005
    sentinel_maximum_reseeds: int = 2

    def __post_init__(self) -> None:
        _require(self.total_query_count >= 2, "total_query_count must be at least two")
        _require(
            1 <= self.sentinel_query_count < self.total_query_count,
            "sentinel_query_count must reserve a strict subset of the budget",
        )
        _require(
            np.isfinite(self.sentinel_maximum_motion_m)
            and self.sentinel_maximum_motion_m >= 0.0,
            "sentinel_maximum_motion_m must be finite and nonnegative",
        )
        _require(
            0 <= self.sentinel_maximum_reseeds,
            "sentinel_maximum_reseeds must be nonnegative",
        )

    @property
    def active_query_count(self) -> int:
        """Number of query slots left for predicted response modes."""

        return self.total_query_count - self.sentinel_query_count


@dataclass(frozen=True)
class MotionStratifiedQueryPlan:
    """Disjoint active and sentinel plans under one fixed initial budget."""

    active: PhysicsGuidedQueryPlan
    sentinel: PhysicsGuidedQueryPlan
    config: MotionStratifiedQueryConfig

    def __post_init__(self) -> None:
        _require(
            self.active.requested_active_queries == self.config.active_query_count,
            "active plan does not match the reserved budget",
        )
        _require(
            self.sentinel.requested_active_queries
            == self.config.sentinel_query_count,
            "sentinel plan does not match the reserved budget",
        )
        _require(
            self.active.prefix_frame_count == self.sentinel.prefix_frame_count,
            "active and sentinel plans span different prefixes",
        )
        _require(
            self.active.camera_mask.shape[1] == self.sentinel.camera_mask.shape[1],
            "active and sentinel plans use different camera counts",
        )
        _require(
            not (
                set(map(int, self.active.node_ids))
                & set(map(int, self.sentinel.node_ids))
            ),
            "active and sentinel graph identities overlap",
        )

    @property
    def initial_query_count(self) -> int:
        """Total number of identities seeded at frame zero."""

        return self.active.initial_query_count + self.sentinel.initial_query_count

    @property
    def initial_budget_met(self) -> bool:
        """Whether both query roles received their complete reserved budgets."""

        return self.active.initial_budget_met and self.sentinel.initial_budget_met

    @property
    def active_node_ids(self) -> np.ndarray:
        """Every graph identity ever seeded as an active response query."""

        return self.active.node_ids

    @property
    def sentinel_node_ids(self) -> np.ndarray:
        """Every graph identity ever seeded as a near-static sentinel."""

        return self.sentinel.node_ids

    def camera_queries_txy(
        self,
        camera_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return node IDs, tracker queries, replacements, and query roles."""

        active_ids, active_txy, active_replacements = self.active.camera_queries_txy(
            camera_index
        )
        sentinel_ids, sentinel_txy, sentinel_replacements = (
            self.sentinel.camera_queries_txy(camera_index)
        )
        node_ids = np.concatenate((active_ids, sentinel_ids))
        queries_txy = np.concatenate((active_txy, sentinel_txy), axis=0)
        replacements = np.concatenate(
            (active_replacements, sentinel_replacements)
        )
        roles = np.concatenate(
            (
                np.full(len(active_ids), ACTIVE_QUERY_ROLE, dtype="<U8"),
                np.full(len(sentinel_ids), SENTINEL_QUERY_ROLE, dtype="<U8"),
            )
        )
        if len(node_ids):
            role_order = np.where(roles == ACTIVE_QUERY_ROLE, 0, 1)
            order = np.lexsort((node_ids, role_order, queries_txy[:, 0]))
            node_ids = node_ids[order]
            queries_txy = queries_txy[order]
            replacements = replacements[order]
            roles = roles[order]
        return (
            _readonly(node_ids, dtype=node_ids.dtype),
            _readonly(queries_txy, dtype=queries_txy.dtype),
            _readonly(replacements, dtype=replacements.dtype),
            _readonly(roles, dtype=roles.dtype),
        )


def _empty_plan(
    *,
    camera_count: int,
    prefix_frame_count: int,
    requested_queries: int,
    minimum_camera_support: int,
) -> PhysicsGuidedQueryPlan:
    return PhysicsGuidedQueryPlan(
        node_ids=np.empty(0, dtype=np.int64),
        seed_frames=np.empty(0, dtype=np.int64),
        replaces_node_ids=np.empty(0, dtype=np.int64),
        camera_mask=np.empty((0, camera_count), dtype=bool),
        seed_pixels_xy=np.empty((0, camera_count, 2), dtype=np.float64),
        motion_score=np.empty(0, dtype=np.float64),
        visibility_score=np.empty(0, dtype=np.float64),
        mode_information_gain=np.empty(0, dtype=np.float64),
        spatial_diversity_score=np.empty(0, dtype=np.float64),
        contact_distance_score=np.empty(0, dtype=np.float64),
        total_score=np.empty(0, dtype=np.float64),
        requested_active_queries=requested_queries,
        minimum_camera_support=minimum_camera_support,
        prefix_frame_count=prefix_frame_count,
    )


def _maximum_prefix_motion_m(rollout_m: np.ndarray) -> np.ndarray:
    displacement = rollout_m - rollout_m[0][None]
    return np.max(np.linalg.norm(displacement, axis=2), axis=0)


def plan_motion_stratified_queries(
    physical_rollout_m: np.ndarray,
    projected_pixels_xy: np.ndarray,
    predicted_support_probability: np.ndarray,
    *,
    mode_basis: np.ndarray | None = None,
    tracker_support_probability: np.ndarray | None = None,
    contact_position_m: np.ndarray | None = None,
    candidate_ids: np.ndarray | None = None,
    active_config: PhysicsGuidedQueryConfig | None = None,
    config: MotionStratifiedQueryConfig | None = None,
) -> MotionStratifiedQueryPlan:
    """Plan disjoint moving queries and near-static sentinels causally.

    ``total_query_count`` is fixed.  Sentinel shortfalls are never filled with
    extra active queries, and active shortfalls are never filled with sentinels.
    A downstream guarded update should require ``initial_budget_met`` and retain
    its exact baseline otherwise.
    """

    cfg = config or MotionStratifiedQueryConfig()
    base = active_config or PhysicsGuidedQueryConfig()
    _require(
        cfg.sentinel_maximum_motion_m < base.minimum_motion_m,
        "sentinel and active motion regimes must be separated by a gap",
    )
    _require(
        cfg.sentinel_maximum_reseeds <= base.maximum_reseeds,
        "sentinel reseeds exceed the shared reseed budget",
    )
    rollout = np.asarray(physical_rollout_m, dtype=np.float64)
    pixels = np.asarray(projected_pixels_xy, dtype=np.float64)
    predicted_support = np.asarray(
        predicted_support_probability,
        dtype=np.float64,
    )
    _require(
        rollout.ndim == 3
        and rollout.shape[2] == 3
        and len(rollout) >= 1
        and rollout.shape[1] >= 1,
        "physical_rollout_m must have nonempty shape (T, N, 3)",
    )
    frame_count, node_count, _ = rollout.shape
    _require(
        pixels.ndim == 4
        and pixels.shape[1:] == (frame_count, node_count, 2),
        "projected_pixels_xy must have shape (C, T, N, 2)",
    )
    camera_count = pixels.shape[0]
    _require(
        predicted_support.shape == (camera_count, frame_count, node_count),
        "predicted_support_probability shape changed",
    )
    _require(
        np.all(np.isfinite(predicted_support))
        and np.all((predicted_support >= 0.0) & (predicted_support <= 1.0)),
        "predicted support probabilities must lie in [0, 1]",
    )
    _require(
        camera_count >= base.minimum_camera_support,
        "camera count is below minimum_camera_support",
    )
    candidates = (
        np.arange(node_count, dtype=np.int64)
        if candidate_ids is None
        else np.asarray(candidate_ids, dtype=np.int64)
    )
    _require(
        candidates.ndim == 1
        and len(candidates) >= 1
        and np.all((candidates >= 0) & (candidates < node_count))
        and len(np.unique(candidates)) == len(candidates),
        "candidate_ids is invalid",
    )
    candidates = np.sort(candidates, kind="mergesort")
    motion = _maximum_prefix_motion_m(rollout)
    active_candidates = candidates[
        np.isfinite(motion[candidates])
        & (motion[candidates] >= base.minimum_motion_m)
    ]
    sentinel_candidates = candidates[
        np.isfinite(motion[candidates])
        & (motion[candidates] <= cfg.sentinel_maximum_motion_m)
    ]

    active_reseeds = max(0, base.maximum_reseeds - cfg.sentinel_maximum_reseeds)
    active_cfg = replace(
        base,
        query_count=cfg.active_query_count,
        maximum_reseeds=active_reseeds,
    )
    sentinel_cfg = replace(
        base,
        query_count=cfg.sentinel_query_count,
        maximum_reseeds=cfg.sentinel_maximum_reseeds,
        minimum_motion_m=0.0,
        motion_weight=0.0,
        visibility_weight=max(base.visibility_weight, 1.0),
        mode_information_weight=0.0,
        spatial_diversity_weight=max(base.spatial_diversity_weight, 1.0),
    )

    def make_plan(
        selected_candidates: np.ndarray,
        *,
        planner_config: PhysicsGuidedQueryConfig,
        planner_basis: np.ndarray | None,
    ) -> PhysicsGuidedQueryPlan:
        if not len(selected_candidates):
            return _empty_plan(
                camera_count=camera_count,
                prefix_frame_count=frame_count,
                requested_queries=planner_config.query_count,
                minimum_camera_support=planner_config.minimum_camera_support,
            )
        return plan_physics_guided_queries(
            rollout,
            pixels,
            predicted_support,
            mode_basis=planner_basis,
            tracker_support_probability=tracker_support_probability,
            contact_position_m=contact_position_m,
            candidate_ids=selected_candidates,
            config=planner_config,
        )

    active = make_plan(
        active_candidates,
        planner_config=active_cfg,
        planner_basis=mode_basis,
    )
    sentinel = make_plan(
        sentinel_candidates,
        planner_config=sentinel_cfg,
        planner_basis=None,
    )
    return MotionStratifiedQueryPlan(active=active, sentinel=sentinel, config=cfg)


@dataclass(frozen=True)
class SentinelBiasConfig:
    """Conservative nuisance-estimation settings for sentinel displacements."""

    minimum_reliability: float = 0.05
    covariance_floor_m2: float = 1e-10
    maximum_inconsistency_sigma: float = 4.0
    minimum_correlation_groups: int = 1

    def __post_init__(self) -> None:
        _require(
            np.isfinite(self.minimum_reliability)
            and 0.0 < self.minimum_reliability <= 1.0,
            "minimum_reliability must lie in (0, 1]",
        )
        _require(
            np.isfinite(self.covariance_floor_m2)
            and self.covariance_floor_m2 > 0.0,
            "covariance_floor_m2 must be positive",
        )
        _require(
            np.isfinite(self.maximum_inconsistency_sigma)
            and self.maximum_inconsistency_sigma > 0.0,
            "maximum_inconsistency_sigma must be positive",
        )
        _require(
            self.minimum_correlation_groups >= 1,
            "minimum_correlation_groups must be positive",
        )


@dataclass(frozen=True)
class SentinelBiasEstimate:
    """A shared displacement-bias estimate with explicit abstention state."""

    bias_m: np.ndarray | None
    covariance_m2: np.ndarray | None
    observation_count: int
    correlation_group_count: int
    maximum_inconsistency_sigma: float | None
    usable: bool
    decision: str

    def __post_init__(self) -> None:
        _require(self.observation_count >= 0, "observation_count is negative")
        _require(
            0 <= self.correlation_group_count <= self.observation_count,
            "correlation_group_count is invalid",
        )
        if self.bias_m is None or self.covariance_m2 is None:
            _require(
                self.bias_m is None
                and self.covariance_m2 is None
                and not self.usable,
                "missing bias arrays require an unusable estimate",
            )
            return
        bias = _readonly(self.bias_m, dtype=np.float64)
        covariance = _readonly(self.covariance_m2, dtype=np.float64)
        _require(bias.shape == (3,), "bias_m must have shape (3,)")
        _require(covariance.shape == (3, 3), "covariance_m2 must have shape (3, 3)")
        _require(
            np.all(np.isfinite(bias))
            and np.all(np.isfinite(covariance))
            and np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
            and np.min(np.linalg.eigvalsh(covariance)) > 0.0,
            "bias estimate is not finite positive definite",
        )
        _require(
            self.maximum_inconsistency_sigma is not None
            and np.isfinite(self.maximum_inconsistency_sigma)
            and self.maximum_inconsistency_sigma >= 0.0,
            "maximum_inconsistency_sigma is invalid",
        )
        object.__setattr__(self, "bias_m", bias)
        object.__setattr__(self, "covariance_m2", covariance)


def estimate_sentinel_common_bias(
    observed_displacement_m: np.ndarray,
    predicted_displacement_m: np.ndarray,
    covariance_m2: np.ndarray,
    prior_reliability: np.ndarray,
    correlation_group_ids: np.ndarray,
    *,
    config: SentinelBiasConfig | None = None,
) -> SentinelBiasEstimate:
    """Estimate one shared bias without independent-sample overconfidence.

    Observations inside one declared correlation group are collapsed into one
    conservative estimate.  Group estimates are fused by equal-weight
    covariance intersection because their cross-correlation is unknown.
    """

    cfg = config or SentinelBiasConfig()
    observed = np.asarray(observed_displacement_m, dtype=np.float64)
    predicted = np.asarray(predicted_displacement_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    groups = np.asarray(correlation_group_ids, dtype=np.int64)
    _require(
        observed.ndim == 2
        and observed.shape[1] == 3
        and predicted.shape == observed.shape,
        "sentinel displacements must have shape (S, 3)",
    )
    sentinel_count = len(observed)
    _require(
        covariance.shape == (sentinel_count, 3, 3)
        and reliability.shape == groups.shape == (sentinel_count,),
        "sentinel uncertainty metadata shape changed",
    )
    finite = (
        np.all(np.isfinite(observed), axis=1)
        & np.all(np.isfinite(predicted), axis=1)
        & np.all(np.isfinite(covariance), axis=(1, 2))
        & np.isfinite(reliability)
        & (reliability >= cfg.minimum_reliability)
    )
    valid_rows: list[int] = []
    for row in np.flatnonzero(finite):
        matrix = covariance[row]
        if np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12) and np.min(
            np.linalg.eigvalsh(matrix)
        ) > 0.0:
            valid_rows.append(int(row))
    if not valid_rows:
        return SentinelBiasEstimate(
            bias_m=None,
            covariance_m2=None,
            observation_count=0,
            correlation_group_count=0,
            maximum_inconsistency_sigma=None,
            usable=False,
            decision="no-reliable-sentinel-observation",
        )

    rows = np.asarray(valid_rows, dtype=np.int64)
    residual = observed[rows] - predicted[rows]
    adjusted_covariance = covariance[rows] / reliability[rows, None, None]
    selected_groups = groups[rows]
    group_means: list[np.ndarray] = []
    group_covariances: list[np.ndarray] = []
    group_inconsistency: list[float] = []
    for group_id in np.unique(selected_groups):
        local = np.flatnonzero(selected_groups == group_id)
        local_residual = residual[local]
        local_reliability = reliability[rows[local]]
        weights = local_reliability / np.sum(local_reliability)
        mean = np.sum(weights[:, None] * local_residual, axis=0)
        noise_bound = max(
            float(np.max(np.linalg.eigvalsh(adjusted_covariance[index])))
            for index in local
        )
        scatter = local_residual - mean
        scatter_bound = (
            float(np.max(np.sum(np.square(scatter), axis=1)))
            if len(scatter)
            else 0.0
        )
        group_covariance = (
            noise_bound + scatter_bound + cfg.covariance_floor_m2
        ) * np.eye(3)
        maximum_sigma = 0.0
        for offset, index in enumerate(local):
            delta = local_residual[offset] - mean
            variance = adjusted_covariance[index] + (
                cfg.covariance_floor_m2 * np.eye(3)
            )
            squared = float(delta @ _solve_spd(variance, delta))
            maximum_sigma = max(maximum_sigma, float(np.sqrt(max(squared, 0.0))))
        group_means.append(mean)
        group_covariances.append(group_covariance)
        group_inconsistency.append(maximum_sigma)

    group_count = len(group_means)
    ci_weight = 1.0 / group_count
    precision: np.ndarray = np.zeros((3, 3), dtype=np.float64)
    information: np.ndarray = np.zeros(3, dtype=np.float64)
    for mean, group_covariance in zip(
        group_means,
        group_covariances,
        strict=True,
    ):
        group_precision = _solve_spd(group_covariance, np.eye(3))
        precision += ci_weight * group_precision
        information += ci_weight * (group_precision @ mean)
    fused_covariance = _solve_spd(precision, np.eye(3))
    fused_mean = fused_covariance @ information
    between_group_scatter_bound = max(
        float(np.sum(np.square(mean - fused_mean))) for mean in group_means
    )
    fused_covariance = fused_covariance + (
        between_group_scatter_bound * np.eye(3)
    )

    between_group_sigma = 0.0
    for mean, group_covariance in zip(
        group_means,
        group_covariances,
        strict=True,
    ):
        delta = mean - fused_mean
        squared = float(
            delta
            @ _solve_spd(group_covariance + fused_covariance, delta)
        )
        between_group_sigma = max(
            between_group_sigma,
            float(np.sqrt(max(squared, 0.0))),
        )
    maximum_sigma = max([between_group_sigma, *group_inconsistency])
    enough_groups = group_count >= cfg.minimum_correlation_groups
    consistent = maximum_sigma <= cfg.maximum_inconsistency_sigma
    usable = enough_groups and consistent
    if not enough_groups:
        decision = "insufficient-correlation-groups"
    elif not consistent:
        decision = "sentinel-common-mode-inconsistent"
    else:
        decision = "sentinel-common-mode-estimated"
    return SentinelBiasEstimate(
        bias_m=fused_mean,
        covariance_m2=fused_covariance,
        observation_count=len(rows),
        correlation_group_count=group_count,
        maximum_inconsistency_sigma=maximum_sigma,
        usable=usable,
        decision=decision,
    )


def debias_active_displacements(
    observed_displacement_m: np.ndarray,
    covariance_m2: np.ndarray,
    bias: SentinelBiasEstimate,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract an admitted sentinel bias and propagate its uncertainty."""

    bias_m = bias.bias_m
    bias_covariance = bias.covariance_m2
    _require(
        bias.usable and bias_m is not None and bias_covariance is not None,
        "sentinel bias is not usable; retain the exact baseline",
    )
    assert bias_m is not None and bias_covariance is not None
    observed = np.asarray(observed_displacement_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    _require(
        observed.ndim == 2
        and observed.shape[1] == 3
        and covariance.shape == (len(observed), 3, 3),
        "active displacement arrays have incompatible shapes",
    )
    _require(
        np.all(np.isfinite(observed)) and np.all(np.isfinite(covariance)),
        "active displacement arrays are not finite",
    )
    corrected = observed - bias_m[None]
    corrected_covariance = covariance + bias_covariance[None]
    corrected = _readonly(corrected, dtype=np.float64)
    corrected_covariance = _readonly(corrected_covariance, dtype=np.float64)
    return corrected, corrected_covariance


__all__ = [
    "ACTIVE_QUERY_ROLE",
    "SENTINEL_QUERY_ROLE",
    "MotionStratifiedQueryConfig",
    "MotionStratifiedQueryPlan",
    "SentinelBiasConfig",
    "SentinelBiasEstimate",
    "debias_active_displacements",
    "estimate_sentinel_common_bias",
    "plan_motion_stratified_queries",
]
