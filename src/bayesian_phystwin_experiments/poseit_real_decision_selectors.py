"""Target-blind PoseIt probe selectors sharing one latent Gaussian belief."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .poseit_real_decision_protocol import MANDATORY_ANCHOR, SELECTABLE_POSES

POSE_COUNT = 16
PREDICTIVE_DRAW_COUNT = 4096
PREDICTIVE_SEED = 20260902
REGISTERED_PROBE_BUDGET = 3
SelectorName = Literal["decision_directed", "system_identification", "fixed"]
FloatArray: TypeAlias = NDArray[np.float64]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pose_offset(pose: int, feature_dimension: int) -> int:
    _require(1 <= pose <= POSE_COUNT, "holding pose is out of range")
    _require(feature_dimension > 0, "feature dimension must be positive")
    return (pose - 1) * (feature_dimension + 1)


def pose_feature_indices(pose: int, feature_dimension: int) -> tuple[int, ...]:
    """Return the registered pre-shake feature coordinates for one pose."""

    start = _pose_offset(pose, feature_dimension)
    return tuple(range(start, start + feature_dimension))


def pose_stability_index(pose: int, feature_dimension: int) -> int:
    """Return the latent shake-stability coordinate for one pose."""

    return _pose_offset(pose, feature_dimension) + feature_dimension


@dataclass(frozen=True)
class PoseItGaussianState:
    """Joint belief over pre-shake features and latent shake stability."""

    mean: FloatArray
    covariance: FloatArray
    feature_dimension: int
    available_poses: tuple[int, ...] = tuple(range(1, POSE_COUNT + 1))
    observed_poses: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require(self.feature_dimension > 0, "feature dimension must be positive")
        coordinate_count = POSE_COUNT * (self.feature_dimension + 1)
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        _require(mean.shape == (coordinate_count,), "Gaussian mean shape changed")
        _require(
            covariance.shape == (coordinate_count, coordinate_count),
            "Gaussian covariance shape changed",
        )
        _require(bool(np.all(np.isfinite(mean))), "Gaussian mean is non-finite")
        _require(
            bool(np.all(np.isfinite(covariance))),
            "Gaussian covariance is non-finite",
        )
        _require(
            np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10),
            "Gaussian covariance is not symmetric",
        )
        available = tuple(int(pose) for pose in self.available_poses)
        observed = tuple(int(pose) for pose in self.observed_poses)
        _require(
            available == tuple(sorted(set(available))),
            "available poses are not canonical",
        )
        _require(MANDATORY_ANCHOR in available, "mandatory anchor is unavailable")
        _require(
            set(available) <= set(range(1, POSE_COUNT + 1)),
            "available pose is out of range",
        )
        _require(len(observed) == len(set(observed)), "pose was observed twice")
        _require(set(observed) <= set(available), "unavailable pose was observed")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "available_poses", available)
        object.__setattr__(self, "observed_poses", observed)

    @property
    def action_poses(self) -> tuple[int, ...]:
        """Return structurally available non-reference actions."""

        return tuple(pose for pose in self.available_poses if pose != MANDATORY_ANCHOR)


def _positive_definite_solve(covariance: FloatArray, right: FloatArray) -> FloatArray:
    covariance = np.asarray(covariance, dtype=np.float64)
    _require(
        np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10),
        "conditioning covariance is not symmetric",
    )
    try:
        np.linalg.cholesky(covariance)
        return np.asarray(np.linalg.solve(covariance, right), dtype=np.float64)
    except np.linalg.LinAlgError as error:
        raise ValueError("conditioning covariance is not positive definite") from error


def condition_on_pose_features(
    state: PoseItGaussianState,
    pose: int,
    values: Sequence[float] | FloatArray,
) -> PoseItGaussianState:
    """Condition only on one pose's pre-shake features, never its shake label."""

    _require(pose in state.available_poses, "candidate pose is unavailable")
    _require(pose not in state.observed_poses, "pose features were observed twice")
    observations = np.asarray(values, dtype=np.float64)
    _require(
        observations.shape == (state.feature_dimension,),
        "pose feature shape changed",
    )
    _require(bool(np.all(np.isfinite(observations))), "pose feature is non-finite")
    indices = np.asarray(pose_feature_indices(pose, state.feature_dimension), dtype=int)
    covariance_yy = state.covariance[np.ix_(indices, indices)]
    cross = state.covariance[:, indices]
    gain = _positive_definite_solve(covariance_yy, cross.T).T
    residual = observations - state.mean[indices]
    mean = state.mean + gain @ residual
    covariance = state.covariance - gain @ cross.T
    covariance = 0.5 * (covariance + covariance.T)
    covariance[indices, :] = 0.0
    covariance[:, indices] = 0.0
    return PoseItGaussianState(
        mean=mean,
        covariance=covariance,
        feature_dimension=state.feature_dimension,
        available_poses=state.available_poses,
        observed_poses=(*state.observed_poses, pose),
    )


def _normal_cdf(value: FloatArray) -> FloatArray:
    values = np.asarray(value, dtype=np.float64)
    return np.asarray(
        0.5
        * (
            1.0
            + np.asarray(
                [math.erf(float(item) / math.sqrt(2.0)) for item in values.ravel()],
                dtype=np.float64,
            ).reshape(values.shape)
        ),
        dtype=np.float64,
    )


def _stability_probabilities(
    latent_mean: FloatArray,
    latent_variance: FloatArray,
) -> FloatArray:
    mean = np.asarray(latent_mean, dtype=np.float64)
    variance = np.asarray(latent_variance, dtype=np.float64)
    _require(mean.shape == variance.shape, "latent moment shape changed")
    _require(bool(np.all(variance >= -1e-10)), "latent variance is negative")
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    deterministic = standard_deviation <= np.finfo(np.float64).eps
    safe_scale = np.where(deterministic, 1.0, standard_deviation)
    probability = _normal_cdf(mean / safe_scale)
    deterministic_probability = np.where(
        mean > 0.0, 1.0, np.where(mean < 0.0, 0.0, 0.5)
    )
    return np.asarray(
        np.where(deterministic, deterministic_probability, probability),
        dtype=np.float64,
    )


def stability_probabilities(
    state: PoseItGaussianState,
    poses: Sequence[int] | None = None,
) -> FloatArray:
    """Return posterior Pass probabilities for available action poses."""

    selected = (
        state.action_poses if poses is None else tuple(int(pose) for pose in poses)
    )
    _require(set(selected) <= set(state.action_poses), "stability pose is unavailable")
    if not selected:
        return np.asarray([], dtype=np.float64)
    indices = np.asarray(
        [pose_stability_index(pose, state.feature_dimension) for pose in selected],
        dtype=int,
    )
    return _stability_probabilities(
        state.mean[indices],
        np.diag(state.covariance[np.ix_(indices, indices)]),
    )


def expected_best_utility(state: PoseItGaussianState) -> float:
    """Return Bayes utility of the best action, including zero-utility abstention."""

    probabilities = stability_probabilities(state)
    if not len(probabilities):
        return 0.0
    expected_action_utilities = 2.0 * probabilities - 1.0
    return max(0.0, float(np.max(expected_action_utilities)))


def _common_standard_draws(feature_dimension: int) -> FloatArray:
    _require(feature_dimension > 0, "feature dimension must be positive")
    generator = np.random.default_rng(PREDICTIVE_SEED)
    half = generator.standard_normal((PREDICTIVE_DRAW_COUNT // 2, feature_dimension))
    return np.asarray(np.concatenate((half, -half), axis=0), dtype=np.float64)


def _covariance_factor(covariance: FloatArray) -> FloatArray:
    covariance = np.asarray(covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    _require(
        float(np.min(eigenvalues)) >= -1e-10 * scale,
        "predictive covariance is not positive semidefinite",
    )
    return np.asarray(
        eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))),
        dtype=np.float64,
    )


def decision_value_of_probe(
    state: PoseItGaussianState,
    pose: int,
    *,
    standard_draws: FloatArray | None = None,
) -> float:
    """Expected reduction in downstream Bayes decision regret from one probe."""

    _require(pose in state.action_poses, "candidate probe is unavailable")
    _require(pose not in state.observed_poses, "candidate probe was already observed")
    feature_indices = np.asarray(
        pose_feature_indices(pose, state.feature_dimension), dtype=int
    )
    feature_mean = state.mean[feature_indices]
    feature_covariance = state.covariance[np.ix_(feature_indices, feature_indices)]
    cross = state.covariance[:, feature_indices]
    gain = _positive_definite_solve(feature_covariance, cross.T).T
    posterior_covariance = state.covariance - gain @ cross.T
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)

    draws = (
        _common_standard_draws(state.feature_dimension)
        if standard_draws is None
        else np.asarray(standard_draws, dtype=np.float64)
    )
    _require(
        draws.shape == (PREDICTIVE_DRAW_COUNT, state.feature_dimension),
        "common-random-number shape changed",
    )
    feature_samples = feature_mean + draws @ _covariance_factor(feature_covariance).T
    latent_indices = np.asarray(
        [
            pose_stability_index(action, state.feature_dimension)
            for action in state.action_poses
        ],
        dtype=int,
    )
    posterior_latent_mean = (
        state.mean[latent_indices]
        + (feature_samples - feature_mean) @ gain[latent_indices, :].T
    )
    posterior_latent_variance = np.diag(
        posterior_covariance[np.ix_(latent_indices, latent_indices)]
    )
    posterior_probability = _stability_probabilities(
        posterior_latent_mean,
        np.broadcast_to(posterior_latent_variance, posterior_latent_mean.shape),
    )
    posterior_best = np.maximum(
        0.0,
        np.max(2.0 * posterior_probability - 1.0, axis=1),
    )
    return max(0.0, float(np.mean(posterior_best)) - expected_best_utility(state))


def _positive_logdet(covariance: FloatArray) -> float:
    sign, value = np.linalg.slogdet(covariance)
    _require(sign > 0 and np.isfinite(value), "covariance log determinant failed")
    return float(value)


def system_identification_value_of_probe(
    state: PoseItGaussianState,
    pose: int,
) -> float:
    """Information gain about latent responses, independent of downstream utility."""

    _require(pose in state.action_poses, "candidate probe is unavailable")
    _require(pose not in state.observed_poses, "candidate probe was already observed")
    feature_indices = np.asarray(
        pose_feature_indices(pose, state.feature_dimension), dtype=int
    )
    latent_indices = np.asarray(
        [
            pose_stability_index(action, state.feature_dimension)
            for action in state.action_poses
        ],
        dtype=int,
    )
    covariance_ff = state.covariance[np.ix_(feature_indices, feature_indices)]
    covariance_ll = state.covariance[np.ix_(latent_indices, latent_indices)]
    cross = state.covariance[np.ix_(latent_indices, feature_indices)]
    conditional = covariance_ll - cross @ _positive_definite_solve(
        covariance_ff, cross.T
    )
    conditional = 0.5 * (conditional + conditional.T)
    return max(
        0.0,
        0.5 * (_positive_logdet(covariance_ll) - _positive_logdet(conditional)),
    )


def select_next_probe(
    state: PoseItGaussianState,
    remaining: Sequence[int],
    *,
    selector: SelectorName,
    standard_draws: FloatArray | None = None,
) -> int:
    """Select one probe with the registered lowest-pose tie break."""

    candidates = tuple(int(pose) for pose in remaining)
    _require(bool(candidates), "candidate probe set is empty")
    _require(len(candidates) == len(set(candidates)), "candidate probe repeated")
    _require(set(candidates) <= set(state.action_poses), "candidate is unavailable")
    _require(
        not (set(candidates) & set(state.observed_poses)),
        "observed candidate was retained",
    )
    scores: list[tuple[float, int]] = []
    for pose in candidates:
        if selector == "decision_directed":
            value = decision_value_of_probe(
                state,
                pose,
                standard_draws=standard_draws,
            )
        elif selector == "system_identification":
            value = system_identification_value_of_probe(state, pose)
        elif selector == "fixed":
            value = 0.0
        else:  # pragma: no cover
            raise ValueError(f"unknown selector: {selector}")
        scores.append((float(value), pose))
    return min(scores, key=lambda item: (-item[0], item[1]))[1]


@dataclass(frozen=True)
class PoseItPolicyTrace:
    """Posterior states at registered budgets zero through three."""

    selector: str
    selected_poses: tuple[int, ...]
    states: tuple[PoseItGaussianState, ...]

    def __post_init__(self) -> None:
        _require(
            len(self.states) == REGISTERED_PROBE_BUDGET + 1, "budget roster changed"
        )
        _require(
            len(self.selected_poses) <= REGISTERED_PROBE_BUDGET,
            "probe budget exceeded",
        )
        _require(
            len(self.selected_poses) == len(set(self.selected_poses)),
            "selected pose repeated",
        )
        _require(
            all(pose in SELECTABLE_POSES for pose in self.selected_poses),
            "unregistered pose selected",
        )
        _require(
            self.states[0].observed_poses == (MANDATORY_ANCHOR,),
            "trace does not begin after the mandatory anchor",
        )
        for budget, state in enumerate(self.states):
            expected = (
                MANDATORY_ANCHOR,
                *self.selected_poses[: min(budget, len(self.selected_poses))],
            )
            _require(
                state.observed_poses == expected,
                "trace state contains an unselected observation",
            )


def _validated_feature_map(
    state: PoseItGaussianState,
    pre_shake_features: Mapping[int, Sequence[float] | FloatArray],
) -> dict[int, FloatArray]:
    features = {
        int(pose): np.asarray(values, dtype=np.float64)
        for pose, values in pre_shake_features.items()
    }
    _require(
        set(features) == set(state.available_poses),
        "pre-shake feature roster differs from structurally available poses",
    )
    for values in features.values():
        _require(
            values.shape == (state.feature_dimension,),
            "pre-shake feature shape changed",
        )
        _require(bool(np.all(np.isfinite(values))), "pre-shake feature is non-finite")
    return features


def trace_policy(
    prior: PoseItGaussianState,
    pre_shake_features: Mapping[int, Sequence[float] | FloatArray],
    *,
    selector: SelectorName,
) -> PoseItPolicyTrace:
    """Trace a selector using pre-shake features only; no outcome input exists."""

    _require(not prior.observed_poses, "policy prior already contains observations")
    features = _validated_feature_map(prior, pre_shake_features)
    state = condition_on_pose_features(
        prior,
        MANDATORY_ANCHOR,
        features[MANDATORY_ANCHOR],
    )
    states = [state]
    remaining = list(state.action_poses)
    selected: list[int] = []
    standard_draws = _common_standard_draws(state.feature_dimension)
    for _ in range(REGISTERED_PROBE_BUDGET):
        if remaining:
            pose = select_next_probe(
                state,
                remaining,
                selector=selector,
                standard_draws=standard_draws,
            )
            selected.append(pose)
            state = condition_on_pose_features(state, pose, features[pose])
            remaining.remove(pose)
        states.append(state)
    return PoseItPolicyTrace(
        selector=selector,
        selected_poses=tuple(selected),
        states=tuple(states),
    )


def trace_probe_order(
    prior: PoseItGaussianState,
    pre_shake_features: Mapping[int, Sequence[float] | FloatArray],
    probe_order: Sequence[int],
    *,
    selector: str,
) -> PoseItPolicyTrace:
    """Trace one fixed control order without opening any shake outcome."""

    order = tuple(int(pose) for pose in probe_order)
    _require(len(order) == len(set(order)), "fixed probe order repeats a pose")
    _require(set(order) <= set(prior.action_poses), "fixed probe is unavailable")
    features = _validated_feature_map(prior, pre_shake_features)
    state = condition_on_pose_features(
        prior,
        MANDATORY_ANCHOR,
        features[MANDATORY_ANCHOR],
    )
    states = [state]
    selected: list[int] = []
    for pose in order[:REGISTERED_PROBE_BUDGET]:
        selected.append(pose)
        state = condition_on_pose_features(state, pose, features[pose])
        states.append(state)
    while len(states) < REGISTERED_PROBE_BUDGET + 1:
        states.append(state)
    return PoseItPolicyTrace(
        selector=selector,
        selected_poses=tuple(selected),
        states=tuple(states),
    )


__all__ = [
    "POSE_COUNT",
    "PREDICTIVE_DRAW_COUNT",
    "PREDICTIVE_SEED",
    "REGISTERED_PROBE_BUDGET",
    "PoseItGaussianState",
    "PoseItPolicyTrace",
    "condition_on_pose_features",
    "decision_value_of_probe",
    "expected_best_utility",
    "pose_feature_indices",
    "pose_stability_index",
    "select_next_probe",
    "stability_probabilities",
    "system_identification_value_of_probe",
    "trace_policy",
    "trace_probe_order",
]
