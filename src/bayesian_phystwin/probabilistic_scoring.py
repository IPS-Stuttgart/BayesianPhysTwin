"""Proper probabilistic scores for registered physical-query predictions.

The functions in this module score predictive distributions in a caller-frozen
query space.  They deliberately do not reinterpret readout discrepancy as a
latent physical-state posterior, calibrate any covariance, or authorize a
scientific claim.  Claim-bearing use requires a separately registered cohort,
method lock, information boundary, and decision rule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    immutable_array,
    plain_json,
)
from ._portable_contracts import content_id

PROBABILISTIC_SCORE_INPUT_CONTRACT: Final = (
    "bayesian-phystwin-probabilistic-score-input-v1"
)
PROBABILISTIC_SCORE_REPORT_CONTRACT: Final = (
    "bayesian-phystwin-probabilistic-score-report-v1"
)
PROBABILISTIC_SCORE_REPORT_VERSION: Final = 1
PROBABILISTIC_SCORE_IMPLEMENTATION: Final = "registered-query-proper-scoring-v1"
PROBABILISTIC_SCORE_CLAIM_BOUNDARY: Final = (
    "Diagnostic scoring infrastructure only. Proper scores compare predictive "
    "distributions in the registered query space, but do not establish raw "
    "covariance calibration, physical-state identification, fresh-object "
    "transfer, intervention benefit, deployment safety, or state of the art."
)
ENERGY_SCORE: Final = "energy_score"
VARIOGRAM_SCORE: Final = "variogram_score"
GAUSSIAN_NLL_PER_DIMENSION: Final = "gaussian_nll_per_dimension"
WEIGHTED_INTERVAL_SCORE: Final = "weighted_interval_score"
SCORE_ORDER: Final = (
    ENERGY_SCORE,
    VARIOGRAM_SCORE,
    GAUSSIAN_NLL_PER_DIMENSION,
    WEIGHTED_INTERVAL_SCORE,
)
GENERAL_PROBABILISTIC_SCORING_PROFILE: Final = "general-probabilistic-scoring-v1"
BAYESIAN_VALUE_DECOMPOSITION_PROFILE: Final = "bayesian-value-decomposition-v1"
BAYESIAN_VALUE_REQUIRED_METHODS: Final = (
    "bayesian_full_guarded",
    "bayesian_mean_guarded",
    "last_residual",
    "last_residual_guarded",
    "physical_fallback",
)
BAYESIAN_VALUE_REQUIRED_COMPARISONS: Final = (
    (
        "bayesian-full-vs-bayesian-mean",
        "bayesian_full_guarded",
        "bayesian_mean_guarded",
    ),
    (
        "bayesian-mean-vs-guarded-last-residual",
        "bayesian_mean_guarded",
        "last_residual_guarded",
    ),
    (
        "guarded-last-residual-vs-last-residual",
        "last_residual_guarded",
        "last_residual",
    ),
    (
        "last-residual-vs-physical-fallback",
        "last_residual",
        "physical_fallback",
    ),
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _fields(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    name: str,
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result <= minimum if minimum_exclusive else result < minimum
        if invalid:
            relation = "greater than" if minimum_exclusive else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _optional_reliability(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name=name, minimum=0.0, maximum=1.0)


def _optional_rank(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer or null")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer or null")
    return result


def _horizon(value: object, *, name: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a label or nonnegative number")
    if type(value) is str:
        return _text(value, name=name)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = _number(value, name=name, minimum=0.0)
        return f"{number:.12g}"
    raise ValueError(f"{name} must be a label or nonnegative number")


def _finite_numeric_array(
    value: object,
    *,
    name: str,
    minimum_ndim: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim < minimum_ndim:
        raise ValueError(f"{name} must have at least {minimum_ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _event_array(value: object, *, name: str) -> np.ndarray:
    array = _finite_numeric_array(value, name=name, minimum_ndim=1)
    if array.size < 1:
        raise ValueError(f"{name} must not be empty")
    return immutable_array(array, dtype=np.float64)


def _normalized_sample_weights(
    value: object | None,
    *,
    sample_count: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return immutable_array(
            np.full(sample_count, 1.0 / sample_count, dtype=np.float64)
        )
    raw = _finite_numeric_array(value, name=name, minimum_ndim=1)
    if raw.shape != (sample_count,):
        raise ValueError(f"{name} must have shape ({sample_count},)")
    if np.any(raw < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    weights = raw / total
    return immutable_array(weights, dtype=np.float64)


def _predictive_samples(
    value: object,
    *,
    observation_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    samples = _finite_numeric_array(value, name=name, minimum_ndim=2)
    if samples.shape[0] < 1 or samples.shape[1:] != observation_shape:
        raise ValueError(f"{name} must have shape (sample, {observation_shape!r})")
    return immutable_array(samples, dtype=np.float64)


def _symmetric_positive_definite_covariance(
    value: object,
    *,
    dimension: int,
    name: str,
    maximum_condition_number: float,
) -> np.ndarray:
    covariance = _finite_numeric_array(value, name=name, minimum_ndim=2)
    if covariance.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    condition_number = float(np.linalg.cond(covariance))
    if not np.isfinite(condition_number) or condition_number > maximum_condition_number:
        raise ValueError(f"{name} exceeds the registered condition-number limit")
    return immutable_array(covariance, dtype=np.float64)


def energy_score(
    samples: object,
    observation: object,
    *,
    beta: float = 1.0,
    sample_weights: object | None = None,
    block_size: int = 256,
) -> float:
    """Return the weighted empirical energy score.

    Pairwise distances are accumulated in bounded blocks, avoiding a complete
    ``sample_count x sample_count`` temporary. ``beta`` must lie in ``(0, 2)``.
    """

    target = _event_array(observation, name="observation")
    draws = _predictive_samples(
        samples,
        observation_shape=target.shape,
        name="samples",
    )
    exponent = _number(
        beta,
        name="beta",
        minimum=0.0,
        maximum=2.0,
        minimum_exclusive=True,
    )
    if exponent >= 2.0:
        raise ValueError("beta must lie strictly inside (0, 2)")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size must be a genuine integer")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    weights = _normalized_sample_weights(
        sample_weights,
        sample_count=len(draws),
        name="sample_weights",
    )
    flattened = np.asarray(draws).reshape(len(draws), -1)
    target_flat = np.asarray(target).reshape(-1)
    target_distance = np.linalg.norm(flattened - target_flat, axis=1) ** exponent
    first = float(weights @ target_distance)
    second = 0.0
    for start in range(0, len(flattened), block_size):
        stop = min(start + block_size, len(flattened))
        difference = flattened[start:stop, None, :] - flattened[None, :, :]
        distance = np.linalg.norm(difference, axis=2) ** exponent
        second += float(np.sum(weights[start:stop, None] * weights[None, :] * distance))
    score = first - 0.5 * second
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, first, second)
    if score < 0.0 and abs(score) <= tolerance:
        score = 0.0
    if not np.isfinite(score) or score < 0.0:
        raise FloatingPointError("energy score became invalid")
    return float(score)


def variogram_score(
    samples: object,
    observation: object,
    *,
    pair_indices: object,
    order: float = 0.5,
    pair_weights: object | None = None,
    sample_weights: object | None = None,
) -> float:
    """Return a pair-weight-normalized empirical variogram score."""

    target = _event_array(observation, name="observation")
    draws = _predictive_samples(
        samples,
        observation_shape=target.shape,
        name="samples",
    )
    exponent = _number(
        order,
        name="order",
        minimum=0.0,
        maximum=2.0,
        minimum_exclusive=True,
    )
    raw_pairs = np.asarray(pair_indices)
    if (
        raw_pairs.ndim != 2
        or raw_pairs.shape[1] != 2
        or raw_pairs.dtype.kind not in {"i", "u"}
    ):
        raise ValueError("pair_indices must be an integer array with shape (P, 2)")
    pairs = np.asarray(raw_pairs, dtype=np.int64)
    if len(pairs) < 1:
        raise ValueError("pair_indices must not be empty")
    dimension = target.size
    if np.any(pairs < 0) or np.any(pairs >= dimension):
        raise ValueError("pair_indices contain an out-of-range component")
    if np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError("pair_indices must satisfy left < right")
    if len({tuple(pair) for pair in pairs.tolist()}) != len(pairs):
        raise ValueError("pair_indices must not contain duplicates")
    sample_mass = _normalized_sample_weights(
        sample_weights,
        sample_count=len(draws),
        name="sample_weights",
    )
    pair_mass = _normalized_sample_weights(
        pair_weights,
        sample_count=len(pairs),
        name="pair_weights",
    )
    flattened = np.asarray(draws).reshape(len(draws), -1)
    target_flat = np.asarray(target).reshape(-1)
    observed = np.abs(target_flat[pairs[:, 0]] - target_flat[pairs[:, 1]]) ** exponent
    sample_terms = (
        np.abs(flattened[:, pairs[:, 0]] - flattened[:, pairs[:, 1]]) ** exponent
    )
    expected = sample_mass @ sample_terms
    score = float(pair_mass @ np.square(observed - expected))
    if not np.isfinite(score) or score < 0.0:
        raise FloatingPointError("variogram score became invalid")
    return score


def gaussian_nll_per_dimension(
    mean: object,
    covariance: object,
    observation: object,
    *,
    maximum_condition_number: float = 1e14,
) -> float:
    """Return the exact Gaussian negative log score per registered dimension."""

    target = _event_array(observation, name="observation")
    location = _event_array(mean, name="mean")
    if location.shape != target.shape:
        raise ValueError("mean shape differs from observation shape")
    condition_limit = _number(
        maximum_condition_number,
        name="maximum_condition_number",
        minimum=1.0,
    )
    dimension = target.size
    matrix = _symmetric_positive_definite_covariance(
        covariance,
        dimension=dimension,
        name="covariance",
        maximum_condition_number=condition_limit,
    )
    factor = np.linalg.cholesky(np.asarray(matrix))
    difference = np.asarray(target - location).reshape(-1)
    whitened = np.linalg.solve(factor, difference)
    quadratic = float(whitened @ whitened)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
    score = 0.5 * (dimension * math.log(2.0 * math.pi) + log_determinant + quadratic)
    score /= dimension
    if not np.isfinite(score):
        raise FloatingPointError("Gaussian negative log score became invalid")
    return float(score)


def interval_score(
    lower: object,
    upper: object,
    observation: object,
    *,
    nominal_coverage: float,
) -> float:
    """Return the component-mean central interval score."""

    target = _event_array(observation, name="observation")
    lower_array = _event_array(lower, name="lower")
    upper_array = _event_array(upper, name="upper")
    if lower_array.shape != target.shape or upper_array.shape != target.shape:
        raise ValueError("interval bounds must match the observation shape")
    if np.any(lower_array > upper_array):
        raise ValueError("interval lower bounds exceed upper bounds")
    coverage = _number(
        nominal_coverage,
        name="nominal_coverage",
        minimum=0.0,
        maximum=1.0,
        minimum_exclusive=True,
    )
    if coverage >= 1.0:
        raise ValueError("nominal_coverage must lie strictly inside (0, 1)")
    alpha = 1.0 - coverage
    width = np.asarray(upper_array - lower_array)
    below = np.maximum(np.asarray(lower_array - target), 0.0)
    above = np.maximum(np.asarray(target - upper_array), 0.0)
    component_score = width + 2.0 / alpha * (below + above)
    result = float(np.mean(component_score))
    if not np.isfinite(result) or result < 0.0:
        raise FloatingPointError("interval score became invalid")
    return result


def weighted_interval_score(
    median: object,
    lower_bounds: object,
    upper_bounds: object,
    observation: object,
    *,
    nominal_coverages: Sequence[float],
) -> float:
    """Return the component-mean weighted interval score."""

    target = _event_array(observation, name="observation")
    median_array = _event_array(median, name="median")
    if median_array.shape != target.shape:
        raise ValueError("median shape differs from observation shape")
    lower = _finite_numeric_array(
        lower_bounds,
        name="lower_bounds",
        minimum_ndim=2,
    )
    upper = _finite_numeric_array(
        upper_bounds,
        name="upper_bounds",
        minimum_ndim=2,
    )
    if lower.shape != upper.shape or lower.shape[1:] != target.shape:
        raise ValueError(
            "interval arrays must have shape (coverage, *observation.shape)"
        )
    coverages = tuple(
        _number(
            value,
            name=f"nominal_coverages[{index}]",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        )
        for index, value in enumerate(nominal_coverages)
    )
    if len(coverages) != len(lower):
        raise ValueError("nominal_coverages length differs from interval arrays")
    if not coverages or tuple(sorted(set(coverages))) != coverages:
        raise ValueError("nominal_coverages must be strictly increasing")
    if any(value >= 1.0 for value in coverages):
        raise ValueError("nominal_coverages must lie strictly inside (0, 1)")
    if np.any(lower > upper):
        raise ValueError("interval lower bounds exceed upper bounds")
    component_score = 0.5 * np.abs(np.asarray(target - median_array))
    for index, coverage in enumerate(coverages):
        alpha = 1.0 - coverage
        width = upper[index] - lower[index]
        below = np.maximum(lower[index] - target, 0.0)
        above = np.maximum(target - upper[index], 0.0)
        interval_component = width + 2.0 / alpha * (below + above)
        component_score += 0.5 * alpha * interval_component
    result = float(np.mean(component_score / (len(coverages) + 0.5)))
    if not np.isfinite(result) or result < 0.0:
        raise FloatingPointError("weighted interval score became invalid")
    return result


@dataclass(frozen=True, slots=True)
class PredictiveIntervalV1:
    nominal_coverage: float
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        coverage = _number(
            self.nominal_coverage,
            name="nominal_coverage",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        )
        if coverage >= 1.0:
            raise ValueError("nominal_coverage must lie strictly inside (0, 1)")
        lower = _event_array(self.lower, name="lower")
        upper = _event_array(self.upper, name="upper")
        if lower.shape != upper.shape or np.any(lower > upper):
            raise ValueError("predictive interval bounds are inconsistent")
        object.__setattr__(self, "nominal_coverage", coverage)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class PredictiveArmV1:
    method: str
    accepted: bool
    risk_score: float
    reliability: float | None
    identifiable_rank: int | None
    samples: np.ndarray | None
    sample_weights: np.ndarray | None
    gaussian_mean: np.ndarray | None
    gaussian_covariance: np.ndarray | None
    median: np.ndarray | None
    intervals: tuple[PredictiveIntervalV1, ...]


@dataclass(frozen=True, slots=True)
class PredictiveUnitV1:
    unit_id: str
    group_id: str
    horizon: str
    observation: np.ndarray
    predictions: tuple[PredictiveArmV1, ...]


@dataclass(frozen=True, slots=True)
class ProbabilisticScoreConfigurationV1:
    score_names: tuple[str, ...]
    energy_beta: float | None
    variogram_order: float | None
    variogram_pairs: np.ndarray | None
    variogram_pair_weights: np.ndarray | None
    gaussian_maximum_condition_number: float


@dataclass(frozen=True, slots=True)
class ScoreComparisonV1:
    comparison_id: str
    candidate_method: str
    reference_method: str


@dataclass(frozen=True, slots=True)
class ProbabilisticScoreBundleV1:
    protocol_id: str
    statistical_unit: str
    claim_boundary: str
    analysis_profile: str
    fallback_method: str
    reference_method: str
    configuration: ProbabilisticScoreConfigurationV1
    comparison_pairs: tuple[ScoreComparisonV1, ...]
    units: tuple[PredictiveUnitV1, ...]


def _same_optional_array(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.array_equal(left, right))


def _same_intervals(
    left: tuple[PredictiveIntervalV1, ...],
    right: tuple[PredictiveIntervalV1, ...],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        first.nominal_coverage == second.nominal_coverage
        and np.array_equal(first.lower, second.lower)
        and np.array_equal(first.upper, second.upper)
        for first, second in zip(left, right, strict=True)
    )


def _validate_bayesian_value_profile(
    bundle: ProbabilisticScoreBundleV1,
) -> None:
    if bundle.fallback_method != "physical_fallback":
        raise ValueError(
            "Bayesian-value profile requires physical_fallback as fallback_method"
        )
    if bundle.reference_method != "last_residual":
        raise ValueError(
            "Bayesian-value profile requires last_residual as reference_method"
        )
    methods = tuple(arm.method for arm in bundle.units[0].predictions)
    if methods != BAYESIAN_VALUE_REQUIRED_METHODS:
        raise ValueError(
            "Bayesian-value profile requires the exact five registered methods"
        )
    if bundle.configuration.score_names != SCORE_ORDER:
        raise ValueError(
            "Bayesian-value profile requires all four registered proper scores"
        )
    expected_comparisons = tuple(
        ScoreComparisonV1(
            comparison_id=comparison_id,
            candidate_method=candidate_method,
            reference_method=reference_method,
        )
        for comparison_id, candidate_method, reference_method in (
            BAYESIAN_VALUE_REQUIRED_COMPARISONS
        )
    )
    if bundle.comparison_pairs != expected_comparisons:
        raise ValueError(
            "Bayesian-value profile requires the four registered attribution pairs"
        )
    for unit in bundle.units:
        by_method = {arm.method: arm for arm in unit.predictions}
        last = by_method["last_residual"]
        guarded = by_method["last_residual_guarded"]
        if not last.accepted:
            raise ValueError(
                "Bayesian-value profile requires last_residual to be unguarded"
            )
        raw_equal = all(
            (
                _same_optional_array(last.samples, guarded.samples),
                _same_optional_array(
                    last.sample_weights,
                    guarded.sample_weights,
                ),
                _same_optional_array(
                    last.gaussian_mean,
                    guarded.gaussian_mean,
                ),
                _same_optional_array(
                    last.gaussian_covariance,
                    guarded.gaussian_covariance,
                ),
                _same_optional_array(last.median, guarded.median),
                _same_intervals(last.intervals, guarded.intervals),
            )
        )
        if not raw_equal:
            raise ValueError(
                "last_residual_guarded raw prediction must equal last_residual"
            )


def validate_bayesian_value_decomposition_bundle(
    payload: Mapping[str, object] | ProbabilisticScoreBundleV1,
) -> ProbabilisticScoreBundleV1:
    """Validate and return the exact five-arm Bayesian-value profile."""

    bundle = (
        payload
        if isinstance(payload, ProbabilisticScoreBundleV1)
        else parse_probabilistic_score_bundle(payload)
    )
    if bundle.analysis_profile != BAYESIAN_VALUE_DECOMPOSITION_PROFILE:
        raise ValueError("analysis_profile must select bayesian-value-decomposition-v1")
    _validate_bayesian_value_profile(bundle)
    return bundle


def _parse_score_configuration(
    value: object,
) -> ProbabilisticScoreConfigurationV1:
    name = "score_configuration"
    config = _mapping(value, name=name)
    _fields(
        config,
        required=frozenset({"score_names"}),
        optional=frozenset(
            {
                "energy_beta",
                "variogram_order",
                "variogram_pairs",
                "variogram_pair_weights",
                "gaussian_maximum_condition_number",
            }
        ),
        name=name,
    )
    raw_names = tuple(
        _text(item, name=f"{name}.score_names[{index}]")
        for index, item in enumerate(
            _sequence(config["score_names"], name=f"{name}.score_names")
        )
    )
    if not raw_names or len(set(raw_names)) != len(raw_names):
        raise ValueError("score_configuration.score_names must be unique and nonempty")
    unknown = sorted(set(raw_names) - set(SCORE_ORDER))
    if unknown:
        raise ValueError(f"unknown probabilistic scores: {unknown}")
    canonical = tuple(score for score in SCORE_ORDER if score in raw_names)
    if raw_names != canonical:
        raise ValueError(
            "score_configuration.score_names must follow the canonical score order"
        )

    energy_beta: float | None = None
    if ENERGY_SCORE in raw_names:
        if "energy_beta" not in config:
            raise ValueError("energy_beta is required for energy_score")
        energy_beta = _number(
            config["energy_beta"],
            name=f"{name}.energy_beta",
            minimum=0.0,
            maximum=2.0,
            minimum_exclusive=True,
        )
        if energy_beta >= 2.0:
            raise ValueError("energy_beta must lie strictly inside (0, 2)")
    elif "energy_beta" in config:
        raise ValueError("energy_beta is unused without energy_score")

    variogram_order: float | None = None
    variogram_pairs: np.ndarray | None = None
    variogram_pair_weights: np.ndarray | None = None
    variogram_fields = {
        "variogram_order",
        "variogram_pairs",
        "variogram_pair_weights",
    }
    if VARIOGRAM_SCORE in raw_names:
        missing = sorted({"variogram_order", "variogram_pairs"} - set(config))
        if missing:
            raise ValueError(f"variogram score configuration missing {missing}")
        variogram_order = _number(
            config["variogram_order"],
            name=f"{name}.variogram_order",
            minimum=0.0,
            maximum=2.0,
            minimum_exclusive=True,
        )
        raw_pairs = np.asarray(config["variogram_pairs"])
        if (
            raw_pairs.ndim != 2
            or raw_pairs.shape[1] != 2
            or raw_pairs.dtype.kind not in {"i", "u"}
        ):
            raise ValueError("variogram_pairs must have integer shape (P, 2)")
        pairs = np.asarray(raw_pairs, dtype=np.int64)
        if len(pairs) < 1:
            raise ValueError("variogram_pairs must not be empty")
        if np.any(pairs < 0):
            raise ValueError("variogram_pairs must be nonnegative")
        if np.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError("variogram_pairs must satisfy left < right")
        if len({tuple(pair) for pair in pairs.tolist()}) != len(pairs):
            raise ValueError("variogram_pairs must not contain duplicates")
        variogram_pairs = immutable_array(pairs, dtype=np.int64)
        if "variogram_pair_weights" in config:
            variogram_pair_weights = _normalized_sample_weights(
                config["variogram_pair_weights"],
                sample_count=len(pairs),
                name=f"{name}.variogram_pair_weights",
            )
    elif set(config) & variogram_fields:
        raise ValueError("variogram fields are unused without variogram_score")

    condition = _number(
        config.get("gaussian_maximum_condition_number", 1e14),
        name=f"{name}.gaussian_maximum_condition_number",
        minimum=1.0,
    )
    if (
        GAUSSIAN_NLL_PER_DIMENSION not in raw_names
        and "gaussian_maximum_condition_number" in config
    ):
        raise ValueError(
            "gaussian_maximum_condition_number is unused without Gaussian NLL"
        )
    return ProbabilisticScoreConfigurationV1(
        score_names=raw_names,
        energy_beta=energy_beta,
        variogram_order=variogram_order,
        variogram_pairs=variogram_pairs,
        variogram_pair_weights=variogram_pair_weights,
        gaussian_maximum_condition_number=condition,
    )


def _parse_interval(
    value: object,
    *,
    name: str,
    observation_shape: tuple[int, ...],
) -> PredictiveIntervalV1:
    interval = _mapping(value, name=name)
    _fields(
        interval,
        required=frozenset({"nominal_coverage", "lower", "upper"}),
        optional=frozenset(),
        name=name,
    )
    result = PredictiveIntervalV1(
        nominal_coverage=_number(
            interval["nominal_coverage"],
            name=f"{name}.nominal_coverage",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        ),
        lower=_event_array(interval["lower"], name=f"{name}.lower"),
        upper=_event_array(interval["upper"], name=f"{name}.upper"),
    )
    if result.lower.shape != observation_shape:
        raise ValueError(f"{name} shape differs from observation shape")
    return result


def _parse_arm(
    value: object,
    *,
    name: str,
    observation: np.ndarray,
    configuration: ProbabilisticScoreConfigurationV1,
) -> PredictiveArmV1:
    arm = _mapping(value, name=name)
    common_required = {"method", "accepted", "risk_score"}
    optional = {"reliability", "identifiable_rank"}
    required = set(common_required)
    if ENERGY_SCORE in configuration.score_names or VARIOGRAM_SCORE in (
        configuration.score_names
    ):
        required.add("samples")
        optional.add("sample_weights")
    if GAUSSIAN_NLL_PER_DIMENSION in configuration.score_names:
        required.update({"gaussian_mean", "gaussian_covariance"})
    if WEIGHTED_INTERVAL_SCORE in configuration.score_names:
        required.update({"median", "intervals"})
    _fields(
        arm,
        required=frozenset(required),
        optional=frozenset(optional),
        name=name,
    )
    samples: np.ndarray | None = None
    sample_weights: np.ndarray | None = None
    if "samples" in arm:
        samples = _predictive_samples(
            arm["samples"],
            observation_shape=observation.shape,
            name=f"{name}.samples",
        )
        sample_weights = _normalized_sample_weights(
            arm.get("sample_weights"),
            sample_count=len(samples),
            name=f"{name}.sample_weights",
        )
    gaussian_mean: np.ndarray | None = None
    gaussian_covariance: np.ndarray | None = None
    if "gaussian_mean" in arm:
        gaussian_mean = _event_array(
            arm["gaussian_mean"],
            name=f"{name}.gaussian_mean",
        )
        if gaussian_mean.shape != observation.shape:
            raise ValueError(f"{name}.gaussian_mean shape changed")
        gaussian_covariance = _symmetric_positive_definite_covariance(
            arm["gaussian_covariance"],
            dimension=observation.size,
            name=f"{name}.gaussian_covariance",
            maximum_condition_number=(configuration.gaussian_maximum_condition_number),
        )
    median: np.ndarray | None = None
    intervals: tuple[PredictiveIntervalV1, ...] = ()
    if "median" in arm:
        median = _event_array(arm["median"], name=f"{name}.median")
        if median.shape != observation.shape:
            raise ValueError(f"{name}.median shape changed")
        intervals = tuple(
            _parse_interval(
                raw,
                name=f"{name}.intervals[{index}]",
                observation_shape=observation.shape,
            )
            for index, raw in enumerate(
                _sequence(arm["intervals"], name=f"{name}.intervals")
            )
        )
        coverages = tuple(item.nominal_coverage for item in intervals)
        if not coverages or tuple(sorted(set(coverages))) != coverages:
            raise ValueError(f"{name}.intervals must have increasing coverages")
    return PredictiveArmV1(
        method=_text(arm["method"], name=f"{name}.method"),
        accepted=_boolean(arm["accepted"], name=f"{name}.accepted"),
        risk_score=_number(arm["risk_score"], name=f"{name}.risk_score"),
        reliability=_optional_reliability(
            arm.get("reliability"),
            name=f"{name}.reliability",
        ),
        identifiable_rank=_optional_rank(
            arm.get("identifiable_rank"),
            name=f"{name}.identifiable_rank",
        ),
        samples=samples,
        sample_weights=sample_weights,
        gaussian_mean=gaussian_mean,
        gaussian_covariance=gaussian_covariance,
        median=median,
        intervals=intervals,
    )


def _parse_comparison_pairs(
    value: object | None,
    *,
    methods: tuple[str, ...],
) -> tuple[ScoreComparisonV1, ...]:
    if value is None:
        return ()
    pairs: list[ScoreComparisonV1] = []
    seen_method_pairs: set[tuple[str, str]] = set()
    for index, raw_pair in enumerate(_sequence(value, name="comparison_pairs")):
        name = f"comparison_pairs[{index}]"
        pair = _mapping(raw_pair, name=name)
        _fields(
            pair,
            required=frozenset(
                {"comparison_id", "candidate_method", "reference_method"}
            ),
            optional=frozenset(),
            name=name,
        )
        result = ScoreComparisonV1(
            comparison_id=_text(
                pair["comparison_id"],
                name=f"{name}.comparison_id",
            ),
            candidate_method=_text(
                pair["candidate_method"],
                name=f"{name}.candidate_method",
            ),
            reference_method=_text(
                pair["reference_method"],
                name=f"{name}.reference_method",
            ),
        )
        if result.candidate_method == result.reference_method:
            raise ValueError(f"{name} must compare distinct methods")
        missing = sorted(
            {result.candidate_method, result.reference_method} - set(methods)
        )
        if missing:
            raise ValueError(f"{name} references unknown methods {missing}")
        method_pair = (result.candidate_method, result.reference_method)
        if method_pair in seen_method_pairs:
            raise ValueError(f"{name} repeats a candidate/reference pair")
        seen_method_pairs.add(method_pair)
        pairs.append(result)
    identifiers = tuple(pair.comparison_id for pair in pairs)
    if len(set(identifiers)) != len(identifiers) or identifiers != tuple(
        sorted(identifiers)
    ):
        raise ValueError("comparison_pairs IDs must be unique and sorted")
    return tuple(pairs)


def parse_probabilistic_score_bundle(
    payload: object,
) -> ProbabilisticScoreBundleV1:
    """Validate a matched predictive-distribution scoring bundle."""

    root = _mapping(payload, name="input")
    _fields(
        root,
        required=frozenset(
            {
                "contract",
                "schema_version",
                "protocol_id",
                "statistical_unit",
                "claim_boundary",
                "fallback_method",
                "reference_method",
                "score_configuration",
                "units",
            }
        ),
        optional=frozenset({"analysis_profile", "comparison_pairs"}),
        name="input",
    )
    if root["contract"] != PROBABILISTIC_SCORE_INPUT_CONTRACT:
        raise ValueError(f"contract must be {PROBABILISTIC_SCORE_INPUT_CONTRACT!r}")
    if isinstance(root["schema_version"], bool) or root["schema_version"] != 1:
        raise ValueError("schema_version must be the integer 1")
    configuration = _parse_score_configuration(root["score_configuration"])
    fallback_method = _text(root["fallback_method"], name="fallback_method")
    reference_method = _text(root["reference_method"], name="reference_method")
    units: list[PredictiveUnitV1] = []
    expected_methods: tuple[str, ...] | None = None
    expected_coverages: tuple[float, ...] | None = None
    seen_units: set[str] = set()
    for unit_index, raw_unit in enumerate(_sequence(root["units"], name="units")):
        name = f"units[{unit_index}]"
        unit = _mapping(raw_unit, name=name)
        _fields(
            unit,
            required=frozenset(
                {"unit_id", "group_id", "horizon", "observation", "predictions"}
            ),
            optional=frozenset(),
            name=name,
        )
        unit_id = _text(unit["unit_id"], name=f"{name}.unit_id")
        if unit_id in seen_units:
            raise ValueError(f"duplicate unit_id {unit_id!r}")
        seen_units.add(unit_id)
        observation = _event_array(
            unit["observation"],
            name=f"{name}.observation",
        )
        arms = tuple(
            _parse_arm(
                raw_arm,
                name=f"{name}.predictions[{arm_index}]",
                observation=observation,
                configuration=configuration,
            )
            for arm_index, raw_arm in enumerate(
                _sequence(unit["predictions"], name=f"{name}.predictions")
            )
        )
        if not arms:
            raise ValueError(f"{name}.predictions must not be empty")
        methods = tuple(arm.method for arm in arms)
        if len(set(methods)) != len(methods) or tuple(sorted(methods)) != methods:
            raise ValueError(f"{name}.predictions methods must be unique and sorted")
        if expected_methods is None:
            expected_methods = methods
        elif methods != expected_methods:
            raise ValueError("every unit must contain the same sorted methods")
        if fallback_method not in methods:
            raise ValueError(f"{name} lacks fallback_method {fallback_method!r}")
        if reference_method not in methods:
            raise ValueError(f"{name} lacks reference_method {reference_method!r}")
        fallback = next(arm for arm in arms if arm.method == fallback_method)
        if not fallback.accepted:
            raise ValueError("fallback_method must be marked accepted")
        if WEIGHTED_INTERVAL_SCORE in configuration.score_names:
            coverages = tuple(item.nominal_coverage for item in arms[0].intervals)
            for arm in arms[1:]:
                if tuple(item.nominal_coverage for item in arm.intervals) != coverages:
                    raise ValueError(
                        "all methods and units must use the same interval coverages"
                    )
            if expected_coverages is None:
                expected_coverages = coverages
            elif coverages != expected_coverages:
                raise ValueError(
                    "all methods and units must use the same interval coverages"
                )
        if configuration.variogram_pairs is not None:
            pairs = np.asarray(configuration.variogram_pairs)
            if np.any(pairs < 0) or np.any(pairs >= observation.size):
                raise ValueError(
                    "variogram_pairs exceed the registered query dimension"
                )
        units.append(
            PredictiveUnitV1(
                unit_id=unit_id,
                group_id=_text(unit["group_id"], name=f"{name}.group_id"),
                horizon=_horizon(unit["horizon"], name=f"{name}.horizon"),
                observation=observation,
                predictions=arms,
            )
        )
    if not units:
        raise ValueError("units must not be empty")
    if fallback_method == reference_method:
        raise ValueError("reference_method must differ from fallback_method")
    if expected_methods is None:
        raise AssertionError("validated score input produced no method inventory")
    analysis_profile = _text(
        root.get(
            "analysis_profile",
            GENERAL_PROBABILISTIC_SCORING_PROFILE,
        ),
        name="analysis_profile",
    )
    if analysis_profile not in {
        GENERAL_PROBABILISTIC_SCORING_PROFILE,
        BAYESIAN_VALUE_DECOMPOSITION_PROFILE,
    }:
        raise ValueError(f"unknown analysis_profile {analysis_profile!r}")
    comparison_pairs = _parse_comparison_pairs(
        root.get("comparison_pairs"),
        methods=expected_methods,
    )
    result = ProbabilisticScoreBundleV1(
        protocol_id=_text(root["protocol_id"], name="protocol_id"),
        statistical_unit=_text(
            root["statistical_unit"],
            name="statistical_unit",
        ),
        claim_boundary=_text(root["claim_boundary"], name="claim_boundary"),
        analysis_profile=analysis_profile,
        fallback_method=fallback_method,
        reference_method=reference_method,
        configuration=configuration,
        comparison_pairs=comparison_pairs,
        units=tuple(units),
    )
    if analysis_profile == BAYESIAN_VALUE_DECOMPOSITION_PROFILE:
        _validate_bayesian_value_profile(result)
    return result


def _score_arm(
    arm: PredictiveArmV1,
    observation: np.ndarray,
    configuration: ProbabilisticScoreConfigurationV1,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for score_name in configuration.score_names:
        if score_name == ENERGY_SCORE:
            if arm.samples is None or configuration.energy_beta is None:
                raise AssertionError("validated energy-score input is missing")
            result[score_name] = energy_score(
                arm.samples,
                observation,
                beta=configuration.energy_beta,
                sample_weights=arm.sample_weights,
            )
        elif score_name == VARIOGRAM_SCORE:
            if (
                arm.samples is None
                or configuration.variogram_pairs is None
                or configuration.variogram_order is None
            ):
                raise AssertionError("validated variogram-score input is missing")
            result[score_name] = variogram_score(
                arm.samples,
                observation,
                pair_indices=configuration.variogram_pairs,
                order=configuration.variogram_order,
                pair_weights=configuration.variogram_pair_weights,
                sample_weights=arm.sample_weights,
            )
        elif score_name == GAUSSIAN_NLL_PER_DIMENSION:
            if arm.gaussian_mean is None or arm.gaussian_covariance is None:
                raise AssertionError("validated Gaussian-score input is missing")
            result[score_name] = gaussian_nll_per_dimension(
                arm.gaussian_mean,
                arm.gaussian_covariance,
                observation,
                maximum_condition_number=(
                    configuration.gaussian_maximum_condition_number
                ),
            )
        elif score_name == WEIGHTED_INTERVAL_SCORE:
            if arm.median is None or not arm.intervals:
                raise AssertionError("validated interval-score input is missing")
            result[score_name] = weighted_interval_score(
                arm.median,
                np.stack([item.lower for item in arm.intervals]),
                np.stack([item.upper for item in arm.intervals]),
                observation,
                nominal_coverages=tuple(
                    item.nominal_coverage for item in arm.intervals
                ),
            )
        else:
            raise AssertionError(f"unhandled probabilistic score {score_name!r}")
    return result


def _interval_diagnostics(
    arm: PredictiveArmV1,
    observation: np.ndarray,
) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for interval in arm.intervals:
        covered = (observation >= interval.lower) & (observation <= interval.upper)
        diagnostics.append(
            {
                "nominal_coverage": interval.nominal_coverage,
                "simultaneous_coverage": bool(np.all(covered)),
                "component_coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(interval.upper - interval.lower)),
            }
        )
    return diagnostics


def _equal_group_mean(
    rows: Sequence[Mapping[str, object]],
    *,
    key: str,
) -> float:
    by_group: dict[str, list[float]] = {}
    for row in rows:
        group_id = row["group_id"]
        value = row[key]
        if not isinstance(group_id, str) or not isinstance(value, (int, float)):
            raise AssertionError("validated score report row changed type")
        by_group.setdefault(group_id, []).append(float(value))
    return float(np.mean([np.mean(values) for values in by_group.values()]))


def score_probabilistic_bundle(
    payload: Mapping[str, object] | ProbabilisticScoreBundleV1,
) -> dict[str, object]:
    """Score every matched arm and return a content-addressed diagnostic report."""

    bundle = (
        payload
        if isinstance(payload, ProbabilisticScoreBundleV1)
        else parse_probabilistic_score_bundle(payload)
    )
    rows: list[dict[str, object]] = []
    for unit in bundle.units:
        arm_scores = {
            arm.method: _score_arm(arm, unit.observation, bundle.configuration)
            for arm in unit.predictions
        }
        fallback_scores = arm_scores[bundle.fallback_method]
        fallback_arm = next(
            arm for arm in unit.predictions if arm.method == bundle.fallback_method
        )
        for arm in unit.predictions:
            raw_scores = arm_scores[arm.method]
            deployed_scores = raw_scores if arm.accepted else fallback_scores
            deployed_arm = arm if arm.accepted else fallback_arm
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "group_id": unit.group_id,
                    "horizon": unit.horizon,
                    "method": arm.method,
                    "accepted": arm.accepted,
                    "risk_score": arm.risk_score,
                    "reliability": arm.reliability,
                    "identifiable_rank": arm.identifiable_rank,
                    "raw_scores": dict(raw_scores),
                    "deployed_scores": dict(deployed_scores),
                    "intervals": _interval_diagnostics(
                        deployed_arm,
                        unit.observation,
                    ),
                }
            )

    methods = tuple(arm.method for arm in bundle.units[0].predictions)
    aggregate: dict[str, dict[str, object]] = {}
    for score_name in bundle.configuration.score_names:
        score_methods: dict[str, object] = {}
        fallback_rows = {
            row["unit_id"]: row
            for row in rows
            if row["method"] == bundle.fallback_method
        }
        reference_rows = {
            row["unit_id"]: row
            for row in rows
            if row["method"] == bundle.reference_method
        }
        for method in methods:
            selected = [row for row in rows if row["method"] == method]
            normalized: list[dict[str, object]] = []
            for row in selected:
                row_raw_scores = row["raw_scores"]
                row_deployed_scores = row["deployed_scores"]
                if not isinstance(row_raw_scores, Mapping) or not isinstance(
                    row_deployed_scores, Mapping
                ):
                    raise AssertionError("validated score report mapping changed")
                fallback = fallback_rows[row["unit_id"]]
                reference = reference_rows[row["unit_id"]]
                fallback_deployed = fallback["deployed_scores"]
                reference_deployed = reference["deployed_scores"]
                if not isinstance(fallback_deployed, Mapping) or not isinstance(
                    reference_deployed, Mapping
                ):
                    raise AssertionError("validated comparator score changed")
                normalized.append(
                    {
                        "group_id": row["group_id"],
                        "raw": _number(
                            row_raw_scores.get(score_name), name="raw score"
                        ),
                        "deployed": _number(
                            row_deployed_scores.get(score_name), name="deployed score"
                        ),
                        "deployed_minus_fallback": (
                            _number(
                                row_deployed_scores.get(score_name),
                                name="deployed score",
                            )
                            - _number(
                                fallback_deployed.get(score_name),
                                name="fallback deployed score",
                            )
                        ),
                        "deployed_minus_reference": (
                            _number(
                                row_deployed_scores.get(score_name),
                                name="deployed score",
                            )
                            - _number(
                                reference_deployed.get(score_name),
                                name="reference deployed score",
                            )
                        ),
                    }
                )
            group_count = len({str(row["group_id"]) for row in normalized})
            score_methods[method] = {
                "unit_count": len(normalized),
                "group_count": group_count,
                "mean_raw_score": float(
                    np.mean(
                        [
                            _number(row.get("raw"), name="normalized raw score")
                            for row in normalized
                        ]
                    )
                ),
                "mean_deployed_score": float(
                    np.mean(
                        [
                            _number(
                                row.get("deployed"), name="normalized deployed score"
                            )
                            for row in normalized
                        ]
                    )
                ),
                "equal_group_mean_raw_score": _equal_group_mean(
                    normalized,
                    key="raw",
                ),
                "equal_group_mean_deployed_score": _equal_group_mean(
                    normalized,
                    key="deployed",
                ),
                "equal_group_mean_deployed_minus_fallback": _equal_group_mean(
                    normalized,
                    key="deployed_minus_fallback",
                ),
                "equal_group_mean_deployed_minus_reference": _equal_group_mean(
                    normalized,
                    key="deployed_minus_reference",
                ),
            }
        aggregate[score_name] = score_methods

    pairwise_attribution: dict[str, object] = {}
    rows_by_method_unit = {
        (str(row["method"]), str(row["unit_id"])): row for row in rows
    }
    for pair in bundle.comparison_pairs:
        score_results: dict[str, object] = {}
        for score_name in bundle.configuration.score_names:
            differences: list[dict[str, object]] = []
            for unit in bundle.units:
                candidate = rows_by_method_unit[(pair.candidate_method, unit.unit_id)]
                reference = rows_by_method_unit[(pair.reference_method, unit.unit_id)]
                candidate_raw = _mapping(
                    candidate["raw_scores"],
                    name="candidate.raw_scores",
                )
                candidate_deployed = _mapping(
                    candidate["deployed_scores"],
                    name="candidate.deployed_scores",
                )
                reference_raw = _mapping(
                    reference["raw_scores"],
                    name="reference.raw_scores",
                )
                reference_deployed = _mapping(
                    reference["deployed_scores"],
                    name="reference.deployed_scores",
                )
                raw_difference = _number(
                    candidate_raw.get(score_name),
                    name="candidate raw score",
                ) - _number(
                    reference_raw.get(score_name),
                    name="reference raw score",
                )
                deployed_difference = _number(
                    candidate_deployed.get(score_name),
                    name="candidate deployed score",
                ) - _number(
                    reference_deployed.get(score_name),
                    name="reference deployed score",
                )
                differences.append(
                    {
                        "group_id": unit.group_id,
                        "raw_difference": raw_difference,
                        "deployed_difference": deployed_difference,
                    }
                )
            deployed_values = np.asarray(
                [
                    _number(
                        row.get("deployed_difference"), name="deployed score difference"
                    )
                    for row in differences
                ]
            )
            score_results[score_name] = {
                "unit_count": len(differences),
                "group_count": len({str(row["group_id"]) for row in differences}),
                "mean_raw_score_difference": float(
                    np.mean(
                        [
                            _number(
                                row.get("raw_difference"), name="raw score difference"
                            )
                            for row in differences
                        ]
                    )
                ),
                "mean_deployed_score_difference": float(np.mean(deployed_values)),
                "equal_group_mean_raw_score_difference": _equal_group_mean(
                    differences,
                    key="raw_difference",
                ),
                "equal_group_mean_deployed_score_difference": (
                    _equal_group_mean(
                        differences,
                        key="deployed_difference",
                    )
                ),
                "candidate_better_unit_count": int(np.sum(deployed_values < 0.0)),
                "tied_unit_count": int(np.sum(deployed_values == 0.0)),
                "candidate_worse_unit_count": int(np.sum(deployed_values > 0.0)),
            }
        pairwise_attribution[pair.comparison_id] = {
            "candidate_method": pair.candidate_method,
            "reference_method": pair.reference_method,
            "difference_semantics": "candidate-minus-reference; lower is better",
            "scores": score_results,
        }

    configuration: dict[str, object] = {
        "score_names": list(bundle.configuration.score_names),
        "energy_beta": bundle.configuration.energy_beta,
        "variogram_order": bundle.configuration.variogram_order,
        "variogram_pairs": (
            None
            if bundle.configuration.variogram_pairs is None
            else bundle.configuration.variogram_pairs.tolist()
        ),
        "variogram_pair_weights": (
            None
            if bundle.configuration.variogram_pair_weights is None
            else bundle.configuration.variogram_pair_weights.tolist()
        ),
        "gaussian_maximum_condition_number": (
            bundle.configuration.gaussian_maximum_condition_number
        ),
        "energy_pairwise_accumulation": "bounded-block-exact",
        "variogram_pair_weighting": "normalized",
        "gaussian_covariance_regularization": "none-fail-closed",
        "interval_aggregation": "component-mean",
        "coverage_aggregation": "simultaneous-and-component-wise",
    }
    identity: dict[str, object] = {
        "contract": PROBABILISTIC_SCORE_REPORT_CONTRACT,
        "schema_version": PROBABILISTIC_SCORE_REPORT_VERSION,
        "implementation": PROBABILISTIC_SCORE_IMPLEMENTATION,
        "protocol_id": bundle.protocol_id,
        "statistical_unit": bundle.statistical_unit,
        "claim_boundary": bundle.claim_boundary,
        "analysis_profile": bundle.analysis_profile,
        "infrastructure_claim_boundary": PROBABILISTIC_SCORE_CLAIM_BOUNDARY,
        "fallback_method": bundle.fallback_method,
        "reference_method": bundle.reference_method,
        "methods": list(methods),
        "comparison_pairs": [
            {
                "comparison_id": pair.comparison_id,
                "candidate_method": pair.candidate_method,
                "reference_method": pair.reference_method,
            }
            for pair in bundle.comparison_pairs
        ],
        "score_configuration": configuration,
        "unit_score_rows": rows,
        "aggregate": aggregate,
        "pairwise_attribution": pairwise_attribution,
        "claim_authorized": False,
    }
    report = {**identity, "report_id": content_id(identity)}
    return plain_json(frozen_finite_json_mapping(report, name="score report"))


def _nonnegative_score(value: float, *, offset: float, name: str) -> float:
    shifted = value + offset
    tolerance = (
        32.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            abs(value),
            abs(offset),
        )
    )
    if shifted < 0.0 and abs(shifted) <= tolerance:
        shifted = 0.0
    if not np.isfinite(shifted) or shifted < 0.0:
        raise ValueError(f"{name} could not be shifted to a nonnegative loss")
    return float(shifted)


_SCORE_REPORT_REQUIRED_FIELDS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "implementation",
        "protocol_id",
        "statistical_unit",
        "claim_boundary",
        "analysis_profile",
        "infrastructure_claim_boundary",
        "fallback_method",
        "reference_method",
        "methods",
        "comparison_pairs",
        "score_configuration",
        "unit_score_rows",
        "aggregate",
        "pairwise_attribution",
        "claim_authorized",
        "report_id",
    }
)
_SCORE_REPORT_PUBLICATION_FIELDS: Final = frozenset({"input_artifact", "status_sha256"})


def _validated_score_report_root(
    report: object,
) -> Mapping[str, object]:
    root = _mapping(report, name="report")
    _fields(
        root,
        required=_SCORE_REPORT_REQUIRED_FIELDS,
        optional=_SCORE_REPORT_PUBLICATION_FIELDS,
        name="report",
    )
    if root["contract"] != PROBABILISTIC_SCORE_REPORT_CONTRACT:
        raise ValueError("report contract is not probabilistic-score-report-v1")
    if isinstance(root["schema_version"], bool) or root["schema_version"] != 1:
        raise ValueError("report schema_version must be the integer 1")
    if root["implementation"] != PROBABILISTIC_SCORE_IMPLEMENTATION:
        raise ValueError("report implementation is not the registered scorer")
    if root["infrastructure_claim_boundary"] != PROBABILISTIC_SCORE_CLAIM_BOUNDARY:
        raise ValueError("report infrastructure claim boundary changed")
    if root["claim_authorized"] is not False:
        raise ValueError("probabilistic score reports must not authorize claims")
    return root


def _verify_score_report_identity(root: Mapping[str, object]) -> None:
    report_id = _text(root.get("report_id"), name="report.report_id")
    identity = {
        key: value
        for key, value in root.items()
        if key
        not in {
            "report_id",
            "input_artifact",
            "status_sha256",
        }
    }
    if content_id(identity) != report_id:
        raise ValueError("report_id does not match the score report content")
    if "status_sha256" in root:
        status_sha256 = _text(
            root["status_sha256"],
            name="report.status_sha256",
        )
        published = {
            key: value for key, value in root.items() if key != "status_sha256"
        }
        if content_id(published) != status_sha256:
            raise ValueError("status_sha256 does not match the published score report")


def build_decisive_evidence_from_score_report(
    report: object,
) -> dict[str, object]:
    """Convert one score report into matched decisive-evidence records.

    Gaussian log scores may be negative because a sufficiently concentrated
    density can exceed one in the registered physical units.  The decisive-
    evidence schema represents losses as nonnegative numbers.  For each
    unit/score pair this adapter therefore adds one common, recorded constant to
    every method.  The transformation preserves all method ordering, paired
    differences, exact-fallback equality, and proper-score comparisons.
    """

    root = _validated_score_report_root(report)
    score_configuration = _mapping(
        root.get("score_configuration"),
        name="score_configuration",
    )
    score_names = tuple(
        _text(value, name=f"score_names[{index}]")
        for index, value in enumerate(
            _sequence(score_configuration.get("score_names"), name="score_names")
        )
    )
    fallback_method = _text(
        root.get("fallback_method"),
        name="fallback_method",
    )
    raw_rows = _sequence(root.get("unit_score_rows"), name="unit_score_rows")
    by_unit_method: dict[tuple[str, str], Mapping[str, object]] = {}
    by_unit: dict[str, list[Mapping[str, object]]] = {}
    for index, raw_row in enumerate(raw_rows):
        row = _mapping(raw_row, name=f"unit_score_rows[{index}]")
        unit_id = _text(row.get("unit_id"), name=f"row[{index}].unit_id")
        method = _text(row.get("method"), name=f"row[{index}].method")
        key = (unit_id, method)
        if key in by_unit_method:
            raise ValueError(f"duplicate score-report row for {unit_id}/{method}")
        by_unit_method[key] = row
        by_unit.setdefault(unit_id, []).append(row)

    offsets: dict[tuple[str, str], float] = {}
    offset_records: list[dict[str, object]] = []
    for unit_id, unit_rows in sorted(by_unit.items()):
        for score_name in score_names:
            values: list[float] = []
            for row_index, row in enumerate(unit_rows):
                raw_scores = _mapping(
                    row.get("raw_scores"),
                    name=f"{unit_id}.rows[{row_index}].raw_scores",
                )
                value = _number(
                    raw_scores.get(score_name),
                    name=f"{unit_id}/{score_name}/raw_score",
                )
                values.append(value)
            offset = max(0.0, -min(values))
            offsets[(unit_id, score_name)] = offset
            offset_records.append(
                {
                    "unit_id": unit_id,
                    "score_name": score_name,
                    "common_additive_offset": offset,
                }
            )

    records: list[dict[str, object]] = []
    for index, raw_row in enumerate(raw_rows):
        row = _mapping(raw_row, name=f"unit_score_rows[{index}]")
        unit_id = _text(row.get("unit_id"), name=f"row[{index}].unit_id")
        method = _text(row.get("method"), name=f"row[{index}].method")
        fallback = by_unit_method.get((unit_id, fallback_method))
        if fallback is None:
            raise ValueError(f"unit {unit_id!r} lacks the fallback method")
        raw_scores = _mapping(row.get("raw_scores"), name=f"row[{index}].raw_scores")
        deployed_scores = _mapping(
            row.get("deployed_scores"),
            name=f"row[{index}].deployed_scores",
        )
        fallback_scores = _mapping(
            fallback.get("deployed_scores"),
            name=f"fallback[{unit_id}].deployed_scores",
        )
        for score_name in score_names:
            intervals: list[dict[str, object]] = []
            if score_name == WEIGHTED_INTERVAL_SCORE:
                for interval_index, raw_interval in enumerate(
                    _sequence(row.get("intervals"), name=f"row[{index}].intervals")
                ):
                    interval = _mapping(
                        raw_interval,
                        name=f"row[{index}].intervals[{interval_index}]",
                    )
                    intervals.append(
                        {
                            "nominal_coverage": interval["nominal_coverage"],
                            "covered": interval["simultaneous_coverage"],
                            "width": interval["mean_width"],
                        }
                    )
            accepted = _boolean(
                row.get("accepted"),
                name=f"row[{index}].accepted",
            )
            raw_loss = _number(
                raw_scores.get(score_name),
                name=f"row[{index}].raw_scores[{score_name}]",
            )
            raw_fallback = _number(
                fallback_scores.get(score_name),
                name=f"fallback[{unit_id}].deployed_scores[{score_name}]",
            )
            raw_deployed = _number(
                deployed_scores.get(score_name),
                name=f"row[{index}].deployed_scores[{score_name}]",
            )
            expected = raw_loss if accepted else raw_fallback
            if raw_deployed != expected:
                raise ValueError("score report violates exact fallback semantics")
            offset = offsets[(unit_id, score_name)]
            loss = _nonnegative_score(
                raw_loss,
                offset=offset,
                name=f"{unit_id}/{method}/{score_name}/loss",
            )
            fallback_loss = _nonnegative_score(
                raw_fallback,
                offset=offset,
                name=f"{unit_id}/{method}/{score_name}/fallback_loss",
            )
            deployed_loss = _nonnegative_score(
                raw_deployed,
                offset=offset,
                name=f"{unit_id}/{method}/{score_name}/deployed_loss",
            )
            shifted_expected = loss if accepted else fallback_loss
            if deployed_loss != shifted_expected:
                raise ValueError("score offset violated exact fallback semantics")
            records.append(
                {
                    "unit_id": unit_id,
                    "group_id": row["group_id"],
                    "metric": f"probabilistic/{score_name}",
                    "method": method,
                    "loss": loss,
                    "fallback_loss": fallback_loss,
                    "risk_score": row["risk_score"],
                    "accepted": accepted,
                    "deployed_loss": deployed_loss,
                    "horizon": row["horizon"],
                    "reliability": row["reliability"],
                    "identifiable_rank": row["identifiable_rank"],
                    "intervals": intervals,
                }
            )
    _verify_score_report_identity(root)
    return {
        "contract": "bayesian-phystwin-decisive-evidence-v1",
        "schema_version": 1,
        "protocol_id": root["protocol_id"],
        "statistical_unit": root["statistical_unit"],
        "claim_boundary": root["claim_boundary"],
        "reference_method": root["reference_method"],
        "loss_offset_semantics": (
            "one common additive constant per unit and probabilistic score; "
            "preserves method ordering, paired differences, and exact fallback"
        ),
        "loss_offsets": offset_records,
        "records": records,
    }


__all__ = [
    "BAYESIAN_VALUE_DECOMPOSITION_PROFILE",
    "BAYESIAN_VALUE_REQUIRED_COMPARISONS",
    "BAYESIAN_VALUE_REQUIRED_METHODS",
    "ENERGY_SCORE",
    "GAUSSIAN_NLL_PER_DIMENSION",
    "GENERAL_PROBABILISTIC_SCORING_PROFILE",
    "PROBABILISTIC_SCORE_CLAIM_BOUNDARY",
    "PROBABILISTIC_SCORE_IMPLEMENTATION",
    "PROBABILISTIC_SCORE_INPUT_CONTRACT",
    "PROBABILISTIC_SCORE_REPORT_CONTRACT",
    "PROBABILISTIC_SCORE_REPORT_VERSION",
    "SCORE_ORDER",
    "ScoreComparisonV1",
    "VARIOGRAM_SCORE",
    "WEIGHTED_INTERVAL_SCORE",
    "ProbabilisticScoreBundleV1",
    "ProbabilisticScoreConfigurationV1",
    "PredictiveArmV1",
    "PredictiveIntervalV1",
    "PredictiveUnitV1",
    "build_decisive_evidence_from_score_report",
    "energy_score",
    "gaussian_nll_per_dimension",
    "interval_score",
    "parse_probabilistic_score_bundle",
    "score_probabilistic_bundle",
    "validate_bayesian_value_decomposition_bundle",
    "variogram_score",
    "weighted_interval_score",
]
