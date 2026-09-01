"""Query-quotient mechanics for the Tracking Cloth public-data study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.query_quotient_belief_v1 import (
    aggregate_to_query_quotient,
    minimum_information_query_lift,
    query_ambiguity_envelope,
    query_quotient_information_decomposition,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

SAME_QUOTIENT_LIFTS = (
    "full_source_posterior",
    "jeffrey_i_projection",
    "uniform_within_class",
    "prior_map_within_class",
    "prior_antimap_within_class",
)


def _probability(value: object, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return array / total


def registered_prior(protocol: Mapping[str, Any]) -> FloatArray:
    """Return the frozen product prior in the base parameter-bank order."""

    stiffness = _probability(
        protocol["prior"]["stiffness_weights"],
        name="stiffness prior",
    )
    damping = _probability(
        protocol["prior"]["damping_weights"],
        name="damping prior",
    )
    prior = np.outer(stiffness, damping).reshape(-1)
    if prior.size != 9:
        raise ValueError("the registered prior must cover the 3x3 parameter bank")
    return prior


def prior_aware_source_posterior(
    losses_m2: object,
    prior_weights: object,
    *,
    measurement_floor_m: float,
) -> tuple[FloatArray, float]:
    """Update a registered prior from one normalized loss per source recording."""

    losses = np.asarray(losses_m2, dtype=np.float64)
    prior = _probability(prior_weights, name="prior_weights")
    if losses.ndim != 2 or losses.shape[1] != prior.size or losses.shape[0] == 0:
        raise ValueError("losses must have shape (recordings, hypotheses)")
    if not np.all(np.isfinite(losses)) or np.any(losses < 0.0):
        raise ValueError("losses must be finite and nonnegative")
    if not np.isfinite(measurement_floor_m) or measurement_floor_m <= 0.0:
        raise ValueError("measurement_floor_m must be positive")
    temperature = max(
        float(np.median(np.min(losses, axis=1))),
        float(measurement_floor_m) ** 2,
    )
    logits = np.log(prior) - np.sum(losses, axis=0) / (2.0 * temperature)
    logits -= float(np.max(logits))
    posterior = np.exp(logits)
    posterior /= float(posterior.sum())
    return posterior, temperature


def centered_shape_rms_m(
    trajectory: object,
    *,
    cutoff: int,
    corners: object,
    tail_fraction: float,
) -> float:
    """RMS nonrigid displacement after removing framewise centroid motion."""

    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("trajectory must have shape (frames, markers, 3)")
    if cutoff < 0 or cutoff >= values.shape[0] - 1:
        raise ValueError("cutoff must leave at least one forecast frame")
    corner_index = np.asarray(corners, dtype=np.int64)
    if corner_index.ndim != 1 or corner_index.size == 0:
        raise ValueError("corners must be a nonempty vector")
    if np.any(corner_index < 0) or np.any(corner_index >= values.shape[1]):
        raise ValueError("corner index is out of range")
    if not np.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")

    free = np.ones(values.shape[1], dtype=bool)
    free[corner_index] = False
    if int(free.sum()) < 2:
        raise ValueError("at least two free markers are required")
    base = values[cutoff, free]
    forecast_count = values.shape[0] - cutoff - 1
    tail_count = max(1, int(np.ceil(tail_fraction * forecast_count)))
    start = values.shape[0] - tail_count

    squared_norms: list[FloatArray] = []
    for frame in values[start:]:
        current = frame[free]
        valid = np.all(np.isfinite(current), axis=1) & np.all(
            np.isfinite(base), axis=1
        )
        if int(valid.sum()) < 2:
            continue
        current_centered = current[valid] - np.mean(current[valid], axis=0)
        base_centered = base[valid] - np.mean(base[valid], axis=0)
        squared_norms.append(
            np.sum(np.square(current_centered - base_centered), axis=1)
        )
    if not squared_norms:
        raise ValueError("no finite free-marker samples for the query")
    return float(np.sqrt(np.mean(np.concatenate(squared_norms))))


def query_partition(
    hypothesis_values_m: object,
    *,
    requested_class_count: int,
    minimum_gap_m: float,
) -> tuple[IntArray, FloatArray]:
    """Partition sorted query values at registered rank boundaries.

    Boundaries that fall inside a numerical tie are omitted. The result therefore
    contains between one and ``requested_class_count`` nonempty contiguous classes.
    """

    values = np.asarray(hypothesis_values_m, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("hypothesis query values must be a finite vector")
    if requested_class_count < 1 or requested_class_count > values.size:
        raise ValueError("requested_class_count is invalid")
    if not np.isfinite(minimum_gap_m) or minimum_gap_m < 0.0:
        raise ValueError("minimum_gap_m must be nonnegative")

    ordered = np.sort(values, kind="stable")
    thresholds: list[float] = []
    for group in range(1, requested_class_count):
        index = (group * values.size) // requested_class_count
        if index <= 0 or index >= values.size:
            continue
        left = float(ordered[index - 1])
        right = float(ordered[index])
        if right - left <= minimum_gap_m:
            continue
        threshold = 0.5 * (left + right)
        if not thresholds or threshold > thresholds[-1]:
            thresholds.append(threshold)
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    classes = np.searchsorted(threshold_array, values, side="right").astype(
        np.int64
    )
    if not np.array_equal(
        np.unique(classes),
        np.arange(int(classes.max()) + 1, dtype=np.int64),
    ):
        raise RuntimeError("query partition is not canonical")
    return classes, threshold_array


def observed_query_class(value_m: float, thresholds_m: object) -> int:
    if not np.isfinite(value_m):
        raise ValueError("observed query value must be finite")
    thresholds = np.asarray(thresholds_m, dtype=np.float64)
    if thresholds.ndim != 1 or not np.all(np.isfinite(thresholds)):
        raise ValueError("thresholds must be a finite vector")
    if thresholds.size and np.any(np.diff(thresholds) <= 0.0):
        raise ValueError("thresholds must be strictly increasing")
    return int(np.searchsorted(thresholds, value_m, side="right"))


def _concentrated_lift(
    quotient: FloatArray,
    prior: FloatArray,
    classes: IntArray,
    *,
    choose_maximum: bool,
) -> FloatArray:
    result = np.zeros_like(prior)
    for class_id, mass in enumerate(quotient):
        members = np.flatnonzero(classes == class_id)
        values = prior[members]
        local = int(np.argmax(values) if choose_maximum else np.argmin(values))
        result[members[local]] = mass
    return result


def same_quotient_lifts(
    prior_weights: object,
    full_posterior_weights: object,
    class_index: object,
) -> dict[str, FloatArray]:
    """Construct complete beliefs sharing the full posterior's quotient masses."""

    prior = _probability(prior_weights, name="prior_weights")
    posterior = _probability(
        full_posterior_weights,
        name="full_posterior_weights",
    )
    classes = np.asarray(class_index, dtype=np.int64)
    if classes.ndim != 1 or classes.size != prior.size:
        raise ValueError("class_index must align with the hypothesis bank")
    quotient = aggregate_to_query_quotient(posterior, classes)
    jeffrey = minimum_information_query_lift(
        prior,
        classes,
        quotient,
    ).lifted_weights

    uniform = np.zeros_like(prior)
    for class_id, mass in enumerate(quotient):
        members = np.flatnonzero(classes == class_id)
        uniform[members] = mass / members.size

    lifts = {
        "full_source_posterior": posterior,
        "jeffrey_i_projection": np.asarray(jeffrey),
        "uniform_within_class": uniform,
        "prior_map_within_class": _concentrated_lift(
            quotient,
            prior,
            classes,
            choose_maximum=True,
        ),
        "prior_antimap_within_class": _concentrated_lift(
            quotient,
            prior,
            classes,
            choose_maximum=False,
        ),
    }
    for name, weights in lifts.items():
        actual = aggregate_to_query_quotient(weights, classes)
        if not np.allclose(actual, quotient, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{name} does not preserve the query quotient")
    return lifts


def jeffrey_control_lift(
    prior_weights: object,
    evidence_weights: object,
    class_index: object,
) -> FloatArray:
    prior = _probability(prior_weights, name="prior_weights")
    evidence = _probability(evidence_weights, name="evidence_weights")
    classes = np.asarray(class_index, dtype=np.int64)
    quotient = aggregate_to_query_quotient(evidence, classes)
    return np.asarray(
        minimum_information_query_lift(prior, classes, quotient).lifted_weights
    )


def unsupported_specificity_nats(
    prior_weights: object,
    posterior_weights: object,
    class_index: object,
) -> float:
    return float(
        query_quotient_information_decomposition(
            prior_weights,
            posterior_weights,
            class_index,
        ).unsupported_specificity_nats
    )


def ambiguity_envelopes(
    quotient_weights: object,
    class_index: object,
    query_values_m: object,
    stiffness_values: object,
    damping_values: object,
) -> dict[str, dict[str, float]]:
    values = {
        "query_m": query_values_m,
        "stiffness_per_mass": stiffness_values,
        "damping_per_mass": damping_values,
    }
    result: dict[str, dict[str, float]] = {}
    for name, endpoint in values.items():
        envelope = query_ambiguity_envelope(
            quotient_weights,
            class_index,
            endpoint,
        )
        result[name] = {
            "lower": float(envelope.lower[0]),
            "upper": float(envelope.upper[0]),
            "width": float(envelope.width[0]),
        }
    return result


def categorical_scores(
    probabilities: object,
    observed_class: int,
    *,
    probability_floor: float,
) -> dict[str, float | int]:
    probs = _probability(probabilities, name="query probabilities")
    if observed_class < 0 or observed_class >= probs.size:
        raise ValueError("observed_class is out of range")
    if not np.isfinite(probability_floor) or not 0.0 < probability_floor < 1.0:
        raise ValueError("probability_floor must lie in (0, 1)")
    one_hot = np.zeros_like(probs)
    one_hot[observed_class] = 1.0
    return {
        "query_log_score_nats": float(
            -np.log(max(float(probs[observed_class]), probability_floor))
        ),
        "query_brier_score": float(np.sum(np.square(probs - one_hot))),
        "query_class_correct": int(int(np.argmax(probs)) == observed_class),
    }


def trajectory_mask(
    truth: object,
    *,
    cutoff: int,
    corners: object,
    time_stride: int,
) -> BoolArray:
    values = np.asarray(truth, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("truth must have shape (frames, markers, 3)")
    if time_stride < 1:
        raise ValueError("time_stride must be positive")
    mask = np.all(np.isfinite(values), axis=2)
    mask[: cutoff + 1] = False
    mask[:, np.asarray(corners, dtype=np.int64)] = False
    frame_keep = np.zeros(values.shape[0], dtype=bool)
    frame_keep[cutoff + 1 :: time_stride] = True
    mask &= frame_keep[:, None]
    if not np.any(mask):
        raise ValueError("no valid trajectory samples remain")
    return mask


def mixture_mean(bank: object, weights: object) -> FloatArray:
    trajectories = np.asarray(bank, dtype=np.float64)
    probabilities = _probability(weights, name="weights")
    if trajectories.ndim != 4 or trajectories.shape[0] != probabilities.size:
        raise ValueError("bank must have shape (hypotheses, frames, markers, 3)")
    if not np.all(np.isfinite(trajectories)):
        raise ValueError("bank trajectories must be finite")
    return np.einsum("k,ktnd->tnd", probabilities, trajectories)


def trajectory_rmse_mm(
    bank: object,
    weights: object,
    truth: object,
    mask: object,
) -> float:
    mean = mixture_mean(bank, weights)
    target = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    error = mean[valid] - target[valid]
    return 1000.0 * float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def trajectory_energy_score_mm(
    bank: object,
    weights: object,
    truth: object,
    mask: object,
) -> float:
    """Energy score for a finite trajectory belief, normalized per coordinate."""

    trajectories = np.asarray(bank, dtype=np.float64)
    probabilities = _probability(weights, name="weights")
    target = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if trajectories.ndim != 4 or trajectories.shape[0] != probabilities.size:
        raise ValueError("bank must align with weights")
    if target.shape != trajectories.shape[1:] or valid.shape != target.shape[:2]:
        raise ValueError("target or mask shape is incompatible with bank")
    if not np.any(valid):
        raise ValueError("energy score mask is empty")

    samples = trajectories[:, valid].reshape(probabilities.size, -1)
    observation = target[valid].reshape(-1)
    dimension_scale = np.sqrt(observation.size)
    to_truth = np.linalg.norm(samples - observation[None], axis=1) / dimension_scale
    pairwise = np.linalg.norm(
        samples[:, None] - samples[None, :],
        axis=2,
    ) / dimension_scale
    score_m = float(
        probabilities @ to_truth
        - 0.5 * probabilities @ pairwise @ probabilities
    )
    return 1000.0 * max(0.0, score_m)


def parameter_expectations(
    weights: object,
    stiffness_values: object,
    damping_values: object,
) -> dict[str, float]:
    probabilities = _probability(weights, name="weights")
    stiffness = np.asarray(stiffness_values, dtype=np.float64)
    damping = np.asarray(damping_values, dtype=np.float64)
    if stiffness.shape != probabilities.shape or damping.shape != probabilities.shape:
        raise ValueError("parameter values must align with weights")
    return {
        "expected_stiffness_per_mass": float(probabilities @ stiffness),
        "expected_damping_per_mass": float(probabilities @ damping),
    }
