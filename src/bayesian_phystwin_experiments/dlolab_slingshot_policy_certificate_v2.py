"""Source-development utilities for a posterior-aware Slingshot certificate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin.policy_gain_certificate import (
    apply_policy_gain_guard,
    calibrate_policy_gain_lower_bound,
    fit_local_policy_gain_predictor,
    predict_distance_weighted_local_policy_gain,
    predict_local_policy_gain,
)

from .dlolab_slingshot_belief import BASELINE, REWARD_MARGIN
from .dlolab_slingshot_policy_certificate_v1 import bias_invariant_features

Array: TypeAlias = NDArray[Any]
SOURCE_COUNT = 147
FOLD_COUNT = 7
ROTATION_COUNT = 30
ROTATION_SEED = 261945
NEIGHBOR_COUNT = 7
MISCOVERAGE = 0.10

_VECTOR_FIELDS = (
    "weights",
    "iid_weights",
    "expected_losses",
    "iid_expected_losses",
    "map_losses",
    "nominal_losses",
    "prior_losses",
)
_LOSS_FIELDS = _VECTOR_FIELDS[2:]


def posterior_diagnostic_features(inference: Mapping[str, object]) -> NDArray[np.float64]:
    """Flatten residual-independent posterior shape and policy diagnostics.

    Loss-like vectors are expressed relative to the registered incumbent entry.
    This keeps the feature insensitive to a common reward offset while retaining
    posterior spread, action ranking, and regret information.
    """

    if not isinstance(inference, Mapping):
        raise TypeError("inference must be a mapping")
    values: dict[str, NDArray[np.float64]] = {}
    for name in _VECTOR_FIELDS:
        value = np.asarray(inference.get(name), dtype=np.float64)
        expected_shape = (27,) if name in _VECTOR_FIELDS[:2] else (7,)
        if value.shape != expected_shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must have finite shape {expected_shape}")
        values[name] = value
    raw_upper = np.asarray(inference.get("raw_upper"), dtype=np.float64)
    if raw_upper.shape != (3, 7) or not np.all(np.isfinite(raw_upper)):
        raise ValueError("raw_upper must have finite shape (3, 7)")
    if (
        np.any(values["weights"] < 0.0)
        or np.any(values["iid_weights"] < 0.0)
        or not np.isclose(values["weights"].sum(), 1.0, rtol=0.0, atol=1e-10)
        or not np.isclose(values["iid_weights"].sum(), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("posterior weights must be nonnegative and normalized")
    centered = [values[name] - values[name][0] for name in _LOSS_FIELDS]
    result = np.concatenate(
        (values["weights"], values["iid_weights"], *centered, raw_upper.reshape(-1))
    )
    if result.shape != (110,):
        raise AssertionError("posterior diagnostic feature dimension changed")
    return result


def combined_competence_features(
    observation: object, inference: Mapping[str, object]
) -> NDArray[np.float64]:
    """Combine causal bias-invariant geometry with posterior diagnostics."""

    geometry = bias_invariant_features(observation)
    if geometry.shape != (51,):
        raise ValueError("one registered Slingshot prefix observation is required")
    result = np.concatenate((geometry, posterior_diagnostic_features(inference)))
    if result.shape != (161,):
        raise AssertionError("combined competence feature dimension changed")
    return result


def _source_arrays(
    *,
    case_ids: Sequence[str],
    features: object,
    candidate_actions: object,
    action_gains: object,
) -> tuple[tuple[str, ...], Array, NDArray[np.int64], Array, Array]:
    ids = tuple(case_ids)
    feature = np.asarray(features, dtype=np.float64)
    actions = np.asarray(candidate_actions)
    gains = np.asarray(action_gains, dtype=np.float64)
    if (
        len(ids) != SOURCE_COUNT
        or len(set(ids)) != SOURCE_COUNT
        or any(type(value) is not str or not value for value in ids)
        or feature.ndim != 2
        or feature.shape[0] != SOURCE_COUNT
        or actions.shape != (SOURCE_COUNT,)
        or actions.dtype.kind not in "iu"
        or gains.ndim != 2
        or gains.shape[0] != SOURCE_COUNT
        or gains.shape[1] < 2
        or not np.all(np.isfinite(feature))
        or not np.all(np.isfinite(gains))
    ):
        raise ValueError("complete finite 147-world source arrays are required")
    actions = np.asarray(actions, dtype=np.int64)
    if np.any((actions < 0) | (actions >= gains.shape[1])):
        raise ValueError("candidate action lies outside source action gains")
    realized = gains[np.arange(SOURCE_COUNT), actions]
    return ids, feature, actions, gains, realized


def _folds(seed: int) -> NDArray[np.int64]:
    permutation = np.random.default_rng(seed).permutation(SOURCE_COUNT)
    folds = np.empty(SOURCE_COUNT, dtype=np.int64)
    for fold, indices in enumerate(np.array_split(permutation, FOLD_COUNT)):
        folds[indices] = fold
    if not np.all(np.bincount(folds) == SOURCE_COUNT // FOLD_COUNT):
        raise AssertionError("registered equal source folds changed")
    return folds


def _predict(
    *,
    reference_ids: tuple[str, ...],
    reference_features: Array,
    reference_gains: Array,
    query_features: Array,
    query_actions: NDArray[np.int64],
    neighbor_count: int,
    distance_weighted: bool,
) -> NDArray[np.float64]:
    predictor = fit_local_policy_gain_predictor(
        reference_ids=reference_ids,
        reference_features=reference_features,
        reference_action_gains=reference_gains,
        neighbor_count=neighbor_count,
    )
    function = (
        predict_distance_weighted_local_policy_gain
        if distance_weighted
        else predict_local_policy_gain
    )
    return function(
        predictor,
        query_features=query_features,
        candidate_actions=query_actions,
    ).predicted_gain


def repeated_rotation_diagnostic(
    *,
    case_ids: Sequence[str],
    features: object,
    candidate_actions: object,
    action_gains: object,
    neighbor_count: int,
    distance_weighted: bool,
) -> dict[str, Any]:
    """Run fixed train/calibrate/evaluate rotations on opened source worlds."""

    ids, feature, actions, gains, realized = _source_arrays(
        case_ids=case_ids,
        features=features,
        candidate_actions=candidate_actions,
        action_gains=action_gains,
    )
    if type(neighbor_count) is not int or not 1 <= neighbor_count <= 105:
        raise ValueError("neighbor_count must fit every registered training split")

    rows: list[dict[str, Any]] = []
    for repetition in range(ROTATION_COUNT):
        fold = _folds(ROTATION_SEED + repetition)
        ordered_realized: list[Array] = []
        ordered_lower: list[Array] = []
        ordered_accepted: list[Array] = []
        ordered_prediction: list[Array] = []
        for evaluation_fold in range(FOLD_COUNT):
            calibration_fold = (evaluation_fold + 1) % FOLD_COUNT
            training = (fold != evaluation_fold) & (fold != calibration_fold)
            calibration = fold == calibration_fold
            evaluation = fold == evaluation_fold
            training_indices = np.flatnonzero(training)
            calibration_prediction = _predict(
                reference_ids=tuple(ids[index] for index in training_indices),
                reference_features=feature[training],
                reference_gains=gains[training],
                query_features=feature[calibration],
                query_actions=actions[calibration],
                neighbor_count=neighbor_count,
                distance_weighted=distance_weighted,
            )
            evaluation_prediction = _predict(
                reference_ids=tuple(ids[index] for index in training_indices),
                reference_features=feature[training],
                reference_gains=gains[training],
                query_features=feature[evaluation],
                query_actions=actions[evaluation],
                neighbor_count=neighbor_count,
                distance_weighted=distance_weighted,
            )
            fitted = calibrate_policy_gain_lower_bound(
                predicted_gain=calibration_prediction,
                realized_gain=realized[calibration],
                miscoverage=MISCOVERAGE,
            )
            decision = apply_policy_gain_guard(
                candidate_actions=actions[evaluation],
                predicted_gain=evaluation_prediction,
                calibration=fitted,
                fallback_action=BASELINE,
                harm_margin=REWARD_MARGIN,
            )
            ordered_realized.append(realized[evaluation])
            ordered_lower.append(decision.lower_gain_bound)
            ordered_accepted.append(decision.accepted_mask)
            ordered_prediction.append(evaluation_prediction)

        y = np.concatenate(ordered_realized)
        lower = np.concatenate(ordered_lower)
        accepted = np.concatenate(ordered_accepted)
        prediction = np.concatenate(ordered_prediction)
        guarded_gain = np.where(accepted, y, 0.0)
        rows.append(
            {
                "repetition": repetition,
                "mean_guarded_gain": float(guarded_gain.mean()),
                "accepted_count": int(np.count_nonzero(accepted)),
                "harmful_accepted_count": int(
                    np.count_nonzero(accepted & (y < -REWARD_MARGIN))
                ),
                "marginal_lower_bound_coverage": float(np.mean(y >= lower)),
                "prediction_correlation": float(np.corrcoef(prediction, y)[0, 1]),
            }
        )

    def quantiles(name: str) -> list[float]:
        value = np.asarray([row[name] for row in rows], dtype=np.float64)
        return [
            float(item)
            for item in np.quantile(value, [0.0, 0.25, 0.5, 0.75, 1.0])
        ]

    return {
        "neighbor_count": neighbor_count,
        "distance_weighted": bool(distance_weighted),
        "source_count": SOURCE_COUNT,
        "fold_count": FOLD_COUNT,
        "training_count_per_fold": 105,
        "calibration_count_per_fold": 21,
        "evaluation_count_per_fold": 21,
        "rotation_count": ROTATION_COUNT,
        "rotation_seed_first": ROTATION_SEED,
        "rotation_seed_last": ROTATION_SEED + ROTATION_COUNT - 1,
        "miscoverage": MISCOVERAGE,
        "harm_margin": REWARD_MARGIN,
        "mean_guarded_gain_quantiles": quantiles("mean_guarded_gain"),
        "accepted_count_quantiles": quantiles("accepted_count"),
        "harmful_accepted_count_quantiles": quantiles("harmful_accepted_count"),
        "marginal_lower_bound_coverage_quantiles": quantiles(
            "marginal_lower_bound_coverage"
        ),
        "prediction_correlation_quantiles": quantiles("prediction_correlation"),
        "repetitions": rows,
        "prospective_coverage_claim": False,
    }


def descriptive_prefix_capacity(
    *,
    case_ids: Sequence[str],
    features: object,
    candidate_actions: object,
    action_gains: object,
    prefix_features: object,
    prefix_actions: object,
    neighbor_count: int = NEIGHBOR_COUNT,
) -> dict[str, Any]:
    """Estimate admission capacity without reading any prefix-panel future."""

    ids, feature, actions, gains, realized = _source_arrays(
        case_ids=case_ids,
        features=features,
        candidate_actions=candidate_actions,
        action_gains=action_gains,
    )
    query = np.asarray(prefix_features, dtype=np.float64)
    query_actions = np.asarray(prefix_actions)
    if (
        query.ndim != 2
        or query.shape[1] != feature.shape[1]
        or query_actions.shape != (len(query),)
        or query_actions.dtype.kind not in "iu"
        or not np.all(np.isfinite(query))
    ):
        raise ValueError("complete finite prefix-only capacity inputs required")
    query_actions = np.asarray(query_actions, dtype=np.int64)

    fold = _folds(ROTATION_SEED)
    oof_prediction = np.empty(SOURCE_COUNT, dtype=np.float64)
    for held_out in range(FOLD_COUNT):
        training = fold != held_out
        training_indices = np.flatnonzero(training)
        oof_prediction[~training] = _predict(
            reference_ids=tuple(ids[index] for index in training_indices),
            reference_features=feature[training],
            reference_gains=gains[training],
            query_features=feature[~training],
            query_actions=actions[~training],
            neighbor_count=neighbor_count,
            distance_weighted=True,
        )
    rank = int(math.ceil((SOURCE_COUNT + 1) * (1.0 - MISCOVERAGE)))
    offset = float(np.sort(oof_prediction - realized, kind="stable")[rank - 1])
    prefix_prediction = _predict(
        reference_ids=ids,
        reference_features=feature,
        reference_gains=gains,
        query_features=query,
        query_actions=query_actions,
        neighbor_count=neighbor_count,
        distance_weighted=True,
    )
    lower = prefix_prediction - offset
    accepted = lower >= -REWARD_MARGIN
    return {
        "status": "descriptive_oof_capacity_only",
        "prospective_coverage_claim": False,
        "source_count": SOURCE_COUNT,
        "prefix_count": len(query),
        "neighbor_count": neighbor_count,
        "oof_fold_count": FOLD_COUNT,
        "oof_seed": ROTATION_SEED,
        "oof_rank": rank,
        "oof_offset": offset,
        "accepted_prefix_count": int(np.count_nonzero(accepted)),
        "fallback_prefix_count": int(len(accepted) - np.count_nonzero(accepted)),
        "prefix_futures_read": False,
    }
