"""Policy-level split-conformal gain certificates with exact fallback.

The candidate policy and gain predictor must be fixed before calibration.  For
calibration scores ``s_i = predicted_gain_i - realized_gain_i``, the standard
split-conformal order statistic gives a marginal lower bound on a new policy
gain.  If the candidate is admitted only when that lower bound exceeds the
negative harm margin, every harmful admitted update is a coverage failure.

This module implements the arithmetic and a deterministic local predictor.  It
does not claim conditional coverage, physical safety, or validity after policy
selection on calibration outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


def _finite_matrix(value: object, *, name: str) -> FloatArray:
    try:
        result = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real matrix") from error
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real matrix")
    return result


def _finite_vector(value: object, *, name: str) -> FloatArray:
    try:
        result = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real vector") from error
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real vector")
    return result


def _integer_vector(value: object, *, name: str) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector")
    return np.array(raw, dtype=np.int64, copy=True, order="C")


def _immutable_float(value: FloatArray) -> FloatArray:
    canonical = np.asarray(value, dtype=np.dtype("<f8"), order="C")
    return np.frombuffer(canonical.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
        canonical.shape
    )


def _immutable_int(value: IntArray) -> IntArray:
    canonical = np.asarray(value, dtype=np.dtype("<i8"), order="C")
    return np.frombuffer(canonical.tobytes(order="C"), dtype=np.dtype("<i8")).reshape(
        canonical.shape
    )


def _immutable_bool(value: BoolArray) -> BoolArray:
    canonical = np.asarray(value, dtype=np.bool_, order="C")
    return np.frombuffer(canonical.tobytes(order="C"), dtype=np.bool_).reshape(
        canonical.shape
    )


def _canonical_ids(value: Sequence[str], *, expected_count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("reference_ids must be a sequence of canonical strings")
    result = tuple(value)
    if len(result) != expected_count:
        raise ValueError("reference_ids length must match reference rows")
    if any(type(item) is not str or not item or item.strip() != item for item in result):
        raise ValueError("reference_ids must contain canonical nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("reference_ids must not contain duplicates")
    return result


def _open_probability(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


@dataclass(frozen=True, slots=True)
class LocalPolicyGainPredictor:
    """Frozen nearest-neighbor predictor fitted without calibration outcomes."""

    reference_ids: tuple[str, ...]
    feature_mean: FloatArray
    feature_scale: FloatArray
    standardized_reference_features: FloatArray
    reference_action_gains: FloatArray
    neighbor_count: int


@dataclass(frozen=True, slots=True)
class LocalPolicyGainPrediction:
    """Predicted gain and auditable local support for candidate policy actions."""

    predicted_gain: FloatArray
    neighbor_indices: IntArray
    neighbor_squared_distances: FloatArray


@dataclass(frozen=True, slots=True)
class PolicyGainCalibration:
    """One-sided split-conformal calibration for one fixed candidate policy."""

    miscoverage: float
    calibration_count: int
    rank: int
    offset: float


@dataclass(frozen=True, slots=True)
class PolicyGainGuardDecision:
    """Candidate actions admitted by the calibrated lower gain bound."""

    selected_actions: IntArray
    accepted_mask: BoolArray
    lower_gain_bound: FloatArray


def fit_local_policy_gain_predictor(
    *,
    reference_ids: Sequence[str],
    reference_features: object,
    reference_action_gains: object,
    neighbor_count: int,
) -> LocalPolicyGainPredictor:
    """Fit a deterministic local predictor on source-only reference outcomes."""

    features = _finite_matrix(reference_features, name="reference_features")
    gains = _finite_matrix(reference_action_gains, name="reference_action_gains")
    ids = _canonical_ids(reference_ids, expected_count=len(features))
    if gains.shape[0] != len(features) or gains.shape[1] < 2:
        raise ValueError("reference gains must align with rows and contain actions")
    if type(neighbor_count) is not int or not 1 <= neighbor_count <= len(features):
        raise ValueError("neighbor_count must be supported by the reference rows")

    order = np.argsort(np.asarray(ids), kind="stable")
    sorted_ids = tuple(ids[int(index)] for index in order)
    sorted_features = features[order]
    sorted_gains = gains[order]
    mean = sorted_features.mean(axis=0)
    scale = sorted_features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (sorted_features - mean) / scale
    return LocalPolicyGainPredictor(
        reference_ids=sorted_ids,
        feature_mean=_immutable_float(mean),
        feature_scale=_immutable_float(scale),
        standardized_reference_features=_immutable_float(standardized),
        reference_action_gains=_immutable_float(sorted_gains),
        neighbor_count=neighbor_count,
    )


def predict_local_policy_gain(
    predictor: LocalPolicyGainPredictor,
    *,
    query_features: object,
    candidate_actions: object,
) -> LocalPolicyGainPrediction:
    """Predict candidate-policy gains from the mean of local source outcomes."""

    features = _finite_matrix(query_features, name="query_features")
    actions = _integer_vector(candidate_actions, name="candidate_actions")
    if features.shape[1:] != predictor.feature_mean.shape or len(actions) != len(features):
        raise ValueError("query features and actions must align with the predictor")
    action_count = predictor.reference_action_gains.shape[1]
    if np.any((actions < 0) | (actions >= action_count)):
        raise ValueError("candidate action is outside the reference action bank")

    standardized = (features - predictor.feature_mean) / predictor.feature_scale
    distance = np.mean(
        (standardized[:, None] - predictor.standardized_reference_features[None]) ** 2,
        axis=2,
    )
    neighbors = np.argsort(distance, axis=1, kind="stable")[
        :, : predictor.neighbor_count
    ]
    row = np.arange(len(features))[:, None]
    local_gain = predictor.reference_action_gains[neighbors, actions[:, None]]
    return LocalPolicyGainPrediction(
        predicted_gain=_immutable_float(local_gain.mean(axis=1)),
        neighbor_indices=_immutable_int(neighbors),
        neighbor_squared_distances=_immutable_float(distance[row, neighbors]),
    )


def calibrate_policy_gain_lower_bound(
    *,
    predicted_gain: object,
    realized_gain: object,
    miscoverage: float,
) -> PolicyGainCalibration:
    """Calibrate a one-sided marginal lower bound for a fixed policy predictor."""

    predicted = _finite_vector(predicted_gain, name="predicted_gain")
    realized = _finite_vector(realized_gain, name="realized_gain")
    alpha = _open_probability(miscoverage, name="miscoverage")
    if len(predicted) == 0 or predicted.shape != realized.shape:
        raise ValueError("complete aligned calibration gains are required")
    rank = int(math.ceil((len(predicted) + 1) * (1.0 - alpha)))
    if rank > len(predicted):
        raise ValueError("calibration count cannot support the requested miscoverage")
    offset = float(np.sort(predicted - realized, kind="stable")[rank - 1])
    return PolicyGainCalibration(
        miscoverage=alpha,
        calibration_count=len(predicted),
        rank=rank,
        offset=offset,
    )


def apply_policy_gain_guard(
    *,
    candidate_actions: object,
    predicted_gain: object,
    calibration: PolicyGainCalibration,
    fallback_action: int,
    harm_margin: float,
) -> PolicyGainGuardDecision:
    """Admit the fixed candidate policy only when its lower bound is safe."""

    actions = _integer_vector(candidate_actions, name="candidate_actions")
    predicted = _finite_vector(predicted_gain, name="predicted_gain")
    if actions.shape != predicted.shape:
        raise ValueError("candidate actions and predicted gains must align")
    if type(fallback_action) is not int:
        raise ValueError("fallback_action must be an integer")
    if isinstance(harm_margin, (bool, np.bool_)) or not isinstance(harm_margin, Real):
        raise ValueError("harm_margin must be a finite nonnegative real")
    margin = float(harm_margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("harm_margin must be a finite nonnegative real")
    lower = predicted - calibration.offset
    accepted = lower >= -margin
    selected = np.where(accepted, actions, fallback_action).astype(np.int64)
    return PolicyGainGuardDecision(
        selected_actions=_immutable_int(selected),
        accepted_mask=_immutable_bool(accepted),
        lower_gain_bound=_immutable_float(lower),
    )
