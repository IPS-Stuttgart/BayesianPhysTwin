"""Whole-trajectory proper scoring and frozen action decision value.

These target-closed utilities evaluate predictive trajectory distributions and
the downstream value of a predeclared finite action loss. They do not choose a
model, fit a guard, execute an action, or authorize target access.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id

TRAJECTORY_PROPER_SCORE_SCHEMA: Final = (
    "bayesian_phystwin.trajectory_proper_score"
)
TRAJECTORY_PROPER_SCORE_VERSION: Final = 1
TRAJECTORY_PROPER_SCORE_SEMANTICS: Final = (
    "scaled-energy-and-registered-variogram-score-v1"
)
TRAJECTORY_DECISION_VALUE_SCHEMA: Final = (
    "bayesian_phystwin.frozen_action_decision_value"
)
TRAJECTORY_DECISION_VALUE_VERSION: Final = 1
TRAJECTORY_VALUE_CLAIM_BOUNDARY: Final = (
    "The records establish only the reported whole-trajectory scores and "
    "finite-action decision regret under the exact predictive samples, target, "
    "coordinate scale, registered variogram pairs, loss definition, and target "
    "access boundary. They do not establish model correctness, calibrated "
    "deployment uncertainty, safe action execution, causal identification, "
    "unseen-object transfer, provider competence, universal decision benefit, "
    "Causal4D physical evidence, deployment safety, or state of the art."
)


def _digest(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={64}))


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _immutable(value: object) -> np.ndarray:
    return cast(np.ndarray, immutable_array(value, dtype=np.float64))


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _labels(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if not result or any(type(value) is not str or not value for value in result):
        raise ValueError(f"{name} must contain nonempty literal strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must be sorted")
    return result


def _mean_pairwise_distance(samples: np.ndarray, *, block_size: int = 256) -> float:
    count = samples.shape[0]
    total = 0.0
    for start_i in range(0, count, block_size):
        block_i = samples[start_i : start_i + block_size]
        for start_j in range(0, count, block_size):
            block_j = samples[start_j : start_j + block_size]
            differences = block_i[:, None, :] - block_j[None, :, :]
            total += float(np.linalg.norm(differences, axis=2).sum())
    return total / float(count * count)


@dataclass(frozen=True, slots=True, order=True)
class VariogramPairV1:
    """One registered flattened-coordinate pair and nonnegative weight."""

    first_index: int
    second_index: int
    weight: float

    def __post_init__(self) -> None:
        if isinstance(self.first_index, (bool, np.bool_)) or not isinstance(
            self.first_index,
            (int, np.integer),
        ):
            raise ValueError("first_index must be an integer")
        if isinstance(self.second_index, (bool, np.bool_)) or not isinstance(
            self.second_index,
            (int, np.integer),
        ):
            raise ValueError("second_index must be an integer")
        first = int(self.first_index)
        second = int(self.second_index)
        if first < 0 or second <= first:
            raise ValueError(
                "variogram indices must satisfy 0 <= first_index < second_index"
            )
        object.__setattr__(self, "first_index", first)
        object.__setattr__(self, "second_index", second)
        object.__setattr__(
            self,
            "weight",
            _finite(self.weight, name="weight", minimum=0.0),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "first_index": self.first_index,
            "second_index": self.second_index,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryProperScoreConfigV1:
    """Frozen score definition for one vectorized trajectory geometry."""

    score_definition_id: str
    coordinate_scale_id: str
    coordinate_scale: np.ndarray
    energy_weight: float = 1.0
    variogram_weight: float = 1.0
    variogram_power: float = 0.5
    variogram_pairs: Sequence[VariogramPairV1] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "score_definition_id",
            _digest(self.score_definition_id, name="score_definition_id"),
        )
        object.__setattr__(
            self,
            "coordinate_scale_id",
            _digest(self.coordinate_scale_id, name="coordinate_scale_id"),
        )
        scale = _vector(self.coordinate_scale, name="coordinate_scale")
        if len(scale) == 0 or np.any(scale <= 0.0):
            raise ValueError("coordinate_scale must be nonempty and strictly positive")
        energy_weight = _finite(
            self.energy_weight,
            name="energy_weight",
            minimum=0.0,
        )
        variogram_weight = _finite(
            self.variogram_weight,
            name="variogram_weight",
            minimum=0.0,
        )
        if energy_weight == 0.0 and variogram_weight == 0.0:
            raise ValueError("at least one score component must have positive weight")
        power = _finite(
            self.variogram_power,
            name="variogram_power",
            minimum=float(np.nextafter(0.0, 1.0)),
            maximum=float(np.nextafter(2.0, 0.0)),
        )
        if isinstance(self.variogram_pairs, (str, bytes)):
            raise TypeError("variogram_pairs must be a sequence")
        pairs = tuple(self.variogram_pairs)
        if any(not isinstance(pair, VariogramPairV1) for pair in pairs):
            raise TypeError("variogram_pairs must contain VariogramPairV1 values")
        if pairs != tuple(sorted(pairs)):
            raise ValueError("variogram_pairs must be sorted")
        pair_keys = [(pair.first_index, pair.second_index) for pair in pairs]
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("variogram_pairs must not repeat coordinate pairs")
        if any(pair.second_index >= len(scale) for pair in pairs):
            raise ValueError("variogram pair index exceeds coordinate_scale length")
        if variogram_weight > 0.0 and not pairs:
            raise ValueError(
                "positive variogram_weight requires registered variogram_pairs"
            )
        object.__setattr__(self, "coordinate_scale", _immutable(scale))
        object.__setattr__(self, "energy_weight", energy_weight)
        object.__setattr__(self, "variogram_weight", variogram_weight)
        object.__setattr__(self, "variogram_power", power)
        object.__setattr__(self, "variogram_pairs", pairs)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="trajectory score configuration metadata",
            ),
        )
        expected = cast(str, content_id(self.descriptor()))
        supplied = self.artifact_id
        if supplied is not None:
            supplied = _digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError(
                    "trajectory score configuration artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected)

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinate_scale)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": TRAJECTORY_PROPER_SCORE_SCHEMA,
            "schema_version": TRAJECTORY_PROPER_SCORE_VERSION,
            "artifact_kind": "TrajectoryProperScoreConfigV1",
            "semantics": TRAJECTORY_PROPER_SCORE_SEMANTICS,
            "score_definition_id": self.score_definition_id,
            "coordinate_scale_id": self.coordinate_scale_id,
            "coordinate_scale": _array_record(self.coordinate_scale),
            "energy_weight": self.energy_weight,
            "variogram_weight": self.variogram_weight,
            "variogram_power": self.variogram_power,
            "variogram_pairs": [pair.to_record() for pair in self.variogram_pairs],
            "metadata": plain_json(self.metadata),
            "claim_boundary": TRAJECTORY_VALUE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class TrajectoryProperScoreV1:
    """Energy and variogram score for one sealed trajectory distribution."""

    config: TrajectoryProperScoreConfigV1
    prediction_artifact_id: str
    target_artifact_id: str
    object_session_id: str
    action_id: str
    arm_id: str
    predictive_samples: np.ndarray
    target_trajectory: np.ndarray
    prediction_sealed_before_target: bool
    target_outcomes_used_for_prediction: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    energy_score: float = field(init=False)
    variogram_score: float = field(init=False)
    total_score: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, TrajectoryProperScoreConfigV1):
            raise TypeError("config must be a TrajectoryProperScoreConfigV1")
        for name in ("prediction_artifact_id", "target_artifact_id"):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name=name),
            )
        for name in ("object_session_id", "action_id", "arm_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty literal string")
        raw_samples = np.asarray(self.predictive_samples)
        if raw_samples.dtype.kind not in "iuf":
            raise ValueError("predictive_samples must contain real numeric values")
        samples = np.ascontiguousarray(raw_samples, dtype=np.float64)
        target = _matrix(self.target_trajectory, name="target_trajectory")
        if samples.ndim != 3 or samples.shape[0] < 2:
            raise ValueError(
                "predictive_samples must have shape (sample, time, coordinate) "
                "with at least two samples"
            )
        if samples.shape[1:] != target.shape:
            raise ValueError(
                "predictive_samples and target_trajectory shapes are incompatible"
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError("predictive_samples must be finite")
        flat_samples = samples.reshape(samples.shape[0], -1)
        flat_target = target.reshape(-1)
        if flat_samples.shape[1] != self.config.coordinate_count:
            raise ValueError(
                "trajectory coordinate count does not match score configuration"
            )
        scaled_samples = flat_samples / self.config.coordinate_scale
        scaled_target = flat_target / self.config.coordinate_scale

        target_distances = np.linalg.norm(
            scaled_samples - scaled_target[None, :],
            axis=1,
        )
        energy = float(target_distances.mean()) - 0.5 * _mean_pairwise_distance(
            scaled_samples
        )
        energy = max(0.0, energy)

        variogram = 0.0
        for pair in self.config.variogram_pairs:
            target_increment = abs(
                scaled_target[pair.first_index]
                - scaled_target[pair.second_index]
            ) ** self.config.variogram_power
            sample_increment = np.abs(
                scaled_samples[:, pair.first_index]
                - scaled_samples[:, pair.second_index]
            ) ** self.config.variogram_power
            variogram += pair.weight * (
                target_increment - float(sample_increment.mean())
            ) ** 2
        total = (
            self.config.energy_weight * energy
            + self.config.variogram_weight * variogram
        )
        if not np.isfinite(total):
            raise ValueError("trajectory score is nonfinite")

        sealed = genuine_boolean(
            self.prediction_sealed_before_target,
            name="prediction_sealed_before_target",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used_for_prediction,
            name="target_outcomes_used_for_prediction",
        )
        if not sealed or target_used:
            raise ValueError(
                "prediction must be sealed before target and target-outcome free"
            )

        object.__setattr__(self, "predictive_samples", _immutable(samples))
        object.__setattr__(self, "target_trajectory", _immutable(target))
        object.__setattr__(self, "prediction_sealed_before_target", sealed)
        object.__setattr__(
            self,
            "target_outcomes_used_for_prediction",
            target_used,
        )
        object.__setattr__(self, "energy_score", energy)
        object.__setattr__(self, "variogram_score", variogram)
        object.__setattr__(self, "total_score", total)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="trajectory score metadata",
            ),
        )
        expected = cast(str, content_id(self.descriptor()))
        supplied = self.artifact_id
        if supplied is not None:
            supplied = _digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError("trajectory score artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": TRAJECTORY_PROPER_SCORE_SCHEMA,
            "schema_version": TRAJECTORY_PROPER_SCORE_VERSION,
            "artifact_kind": "TrajectoryProperScoreV1",
            "semantics": TRAJECTORY_PROPER_SCORE_SEMANTICS,
            "config_id": self.config.artifact_id,
            "prediction_artifact_id": self.prediction_artifact_id,
            "target_artifact_id": self.target_artifact_id,
            "object_session_id": self.object_session_id,
            "action_id": self.action_id,
            "arm_id": self.arm_id,
            "predictive_samples": _array_record(self.predictive_samples),
            "target_trajectory": _array_record(self.target_trajectory),
            "prediction_sealed_before_target": self.prediction_sealed_before_target,
            "target_outcomes_used_for_prediction": (
                self.target_outcomes_used_for_prediction
            ),
            "energy_score": self.energy_score,
            "variogram_score": self.variogram_score,
            "total_score": self.total_score,
            "metadata": plain_json(self.metadata),
            "claim_boundary": TRAJECTORY_VALUE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    def summary(self) -> dict[str, object]:
        return {
            "schema": TRAJECTORY_PROPER_SCORE_SCHEMA,
            "schema_version": TRAJECTORY_PROPER_SCORE_VERSION,
            "artifact_id": self.artifact_id,
            "object_session_id": self.object_session_id,
            "action_id": self.action_id,
            "arm_id": self.arm_id,
            "sample_count": int(self.predictive_samples.shape[0]),
            "energy_score": self.energy_score,
            "variogram_score": self.variogram_score,
            "total_score": self.total_score,
            "claim_boundary": TRAJECTORY_VALUE_CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class FrozenActionDecisionValueV1:
    """Select an action from predictive loss samples and report realized regret."""

    decision_protocol_id: str
    loss_definition_id: str
    prediction_batch_id: str
    target_access_attestation_id: str
    object_session_id: str
    method_id: str
    action_ids: Sequence[str]
    predictive_loss_samples: np.ndarray
    realized_losses: np.ndarray
    predictions_sealed_before_target: bool
    target_outcomes_used_for_prediction: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    predictive_mean_losses: np.ndarray = field(init=False, repr=False)
    selected_action_id: str = field(init=False)
    oracle_action_id: str = field(init=False)
    selected_realized_loss: float = field(init=False)
    oracle_realized_loss: float = field(init=False)
    realized_regret: float = field(init=False)
    predictive_selection_margin: float = field(init=False)
    oracle_match: bool = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "decision_protocol_id",
            "loss_definition_id",
            "prediction_batch_id",
            "target_access_attestation_id",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name=name),
            )
        for name in ("object_session_id", "method_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty literal string")
        actions = _labels(self.action_ids, name="action_ids")
        if len(actions) < 2:
            raise ValueError("decision value requires at least two actions")
        predictive = _matrix(
            self.predictive_loss_samples,
            name="predictive_loss_samples",
        )
        realized = _vector(self.realized_losses, name="realized_losses")
        if predictive.shape[0] != len(actions) or predictive.shape[1] == 0:
            raise ValueError(
                "predictive_loss_samples must have one nonempty row per action"
            )
        if realized.shape != (len(actions),):
            raise ValueError("realized_losses must have one value per action")
        means = predictive.mean(axis=1)
        selected_index = int(np.argmin(means))
        oracle_index = int(np.argmin(realized))
        selected_loss = float(realized[selected_index])
        oracle_loss = float(realized[oracle_index])
        regret = selected_loss - oracle_loss
        tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0,
            abs(selected_loss),
            abs(oracle_loss),
        )
        if regret < -tolerance:
            raise AssertionError("realized regret cannot be negative")
        regret = max(0.0, regret)
        ordered_means = np.sort(means)
        selection_margin = float(ordered_means[1] - ordered_means[0])

        sealed = genuine_boolean(
            self.predictions_sealed_before_target,
            name="predictions_sealed_before_target",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used_for_prediction,
            name="target_outcomes_used_for_prediction",
        )
        if not sealed or target_used:
            raise ValueError(
                "decision predictions must be sealed and target-outcome free"
            )
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(self, "predictive_loss_samples", _immutable(predictive))
        object.__setattr__(self, "realized_losses", _immutable(realized))
        object.__setattr__(self, "predictive_mean_losses", _immutable(means))
        object.__setattr__(self, "selected_action_id", actions[selected_index])
        object.__setattr__(self, "oracle_action_id", actions[oracle_index])
        object.__setattr__(self, "selected_realized_loss", selected_loss)
        object.__setattr__(self, "oracle_realized_loss", oracle_loss)
        object.__setattr__(self, "realized_regret", regret)
        object.__setattr__(
            self,
            "predictive_selection_margin",
            selection_margin,
        )
        object.__setattr__(self, "oracle_match", selected_index == oracle_index)
        object.__setattr__(
            self,
            "predictions_sealed_before_target",
            sealed,
        )
        object.__setattr__(
            self,
            "target_outcomes_used_for_prediction",
            target_used,
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="frozen action decision metadata",
            ),
        )
        expected = cast(str, content_id(self.descriptor()))
        supplied = self.artifact_id
        if supplied is not None:
            supplied = _digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError(
                    "frozen action decision artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected)

    def arrays(self) -> Mapping[str, np.ndarray]:
        return {
            "predictive_loss_samples": self.predictive_loss_samples,
            "realized_losses": self.realized_losses,
            "predictive_mean_losses": self.predictive_mean_losses,
        }

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": TRAJECTORY_DECISION_VALUE_SCHEMA,
            "schema_version": TRAJECTORY_DECISION_VALUE_VERSION,
            "artifact_kind": "FrozenActionDecisionValueV1",
            "decision_protocol_id": self.decision_protocol_id,
            "loss_definition_id": self.loss_definition_id,
            "prediction_batch_id": self.prediction_batch_id,
            "target_access_attestation_id": self.target_access_attestation_id,
            "object_session_id": self.object_session_id,
            "method_id": self.method_id,
            "action_ids": list(self.action_ids),
            "predictive_loss_samples": _array_record(
                self.predictive_loss_samples
            ),
            "realized_losses": _array_record(self.realized_losses),
            "predictive_mean_losses": _array_record(
                self.predictive_mean_losses
            ),
            "selected_action_id": self.selected_action_id,
            "oracle_action_id": self.oracle_action_id,
            "selected_realized_loss": self.selected_realized_loss,
            "oracle_realized_loss": self.oracle_realized_loss,
            "realized_regret": self.realized_regret,
            "predictive_selection_margin": self.predictive_selection_margin,
            "oracle_match": self.oracle_match,
            "predictions_sealed_before_target": (
                self.predictions_sealed_before_target
            ),
            "target_outcomes_used_for_prediction": (
                self.target_outcomes_used_for_prediction
            ),
            "metadata": plain_json(self.metadata),
            "claim_boundary": TRAJECTORY_VALUE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    def summary(self) -> dict[str, object]:
        return {
            "schema": TRAJECTORY_DECISION_VALUE_SCHEMA,
            "schema_version": TRAJECTORY_DECISION_VALUE_VERSION,
            "artifact_id": self.artifact_id,
            "object_session_id": self.object_session_id,
            "method_id": self.method_id,
            "selected_action_id": self.selected_action_id,
            "oracle_action_id": self.oracle_action_id,
            "realized_regret": self.realized_regret,
            "predictive_selection_margin": self.predictive_selection_margin,
            "oracle_match": self.oracle_match,
            "claim_boundary": TRAJECTORY_VALUE_CLAIM_BOUNDARY,
        }
