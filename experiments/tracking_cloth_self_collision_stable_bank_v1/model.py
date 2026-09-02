"""Stable-subset wrapper for the registered Tracking Cloth contact-model bank.

The parent experiment originally treated every registered spring parameter as
mandatory. One numerically unstable member therefore aborted the complete source
fit before any target access. This module changes only that execution rule:

* only the two explicit numerical-domain failures already raised by the parent
  simulator may be pruned;
* parsing, geometry, input, and all other errors remain fatal;
* the registered nominal hypothesis must remain valid;
* at least half of the preregistered bank must survive; and
* every rejected parameter and reason is serialized in the source fit.

No target outcome is consulted when constructing the stable subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.tracking_cloth_self_collision_selective_twin_v1 import (
    model as parent_model,
)
from experiments.tracking_cloth_self_collision_selective_twin_v1.data import InputView

KINEMATIC_ARMS = parent_model.KINEMATIC_ARMS
PHYSICS_ARM = parent_model.PHYSICS_ARM
AUXILIARY_ARMS = parent_model.AUXILIARY_ARMS
ALL_ARMS = parent_model.ALL_ARMS

_PRUNABLE_MESSAGES = (
    "nonfinite contact rollout",
    "contact rollout escaped the registered domain",
)


class UnstablePhysicsHypothesisError(ValueError):
    """One registered hypothesis left the parent's numerical domain."""


@dataclass(frozen=True)
class PhysicsFit:
    """Posterior over the source-stable subset of a registered candidate bank."""

    parameters: tuple[tuple[float, float, float], ...]
    weights: np.ndarray
    losses_m2: np.ndarray
    temperature_m2: float
    rejected_parameters: tuple[tuple[float, float, float], ...]
    rejection_reasons: tuple[str, ...]
    candidate_count: int

    @property
    def valid_fraction(self) -> float:
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        return len(self.parameters) / self.candidate_count

    def record(self) -> dict[str, Any]:
        self._validate()
        return {
            "parameters": [list(item) for item in self.parameters],
            "weights": self.weights.tolist(),
            "losses_m2": self.losses_m2.tolist(),
            "temperature_m2": self.temperature_m2,
            "rejected_parameters": [list(item) for item in self.rejected_parameters],
            "rejection_reasons": list(self.rejection_reasons),
            "candidate_count": self.candidate_count,
            "valid_fraction": self.valid_fraction,
            "stable_bank_semantics": "explicit-numerical-domain-pruning-v1",
        }

    def _validate(self) -> None:
        if not self.parameters:
            raise ValueError("physics fit contains no valid parameters")
        if self.candidate_count != len(self.parameters) + len(self.rejected_parameters):
            raise ValueError("physics fit candidate accounting changed")
        if len(self.rejected_parameters) != len(self.rejection_reasons):
            raise ValueError("physics fit rejection accounting changed")
        if set(self.parameters).intersection(self.rejected_parameters):
            raise ValueError("valid and rejected parameters overlap")
        if self.weights.shape != (len(self.parameters),):
            raise ValueError("physics weight dimension changed")
        if self.losses_m2.shape != self.weights.shape:
            raise ValueError("physics loss dimension changed")
        if (
            not np.isfinite(self.weights).all()
            or np.any(self.weights < 0.0)
            or not np.isclose(np.sum(self.weights), 1.0)
        ):
            raise ValueError("invalid physics weights")
        if not np.isfinite(self.losses_m2).all() or np.any(self.losses_m2 < 0.0):
            raise ValueError("invalid physics losses")
        if not np.isfinite(self.temperature_m2) or self.temperature_m2 <= 0.0:
            raise ValueError("invalid physics temperature")
        if any(reason not in _PRUNABLE_MESSAGES for reason in self.rejection_reasons):
            raise ValueError("unregistered rejection reason")

    @classmethod
    def from_record(cls, value: dict[str, Any]) -> PhysicsFit:
        parameters = tuple(
            tuple(float(component) for component in item)
            for item in value["parameters"]
        )
        rejected = tuple(
            tuple(float(component) for component in item)
            for item in value["rejected_parameters"]
        )
        result = cls(
            parameters=parameters,
            weights=np.asarray(value["weights"], dtype=float),
            losses_m2=np.asarray(value["losses_m2"], dtype=float),
            temperature_m2=float(value["temperature_m2"]),
            rejected_parameters=rejected,
            rejection_reasons=tuple(str(item) for item in value["rejection_reasons"]),
            candidate_count=int(value["candidate_count"]),
        )
        result._validate()
        if value.get("stable_bank_semantics") != (
            "explicit-numerical-domain-pruning-v1"
        ):
            raise ValueError("stable-bank semantics changed")
        if not np.isclose(float(value["valid_fraction"]), result.valid_fraction):
            raise ValueError("serialized valid fraction changed")
        return result


def _stable_bank_contract(protocol: dict[str, Any]) -> tuple[float, bool]:
    contract = protocol.get("stable_physics_bank")
    if not isinstance(contract, dict):
        raise ValueError("stable_physics_bank contract is missing")
    minimum_fraction = float(contract.get("minimum_valid_fraction", -1.0))
    nominal_required = contract.get("nominal_must_survive")
    prunable = tuple(contract.get("prunable_error_messages", ()))
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must lie in (0, 1]")
    if nominal_required is not True:
        raise ValueError("nominal hypothesis must remain required")
    if prunable != _PRUNABLE_MESSAGES:
        raise ValueError("prunable numerical-domain errors changed")
    return minimum_fraction, nominal_required


def parameter_bank(protocol: dict[str, Any]) -> tuple[tuple[float, float, float], ...]:
    return parent_model.parameter_bank(protocol)


def contact_rollout(
    inputs: InputView,
    parameters: tuple[float, float, float],
    protocol: dict[str, Any],
) -> np.ndarray:
    """Run the parent simulator and type only its explicit instability failures."""

    try:
        return parent_model.contact_rollout(inputs, parameters, protocol)
    except ValueError as error:
        if str(error) in _PRUNABLE_MESSAGES:
            raise UnstablePhysicsHypothesisError(str(error)) from error
        raise


def fit_physics(
    inputs: InputView,
    truth: np.ndarray,
    protocol: dict[str, Any],
) -> PhysicsFit:
    """Fit the posterior after source-only, explicitly bounded bank pruning."""

    minimum_fraction, _ = _stable_bank_contract(protocol)
    candidates = parameter_bank(protocol)
    if not candidates:
        raise ValueError("registered physics bank is empty")

    valid_parameters: list[tuple[float, float, float]] = []
    predictions: list[np.ndarray] = []
    rejected_parameters: list[tuple[float, float, float]] = []
    rejection_reasons: list[str] = []
    for parameters in candidates:
        try:
            prediction = contact_rollout(inputs, parameters, protocol)
        except UnstablePhysicsHypothesisError as error:
            rejected_parameters.append(parameters)
            rejection_reasons.append(str(error))
            continue
        valid_parameters.append(parameters)
        predictions.append(prediction)

    nominal = tuple(float(value) for value in protocol["nominal_parameters"])
    if nominal not in valid_parameters:
        raise ValueError("nominal contact hypothesis is unstable on source data")
    valid_fraction = len(valid_parameters) / len(candidates)
    if valid_fraction + 1e-12 < minimum_fraction:
        raise ValueError(
            "stable source physics bank is below the registered minimum: "
            f"{len(valid_parameters)}/{len(candidates)}"
        )

    losses = np.asarray(
        [
            parent_model.trajectory_mse(prediction, truth, inputs)
            for prediction in predictions
        ],
        dtype=float,
    )
    temperature = max(
        float(np.min(losses)),
        float(protocol["measurement_floor_m"]) ** 2,
    )
    logits = -losses / (2.0 * temperature)
    weights = np.exp(logits - np.max(logits))
    weights /= np.sum(weights)
    fit = PhysicsFit(
        parameters=tuple(valid_parameters),
        weights=weights,
        losses_m2=losses,
        temperature_m2=temperature,
        rejected_parameters=tuple(rejected_parameters),
        rejection_reasons=tuple(rejection_reasons),
        candidate_count=len(candidates),
    )
    fit._validate()
    return fit


def all_predictions(
    inputs: InputView,
    fit: PhysicsFit,
    protocol: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Evaluate the source-sealed valid subset without target-side pruning."""

    _stable_bank_contract(protocol)
    fit._validate()
    predictions = parent_model.kinematic_predictions(inputs, protocol)
    bank = np.stack(
        [contact_rollout(inputs, item, protocol) for item in fit.parameters],
        axis=0,
    )
    nominal = tuple(float(value) for value in protocol["nominal_parameters"])
    try:
        nominal_index = fit.parameters.index(nominal)
    except ValueError as error:
        raise ValueError("nominal contact parameters are absent from the stable bank") from error
    predictions["nominal_contact_physics"] = bank[nominal_index]
    predictions["map_contact_physics"] = bank[int(np.argmax(fit.weights))]
    predictions[PHYSICS_ARM] = np.einsum("k,ktnd->tnd", fit.weights, bank)
    return predictions


__all__ = [
    "ALL_ARMS",
    "AUXILIARY_ARMS",
    "KINEMATIC_ARMS",
    "PHYSICS_ARM",
    "PhysicsFit",
    "UnstablePhysicsHypothesisError",
    "all_predictions",
    "contact_rollout",
    "fit_physics",
    "parameter_bank",
]
