"""Leakage-audited fold fitting and active-probe replay for cloth dynamics.

This module is the scientific core of the registered shake-to-twist pilot.  It
keeps three data roles distinct:

* other-material shaking outcomes fit a finite physical-model belief;
* other-material shaking and twisting *input predictions* define prospective
  probe and downstream-task disagreement templates; and
* held-material shaking outcomes are requested only after a policy selects the
  corresponding recorded probe.

No held-material twisting input or outcome enters fold fitting or probe
selection.  Target prediction and post-seal scoring live in the separate CLI.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .active_probe import (
    normalize_weights,
    pairwise_trajectory_mse,
    simulate_policy,
    update_weights,
    weights_from_records,
)
from .active_probe_run import (
    active_mask,
    calibrated_residuals,
    loss_vector,
    posterior_temperature,
    validate_protocol,
)
from .data import object_digest
from .model import Predictions

FOLD_SCHEMA = "tracking-cloth-active-probe-fold-v1"
SPECIMEN_SCHEMA = "tracking-cloth-active-probe-specimen-v1"


@dataclass(frozen=True)
class SourceOutcome:
    """One source prediction paired with its observed free-marker trajectory."""

    recording: str
    material: str
    size: str
    condition: str
    prediction: Predictions
    truth: np.ndarray


@dataclass(frozen=True)
class InputTemplate:
    """One prediction generated from causal inputs, structurally without truth."""

    recording: str
    material: str
    size: str
    condition: str
    prediction: Predictions


def _content_id(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return object_digest(payload)


def _verify_content_id(value: Mapping[str, Any], field: str) -> None:
    identifier = value.get(field)
    if not isinstance(identifier, str) or identifier != _content_id(value, field):
        raise ValueError(f"{field} does not bind the complete payload")


def _labels(protocol: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    validate_protocol(protocol)
    materials = tuple(str(value) for value in protocol["materials"])
    sizes = tuple(str(value) for value in protocol["sizes"])
    conditions = tuple(str(value) for value in protocol["probe_conditions"])
    if len(materials) != 4 or len(set(materials)) != 4:
        raise ValueError("the registered study requires four unique materials")
    if len(sizes) != 2 or len(set(sizes)) != 2:
        raise ValueError("the registered study requires two unique cloth sizes")
    return materials, sizes, conditions


def _prediction_models(prediction: Predictions, *, name: str) -> int:
    bank = np.asarray(prediction.bank, dtype=np.float64)
    nominal = np.asarray(prediction.nominal, dtype=np.float64)
    if bank.ndim != 4 or bank.shape[0] < 2 or bank.shape[-1] != 3:
        raise ValueError(f"{name}.bank must have shape (models>=2, time, markers, 3)")
    if nominal.shape != bank.shape[1:] or not np.all(np.isfinite(nominal)):
        raise ValueError(f"{name}.nominal must be finite and match the model bank")
    if not np.all(np.isfinite(bank)):
        raise ValueError(f"{name}.bank must be finite")
    return int(bank.shape[0])


def _expected_roster(
    materials: Sequence[str], sizes: Sequence[str], conditions: Sequence[str]
) -> set[tuple[str, str, str]]:
    return {
        (material, size, condition)
        for material in materials
        for size in sizes
        for condition in conditions
    }


def _validate_roster(
    records: Sequence[SourceOutcome] | Sequence[InputTemplate],
    *,
    materials: Sequence[str],
    sizes: Sequence[str],
    conditions: Sequence[str],
    name: str,
) -> int:
    expected = _expected_roster(materials, sizes, conditions)
    actual = {(r.material, r.size, r.condition) for r in records}
    recordings = [str(r.recording) for r in records]
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            f"{name} must contain the complete factorial roster exactly once"
        )
    if (
        any(not value for value in recordings)
        or len(set(recordings)) != len(recordings)
    ):
        raise ValueError(f"{name} recording IDs must be nonempty and unique")
    model_counts = {
        _prediction_models(record.prediction, name=f"{name}[{record.recording}]")
        for record in records
    }
    if len(model_counts) != 1:
        raise ValueError(f"{name} predictions disagree about the model count")
    return model_counts.pop()


def _distance(prediction: Predictions) -> np.ndarray:
    return pairwise_trajectory_mse(prediction.bank, active_mask(prediction.inputs))


def _mean_distance(predictions: Sequence[Predictions], *, models: int) -> np.ndarray:
    if not predictions:
        raise ValueError("at least one prediction is required for a template")
    matrices = np.stack([_distance(prediction) for prediction in predictions])
    if matrices.shape[1:] != (models, models):
        raise ValueError("template predictions disagree about the finite-model roster")
    result = np.mean(matrices, axis=0)
    result = np.maximum(0.5 * (result + result.T), 0.0)
    np.fill_diagonal(result, 0.0)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite disagreement template")
    result.setflags(write=False)
    return result


def fit_leave_one_material_out(
    *,
    held_material: str,
    source_outcomes: Sequence[SourceOutcome],
    target_templates: Sequence[InputTemplate],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one complete source-only fold and freeze prospective templates.

    ``target_templates`` contain predictions only.  The type has no target
    trajectory field, making accidental target-outcome fitting impossible at
    this boundary.
    """

    materials, sizes, conditions = _labels(protocol)
    held = str(held_material)
    if held not in materials:
        raise ValueError("held_material is not registered")
    training = tuple(material for material in materials if material != held)
    models = _validate_roster(
        source_outcomes,
        materials=training,
        sizes=sizes,
        conditions=conditions,
        name="source_outcomes",
    )
    target_models = _validate_roster(
        target_templates,
        materials=training,
        sizes=sizes,
        conditions=conditions,
        name="target_templates",
    )
    if target_models != models:
        raise ValueError("source and target templates disagree about model count")
    if any(record.material == held for record in source_outcomes):
        raise ValueError("held-material source outcome entered fold fitting")
    if any(record.material == held for record in target_templates):
        raise ValueError("held-material target input entered probe selection")

    ordered_sources = sorted(source_outcomes, key=lambda record: record.recording)
    losses_by_record: dict[str, list[float]] = {}
    source_pairs: list[tuple[Predictions, np.ndarray]] = []
    losses = []
    for record in ordered_sources:
        truth = np.asarray(record.truth, dtype=np.float64)
        if truth.shape != record.prediction.nominal.shape:
            raise ValueError("source truth must match its prediction trajectory")
        vector = loss_vector(record.prediction, truth)
        if vector.shape != (models,):
            raise ValueError("source loss vector has the wrong model count")
        losses.append(vector)
        losses_by_record[record.recording] = vector.tolist()
        source_pairs.append((record.prediction, truth))
    loss_matrix = np.stack(losses)
    temperature = posterior_temperature(
        loss_matrix, float(protocol["measurement_floor_m"])
    )
    prior = weights_from_records(loss_matrix, temperature)
    residuals = calibrated_residuals(source_pairs, prior, protocol)

    ordered_targets = sorted(target_templates, key=lambda record: record.recording)
    probe_distances = {
        condition: _mean_distance(
            [
                record.prediction
                for record in ordered_sources
                if record.condition == condition
            ],
            models=models,
        ).tolist()
        for condition in conditions
    }
    target_distance = _mean_distance(
        [record.prediction for record in ordered_targets], models=models
    )
    result: dict[str, Any] = {
        "schema": FOLD_SCHEMA,
        "held_material": held,
        "training_materials": list(training),
        "model_count": models,
        "source_record_count": len(source_outcomes),
        "target_template_count": len(target_templates),
        "temperature_m2": temperature,
        "prior_weights": prior.tolist(),
        "source_residual_variance_m2": residuals,
        "probe_distance_m2": probe_distances,
        "target_distance_m2": target_distance.tolist(),
        "source_recordings": [record.recording for record in ordered_sources],
        "target_template_recordings": [
            record.recording for record in ordered_targets
        ],
        "source_loss_vectors_m2": losses_by_record,
        "source_outcomes_used": True,
        "target_outcomes_used": False,
        "held_material_source_outcomes_used": False,
        "held_material_candidate_inputs_used_for_selection": False,
        "held_material_twist_inputs_used_for_selection": False,
        "selection_templates": protocol["selection_templates"],
    }
    result["fold_id"] = _content_id(result, "fold_id")
    return result


class _AuditedLossView(Mapping[str, np.ndarray]):
    """Mapping that records exactly which already-recorded outcomes are requested."""

    def __init__(
        self,
        source: Mapping[str, object],
        actions: Sequence[str],
        models: int,
    ) -> None:
        self._source = source
        self._actions = tuple(actions)
        self._models = models
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._actions:
            raise KeyError(key)
        self.accessed.append(key)
        value = np.asarray(self._source[key], dtype=np.float64).copy()
        if value.shape != (self._models,) or not np.all(np.isfinite(value)):
            raise ValueError(
                f"probe loss for {key} has the wrong shape or is nonfinite"
            )
        if np.any(value < 0.0):
            raise ValueError(f"probe loss for {key} must be nonnegative")
        value.setflags(write=False)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._actions)

    def __len__(self) -> int:
        return len(self._actions)


def _json_step(step: Mapping[str, Any]) -> dict[str, Any]:
    utilities = {
        str(action): None if value is None else float(value)
        for action, value in sorted(step["utilities"].items())
    }
    return {
        "step": int(step["step"]),
        "selected_action": str(step["selected_action"]),
        "utilities": utilities,
        "entropy_before": float(step["entropy_before"]),
        "entropy_after": float(step["entropy_after"]),
        "target_model_spread_before": float(step["target_model_spread_before"]),
        "target_model_spread_after": float(step["target_model_spread_after"]),
    }


def _json_state(state: Any) -> dict[str, Any]:
    return {
        "budget": int(state.budget),
        "selected_actions": list(state.selected_actions),
        "weights": np.asarray(state.weights, dtype=np.float64).tolist(),
        "steps": [_json_step(step) for step in state.steps],
    }


def replay_held_specimen(
    *,
    specimen: str,
    fold: Mapping[str, Any],
    observed_losses: Mapping[str, object],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay all registered policies while auditing selected-outcome access."""

    materials, sizes, conditions = _labels(protocol)
    if fold.get("schema") != FOLD_SCHEMA:
        raise ValueError("unexpected fold schema")
    _verify_content_id(fold, "fold_id")
    held = str(fold["held_material"])
    if held not in materials:
        raise ValueError("fold held material is not registered")
    expected_specimens = {f"{held}_{size}" for size in sizes}
    if specimen not in expected_specimens:
        raise ValueError("specimen does not belong to the held-material fold")
    if set(observed_losses) != set(conditions):
        raise ValueError("observed_losses must expose exactly four registered probes")

    models = int(fold["model_count"])
    prior = normalize_weights(fold["prior_weights"])
    if prior.shape != (models,):
        raise ValueError("fold prior does not match model_count")
    temperature = float(fold["temperature_m2"])
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("fold temperature must be finite and positive")
    probe_distances = {
        condition: np.asarray(fold["probe_distance_m2"][condition], dtype=np.float64)
        for condition in conditions
    }
    target_distance = np.asarray(fold["target_distance_m2"], dtype=np.float64)
    budgets = tuple(int(value) for value in protocol["probe_budgets"])
    policies = tuple(str(value) for value in protocol["probe_policies"])

    policy_states: dict[str, dict[str, Any]] = {}
    access_order: dict[str, list[str]] = {}
    for policy in policies:
        view = _AuditedLossView(observed_losses, conditions, models)
        states = simulate_policy(
            policy=policy,
            initial_weights=prior,
            probe_distances=probe_distances,
            target_distance=target_distance,
            observed_losses=view,
            temperature=temperature,
            fixed_order=tuple(protocol["fixed_probe_order"]),
            budgets=budgets,
        )
        final_actions = list(states[budgets[-1]].selected_actions)
        if view.accessed != final_actions:
            raise ValueError("policy consumed an outcome before or outside selection")
        for budget in budgets:
            if view.accessed[:budget] != list(states[budget].selected_actions):
                raise ValueError(
                    "budget state does not match the outcome-access prefix"
                )
        policy_states[policy] = {
            str(budget): _json_state(states[budget]) for budget in budgets
        }
        access_order[policy] = list(view.accessed)

    single_view = _AuditedLossView(observed_losses, conditions, models)
    losses = {condition: single_view[condition] for condition in conditions}
    single_weights = {
        condition: update_weights(prior, losses[condition], temperature).tolist()
        for condition in conditions
    }
    canonical_all = update_weights(
        prior,
        np.sum(np.stack([losses[condition] for condition in conditions]), axis=0),
        temperature,
    ).tolist()
    for policy in policies:
        zero = policy_states[policy]["0"]
        zero["weights"] = prior.tolist()
        full = policy_states[policy][str(budgets[-1])]
        if set(full["selected_actions"]) != set(conditions):
            raise ValueError(
                "full-budget policy did not consume every registered probe"
            )
        full["weights"] = canonical_all
        full["canonical_all_probe_endpoint"] = True

    zero_weights = [policy_states[policy]["0"]["weights"] for policy in policies]
    full_weights = [
        policy_states[policy][str(budgets[-1])]["weights"] for policy in policies
    ]
    if not all(value == zero_weights[0] for value in zero_weights[1:]):
        raise ValueError("zero-budget policy endpoint parity failed")
    if not all(value == full_weights[0] for value in full_weights[1:]):
        raise ValueError("full-budget policy endpoint parity failed")

    result: dict[str, Any] = {
        "schema": SPECIMEN_SCHEMA,
        "specimen": specimen,
        "held_material": held,
        "fold_id": fold["fold_id"],
        "prior_weights": prior.tolist(),
        "single_probe_weights": single_weights,
        "policy_states": policy_states,
        "policy_outcome_access_order": access_order,
        "single_probe_outcome_access_order": list(single_view.accessed),
        "canonical_all_probe_weights": canonical_all,
        "selection_consumed_only_selected_outcomes": True,
        "held_material_candidate_inputs_used_for_selection": False,
        "held_material_twist_inputs_used_for_selection": False,
        "held_material_twist_outcomes_used": False,
    }
    result["specimen_replay_id"] = _content_id(result, "specimen_replay_id")
    return result


__all__ = [
    "FOLD_SCHEMA",
    "SPECIMEN_SCHEMA",
    "InputTemplate",
    "SourceOutcome",
    "fit_leave_one_material_out",
    "replay_held_specimen",
]
