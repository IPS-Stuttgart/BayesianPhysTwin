"""Checkpoint-belief utilities for exploratory DEFORM model averaging."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

DEFORM_CHECKPOINT_BELIEF_SCHEMA_VERSION = 1
DEFORM_CHECKPOINT_BELIEF_CONTRACT = "deform-dlo-checkpoint-belief-exploratory-v1"
DEFORM_LONGRUN_POSTERIOR_CONTRACT = "deform-dlo-longrun-posterior-v1"


def load_deform_checkpoint_belief_protocol(path: str | Path) -> dict[str, object]:
    """Load and validate the frozen checkpoint-belief arm bank."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_CHECKPOINT_BELIEF_SCHEMA_VERSION:
        raise ValueError("unsupported DEFORM checkpoint-belief schema")
    if payload.get("contract") != DEFORM_CHECKPOINT_BELIEF_CONTRACT:
        raise ValueError("unsupported DEFORM checkpoint-belief contract")
    if payload.get("source_test_status") != "post-open-exploratory-only":
        raise ValueError("DLO1 checkpoint-belief source test must remain exploratory")
    if payload.get("fresh_confirmation_dlo") != "DLO2":
        raise ValueError("checkpoint-belief confirmation must transfer to fresh DLO2")
    source_commit = str(payload.get("source_reproduction_commit", ""))
    if len(source_commit) != 40:
        raise ValueError("checkpoint-belief protocol requires the full source commit")

    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("checkpoint-belief protocol contains no arms")
    names: set[str] = set()
    arms = []
    for raw_arm in raw_arms:
        if not isinstance(raw_arm, Mapping):
            raise ValueError("checkpoint-belief arm must be an object")
        name = str(raw_arm.get("name", ""))
        if not name or name == "selected_single" or name in names:
            raise ValueError("checkpoint-belief arm names must be unique")
        updates = tuple(int(value) for value in raw_arm.get("updates", ()))
        if not updates or any(value <= 0 for value in updates):
            raise ValueError("checkpoint-belief arms require trained checkpoints")
        if tuple(sorted(set(updates))) != updates:
            raise ValueError("checkpoint-belief updates must be sorted and unique")
        weighting = str(raw_arm.get("weighting", ""))
        if weighting not in ("uniform", "validation-softmax"):
            raise ValueError("unsupported checkpoint-belief weighting")
        names.add(name)
        arms.append({**raw_arm, "name": name, "updates": updates})

    gate = payload.get("validation_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("checkpoint-belief protocol omits its validation gate")
    improvement = float(gate.get("minimum_relative_improvement", math.nan))
    if not math.isfinite(improvement) or not 0.0 < improvement < 1.0:
        raise ValueError("checkpoint-belief validation improvement is invalid")
    temperature = float(gate.get("softmax_temperature_m", math.nan))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("checkpoint-belief temperature is invalid")
    if gate.get("fallback") != "selected_single_exact":
        raise ValueError("checkpoint-belief fallback must remain exact")

    result = dict(payload)
    result["arms"] = arms
    result["protocol_path"] = str(source)
    return result


def load_deform_longrun_posterior_protocol(
    path: str | Path,
) -> dict[str, object]:
    """Load the posterior policy separately from its executing long-run parent."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_CHECKPOINT_BELIEF_SCHEMA_VERSION:
        raise ValueError("unsupported DEFORM long-run posterior schema")
    if payload.get("contract") != DEFORM_LONGRUN_POSTERIOR_CONTRACT:
        raise ValueError("unsupported DEFORM long-run posterior contract")
    if payload.get("source_test_status") != "post-open-exploratory-only":
        raise ValueError("long-run posterior DLO1 source test must remain exploratory")
    if payload.get("official_eval_policy") != "forbidden":
        raise ValueError("long-run posterior must forbid official evaluation")
    if payload.get("fresh_confirmation_dlo") != "DLO2":
        raise ValueError("long-run posterior confirmation must use fresh DLO2")
    for key in ("parent_longrun_protocol", "parent_source_result"):
        identity = payload.get(key)
        if (
            not isinstance(identity, Mapping)
            or not str(identity.get("repository_path", ""))
            or len(str(identity.get("sha256", ""))) != 64
        ):
            raise ValueError("long-run posterior parent identity is invalid")
    if tuple(payload.get("operators", ())) != (
        "parameter_mean",
        "predictive_mean",
    ):
        raise ValueError("long-run posterior operators differ")
    if payload.get("fallback") != "selected_single_exact":
        raise ValueError("long-run posterior must preserve exact fallback")
    for key in (
        "validation_improvement_min",
        "source_transfer_improvement_min",
    ):
        value = float(payload.get(key, math.nan))
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("long-run posterior gate is invalid")
    temperature = float(payload.get("softmax_temperature_m", math.nan))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("long-run posterior temperature is invalid")
    if int(payload.get("source_transfer_minimum_case_wins", -1)) not in range(9):
        raise ValueError("long-run posterior win gate is invalid")
    variance_floor = float(payload.get("coordinate_variance_floor_m2", math.nan))
    nominal_coverage = float(
        payload.get("coordinate_interval_nominal_coverage", math.nan)
    )
    if (
        not math.isfinite(variance_floor)
        or variance_floor <= 0.0
        or not math.isfinite(nominal_coverage)
        or not 0.0 < nominal_coverage < 1.0
    ):
        raise ValueError("long-run posterior uncertainty is invalid")
    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("long-run posterior contains no arms")
    arm_names: set[str] = set()
    allowed_updates = {2560, 4000, 5200, 6040, 6400}
    arms = []
    for arm in raw_arms:
        if not isinstance(arm, Mapping):
            raise ValueError("long-run posterior arm is malformed")
        name = str(arm.get("name", ""))
        updates = tuple(int(update) for update in arm.get("updates", ()))
        if not name or name in arm_names:
            raise ValueError("long-run posterior arm names differ")
        if (
            not updates
            or tuple(sorted(set(updates))) != updates
            or not set(updates).issubset(allowed_updates)
        ):
            raise ValueError("long-run posterior arm updates differ")
        if arm.get("weighting") not in ("uniform", "validation-softmax"):
            raise ValueError("long-run posterior weighting differs")
        arm_names.add(name)
        arms.append({**arm, "name": name, "updates": updates})
    fresh = payload.get("fresh_confirmation")
    if not isinstance(fresh, Mapping) or fresh.get("dlo_type") != "DLO2":
        raise ValueError("long-run posterior fresh confirmation differs")

    result = dict(payload)
    result["arms"] = arms
    result["protocol_path"] = str(source)
    return result


def _validation_errors(
    records: Sequence[Mapping[str, object]],
) -> dict[int, float]:
    errors: dict[int, float] = {}
    for record in records:
        update = int(record.get("update", -1))
        error = float(record.get("validation_l1_m", math.nan))
        if update < 0 or update in errors or not math.isfinite(error) or error < 0.0:
            raise ValueError("invalid checkpoint validation record")
        errors[update] = error
    if not errors:
        raise ValueError("checkpoint validation records are empty")
    return errors


def build_deform_checkpoint_belief_arms(
    validation_records: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> dict[str, dict[int, float]]:
    """Resolve registered parameter-averaging weights from validation only."""

    errors = _validation_errors(validation_records)
    selected_update = min(errors, key=lambda update: (errors[update], update))
    candidates: dict[str, dict[int, float]] = {
        "selected_single": {selected_update: 1.0}
    }
    temperature = float(protocol["validation_gate"]["softmax_temperature_m"])
    for arm in protocol["arms"]:
        name = str(arm["name"])
        updates = tuple(int(value) for value in arm["updates"])
        missing = [update for update in updates if update not in errors]
        if missing:
            raise ValueError(f"checkpoint-belief arm {name} is missing {missing}")
        if arm["weighting"] == "uniform":
            weight = 1.0 / len(updates)
            candidates[name] = {update: weight for update in updates}
            continue
        logits = np.asarray(
            [
                -(errors[update] - min(errors[value] for value in updates))
                / temperature
                for update in updates
            ],
            dtype=float,
        )
        unnormalized = np.exp(logits - np.max(logits))
        normalized = unnormalized / np.sum(unnormalized)
        candidates[name] = {
            update: float(weight)
            for update, weight in zip(updates, normalized, strict=True)
        }
    return candidates


def average_deform_checkpoint_states(
    states_by_update: Mapping[int, Mapping[str, Any]],
    weights_by_update: Mapping[int, float],
) -> dict[str, Any]:
    """Average floating checkpoint tensors while preserving exact discrete state."""

    import torch

    updates = tuple(sorted(int(update) for update in weights_by_update))
    if not updates or set(updates) != {int(update) for update in states_by_update}:
        raise ValueError("checkpoint states and weights must use identical updates")
    weights = {}
    for update in updates:
        weight = float(weights_by_update[update])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("checkpoint weights must be finite and positive")
        weights[update] = weight
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("checkpoint weights must sum to one")

    key_sets = {frozenset(states_by_update[update]) for update in updates}
    if len(key_sets) != 1:
        raise ValueError("checkpoint state dictionaries have different keys")
    result: dict[str, Any] = {}
    for name in sorted(next(iter(key_sets))):
        tensors = [states_by_update[update][name] for update in updates]
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError(f"checkpoint state {name} is not a tensor")
        reference = tensors[0]
        if any(
            tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
            or tensor.device != reference.device
            for tensor in tensors[1:]
        ):
            raise ValueError(f"checkpoint state {name} has incompatible tensors")
        if reference.is_floating_point() or reference.is_complex():
            accumulator_dtype = (
                torch.complex128 if reference.is_complex() else torch.float64
            )
            accumulator = torch.zeros_like(reference, dtype=accumulator_dtype)
            for update, tensor in zip(updates, tensors, strict=True):
                accumulator.add_(tensor.to(accumulator_dtype), alpha=weights[update])
            result[name] = accumulator.to(reference.dtype)
            continue
        if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
            raise ValueError(f"discrete checkpoint state {name} differs")
        result[name] = reference.clone()
    return result


def combine_deform_checkpoint_predictions(
    predictions_by_update: Mapping[int, np.ndarray],
    weights_by_update: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the posterior predictive mean and diagonal checkpoint variance."""

    updates = tuple(sorted(int(update) for update in weights_by_update))
    if not updates or set(updates) != {int(update) for update in predictions_by_update}:
        raise ValueError(
            "checkpoint predictions and weights must use identical updates"
        )
    weights = {}
    arrays = []
    for update in updates:
        weight = float(weights_by_update[update])
        array = np.asarray(predictions_by_update[update], dtype=np.float64)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("checkpoint weights must be finite and positive")
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("checkpoint predictions must be finite and nonempty")
        weights[update] = weight
        arrays.append(array)
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("checkpoint weights must sum to one")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("checkpoint predictions have different shapes")

    mean = np.zeros_like(arrays[0], dtype=np.float64)
    for update, array in zip(updates, arrays, strict=True):
        mean += weights[update] * array
    variance = np.zeros_like(mean)
    for update, array in zip(updates, arrays, strict=True):
        variance += weights[update] * np.square(array - mean)
    return mean, variance


def deform_prediction_records(
    predictions: np.ndarray,
    targets: np.ndarray,
    persistence: np.ndarray,
    names: Sequence[str],
) -> list[dict[str, object]]:
    """Compute the exact source metrics from externally combined predictions."""

    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    persisted = np.asarray(persistence, dtype=np.float64)
    if (
        predicted.ndim != 4
        or predicted.shape[-1] != 3
        or predicted.shape != observed.shape
        or predicted.shape != persisted.shape
        or predicted.shape[0] != len(names)
        or not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
        or not np.isfinite(persisted).all()
    ):
        raise ValueError("DEFORM prediction arrays are incompatible")
    normalized_names = tuple(str(name) for name in names)
    if any(not name for name in normalized_names) or len(set(normalized_names)) != len(
        normalized_names
    ):
        raise ValueError("DEFORM prediction names must be nonempty and unique")

    horizon = predicted.shape[1]
    absolute = np.abs(predicted - observed)
    thirds = []
    for third in range(3):
        indices = [
            frame for frame in range(horizon) if min(2, (3 * frame) // horizon) == third
        ]
        if not indices:
            raise ValueError("DEFORM prediction horizon cannot form thirds")
        thirds.append(np.mean(absolute[:, indices], axis=(1, 2, 3)))
    model_error = np.mean(absolute, axis=(1, 2, 3))
    persistence_error = np.mean(np.abs(persisted - observed), axis=(1, 2, 3))
    return [
        {
            "name": name,
            "model_l1_m": float(model_error[index]),
            "persistence_l1_m": float(persistence_error[index]),
            "early_l1_m": float(thirds[0][index]),
            "middle_l1_m": float(thirds[1][index]),
            "late_l1_m": float(thirds[2][index]),
        }
        for index, name in enumerate(normalized_names)
    ]


def calibrate_deform_coordinate_variance(
    predictions: np.ndarray,
    targets: np.ndarray,
    raw_variance_m2: np.ndarray,
    *,
    variance_floor_m2: float,
) -> float:
    """Fit one validation-only variance scale, never shrinking raw uncertainty."""

    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    variance = np.asarray(raw_variance_m2, dtype=np.float64)
    if (
        predicted.shape != observed.shape
        or predicted.shape != variance.shape
        or predicted.size == 0
        or not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
        or not np.isfinite(variance).all()
        or np.any(variance < 0.0)
        or not math.isfinite(variance_floor_m2)
        or variance_floor_m2 <= 0.0
    ):
        raise ValueError("DEFORM variance calibration arrays are invalid")
    effective = np.maximum(variance, variance_floor_m2)
    scale = float(np.mean(np.square(predicted - observed) / effective))
    return max(1.0, scale)


def evaluate_deform_coordinate_uncertainty(
    predictions: np.ndarray,
    targets: np.ndarray,
    raw_variance_m2: np.ndarray,
    *,
    variance_floor_m2: float,
    variance_scale: float,
    nominal_coverage: float,
) -> dict[str, float]:
    """Evaluate coordinate-marginal Gaussian diagnostics for a fixed scale."""

    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    variance = np.asarray(raw_variance_m2, dtype=np.float64)
    if (
        predicted.shape != observed.shape
        or predicted.shape != variance.shape
        or predicted.size == 0
        or not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
        or not np.isfinite(variance).all()
        or np.any(variance < 0.0)
        or not math.isfinite(variance_floor_m2)
        or variance_floor_m2 <= 0.0
        or not math.isfinite(variance_scale)
        or variance_scale < 1.0
        or not math.isfinite(nominal_coverage)
        or not 0.0 < nominal_coverage < 1.0
    ):
        raise ValueError("DEFORM uncertainty inputs are invalid")
    effective = variance_scale * np.maximum(variance, variance_floor_m2)
    residual = predicted - observed
    z_value = NormalDist().inv_cdf(0.5 * (1.0 + nominal_coverage))
    half_width = z_value * np.sqrt(effective)
    nll = 0.5 * (np.log(2.0 * np.pi * effective) + np.square(residual) / effective)
    return {
        "coordinate_coverage": float(np.mean(np.abs(residual) <= half_width)),
        "mean_interval_width_m": float(np.mean(2.0 * half_width)),
        "mean_gaussian_nll": float(np.mean(nll)),
    }


def select_deform_checkpoint_belief_arm(
    validation_arm_errors: Mapping[str, float],
    *,
    minimum_relative_improvement: float,
) -> dict[str, object]:
    """Select a belief arm or return the exact registered single checkpoint."""

    if "selected_single" not in validation_arm_errors:
        raise ValueError("checkpoint-belief selection omits selected_single")
    normalized = {}
    for name, value in validation_arm_errors.items():
        error = float(value)
        if not name or not math.isfinite(error) or error < 0.0:
            raise ValueError("checkpoint-belief validation arm error is invalid")
        normalized[str(name)] = error
    baseline = normalized["selected_single"]
    candidates = [
        (error, name) for name, error in normalized.items() if name != "selected_single"
    ]
    if not candidates:
        return {
            "selected_arm": "selected_single",
            "fallback_used": True,
            "relative_improvement": 0.0,
        }
    candidate_error, candidate_name = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    relative_improvement = (
        (baseline - candidate_error) / baseline if baseline > 0.0 else 0.0
    )
    accepted = relative_improvement >= minimum_relative_improvement
    return {
        "selected_arm": candidate_name if accepted else "selected_single",
        "candidate_arm": candidate_name,
        "fallback_used": not accepted,
        "baseline_validation_l1_m": baseline,
        "candidate_validation_l1_m": candidate_error,
        "relative_improvement": relative_improvement,
    }


def evaluate_deform_checkpoint_belief_transfer(
    candidate_records: Sequence[Mapping[str, object]],
    baseline_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize the post-open DLO1 transfer without changing the selector."""

    def indexed(
        records: Sequence[Mapping[str, object]],
        *,
        label: str,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for record in records:
            name = str(record["name"])
            if not name or name in result:
                raise ValueError(f"{label} checkpoint-belief cases are not unique")
            result[name] = float(record["model_l1_m"])
        return result

    candidate = indexed(candidate_records, label="candidate")
    baseline = indexed(baseline_records, label="baseline")
    if not candidate or set(candidate) != set(baseline):
        raise ValueError("checkpoint-belief transfer cases do not match")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (*candidate.values(), *baseline.values())
    ):
        raise ValueError("checkpoint-belief transfer errors must be finite")
    candidate_mean = float(np.mean(list(candidate.values())))
    baseline_mean = float(np.mean(list(baseline.values())))
    improvement = (
        (baseline_mean - candidate_mean) / baseline_mean if baseline_mean > 0.0 else 0.0
    )
    wins = sum(candidate[name] < baseline[name] for name in candidate)
    return {
        "case_count": len(candidate),
        "candidate_mean_l1_m": candidate_mean,
        "baseline_mean_l1_m": baseline_mean,
        "relative_improvement": improvement,
        "wins": wins,
        "claim_boundary": "post-open DLO1 exploratory; fresh DLO2 required",
    }
