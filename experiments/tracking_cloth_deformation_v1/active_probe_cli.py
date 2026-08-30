"""Retrospective logged-probe analysis for Tracking Cloth Deformation v1.

The registered selection protocol and selection helpers predate the first
shake-to-twist target scoring. This runner was added afterwards to execute that
protocol mechanically. Results are therefore exploratory and cannot constitute
fresh confirmation. Probe selection remains fold-local and target-outcome blind.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .active_probe import (
    pairwise_trajectory_mse,
    simulate_policy,
    update_weights,
    weights_from_records,
)
from .active_probe_run import (
    active_mask,
    arm_specs,
    build_belief_arms,
    calibrated_residuals,
    loss_vector,
    posterior_temperature,
    validate_protocol,
)
from .data import (
    Case,
    Inputs,
    audit_dataset,
    digest,
    infer_source_scale,
    input_view,
    object_digest,
    read_prefix,
    scoring_view,
    write_json,
)
from .model import Predictions, horizon_bins, predict, score

HERE = Path(__file__).resolve().parent
METRICS = (
    "rmse_mm",
    "mean_marker_error_mm",
    "coordinate_nll",
    "coordinate_90_coverage",
    "mean_full_90_width_mm",
)
FROZEN_SELECTION_GIT_BLOBS = {
    "active_probe.py": "bb641789f8387ea7d659ac3f098e5371c1ae1b3f",
    "active_probe_protocol.json": "8ee3f9b47b0faf0c228c738afd3ae6b1905329c4",
    "active_probe_run.py": "ab0bc3c23b2f4d35f0c708d88bb3cf153b5091d9",
}
PRIOR_TARGET_RUN_ID = "33302686759"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def implementation() -> dict[str, str]:
    names = (
        "active_probe.py",
        "active_probe_cli.py",
        "active_probe_run.py",
        "data.py",
        "model.py",
    )
    return {name: digest(HERE / name) for name in names}


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [jsonable(item) for item in value]
    return value


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing an empty result table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def active_arms(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return ("persistence", *arm_specs(protocol))


def source_record(args: tuple[Case, dict[str, Any], float]):
    case, protocol, scale = args
    inputs = input_view(case, protocol, scale)
    prediction = predict(inputs, protocol)
    truth = scoring_view(case, inputs)
    return case, prediction, truth


def input_prediction(args: tuple[Case, dict[str, Any], float]):
    case, protocol, scale = args
    inputs = input_view(case, protocol, scale)
    return case, predict(inputs, protocol)


def map_records(function, tasks: Sequence[Any], workers: int) -> list[Any]:
    if workers == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, tasks))


def average_distance(predictions: Sequence[Predictions]) -> np.ndarray:
    if not predictions:
        raise ValueError("At least one prediction is required for a distance template")
    matrices = [
        pairwise_trajectory_mse(prediction.bank, active_mask(prediction.inputs))
        for prediction in predictions
    ]
    result = np.mean(np.stack(matrices), axis=0)
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)
    return result


def persistence_mean(prediction: Predictions) -> np.ndarray:
    mean = np.broadcast_to(
        prediction.inputs.prefix[-1], prediction.nominal.shape
    ).copy()
    mean[:, prediction.inputs.corners] = prediction.inputs.boundary
    return mean


def persistence_residuals(
    records: Sequence[tuple[Predictions, np.ndarray]], protocol: Mapping[str, Any]
) -> list[float]:
    if not records:
        raise ValueError("At least one source record is required")
    floor2 = float(protocol["measurement_floor_m"]) ** 2
    per_bin: list[list[float]] = [[], [], []]
    for prediction, truth in records:
        mean = persistence_mean(prediction)
        valid = active_mask(prediction.inputs) & np.isfinite(truth).all(axis=2)
        bins = horizon_bins(prediction.inputs)
        error2 = (mean - truth) ** 2
        for bin_index in range(3):
            selected = valid & (bins[:, None] == bin_index)
            if not np.any(selected):
                raise ValueError("Empty persistence calibration horizon bin")
            per_bin[bin_index].append(max(float(np.mean(error2[selected])), floor2))
    return [max(float(np.mean(values)), floor2) for values in per_bin]


def persistence_belief(
    prediction: Predictions, residual_variance: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(residual_variance, dtype=np.float64)
    if residual.shape != (3,) or np.any(residual <= 0.0):
        raise ValueError("Persistence residual variance must contain three positives")
    mean = persistence_mean(prediction)
    bins = horizon_bins(prediction.inputs)
    variance = np.broadcast_to(residual[bins, None, None], mean.shape).copy()
    return mean, variance


def state_record(state) -> dict[str, Any]:
    return {
        "budget": int(state.budget),
        "selected_actions": list(state.selected_actions),
        "weights": state.weights.tolist(),
        "steps": jsonable(state.steps),
    }


def fit_fold(
    held_material: str,
    source_records: Sequence[tuple[Case, Predictions, np.ndarray]],
    target_inputs: Sequence[tuple[Case, Predictions]],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    training = [
        record for record in source_records if record[0].material != held_material
    ]
    expected_training = 3 * 2 * 4
    if len(training) != expected_training:
        raise ValueError("Incomplete leave-one-material-out source fold")
    loss_matrix = np.stack(
        [loss_vector(prediction, truth) for _, prediction, truth in training]
    )
    temperature = posterior_temperature(
        loss_matrix, float(protocol["measurement_floor_m"])
    )
    prior_weights = weights_from_records(loss_matrix, temperature)
    training_pairs = [(prediction, truth) for _, prediction, truth in training]
    residuals = calibrated_residuals(training_pairs, prior_weights, protocol)
    residuals["persistence"] = persistence_residuals(training_pairs, protocol)

    probe_distances: dict[str, np.ndarray] = {}
    for condition in protocol["probe_conditions"]:
        predictions = [
            prediction
            for case, prediction, _ in training
            if case.condition == condition
        ]
        if len(predictions) != 6:
            raise ValueError("Each source probe template requires six recordings")
        probe_distances[condition] = average_distance(predictions)

    target_predictions = [
        prediction
        for case, prediction in target_inputs
        if case.material != held_material
    ]
    if len(target_predictions) != expected_training:
        raise ValueError("Incomplete leave-one-material-out target-input template")
    target_distance = average_distance(target_predictions)

    specimens: dict[str, dict[str, Any]] = {}
    for size in protocol["sizes"]:
        specimen = f"{held_material}_{size}"
        held = {
            case.condition: (prediction, truth)
            for case, prediction, truth in source_records
            if case.material == held_material and case.size == size
        }
        if set(held) != set(protocol["probe_conditions"]):
            raise ValueError(
                "Held specimen does not expose the registered probe roster"
            )
        observed_losses = {
            condition: loss_vector(*held[condition])
            for condition in protocol["probe_conditions"]
        }
        single = {
            condition: update_weights(
                prior_weights, observed_losses[condition], temperature
            ).tolist()
            for condition in protocol["probe_conditions"]
        }
        policies: dict[str, dict[str, Any]] = {}
        for policy in protocol["probe_policies"]:
            states = simulate_policy(
                policy=policy,
                initial_weights=prior_weights,
                probe_distances=probe_distances,
                target_distance=target_distance,
                observed_losses=observed_losses,
                temperature=temperature,
                fixed_order=protocol["fixed_probe_order"],
                budgets=protocol["probe_budgets"],
            )
            policies[policy] = {
                str(budget): state_record(states[int(budget)])
                for budget in protocol["probe_budgets"]
            }
        specimens[specimen] = {
            "single_probe_weights": single,
            "policy_states": policies,
            "held_probe_recordings": {
                condition: next(
                    case.path.name
                    for case, _, _ in source_records
                    if case.material == held_material
                    and case.size == size
                    and case.condition == condition
                )
                for condition in protocol["probe_conditions"]
            },
        }

    fold = {
        "held_material": held_material,
        "source_materials": [
            material
            for material in protocol["materials"]
            if material != held_material
        ],
        "prior_weights": prior_weights.tolist(),
        "posterior_temperature_m2": temperature,
        "source_residual_variance_m2": residuals,
        "source_recordings": [case.path.name for case, _, _ in training],
        "target_input_template_recordings": [
            case.path.name
            for case, _ in target_inputs
            if case.material != held_material
        ],
        "probe_distance_matrices_m2": {
            key: value.tolist() for key, value in probe_distances.items()
        },
        "target_distance_matrix_m2": target_distance.tolist(),
        "held_material_source_outcomes_consumed_only_after_selection": True,
        "held_material_twist_outcomes_used": False,
    }
    return fold, specimens


def prepare(
    root: Path,
    output: Path,
    protocol: dict[str, Any],
    workers: int,
) -> None:
    validate_protocol(protocol)
    root = root.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output and dataset must be disjoint directory trees")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "run_manifest.json",
        {
            "created_at": now(),
            "protocol_id": object_digest(protocol),
            "implementation_sha256": implementation(),
            "frozen_selection_git_blobs": FROZEN_SELECTION_GIT_BLOBS,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "prior_target_run_id": PRIOR_TARGET_RUN_ID,
            "prior_target_outcome_exposure": True,
            "target_numeric_outcomes_read_in_this_stage": False,
            "evidence_class": protocol["evidence_class"],
            "paper_claim_authorized": False,
        },
    )
    cases, inventory = audit_dataset(root, protocol)
    write_json(output / "dataset_manifest.json", inventory)
    (output / "DATA_LICENSE.txt").write_text(inventory["included_license_text"])

    source_cases = [case for case in cases if case.motion == "shake"]
    target_cases = [case for case in cases if case.motion == "twist"]
    scales = [
        infer_source_scale(case, read_prefix(case, protocol["prefix_seconds"])[1])
        for case in source_cases
    ]
    if len(set(scales)) != 1:
        raise ValueError("Source recordings disagree about metric coordinate units")
    scale = scales[0]
    source_records = map_records(
        source_record,
        [(case, protocol, scale) for case in source_cases],
        workers,
    )
    target_inputs = map_records(
        input_prediction,
        [(case, protocol, scale) for case in target_cases],
        workers,
    )

    folds: dict[str, Any] = {}
    specimens: dict[str, Any] = {}
    for held_material in protocol["materials"]:
        fold, fold_specimens = fit_fold(
            held_material, source_records, target_inputs, protocol
        )
        folds[held_material] = fold
        overlap = set(specimens) & set(fold_specimens)
        if overlap:
            raise ValueError(f"Duplicate specimen states: {sorted(overlap)}")
        specimens.update(fold_specimens)
    if len(folds) != 4 or len(specimens) != 8:
        raise ValueError("Incomplete active-probe cross-validation roster")

    source_fit = {
        "created_at": now(),
        "protocol_id": object_digest(protocol),
        "inventory_id": inventory["inventory_id"],
        "coordinate_scale_to_m": scale,
        "folds": folds,
        "specimens": specimens,
        "target_outcomes_used": False,
        "selection_protocol_frozen_before_prior_target_run": True,
        "runner_implementation_added_after_prior_target_run": True,
    }
    write_json(output / "source_fit.json", source_fit)

    source_rows: list[dict[str, Any]] = []
    for material, fold in folds.items():
        for model, weight in enumerate(fold["prior_weights"]):
            source_rows.append(
                {
                    "held_material": material,
                    "model_index": model,
                    "prior_weight": weight,
                    "posterior_temperature_m2": fold["posterior_temperature_m2"],
                }
            )
    save_csv(output / "source_scores.csv", source_rows)

    private = output / "private_predictions"
    private.mkdir(mode=0o700)
    predictions: dict[str, Any] = {}
    all_arms = active_arms(protocol)
    for case, prediction in target_inputs:
        fold = folds[case.material]
        specimen = specimens[case.specimen]
        beliefs = build_belief_arms(
            prediction,
            prefix_last=prediction.inputs.prefix[-1],
            boundary=prediction.inputs.boundary,
            fold=fold,
            specimen=specimen,
            protocol=protocol,
        )
        beliefs["persistence"] = persistence_belief(
            prediction, fold["source_residual_variance_m2"]["persistence"]
        )
        if set(beliefs) != set(all_arms):
            raise ValueError("Active-probe prediction roster is incomplete")
        arrays = {f"{arm}_mean": beliefs[arm][0] for arm in all_arms}
        arrays.update({f"{arm}_variance": beliefs[arm][1] for arm in all_arms})
        arrays.update(
            {
                "times": prediction.inputs.times,
                "order": prediction.inputs.order,
                "corners": prediction.inputs.corners,
                "cutoff": np.array(prediction.inputs.cutoff),
                "scale": np.array(scale),
            }
        )
        artifact = private / f"{case.path.stem}.npz"
        np.savez_compressed(artifact, **arrays)
        predictions[case.path.name] = {
            "artifact": str(artifact.relative_to(output)),
            "sha256": digest(artifact),
            "specimen": case.specimen,
            "material": case.material,
            "policy_choices": {
                policy: {
                    str(budget): specimen["policy_states"][policy][str(budget)][
                        "selected_actions"
                    ]
                    for budget in protocol["probe_budgets"]
                }
                for policy in protocol["probe_policies"]
            },
        }
    if len(predictions) != 32:
        raise ValueError("Refusing an incomplete active-probe target seal")

    seal = {
        "sealed_at": now(),
        "protocol_id": object_digest(protocol),
        "inventory_id": inventory["inventory_id"],
        "source_fit_sha256": digest(output / "source_fit.json"),
        "implementation_sha256": implementation(),
        "frozen_selection_git_blobs": FROZEN_SELECTION_GIT_BLOBS,
        "arms": list(all_arms),
        "predictions": predictions,
        "future_free_marker_outcomes_read_in_this_run": False,
        "future_driven_corner_coordinates_used": True,
        "prior_target_outcome_exposure": True,
        "prior_target_run_id": PRIOR_TARGET_RUN_ID,
        "fresh_confirmation_claim": False,
    }
    write_json(output / "prediction_seal.json", seal)
    report = (
        "# Tracking Cloth Deformation: active-probe replay\n\n"
        f"Study: `{protocol['study_id']}`\n\n"
        "The selection protocol and core selector were frozen before the prior "
        "target-scored run, but this mechanical runner was added afterwards. "
        "The result is a retrospective public-data diagnostic, not fresh "
        "confirmation.\n\n"
        "Four leave-one-material-out folds construct probe and target-query "
        "disagreement templates from the other three materials. Held-material "
        "shake outcomes are consumed only after a policy selects that logged "
        "probe. No twist free-marker outcome entered selection or prediction.\n\n"
        "All 32 twist prediction batches and 19 comparison arms were sealed "
        "before scoring in this run.\n"
    )
    (output / "report.md").write_text(report)


def specimen_table(
    rows: list[dict[str, Any]], arms: Sequence[str]
) -> list[dict[str, Any]]:
    specimens = sorted({str(row["specimen"]) for row in rows})
    if len(specimens) != 8 or len(rows) != 32 * len(arms):
        raise ValueError("Incomplete target roster; no partial result is authorized")
    table: list[dict[str, Any]] = []
    for specimen in specimens:
        for arm in arms:
            subset = [
                row
                for row in rows
                if row["specimen"] == specimen and row["arm"] == arm
            ]
            if len(subset) != 4:
                raise ValueError(
                    "Each specimen-arm pair requires four twist conditions"
                )
            table.append(
                {
                    "specimen": specimen,
                    "material": specimen.split("_", maxsplit=1)[0],
                    "arm": arm,
                    **{
                        metric: float(np.mean([float(row[metric]) for row in subset]))
                        for metric in METRICS
                    },
                }
            )
    return table


def arm_vector(
    table: Sequence[dict[str, Any]], specimens: Sequence[str], arm: str
) -> np.ndarray:
    return np.asarray(
        [
            next(
                float(row["rmse_mm"])
                for row in table
                if row["specimen"] == specimen and row["arm"] == arm
            )
            for specimen in specimens
        ]
    )


def contrast(
    candidate: np.ndarray,
    comparator: np.ndarray,
    specimens: Sequence[str],
    protocol: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    difference = np.asarray(candidate) - np.asarray(comparator)
    if difference.shape != (8,):
        raise ValueError("A contrast requires eight specimen values")
    repetitions = int(protocol["bootstrap_repetitions"])
    resamples = rng.integers(0, 8, size=(repetitions, 8))
    materials = list(protocol["materials"])
    material_difference = np.asarray(
        [
            np.mean(
                [
                    difference[index]
                    for index, specimen in enumerate(specimens)
                    if specimen.startswith(f"{material}_")
                ]
            )
            for material in materials
        ]
    )
    material_resamples = rng.integers(0, 4, size=(repetitions, 4))
    return {
        "candidate_minus_comparator_rmse_mm": float(np.mean(difference)),
        "specimen_bootstrap_95_interval_mm": np.quantile(
            difference[resamples].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "material_cluster_sensitivity_95_interval_mm": np.quantile(
            material_difference[material_resamples].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "specimen_wins": int(np.sum(difference < 0.0)),
        "specimen_ties": int(np.sum(difference == 0.0)),
        "specimen_losses": int(np.sum(difference > 0.0)),
        "worst_specimen_regret_mm": float(np.max(difference)),
    }


def aggregate(
    rows: list[dict[str, Any]],
    source_fit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    arms: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = specimen_table(rows, arms)
    specimens = sorted({str(row["specimen"]) for row in table})
    summaries = {
        arm: {
            metric: float(
                np.mean(
                    [
                        float(row[metric])
                        for row in table
                        if row["arm"] == arm
                    ]
                )
            )
            for metric in METRICS
        }
        for arm in arms
    }
    task = arm_vector(table, specimens, "task_directed_k1")
    parameter = arm_vector(table, specimens, "parameter_information_k1")
    fixed = arm_vector(table, specimens, "fixed_order_k1")
    prior = arm_vector(table, specimens, "fixed_order_k0")
    persistence = arm_vector(table, specimens, "persistence")
    singles = np.stack(
        [
            arm_vector(table, specimens, f"single_probe_{condition}")
            for condition in protocol["probe_conditions"]
        ]
    )
    random_expected = np.mean(singles, axis=0)
    oracle_single = np.min(singles, axis=0)

    specimen_states = source_fit["specimens"]
    task_choices = [
        specimen_states[specimen]["policy_states"]["task_directed"]["1"][
            "selected_actions"
        ][0]
        for specimen in specimens
    ]
    parameter_choices = [
        specimen_states[specimen]["policy_states"]["parameter_information"]["1"][
            "selected_actions"
        ][0]
        for specimen in specimens
    ]
    fixed_choices = [
        specimen_states[specimen]["policy_states"]["fixed_order"]["1"][
            "selected_actions"
        ][0]
        for specimen in specimens
    ]
    disagreement = np.asarray(
        [
            left != right
            for left, right in zip(task_choices, parameter_choices, strict=True)
        ]
    )
    difference = task - parameter
    rng = np.random.default_rng(int(protocol["bootstrap_seed"]))
    contrasts = {
        "parameter_information_k1": contrast(
            task, parameter, specimens, protocol, rng
        ),
        "fixed_order_k1": contrast(task, fixed, specimens, protocol, rng),
        "no_probe_k0": contrast(task, prior, specimens, protocol, rng),
        "persistence": contrast(task, persistence, specimens, protocol, rng),
        "random_expected_k1": contrast(
            task, random_expected, specimens, protocol, rng
        ),
        "single_probe_oracle_k1": contrast(
            task, oracle_single, specimens, protocol, rng
        ),
    }
    selection = {
        "specimens": [
            {
                "specimen": specimen,
                "task_directed_k1": task_choices[index],
                "parameter_information_k1": parameter_choices[index],
                "fixed_order_k1": fixed_choices[index],
                "task_minus_parameter_rmse_mm": float(difference[index]),
            }
            for index, specimen in enumerate(specimens)
        ],
        "task_vs_parameter_disagreement_count": int(np.sum(disagreement)),
        "task_vs_parameter_disagreement_fraction": float(np.mean(disagreement)),
        "task_minus_parameter_rmse_on_disagreements_mm": (
            float(np.mean(difference[disagreement])) if np.any(disagreement) else None
        ),
        "task_choice_counts": {
            condition: task_choices.count(condition)
            for condition in protocol["probe_conditions"]
        },
        "parameter_choice_counts": {
            condition: parameter_choices.count(condition)
            for condition in protocol["probe_conditions"]
        },
    }
    return table, {
        "arms": summaries,
        "derived_endpoints": {
            "random_expected_k1_rmse_mm": float(np.mean(random_expected)),
            "single_probe_oracle_k1_rmse_mm": float(np.mean(oracle_single)),
        },
        "contrasts": contrasts,
        "selection": selection,
        "mechanism_gate": {
            "policies_make_different_k1_choices": bool(np.any(disagreement)),
            "task_directed_beats_parameter_information_overall": bool(
                np.mean(difference) < 0.0
            ),
            "task_directed_beats_parameter_on_disagreements": (
                bool(np.mean(difference[disagreement]) < 0.0)
                if np.any(disagreement)
                else False
            ),
            "task_directed_beats_persistence": bool(
                np.mean(task - persistence) < 0.0
            ),
        },
        "inferential_unit": (
            "8 material-size specimens; 4-material sensitivity also reported"
        ),
        "aggregation": (
            "equal twist conditions within specimen, then equal specimens; "
            "no frame pseudoreplication"
        ),
        "evidence_class": protocol["evidence_class"],
        "paper_claim_authorized": False,
    }


def score_run(root: Path, output: Path) -> None:
    root = root.resolve(strict=True)
    output = output.resolve(strict=True)
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output and dataset must be disjoint directory trees")
    if (output / "target_access.json").exists():
        raise ValueError("This active-probe run already started target scoring")
    protocol = json.loads((output / "protocol.json").read_text())
    validate_protocol(protocol)
    source_fit = json.loads((output / "source_fit.json").read_text())
    seal = json.loads((output / "prediction_seal.json").read_text())
    if seal["protocol_id"] != object_digest(protocol):
        raise ValueError("Protocol changed after active-probe prediction sealing")
    if seal["implementation_sha256"] != implementation():
        raise ValueError("Implementation changed after active-probe prediction sealing")
    if seal["source_fit_sha256"] != digest(output / "source_fit.json"):
        raise ValueError("Source fit changed after active-probe prediction sealing")
    cases, inventory = audit_dataset(root, protocol)
    if inventory["inventory_id"] != seal["inventory_id"]:
        raise ValueError("Dataset changed after active-probe prediction sealing")
    for entry in seal["predictions"].values():
        path = (output / entry["artifact"]).resolve()
        if (
            not path.is_relative_to((output / "private_predictions").resolve())
            or digest(path) != entry["sha256"]
        ):
            raise ValueError("Active-probe prediction artifact identity mismatch")
    write_json(
        output / "target_access.json",
        {
            "started_at": now(),
            "prediction_seal_sha256": digest(output / "prediction_seal.json"),
            "authorized_recordings": sorted(seal["predictions"]),
            "purpose": "retrospective logged active-probe diagnostic",
            "prior_target_run_id": PRIOR_TARGET_RUN_ID,
            "fresh_confirmation_claim": False,
        },
    )

    arms = tuple(str(value) for value in seal["arms"])
    if arms != active_arms(protocol):
        raise ValueError("Sealed arm roster differs from the registered runner")
    rows: list[dict[str, Any]] = []
    for case in (candidate for candidate in cases if candidate.motion == "twist"):
        entry = seal["predictions"][case.path.name]
        with np.load(output / entry["artifact"], allow_pickle=False) as arrays:
            inputs = Inputs(
                arrays["times"],
                np.empty((0, case.markers, 3)),
                np.empty((0, 2, 3)),
                arrays["order"],
                arrays["corners"],
                int(arrays["cutoff"]),
                float(arrays["times"][0]),
                float(arrays["scale"]),
            )
            truth = scoring_view(case, inputs)
            for arm in arms:
                values = score(
                    arrays[f"{arm}_mean"],
                    arrays[f"{arm}_variance"],
                    truth,
                    inputs,
                )
                rows.append(
                    {
                        "recording": case.path.name,
                        "specimen": case.specimen,
                        "material": case.material,
                        "size": case.size,
                        "speed": case.speed,
                        "grasp": case.grasp,
                        "arm": arm,
                        **values,
                    }
                )
    table, metrics = aggregate(rows, source_fit, protocol, arms)
    save_csv(output / "target_scores.csv", rows)
    save_csv(output / "specimen_scores.csv", table)
    write_json(output / "metrics.json", metrics)

    manifest = json.loads((output / "run_manifest.json").read_text())
    manifest.update(
        {
            "completed_at": now(),
            "target_numeric_outcomes_read_in_this_stage": True,
            "prediction_seal_sha256": digest(output / "prediction_seal.json"),
            "metrics_sha256": digest(output / "metrics.json"),
            "status": "completed-retrospective-diagnostic-not-claim-promoted",
        }
    )
    write_json(output / "run_manifest.json", manifest)

    report = (output / "report.md").read_text()
    report += "\n## Logged-probe twist results\n\n"
    report += (
        "| Arm | Specimen-balanced RMSE [mm] | Coordinate NLL | "
        "90% coverage | Full width [mm] |\n"
    )
    report += "| --- | ---: | ---: | ---: | ---: |\n"
    for arm in arms:
        values = metrics["arms"][arm]
        report += (
            f"| {arm} | {values['rmse_mm']:.4f} | "
            f"{values['coordinate_nll']:.4f} | "
            f"{100 * values['coordinate_90_coverage']:.2f}% | "
            f"{values['mean_full_90_width_mm']:.4f} |\n"
        )
    selection = metrics["selection"]
    report += (
        "\nAt budget one, task-directed and parameter-information policies "
        f"disagree on {selection['task_vs_parameter_disagreement_count']}/8 "
        "material-size specimens. "
    )
    disagreement_effect = selection[
        "task_minus_parameter_rmse_on_disagreements_mm"
    ]
    if disagreement_effect is None:
        report += "No disagreement-conditional performance claim is available.\n\n"
    else:
        report += (
            "On those disagreements, task-directed minus parameter-information "
            f"RMSE is {disagreement_effect:.4f} mm.\n\n"
        )
    report += (
        "This run follows a selection protocol frozen before the first target "
        "score, but the execution wrapper was implemented after prior target "
        "exposure. It is therefore exploratory and cannot authorize a fresh "
        "confirmation, online-control, safety, material-identification, or "
        "state-of-the-art claim.\n"
    )
    (output / "report.md").write_text(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("predict", "score"), default="predict")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    try:
        if args.stage == "score":
            score_run(args.dataset_root, args.output)
        else:
            protocol = json.loads((HERE / "active_probe_protocol.json").read_text())
            prepare(args.dataset_root, args.output, protocol, args.workers)
    except Exception as exc:
        if args.output.is_dir() and not args.output.resolve().is_relative_to(
            args.dataset_root.resolve()
        ):
            write_json(
                args.output / "failure.json",
                {
                    "failed_at": now(),
                    "stage": args.stage,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "target_scoring_started": (
                        args.output / "target_access.json"
                    ).exists(),
                    "scientific_decision": "incomplete; no claim",
                },
            )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
