"""Query-quotient validation on public Tracking Cloth Deformation recordings.

The prediction stage consumes shaking-source outcomes and twisting target inputs,
constructs one frozen finite-model posterior per material-size specimen, and
seals every target prediction before any twisting free-marker outcome is read.
The score stage opens only the sealed twisting outcomes.

The study is retrospective public-data evidence. It is not fresh confirmation,
unique material identification, online control, or a safety certificate.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.query_quotient_belief_v1 import (
    aggregate_to_query_quotient,
    minimum_information_query_lift,
    query_ambiguity_envelope,
    query_quotient_information_decomposition,
)
from experiments.tracking_cloth_deformation_v1.active_probe_cli import (
    fit_fold,
    input_prediction,
    map_records,
    source_record,
)
from experiments.tracking_cloth_deformation_v1.active_probe_run import (
    validate_protocol as validate_active_protocol,
)
from experiments.tracking_cloth_deformation_v1.data import (
    Case,
    Inputs,
    audit_dataset,
    digest,
    infer_source_scale,
    object_digest,
    read_prefix,
    scoring_view,
    write_json,
)
from experiments.tracking_cloth_deformation_v1.model import (
    Predictions,
    parameter_bank,
)

HERE = Path(__file__).resolve().parent
BASE_HERE = HERE.parent / "tracking_cloth_deformation_v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/tracking-cloth-query-quotient-v1"
RESULT_SCHEMA = "bayesian-phystwin/tracking-cloth-query-quotient-result-v1"
SCHEMA_VERSION = 1
LIFT_NAMES = (
    "full_source_posterior",
    "jeffrey_i_projection",
    "uniform_within_class",
    "prior_map_concentration",
    "reverse_prior_concentration",
)
ENDPOINT_NAMES = (
    "final_rms_displacement_m",
    "mid_rms_displacement_m",
    "peak_rms_displacement_m",
)
PARAMETER_NAMES = ("stiffness_per_mass", "damping_per_mass")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows do not share one ordered schema")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _probability(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return array / total


def implementation() -> dict[str, str]:
    files = {
        "query_run.py": HERE / "run.py",
        "query_protocol.json": HERE / "protocol.json",
        "active_probe.py": BASE_HERE / "active_probe.py",
        "active_probe_cli.py": BASE_HERE / "active_probe_cli.py",
        "active_probe_protocol.json": BASE_HERE / "active_probe_protocol.json",
        "active_probe_run.py": BASE_HERE / "active_probe_run.py",
        "data.py": BASE_HERE / "data.py",
        "model.py": BASE_HERE / "model.py",
    }
    return {name: digest(path) for name, path in files.items()}


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected query-quotient protocol schema")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected query-quotient protocol version")
    if int(protocol["query"]["class_count"]) != 3:
        raise ValueError("the registered quotient must contain three classes")
    if protocol["query"]["partition_rule"] != (
        "stable sort of model-predicted final free-marker RMS displacement; "
        "three contiguous equal-count finite-diameter classes"
    ):
        raise ValueError("query partition rule changed")
    if tuple(protocol["comparison_lifts"]) != LIFT_NAMES:
        raise ValueError("same-quotient comparison lift roster changed")
    endpoint_names = tuple(protocol["endpoints"]["trajectory"])
    parameter_names = tuple(protocol["endpoints"]["physical_parameters"])
    if endpoint_names != ENDPOINT_NAMES:
        raise ValueError("trajectory endpoint roster changed")
    if parameter_names != PARAMETER_NAMES:
        raise ValueError("physical parameter endpoint roster changed")
    thresholds = protocol["latent_decisions"]
    if thresholds != {
        "stiffness_at_least": 400.0,
        "damping_at_least": 2.0,
        "probability_threshold": 0.5,
    }:
        raise ValueError("latent decision thresholds changed")
    boundary = protocol["information_boundary"]
    required_false = (
        "twist_free_marker_outcomes_used_for_prediction",
        "query_partition_fit_to_twist_outcomes",
        "fresh_confirmation_authorized",
        "paper_claim_authorized",
    )
    if any(boundary.get(name) is not False for name in required_false):
        raise ValueError("protocol widens a closed information boundary")
    if boundary.get("twist_outcomes_used_for_scoring_only") is not True:
        raise ValueError("twist outcomes must remain scoring-only")


def query_class_index(
    final_displacement: object,
    *,
    class_count: int = 3,
) -> np.ndarray:
    """Create deterministic equal-count finite-diameter query classes."""

    values = np.asarray(final_displacement, dtype=np.float64)
    if values.ndim != 1 or values.size < class_count:
        raise ValueError(
            "query values must be a vector with at least one member per class"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("query values must be finite")
    if class_count < 2 or values.size % class_count != 0:
        raise ValueError("hypothesis count must divide evenly into query classes")
    order = np.argsort(values, kind="stable")
    size = values.size // class_count
    classes = np.empty(values.size, dtype=np.int64)
    for class_id in range(class_count):
        classes[order[class_id * size : (class_id + 1) * size]] = class_id
    return classes


def trajectory_endpoints(
    bank: object,
    *,
    cutoff: int,
    corners: object,
) -> tuple[np.ndarray, int]:
    """Return final, middle, and peak free-marker RMS displacement."""

    trajectories = np.asarray(bank, dtype=np.float64)
    corner_index = np.asarray(corners, dtype=np.int64)
    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError("bank must have shape (models, time, markers, 3)")
    if not np.all(np.isfinite(trajectories)):
        raise ValueError("prediction bank must be finite")
    if not 0 <= cutoff < trajectories.shape[1] - 1:
        raise ValueError("cutoff must leave at least one forecast frame")
    if corner_index.shape != (2,):
        raise ValueError("exactly two driven corners are required")
    free = np.ones(trajectories.shape[2], dtype=bool)
    free[corner_index] = False
    if not np.any(free):
        raise ValueError("no free markers remain")
    reference = trajectories[:, cutoff, free]
    delta = trajectories[:, cutoff + 1 :, free] - reference[:, None]
    rms = np.sqrt(np.mean(np.sum(np.square(delta), axis=-1), axis=-1))
    post_count = trajectories.shape[1] - cutoff - 1
    mid_offset = max(1, int(np.ceil(post_count / 2.0)))
    mid_offset = min(mid_offset, post_count)
    mid_index = cutoff + mid_offset
    endpoints = np.column_stack(
        (
            rms[:, -1],
            rms[:, mid_offset - 1],
            np.max(rms, axis=1),
        )
    )
    return endpoints, mid_index


def observed_trajectory_endpoints(
    truth: object,
    *,
    reference: object,
    cutoff: int,
    mid_index: int,
    corners: object,
) -> np.ndarray:
    """Measure the registered endpoints from scored target observations."""

    observed = np.asarray(truth, dtype=np.float64)
    causal_reference = np.asarray(reference, dtype=np.float64)
    corner_index = np.asarray(corners, dtype=np.int64)
    if observed.ndim != 3 or observed.shape[-1] != 3:
        raise ValueError("truth must have shape (time, markers, 3)")
    if causal_reference.shape != observed.shape[1:]:
        raise ValueError("reference shape differs from target marker state")
    if not cutoff < mid_index < observed.shape[0]:
        raise ValueError("midpoint is outside the forecast")
    free = np.ones(observed.shape[1], dtype=bool)
    free[corner_index] = False

    def rms_at(index: int) -> float:
        valid = free & np.all(np.isfinite(observed[index]), axis=1)
        if not np.any(valid):
            raise ValueError("target endpoint contains no finite free marker")
        difference = observed[index, valid] - causal_reference[valid]
        return float(np.sqrt(np.mean(np.sum(np.square(difference), axis=1))))

    post = []
    for index in range(cutoff + 1, observed.shape[0]):
        valid = free & np.all(np.isfinite(observed[index]), axis=1)
        if np.any(valid):
            difference = observed[index, valid] - causal_reference[valid]
            post.append(float(np.sqrt(np.mean(np.sum(np.square(difference), axis=1)))))
    if not post:
        raise ValueError("target trajectory contains no scored free-marker frame")
    return np.asarray((rms_at(observed.shape[0] - 1), rms_at(mid_index), max(post)))


def _concentrated_lift(
    quotient: np.ndarray,
    classes: np.ndarray,
    criterion: np.ndarray,
    *,
    choose_maximum: bool,
) -> np.ndarray:
    result = np.zeros(classes.size, dtype=np.float64)
    for class_id, mass in enumerate(quotient):
        members = np.flatnonzero(classes == class_id)
        values = criterion[members]
        selected = int(
            members[np.argmax(values) if choose_maximum else np.argmin(values)]
        )
        result[selected] = mass
    return result


def same_quotient_lifts(
    prior_weights: object,
    full_posterior_weights: object,
    class_index: object,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Construct the frozen complete-belief comparison roster."""

    prior = _probability(prior_weights, name="prior_weights")
    full = _probability(full_posterior_weights, name="full_posterior_weights")
    classes = np.asarray(class_index, dtype=np.int64)
    if classes.shape != prior.shape or full.shape != prior.shape:
        raise ValueError("weights and class_index must have the same shape")
    quotient = np.asarray(aggregate_to_query_quotient(full, classes))
    jeffrey = np.asarray(
        minimum_information_query_lift(prior, classes, quotient).lifted_weights
    )
    uniform = np.zeros_like(prior)
    for class_id, mass in enumerate(quotient):
        members = np.flatnonzero(classes == class_id)
        uniform[members] = mass / len(members)
    lifts = {
        "full_source_posterior": full,
        "jeffrey_i_projection": jeffrey,
        "uniform_within_class": uniform,
        "prior_map_concentration": _concentrated_lift(
            quotient,
            classes,
            prior,
            choose_maximum=True,
        ),
        "reverse_prior_concentration": _concentrated_lift(
            quotient,
            classes,
            prior,
            choose_maximum=False,
        ),
    }
    if tuple(lifts) != LIFT_NAMES:
        raise RuntimeError("complete lift roster changed")
    for name, weights in lifts.items():
        np.testing.assert_allclose(
            aggregate_to_query_quotient(weights, classes),
            quotient,
            atol=1.0e-12,
            rtol=0.0,
            err_msg=f"{name} does not preserve the registered quotient",
        )
    return lifts, quotient


def categorical_scores(
    probabilities: object,
    observed_class: int,
) -> dict[str, float | int]:
    values = _probability(probabilities, name="class_probabilities")
    if not 0 <= observed_class < values.size:
        raise ValueError("observed class is outside the quotient")
    one_hot = np.zeros(values.size, dtype=np.float64)
    one_hot[observed_class] = 1.0
    return {
        "nll": -float(np.log(max(values[observed_class], np.finfo(float).tiny))),
        "brier": float(np.sum(np.square(values - one_hot))),
        "correct": int(np.argmax(values) == observed_class),
    }


def _envelope_record(envelope: Any, names: Sequence[str]) -> dict[str, Any]:
    return {
        "names": list(names),
        "lower": np.asarray(envelope.lower).tolist(),
        "upper": np.asarray(envelope.upper).tolist(),
        "width": np.asarray(envelope.width).tolist(),
        "identified_mask": np.asarray(envelope.identified_mask).tolist(),
        "maximum_width": envelope.maximum_width,
    }


def _lift_records(
    prior: np.ndarray,
    classes: np.ndarray,
    lifts: Mapping[str, np.ndarray],
    endpoints: np.ndarray,
    parameters: np.ndarray,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in LIFT_NAMES:
        weights = lifts[name]
        information = query_quotient_information_decomposition(
            prior,
            weights,
            classes,
        )
        records[name] = {
            "weights": weights.tolist(),
            "quotient_weights": np.asarray(
                aggregate_to_query_quotient(weights, classes)
            ).tolist(),
            "total_information_nats": information.total_information_nats,
            "quotient_information_nats": information.quotient_information_nats,
            "unsupported_specificity_nats": (
                information.unsupported_specificity_nats
            ),
            "supported_information_fraction": (
                information.supported_information_fraction
            ),
            "expected_trajectory_endpoints": (weights @ endpoints).tolist(),
            "expected_physical_parameters": (weights @ parameters).tolist(),
        }
    return records


def _build_source_fit(
    source_records: Sequence[tuple[Case, Predictions, np.ndarray]],
    target_inputs: Sequence[tuple[Case, Predictions]],
    base_protocol: dict[str, Any],
) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    specimens: dict[str, Any] = {}
    for held_material in base_protocol["materials"]:
        fold, fold_specimens = fit_fold(
            held_material,
            source_records,
            target_inputs,
            base_protocol,
        )
        folds[held_material] = fold
        overlap = set(specimens) & set(fold_specimens)
        if overlap:
            raise ValueError(f"duplicate specimen states: {sorted(overlap)}")
        specimens.update(fold_specimens)
    if len(folds) != 4 or len(specimens) != 8:
        raise ValueError("incomplete leave-one-material-out source fit")
    return {
        "created_at": now(),
        "folds": folds,
        "specimens": specimens,
        "target_outcomes_used": False,
        "selection_protocol_frozen_before_prior_target_run": True,
        "runner_implementation_added_after_prior_target_run": True,
    }


def _predict(
    dataset_root: Path,
    output: Path,
    protocol: dict[str, Any],
    workers: int,
) -> None:
    validate_protocol(protocol)
    base_protocol = _load_json(BASE_HERE / "active_probe_protocol.json")
    validate_active_protocol(base_protocol)
    root = dataset_root.resolve(strict=True)
    destination = output.resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("output and dataset must be disjoint directory trees")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "protocol.json", protocol)
    write_json(destination / "base_protocol.json", base_protocol)
    write_json(
        destination / "run_manifest.json",
        {
            "created_at": now(),
            "stage": "prediction",
            "protocol_id": object_digest(protocol),
            "base_protocol_id": object_digest(base_protocol),
            "implementation_sha256": implementation(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "twist_free_marker_outcomes_read": False,
            "paper_claim_authorized": False,
        },
    )

    cases, inventory = audit_dataset(root, base_protocol)
    write_json(destination / "dataset_manifest.json", inventory)
    (destination / "DATA_LICENSE.txt").write_text(
        inventory["included_license_text"],
        encoding="utf-8",
    )
    source_cases = [case for case in cases if case.motion == "shake"]
    target_cases = [case for case in cases if case.motion == "twist"]
    scales = [
        infer_source_scale(
            case,
            read_prefix(case, float(base_protocol["prefix_seconds"]))[1],
        )
        for case in source_cases
    ]
    if len(set(scales)) != 1:
        raise ValueError("source recordings disagree about coordinate scale")
    scale = scales[0]
    source_records = map_records(
        source_record,
        [(case, base_protocol, scale) for case in source_cases],
        workers,
    )
    target_inputs = map_records(
        input_prediction,
        [(case, base_protocol, scale) for case in target_cases],
        workers,
    )
    source_fit = _build_source_fit(source_records, target_inputs, base_protocol)
    source_fit.update(
        {
            "protocol_id": object_digest(protocol),
            "base_protocol_id": object_digest(base_protocol),
            "inventory_id": inventory["inventory_id"],
            "coordinate_scale_to_m": scale,
        }
    )
    write_json(destination / "source_fit.json", source_fit)

    parameters = np.asarray(parameter_bank(base_protocol), dtype=np.float64)
    if parameters.shape != (9, 2):
        raise ValueError("the registered physical bank must contain nine hypotheses")
    class_count = int(protocol["query"]["class_count"])
    tolerance = float(protocol["query"]["identifiability_tolerance"])
    decision_values = np.column_stack(
        (
            parameters[:, 0]
            >= float(protocol["latent_decisions"]["stiffness_at_least"]),
            parameters[:, 1]
            >= float(protocol["latent_decisions"]["damping_at_least"]),
        )
    ).astype(np.float64)

    private = destination / "private_predictions"
    private.mkdir(mode=0o700)
    predictions: dict[str, Any] = {}
    for case, prediction in target_inputs:
        fold = source_fit["folds"][case.material]
        specimen = source_fit["specimens"][case.specimen]
        prior = _probability(fold["prior_weights"], name="fold prior")
        full = _probability(
            specimen["policy_states"]["task_directed"]["1"]["weights"],
            name="task-directed posterior",
        )
        endpoints, mid_index = trajectory_endpoints(
            prediction.bank,
            cutoff=int(prediction.inputs.cutoff),
            corners=prediction.inputs.corners,
        )
        classes = query_class_index(
            endpoints[:, 0],
            class_count=class_count,
        )
        lifts, quotient = same_quotient_lifts(prior, full, classes)
        prior_quotient = np.asarray(aggregate_to_query_quotient(prior, classes))
        centers = np.asarray(
            [
                np.mean(endpoints[classes == class_id, 0])
                for class_id in range(class_count)
            ],
            dtype=np.float64,
        )
        diameters = np.asarray(
            [
                np.ptp(endpoints[classes == class_id, 0])
                for class_id in range(class_count)
            ],
            dtype=np.float64,
        )
        trajectory_envelope = query_ambiguity_envelope(
            quotient,
            classes,
            endpoints,
            identifiability_tolerance=tolerance,
        )
        parameter_envelope = query_ambiguity_envelope(
            quotient,
            classes,
            parameters,
            identifiability_tolerance=tolerance,
        )
        decision_envelope = query_ambiguity_envelope(
            quotient,
            classes,
            decision_values,
            identifiability_tolerance=tolerance,
        )
        artifact = private / f"{case.path.stem}.npz"
        np.savez_compressed(
            artifact,
            bank=prediction.bank,
            times=prediction.inputs.times,
            order=prediction.inputs.order,
            corners=prediction.inputs.corners,
            cutoff=np.asarray(prediction.inputs.cutoff),
            scale=np.asarray(scale),
        )
        predictions[case.path.name] = {
            "artifact": str(artifact.relative_to(destination)),
            "sha256": digest(artifact),
            "specimen": case.specimen,
            "material": case.material,
            "size": case.size,
            "speed": case.speed,
            "grasp": case.grasp,
            "hypothesis_parameters": parameters.tolist(),
            "trajectory_endpoints": endpoints.tolist(),
            "endpoint_names": list(ENDPOINT_NAMES),
            "mid_index": mid_index,
            "query_class_index": classes.tolist(),
            "query_class_centers_m": centers.tolist(),
            "query_class_diameters_m": diameters.tolist(),
            "maximum_query_class_diameter_m": float(np.max(diameters)),
            "prior_weights": prior.tolist(),
            "full_source_posterior_weights": full.tolist(),
            "prior_quotient_weights": prior_quotient.tolist(),
            "posterior_quotient_weights": quotient.tolist(),
            "lifts": _lift_records(
                prior,
                classes,
                lifts,
                endpoints,
                parameters,
            ),
            "trajectory_ambiguity_envelope": _envelope_record(
                trajectory_envelope,
                ENDPOINT_NAMES,
            ),
            "parameter_ambiguity_envelope": _envelope_record(
                parameter_envelope,
                PARAMETER_NAMES,
            ),
            "latent_decision_ambiguity_envelope": _envelope_record(
                decision_envelope,
                ("stiffness_at_least_threshold", "damping_at_least_threshold"),
            ),
            "query_partition_uses_twist_outcomes": False,
        }
    if len(predictions) != 32:
        raise ValueError("refusing an incomplete 32-recording target seal")

    seal = {
        "sealed_at": now(),
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "TrackingClothQueryQuotientPredictionSealV1",
        "protocol_id": object_digest(protocol),
        "base_protocol_id": object_digest(base_protocol),
        "inventory_id": inventory["inventory_id"],
        "source_fit_sha256": digest(destination / "source_fit.json"),
        "implementation_sha256": implementation(),
        "hypothesis_count": 9,
        "query_class_count": class_count,
        "comparison_lifts": list(LIFT_NAMES),
        "predictions": predictions,
        "twist_free_marker_outcomes_read": False,
        "prior_target_outcome_exposure": True,
        "fresh_confirmation_claim": False,
        "paper_claim_authorized": False,
    }
    write_json(destination / "prediction_seal.json", seal)
    report = (
        "# Tracking Cloth query-quotient prediction seal\n\n"
        "The complete public archive and 120 extracted CSV files were verified. "
        "Shaking-source outcomes supplied leave-one-material-out finite-model "
        "beliefs. Target twisting inputs supplied only the causal prefix and "
        "recorded corner boundary. All 32 twisting prediction banks, query "
        "partitions, quotient posteriors, complete lifts, and ambiguity envelopes "
        "were sealed before twisting free-marker outcomes were opened.\n\n"
        "The three classes are finite-diameter modeled-response classes, not "
        "claims of exact material identity. This is retrospective public-data "
        "evidence and cannot self-authorize a paper claim.\n"
    )
    (destination / "report.md").write_text(report, encoding="utf-8")


def _inputs_from_arrays(case: Case, arrays: Mapping[str, np.ndarray]) -> Inputs:
    return Inputs(
        np.asarray(arrays["times"]),
        np.empty((0, case.markers, 3)),
        np.empty((0, 2, 3)),
        np.asarray(arrays["order"]),
        np.asarray(arrays["corners"]),
        int(np.asarray(arrays["cutoff"])),
        float(np.asarray(arrays["times"])[0]),
        float(np.asarray(arrays["scale"])),
    )


def _mean_by_specimen(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    specimens = sorted({str(row["specimen"]) for row in rows})
    if len(specimens) != 8 or len(rows) != 32:
        raise ValueError("expected 32 target rows across eight specimens")
    numeric = [
        key
        for key, value in rows[0].items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    result = []
    for specimen in specimens:
        subset = [row for row in rows if row["specimen"] == specimen]
        if len(subset) != 4:
            raise ValueError("each specimen must contain four twist conditions")
        result.append(
            {
                "specimen": specimen,
                "material": specimen.split("_", maxsplit=1)[0],
                **{
                    key: float(np.mean([float(row[key]) for row in subset]))
                    for key in numeric
                },
            }
        )
    return result


def _interval(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> list[float]:
    if values.shape != (8,):
        raise ValueError("bootstrap endpoint requires eight specimen values")
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    return np.quantile(values[indices].mean(axis=1), (0.025, 0.975)).tolist()


def _aggregate(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specimens = _mean_by_specimen(rows)
    numeric = [
        key
        for key, value in specimens[0].items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    means = {
        key: float(np.mean([float(row[key]) for row in specimens]))
        for key in numeric
    }
    rng = np.random.default_rng(int(protocol["analysis"]["bootstrap_seed"]))
    repetitions = int(protocol["analysis"]["bootstrap_repetitions"])
    contrasts = {}
    for name, candidate, comparator in (
        (
            "posterior_minus_prior_query_nll",
            "posterior_query_nll",
            "prior_query_nll",
        ),
        (
            "posterior_minus_prior_query_brier",
            "posterior_query_brier",
            "prior_query_brier",
        ),
        (
            "jeffrey_minus_full_final_absolute_error_mm",
            "jeffrey_i_projection_final_rms_displacement_absolute_error_mm",
            "full_source_posterior_final_rms_displacement_absolute_error_mm",
        ),
    ):
        values = np.asarray(
            [float(row[candidate]) - float(row[comparator]) for row in specimens]
        )
        contrasts[name] = {
            "mean": float(np.mean(values)),
            "specimen_bootstrap_95_interval": _interval(
                values,
                rng,
                repetitions,
            ),
            "specimen_wins": int(np.sum(values < 0.0)),
            "specimen_ties": int(np.sum(values == 0.0)),
            "specimen_losses": int(np.sum(values > 0.0)),
            "worst_specimen_regret": float(np.max(values)),
        }
    return specimens, {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "recording_count": 32,
        "specimen_count": 8,
        "specimen_balanced_means": means,
        "contrasts": contrasts,
        "mechanism_checks": {
            "all_complete_lifts_preserved_quotient": bool(
                all(bool(row["same_quotient_verified"]) for row in rows)
            ),
            "jeffrey_unsupported_specificity_numerically_zero": bool(
                max(
                    abs(
                        float(
                            row[
                                "jeffrey_i_projection_unsupported_specificity_nats"
                            ]
                        )
                    )
                    for row in rows
                )
                <= 1.0e-10
            ),
            "full_source_posterior_adds_within_class_information_fraction": float(
                np.mean(
                    [
                        float(
                            row[
                                "full_source_posterior_unsupported_specificity_nats"
                            ]
                        )
                        > 1.0e-10
                        for row in rows
                    ]
                )
            ),
            "latent_stiffness_decision_ambiguous_fraction": means[
                "stiffness_decision_ambiguous"
            ],
            "latent_damping_decision_ambiguous_fraction": means[
                "damping_decision_ambiguous"
            ],
            "complete_lift_decision_disagreement_fraction": means[
                "complete_lift_decision_disagreement"
            ],
        },
        "query_evidence": {
            "posterior_nll_better_than_prior": bool(
                means["posterior_query_nll"] < means["prior_query_nll"]
            ),
            "posterior_brier_better_than_prior": bool(
                means["posterior_query_brier"] < means["prior_query_brier"]
            ),
            "posterior_accuracy_not_worse_than_prior": bool(
                means["posterior_query_correct"] >= means["prior_query_correct"]
            ),
        },
        "inferential_unit": (
            "eight material-size specimens; four twist conditions averaged "
            "within specimen before cross-specimen inference"
        ),
        "evidence_class": protocol["evidence_class"],
        "fresh_confirmation_authorized": False,
        "paper_claim_authorized": False,
    }


def _score(dataset_root: Path, output: Path) -> None:
    root = dataset_root.resolve(strict=True)
    destination = output.resolve(strict=True)
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("output and dataset must be disjoint directory trees")
    if (destination / "target_access.json").exists():
        raise ValueError("target scoring already started for this seal")
    protocol = _load_json(destination / "protocol.json")
    validate_protocol(protocol)
    base_protocol = _load_json(destination / "base_protocol.json")
    validate_active_protocol(base_protocol)
    source_fit = _load_json(destination / "source_fit.json")
    seal = _load_json(destination / "prediction_seal.json")
    if seal["protocol_id"] != object_digest(protocol):
        raise ValueError("protocol changed after prediction sealing")
    if seal["base_protocol_id"] != object_digest(base_protocol):
        raise ValueError("base protocol changed after prediction sealing")
    if seal["implementation_sha256"] != implementation():
        raise ValueError("implementation changed after prediction sealing")
    if seal["source_fit_sha256"] != digest(destination / "source_fit.json"):
        raise ValueError("source fit changed after prediction sealing")
    cases, inventory = audit_dataset(root, base_protocol)
    if inventory["inventory_id"] != seal["inventory_id"]:
        raise ValueError("dataset changed after prediction sealing")
    for entry in seal["predictions"].values():
        path = (destination / entry["artifact"]).resolve()
        if (
            not path.is_relative_to((destination / "private_predictions").resolve())
            or digest(path) != entry["sha256"]
        ):
            raise ValueError("sealed prediction bank identity mismatch")

    write_json(
        destination / "target_access.json",
        {
            "started_at": now(),
            "prediction_seal_sha256": digest(destination / "prediction_seal.json"),
            "authorized_recordings": sorted(seal["predictions"]),
            "purpose": (
                "retrospective public-data query-quotient validation; "
                "target outcomes used for scoring only"
            ),
            "fresh_confirmation_claim": False,
            "paper_claim_authorized": False,
        },
    )
    by_name = {case.path.name: case for case in cases if case.motion == "twist"}
    rows: list[dict[str, Any]] = []
    probability_threshold = float(
        protocol["latent_decisions"]["probability_threshold"]
    )
    for recording, entry in seal["predictions"].items():
        case = by_name[recording]
        with np.load(destination / entry["artifact"], allow_pickle=False) as arrays:
            inputs = _inputs_from_arrays(case, arrays)
            truth = scoring_view(case, inputs)
            bank = np.asarray(arrays["bank"], dtype=np.float64)
            endpoints, mid_index = trajectory_endpoints(
                bank,
                cutoff=int(inputs.cutoff),
                corners=inputs.corners,
            )
            if mid_index != int(entry["mid_index"]):
                raise ValueError("registered midpoint changed after sealing")
            reference = bank[0, int(inputs.cutoff)]
            observed = observed_trajectory_endpoints(
                truth,
                reference=reference,
                cutoff=int(inputs.cutoff),
                mid_index=mid_index,
                corners=inputs.corners,
            )
            classes = np.asarray(entry["query_class_index"], dtype=np.int64)
            centers = np.asarray(entry["query_class_centers_m"], dtype=np.float64)
            observed_class = int(np.argmin(np.abs(centers - observed[0])))
            prior_quotient = np.asarray(
                entry["prior_quotient_weights"], dtype=np.float64
            )
            posterior_quotient = np.asarray(
                entry["posterior_quotient_weights"], dtype=np.float64
            )
            prior_scores = categorical_scores(prior_quotient, observed_class)
            posterior_scores = categorical_scores(
                posterior_quotient,
                observed_class,
            )
            lift_quotients = []
            lift_decisions: list[tuple[bool, bool]] = []
            lift_metrics: dict[str, float] = {}
            parameters = np.asarray(
                entry["hypothesis_parameters"], dtype=np.float64
            )
            for name in LIFT_NAMES:
                lift = entry["lifts"][name]
                weights = _probability(lift["weights"], name=f"{name} weights")
                quotient = np.asarray(
                    aggregate_to_query_quotient(weights, classes)
                )
                lift_quotients.append(quotient)
                np.testing.assert_allclose(
                    quotient,
                    posterior_quotient,
                    atol=1.0e-12,
                    rtol=0.0,
                )
                expected_endpoints = weights @ endpoints
                saved_endpoints = np.asarray(
                    lift["expected_trajectory_endpoints"], dtype=np.float64
                )
                np.testing.assert_allclose(
                    expected_endpoints,
                    saved_endpoints,
                    atol=1.0e-12,
                    rtol=0.0,
                )
                expected_parameters = weights @ parameters
                saved_parameters = np.asarray(
                    lift["expected_physical_parameters"], dtype=np.float64
                )
                np.testing.assert_allclose(
                    expected_parameters,
                    saved_parameters,
                    atol=1.0e-12,
                    rtol=0.0,
                )
                lift_decisions.append(
                    (
                        bool(
                            expected_parameters[0]
                            >= float(
                                protocol["latent_decisions"]["stiffness_at_least"]
                            )
                        ),
                        bool(
                            expected_parameters[1]
                            >= float(
                                protocol["latent_decisions"]["damping_at_least"]
                            )
                        ),
                    )
                )
                for endpoint_index, endpoint_name in enumerate(ENDPOINT_NAMES):
                    short = endpoint_name.removesuffix("_m")
                    lift_metrics[
                        f"{name}_{short}_absolute_error_mm"
                    ] = 1000.0 * abs(
                        float(
                            expected_endpoints[endpoint_index]
                            - observed[endpoint_index]
                        )
                    )
                lift_metrics[f"{name}_unsupported_specificity_nats"] = float(
                    lift["unsupported_specificity_nats"]
                )

            same_quotient = bool(
                all(
                    np.allclose(
                        quotient,
                        posterior_quotient,
                        atol=1.0e-12,
                        rtol=0.0,
                    )
                    for quotient in lift_quotients
                )
            )
            trajectory_envelope = entry["trajectory_ambiguity_envelope"]
            lower = np.asarray(trajectory_envelope["lower"], dtype=np.float64)
            upper = np.asarray(trajectory_envelope["upper"], dtype=np.float64)
            width = np.asarray(trajectory_envelope["width"], dtype=np.float64)
            parameter_envelope = entry["parameter_ambiguity_envelope"]
            parameter_width = np.asarray(
                parameter_envelope["width"], dtype=np.float64
            )
            decision_envelope = entry["latent_decision_ambiguity_envelope"]
            decision_lower = np.asarray(
                decision_envelope["lower"], dtype=np.float64
            )
            decision_upper = np.asarray(
                decision_envelope["upper"], dtype=np.float64
            )
            decision_ambiguous = (
                (decision_lower < probability_threshold)
                & (decision_upper >= probability_threshold)
            )
            rows.append(
                {
                    "recording": recording,
                    "specimen": case.specimen,
                    "material": case.material,
                    "size": case.size,
                    "speed": case.speed,
                    "grasp": case.grasp,
                    "observed_class": observed_class,
                    "prior_query_nll": prior_scores["nll"],
                    "posterior_query_nll": posterior_scores["nll"],
                    "prior_query_brier": prior_scores["brier"],
                    "posterior_query_brier": posterior_scores["brier"],
                    "prior_query_correct": prior_scores["correct"],
                    "posterior_query_correct": posterior_scores["correct"],
                    "same_quotient_verified": int(same_quotient),
                    "maximum_query_class_diameter_mm": (
                        1000.0 * float(entry["maximum_query_class_diameter_m"])
                    ),
                    "final_query_envelope_width_mm": 1000.0 * float(width[0]),
                    "mid_query_envelope_width_mm": 1000.0 * float(width[1]),
                    "peak_query_envelope_width_mm": 1000.0 * float(width[2]),
                    "final_query_envelope_covers_observation": int(
                        lower[0] <= observed[0] <= upper[0]
                    ),
                    "mid_query_envelope_covers_observation": int(
                        lower[1] <= observed[1] <= upper[1]
                    ),
                    "peak_query_envelope_covers_observation": int(
                        lower[2] <= observed[2] <= upper[2]
                    ),
                    "stiffness_envelope_width": float(parameter_width[0]),
                    "damping_envelope_width": float(parameter_width[1]),
                    "stiffness_decision_ambiguous": int(decision_ambiguous[0]),
                    "damping_decision_ambiguous": int(decision_ambiguous[1]),
                    "complete_lift_decision_disagreement": int(
                        len(set(lift_decisions)) > 1
                    ),
                    **lift_metrics,
                }
            )

    specimens, metrics = _aggregate(rows, protocol)
    _save_csv(destination / "recording_scores.csv", rows)
    _save_csv(destination / "specimen_scores.csv", specimens)
    write_json(destination / "metrics.json", metrics)
    manifest = _load_json(destination / "run_manifest.json")
    manifest.update(
        {
            "completed_at": now(),
            "stage": "completed",
            "target_numeric_outcomes_read": True,
            "prediction_seal_sha256": digest(destination / "prediction_seal.json"),
            "metrics_sha256": digest(destination / "metrics.json"),
            "status": "completed-retrospective-public-data-diagnostic",
            "paper_claim_authorized": False,
        }
    )
    write_json(destination / "run_manifest.json", manifest)

    means = metrics["specimen_balanced_means"]
    report = (destination / "report.md").read_text(encoding="utf-8")
    report += "\n## Held-out twist query results\n\n"
    report += (
        "| Metric | Prior / comparator | Posterior / candidate |\n"
        "| --- | ---: | ---: |\n"
        f"| Categorical query NLL | {means['prior_query_nll']:.4f} | "
        f"{means['posterior_query_nll']:.4f} |\n"
        f"| Categorical query Brier | {means['prior_query_brier']:.4f} | "
        f"{means['posterior_query_brier']:.4f} |\n"
        f"| Categorical query accuracy | "
        f"{100 * means['prior_query_correct']:.2f}% | "
        f"{100 * means['posterior_query_correct']:.2f}% |\n"
    )
    report += "\n### Same-quotient complete beliefs\n\n"
    report += (
        "| Lift | Unsupported specificity [nats] | "
        "Final endpoint absolute error [mm] |\n"
        "| --- | ---: | ---: |\n"
    )
    for name in LIFT_NAMES:
        report += (
            f"| {name} | "
            f"{means[f'{name}_unsupported_specificity_nats']:.6f} | "
            f"{means[f'{name}_final_rms_displacement_absolute_error_mm']:.4f} |\n"
        )
    report += (
        "\nAll complete lifts have the same posterior quotient and therefore the "
        "same categorical query score. Their latent parameter expectations and "
        "continuous endpoint predictions can differ. The Jeffrey lift adds zero "
        "within-class KL information by construction; this does not imply that "
        "the full source posterior is false, only that its within-class detail is "
        "not licensed by the quotient alone.\n\n"
        f"The stiffness decision is quotient-ambiguous in "
        f"{100 * means['stiffness_decision_ambiguous']:.2f}% of specimens and "
        f"the damping decision in "
        f"{100 * means['damping_decision_ambiguous']:.2f}%. "
        f"Complete lifts disagree on at least one latent decision in "
        f"{100 * means['complete_lift_decision_disagreement']:.2f}%.\n\n"
        "The reconstructed marker trajectories are public real measurements, "
        "but the experiment is retrospective and the simple spring bank is a "
        "registered diagnostic model. No fresh-confirmation, unique material "
        "identification, online-control, safety, or state-of-the-art claim is "
        "authorized.\n"
    )
    (destination / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--stage", choices=("predict", "score"), default="predict")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    try:
        if args.stage == "score":
            _score(args.dataset_root, args.output)
        else:
            protocol = _load_json(args.protocol)
            _predict(args.dataset_root, args.output, protocol, args.workers)
    except Exception as error:
        if args.output.is_dir() and not args.output.resolve().is_relative_to(
            args.dataset_root.resolve()
        ):
            write_json(
                args.output / "failure.json",
                {
                    "failed_at": now(),
                    "stage": args.stage,
                    "exception": type(error).__name__,
                    "message": str(error),
                    "target_scoring_started": (
                        args.output / "target_access.json"
                    ).exists(),
                    "scientific_decision": "incomplete; no claim",
                    "paper_claim_authorized": False,
                },
            )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
