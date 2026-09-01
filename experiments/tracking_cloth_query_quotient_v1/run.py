"""Run a sealed Tracking Cloth query-quotient public-data evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.query_quotient_belief_v1 import (
    aggregate_to_query_quotient,
    query_ambiguity_envelope,
    query_quotient_information_decomposition,
)
from experiments.tracking_cloth_deformation_v1.data import (
    Inputs,
    audit_dataset,
    digest,
    infer_source_scale,
    input_view,
    object_digest,
    read_prefix,
    scoring_view,
)
from experiments.tracking_cloth_deformation_v1.model import (
    masks,
    parameter_bank,
    predict,
    squared_error,
)

from .core import (
    SAME_QUOTIENT_LIFTS,
    categorical_scores,
    centered_shape_rms_m,
    jeffrey_control_lift,
    parameter_expectations,
    prior_aware_source_posterior,
    query_partition,
    registered_prior,
    same_quotient_lifts,
    trajectory_energy_score_mm,
    trajectory_mask,
    trajectory_rmse_mm,
)

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
BASE_HERE = HERE.parent / "tracking_cloth_deformation_v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/tracking-cloth-query-quotient-v1"
RESULT_SCHEMA = "bayesian-phystwin/tracking-cloth-query-quotient-result-v1"
ALL_ARMS = (
    "prior_physical_belief",
    "nominal_physics",
    *SAME_QUOTIENT_LIFTS,
    "wrong_specimen_jeffrey",
    "identity_permuted_jeffrey",
)
METRICS = (
    "query_log_score_nats",
    "query_brier_score",
    "query_class_correct",
    "continuous_query_abs_error_mm",
    "trajectory_energy_score_mm",
    "trajectory_rmse_mm",
    "unsupported_specificity_nats",
    "quotient_information_nats",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing an empty table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_identity() -> dict[str, str]:
    paths = [
        *sorted(HERE.glob("*.py")),
        BASE_HERE / "data.py",
        BASE_HERE / "model.py",
        REPOSITORY
        / "src"
        / "bayesian_phystwin"
        / "query_quotient_belief_v1.py",
    ]
    return {
        path.relative_to(REPOSITORY).as_posix(): file_sha256(path)
        for path in paths
    }


def load_protocols(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_version") != 1
    ):
        raise ValueError("unexpected query-quotient protocol")
    base_path = REPOSITORY / protocol["base_protocol_path"]
    if file_sha256(base_path) != protocol["base_protocol_sha256"]:
        raise ValueError("base Tracking Cloth protocol changed")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    boundary = protocol.get("information_boundary", {})
    if not (
        boundary.get("public_real_measurements") is True
        and boundary.get("source_motion_only_for_belief_update") is True
        and boundary.get(
            "target_prefix_and_prescribed_corners_only_before_seal"
        )
        is True
    ):
        raise ValueError("required information boundary is missing")
    for key in (
        "target_outcomes_used_for_selection",
        "raw_data_upload",
        "fresh_confirmation_authorized",
        "paper_claim_authorized",
    ):
        if boundary.get(key) is not False:
            raise ValueError("protocol widens a closed information boundary")
    if protocol["query"]["requested_class_count"] != 3:
        raise ValueError("the query must request three regimes")
    permutation = protocol["controls"]["hypothesis_identity_permutation"]
    if sorted(permutation) != list(range(9)):
        raise ValueError("identity control is not a 9-member permutation")
    if registered_prior(protocol).size != len(parameter_bank(base)):
        raise ValueError("prior and parameter bank disagree")
    return protocol, base


def _source_loss(task: tuple[Any, dict[str, Any], float]) -> dict[str, Any]:
    case, base, scale = task
    inputs = input_view(case, base, scale)
    predictions = predict(inputs, base)
    truth = scoring_view(case, inputs)
    valid = masks(inputs, truth)
    return {
        "specimen": case.specimen,
        "recording": case.path.name,
        "losses_m2": [
            squared_error(member, truth, valid)
            for member in predictions.bank
        ],
    }


def fit_sources(
    cases: list[Any],
    protocol: dict[str, Any],
    base: dict[str, Any],
    scale: float,
    workers: int,
) -> dict[str, Any]:
    source = [case for case in cases if case.motion == base["source_motion"]]
    tasks = [(case, base, scale) for case in source]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_source_loss, tasks))
    else:
        records = [_source_loss(task) for task in tasks]
    prior = registered_prior(protocol)
    specimens: dict[str, Any] = {}
    for specimen in sorted({row["specimen"] for row in records}):
        subset = sorted(
            [row for row in records if row["specimen"] == specimen],
            key=lambda row: row["recording"],
        )
        if len(subset) != 4:
            raise ValueError("each specimen requires four source records")
        losses = np.asarray([row["losses_m2"] for row in subset])
        posterior, temperature = prior_aware_source_posterior(
            losses,
            prior,
            measurement_floor_m=float(base["measurement_floor_m"]),
        )
        specimens[specimen] = {
            "recordings": [row["recording"] for row in subset],
            "losses_m2": losses.tolist(),
            "temperature_m2": temperature,
            "source_posterior_weights": posterior.tolist(),
            "target_outcomes_used": False,
        }
    if len(specimens) != 8:
        raise ValueError("the full eight-specimen source roster is required")
    return {
        "created_at": now(),
        "coordinate_scale_to_m": scale,
        "prior_weights": prior.tolist(),
        "specimens": specimens,
        "source_recording_count": len(records),
        "target_outcomes_used": False,
    }


def parameter_arrays(base: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bank = parameter_bank(base)
    return (
        np.asarray([item[0] for item in bank]),
        np.asarray([item[1] for item in bank]),
    )


def target_weights(
    prior: np.ndarray,
    posterior: np.ndarray,
    wrong_posterior: np.ndarray,
    classes: np.ndarray,
    protocol: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, np.ndarray]:
    nominal = np.zeros_like(prior)
    nominal[
        parameter_bank(base).index(tuple(base["nominal_parameters"]))
    ] = 1.0
    permuted = posterior[
        np.asarray(
            protocol["controls"]["hypothesis_identity_permutation"],
            dtype=np.int64,
        )
    ]
    result = {
        "prior_physical_belief": prior,
        "nominal_physics": nominal,
        **same_quotient_lifts(prior, posterior, classes),
        "wrong_specimen_jeffrey": jeffrey_control_lift(
            prior, wrong_posterior, classes
        ),
        "identity_permuted_jeffrey": jeffrey_control_lift(
            prior, permuted, classes
        ),
    }
    if tuple(result) != ALL_ARMS:
        raise RuntimeError("arm order changed")
    return result


def prepare(
    dataset_root: Path,
    output: Path,
    protocol_path: Path,
    workers: int,
) -> None:
    protocol, base = load_protocols(protocol_path)
    root = dataset_root.resolve(strict=True)
    destination = output.resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("output and data must be disjoint")
    destination.mkdir(parents=True, exist_ok=False)
    private = destination / "private_predictions"
    private.mkdir(mode=0o700)
    write_json(destination / "protocol.json", protocol)
    write_json(destination / "base_protocol.json", base)
    write_json(
        destination / "run_manifest.json",
        {
            "created_at": now(),
            "stage": "prepare",
            "repository_revision": os.environ.get("GITHUB_SHA"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_sha256": implementation_identity(),
            "target_numeric_outcomes_read": False,
            "paper_claim_authorized": False,
        },
    )

    cases, inventory = audit_dataset(root, base)
    write_json(destination / "dataset_manifest.json", inventory)
    (destination / "DATA_LICENSE.txt").write_text(
        inventory["included_license_text"], encoding="utf-8"
    )
    source = [case for case in cases if case.motion == base["source_motion"]]
    scales = [
        infer_source_scale(
            case,
            read_prefix(case, float(base["prefix_seconds"]))[1],
        )
        for case in source
    ]
    if len(set(scales)) != 1:
        raise ValueError("source coordinate scales disagree")
    scale = scales[0]
    fit = fit_sources(cases, protocol, base, scale, workers)
    write_json(destination / "source_fit.json", fit)

    prior = np.asarray(fit["prior_weights"])
    stiffness, damping = parameter_arrays(base)
    specimen_order = sorted(fit["specimens"])
    rotation = int(protocol["controls"]["wrong_specimen_rotation"])
    records: dict[str, Any] = {}
    targets = [case for case in cases if case.motion == base["target_motion"]]
    for case in targets:
        inputs = input_view(case, base, scale)
        predictions = predict(inputs, base)
        values = np.asarray(
            [
                centered_shape_rms_m(
                    member,
                    cutoff=inputs.cutoff,
                    corners=inputs.corners,
                    tail_fraction=float(protocol["query"]["tail_fraction"]),
                )
                for member in predictions.bank
            ]
        )
        classes, thresholds = query_partition(
            values,
            requested_class_count=int(
                protocol["query"]["requested_class_count"]
            ),
            minimum_gap_m=float(protocol["query"]["minimum_gap_m"]),
        )
        posterior = np.asarray(
            fit["specimens"][case.specimen]["source_posterior_weights"]
        )
        index = specimen_order.index(case.specimen)
        wrong_specimen = specimen_order[
            (index + rotation) % len(specimen_order)
        ]
        wrong = np.asarray(
            fit["specimens"][wrong_specimen]["source_posterior_weights"]
        )
        weights = target_weights(
            prior, posterior, wrong, classes, protocol, base
        )
        quotient = aggregate_to_query_quotient(posterior, classes)
        mismatch = max(
            float(
                np.sum(
                    np.abs(
                        aggregate_to_query_quotient(
                            weights[name], classes
                        )
                        - quotient
                    )
                )
            )
            for name in SAME_QUOTIENT_LIFTS
        )
        if mismatch > 1e-12:
            raise RuntimeError("same-quotient lift mismatch")

        artifact = private / f"{case.path.stem}.npz"
        np.savez_compressed(
            artifact,
            bank=predictions.bank,
            times=inputs.times,
            prefix=inputs.prefix,
            boundary=inputs.boundary,
            order=inputs.order,
            corners=inputs.corners,
            cutoff=np.asarray(inputs.cutoff),
            scale=np.asarray(inputs.scale),
        )
        records[case.path.name] = {
            "artifact": str(artifact.relative_to(destination)),
            "artifact_sha256": digest(artifact),
            "specimen": case.specimen,
            "material": case.material,
            "condition": case.condition,
            "wrong_specimen": wrong_specimen,
            "query_values_m": values.tolist(),
            "query_thresholds_m": thresholds.tolist(),
            "class_index": classes.tolist(),
            "posterior_quotient_weights": np.asarray(quotient).tolist(),
            "weights": {
                name: np.asarray(value).tolist()
                for name, value in weights.items()
            },
            "pretarget_ambiguity": {
                "query_m": _envelope(quotient, classes, values),
                "stiffness_per_mass": _envelope(
                    quotient, classes, stiffness
                ),
                "damping_per_mass": _envelope(
                    quotient, classes, damping
                ),
            },
            "same_quotient_max_l1": mismatch,
            "future_free_marker_outcomes_read": False,
        }
    if len(records) != 32:
        raise ValueError("refusing an incomplete target seal")
    seal = {
        "sealed_at": now(),
        "protocol_id": object_digest(protocol),
        "base_protocol_id": object_digest(base),
        "inventory_id": inventory["inventory_id"],
        "source_fit_sha256": digest(destination / "source_fit.json"),
        "implementation_sha256": implementation_identity(),
        "targets": records,
        "target_count": 32,
        "future_free_marker_outcomes_read": False,
        "paper_claim_authorized": False,
    }
    write_json(destination / "prediction_seal.json", seal)
    (destination / "report.md").write_text(
        "# Tracking Cloth query-quotient real-data evaluation\n\n"
        "Thirty-two shake recordings updated eight specimen beliefs. "
        "Thirty-two twist predictions were sealed before future free-marker "
        "outcomes were read. The query is a target-action-specific low/mid/high "
        "tail deformation regime over the frozen nine-member parameter bank.\n\n"
        "This is retrospective public-data evidence, not fresh confirmation. "
        "No paper claim is automatically authorized.\n",
        encoding="utf-8",
    )


def _envelope(
    quotient: np.ndarray, classes: np.ndarray, values: np.ndarray
) -> dict[str, float]:
    result = query_ambiguity_envelope(quotient, classes, values)
    return {
        "lower": float(result.lower[0]),
        "upper": float(result.upper[0]),
        "width": float(result.width[0]),
    }


def bootstrap_interval(
    differences: np.ndarray, repetitions: int, seed: int
) -> list[float]:
    if differences.shape != (8,):
        raise ValueError("eight specimen differences are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 8, size=(repetitions, 8))
    return np.quantile(
        differences[indices].mean(axis=1), [0.025, 0.975]
    ).tolist()


def aggregate_results(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specimens = sorted({str(row["specimen"]) for row in rows})
    if len(specimens) != 8:
        raise ValueError("eight specimens are required")
    specimen_rows = []
    for specimen in specimens:
        for arm in ALL_ARMS:
            subset = [
                row
                for row in rows
                if row["specimen"] == specimen and row["arm"] == arm
            ]
            if len(subset) != 4:
                raise ValueError("four target records per specimen are required")
            specimen_rows.append(
                {
                    "specimen": specimen,
                    "material": subset[0]["material"],
                    "arm": arm,
                    **{
                        metric: float(
                            np.mean([float(row[metric]) for row in subset])
                        )
                        for metric in METRICS
                    },
                }
            )
    arms = {
        arm: {
            metric: float(
                np.mean(
                    [
                        row[metric]
                        for row in specimen_rows
                        if row["arm"] == arm
                    ]
                )
            )
            for metric in METRICS
        }
        for arm in ALL_ARMS
    }
    comparisons = (
        ("jeffrey_i_projection", "prior_physical_belief"),
        ("full_source_posterior", "prior_physical_belief"),
        ("jeffrey_i_projection", "full_source_posterior"),
        ("jeffrey_i_projection", "wrong_specimen_jeffrey"),
        ("jeffrey_i_projection", "identity_permuted_jeffrey"),
    )
    comparison_metrics = (
        "query_log_score_nats",
        "query_brier_score",
        "trajectory_energy_score_mm",
        "trajectory_rmse_mm",
        "continuous_query_abs_error_mm",
    )
    contrasts: dict[str, Any] = {}
    for comparison_index, (left, right) in enumerate(comparisons):
        values: dict[str, Any] = {}
        for metric_index, metric in enumerate(comparison_metrics):
            differences = np.asarray(
                [
                    next(
                        row[metric]
                        for row in specimen_rows
                        if row["specimen"] == specimen
                        and row["arm"] == left
                    )
                    - next(
                        row[metric]
                        for row in specimen_rows
                        if row["specimen"] == specimen
                        and row["arm"] == right
                    )
                    for specimen in specimens
                ]
            )
            values[metric] = {
                "mean_difference": float(differences.mean()),
                "specimen_bootstrap_95_interval": bootstrap_interval(
                    differences,
                    int(protocol["analysis"]["bootstrap_repetitions"]),
                    int(protocol["analysis"]["bootstrap_seed"])
                    + 100 * comparison_index
                    + metric_index,
                ),
                "specimen_wins": int(np.sum(differences < 0.0)),
                "specimen_ties": int(np.sum(differences == 0.0)),
                "specimen_losses": int(np.sum(differences > 0.0)),
                "worst_specimen_regret": float(differences.max()),
            }
        contrasts[f"{left}_minus_{right}"] = values

    by_recording: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_recording.setdefault(str(row["recording"]), []).append(row)
    query_spread = max(
        max(
            float(row["query_log_score_nats"])
            for row in group
            if row["arm"] in SAME_QUOTIENT_LIFTS
        )
        - min(
            float(row["query_log_score_nats"])
            for row in group
            if row["arm"] in SAME_QUOTIENT_LIFTS
        )
        for group in by_recording.values()
    )
    jeffrey_specificity = max(
        float(
            next(
                row["unsupported_specificity_nats"]
                for row in group
                if row["arm"] == "jeffrey_i_projection"
            )
        )
        for group in by_recording.values()
    )
    specificity_tolerance = float(
        protocol["analysis"]["specificity_tolerance_nats"]
    )
    contract = {
        "target_cases": len(cases),
        "same_quotient_max_l1": max(
            float(case["same_quotient_max_l1"]) for case in cases
        ),
        "same_quotient_query_log_score_max_spread": query_spread,
        "jeffrey_max_unsupported_specificity_nats": jeffrey_specificity,
        "cases_with_positive_non_jeffrey_specificity": sum(
            any(
                float(row["unsupported_specificity_nats"])
                > specificity_tolerance
                for row in group
                if row["arm"]
                in (
                    "full_source_posterior",
                    "uniform_within_class",
                    "prior_map_within_class",
                    "prior_antimap_within_class",
                )
            )
            for group in by_recording.values()
        ),
        "cases_with_continuous_query_expectation_disagreement": sum(
            float(case["same_quotient_expected_query_range_m"])
            > float(
                protocol["analysis"]["query_expectation_tolerance_m"]
            )
            for case in cases
        ),
        "cases_with_stiffness_decision_disagreement": sum(
            bool(case["same_quotient_stiffness_decision_disagrees"])
            for case in cases
        ),
    }
    contract["same_quotient_contract_passed"] = bool(
        contract["target_cases"] == 32
        and contract["same_quotient_max_l1"] <= 1e-12
        and contract["same_quotient_query_log_score_max_spread"] <= 1e-12
        and contract["jeffrey_max_unsupported_specificity_nats"]
        <= specificity_tolerance
    )
    return specimen_rows, {
        "arms": arms,
        "contrasts": contrasts,
        "contract": contract,
        "inferential_unit": (
            "eight material-size specimens; four target conditions averaged "
            "within specimen"
        ),
        "interval_interpretation": (
            "exploratory paired percentile bootstrap over eight specimens; "
            "not simultaneous and not fresh confirmation"
        ),
    }


def score_run(
    dataset_root: Path, output: Path, protocol_path: Path
) -> None:
    protocol, base = load_protocols(protocol_path)
    root = dataset_root.resolve(strict=True)
    destination = output.resolve(strict=True)
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("output and data must be disjoint")
    if (destination / "target_access.json").exists():
        raise ValueError("target scoring already started")
    seal = json.loads(
        (destination / "prediction_seal.json").read_text(encoding="utf-8")
    )
    if (
        seal["protocol_id"] != object_digest(protocol)
        or seal["base_protocol_id"] != object_digest(base)
        or seal["implementation_sha256"] != implementation_identity()
        or seal["source_fit_sha256"]
        != digest(destination / "source_fit.json")
    ):
        raise ValueError("sealed implementation identity changed")

    all_cases, inventory = audit_dataset(root, base)
    if inventory["inventory_id"] != seal["inventory_id"]:
        raise ValueError("dataset changed after sealing")
    for entry in seal["targets"].values():
        artifact = (destination / entry["artifact"]).resolve()
        if (
            not artifact.is_relative_to(
                (destination / "private_predictions").resolve()
            )
            or digest(artifact) != entry["artifact_sha256"]
        ):
            raise ValueError("private prediction artifact changed")
    write_json(
        destination / "target_access.json",
        {
            "started_at": now(),
            "prediction_seal_sha256": digest(
                destination / "prediction_seal.json"
            ),
            "authorized_recordings": sorted(seal["targets"]),
        },
    )

    prior = registered_prior(protocol)
    stiffness, damping = parameter_arrays(base)
    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    targets = [
        case for case in all_cases if case.motion == base["target_motion"]
    ]
    for case in targets:
        entry = seal["targets"][case.path.name]
        with np.load(
            destination / entry["artifact"], allow_pickle=False
        ) as arrays:
            bank = np.asarray(arrays["bank"])
            inputs = Inputs(
                np.asarray(arrays["times"]),
                np.asarray(arrays["prefix"]),
                np.asarray(arrays["boundary"]),
                np.asarray(arrays["order"]),
                np.asarray(arrays["corners"]),
                int(np.asarray(arrays["cutoff"]).item()),
                float(np.asarray(arrays["times"])[0]),
                float(np.asarray(arrays["scale"]).item()),
            )
        truth = scoring_view(case, inputs)
        observed_query = centered_shape_rms_m(
            truth,
            cutoff=inputs.cutoff,
            corners=inputs.corners,
            tail_fraction=float(protocol["query"]["tail_fraction"]),
        )
        thresholds = np.asarray(entry["query_thresholds_m"])
        classes = np.asarray(entry["class_index"], dtype=np.int64)
        query_values = np.asarray(entry["query_values_m"])
        observed_class = int(
            np.searchsorted(thresholds, observed_query, side="right")
        )
        valid = trajectory_mask(
            truth,
            cutoff=inputs.cutoff,
            corners=inputs.corners,
            time_stride=int(
                protocol["scoring"]["trajectory_time_stride"]
            ),
        )
        correct_quotient = np.asarray(
            entry["posterior_quotient_weights"]
        )
        expected_queries = []
        stiffness_decisions = []
        for arm in ALL_ARMS:
            weights = np.asarray(entry["weights"][arm])
            quotient = aggregate_to_query_quotient(weights, classes)
            categorical = categorical_scores(
                quotient,
                observed_class,
                probability_floor=float(
                    protocol["scoring"]["probability_floor"]
                ),
            )
            information = query_quotient_information_decomposition(
                prior, weights, classes
            )
            parameter = parameter_expectations(
                weights, stiffness, damping
            )
            expected_query = float(weights @ query_values)
            row = {
                "recording": case.path.name,
                "specimen": case.specimen,
                "material": case.material,
                "size": case.size,
                "condition": case.condition,
                "speed": case.speed,
                "grasp": case.grasp,
                "arm": arm,
                "same_quotient_arm": arm in SAME_QUOTIENT_LIFTS,
                "observed_query_class": observed_class,
                "observed_continuous_query_m": observed_query,
                "expected_continuous_query_m": expected_query,
                "continuous_query_abs_error_mm": (
                    1000.0 * abs(expected_query - observed_query)
                ),
                "query_class_count": int(classes.max()) + 1,
                "query_probabilities_json": json.dumps(
                    np.asarray(quotient).tolist(),
                    separators=(",", ":"),
                ),
                "quotient_l1_vs_correct_source": float(
                    np.sum(np.abs(quotient - correct_quotient))
                ),
                "trajectory_energy_score_mm": trajectory_energy_score_mm(
                    bank, weights, truth, valid
                ),
                "trajectory_rmse_mm": trajectory_rmse_mm(
                    bank, weights, truth, valid
                ),
                "unsupported_specificity_nats": float(
                    information.unsupported_specificity_nats
                ),
                "quotient_information_nats": float(
                    information.quotient_information_nats
                ),
                **categorical,
                **parameter,
            }
            rows.append(row)
            if arm in SAME_QUOTIENT_LIFTS:
                expected_queries.append(expected_query)
                stiffness_decisions.append(
                    parameter["expected_stiffness_per_mass"]
                    > float(
                        protocol["analysis"][
                            "stiffness_decision_threshold"
                        ]
                    )
                )
        query_envelope = _envelope(
            correct_quotient, classes, query_values
        )
        case_summaries.append(
            {
                "recording": case.path.name,
                "specimen": case.specimen,
                "wrong_specimen": entry["wrong_specimen"],
                "observed_query_m": observed_query,
                "observed_query_class": observed_class,
                "query_thresholds_m": thresholds.tolist(),
                "hypothesis_query_values_m": query_values.tolist(),
                "class_index": classes.tolist(),
                "prior_quotient_weights": np.asarray(
                    aggregate_to_query_quotient(prior, classes)
                ).tolist(),
                "source_quotient_weights": correct_quotient.tolist(),
                "query_expectation_envelope_m": query_envelope,
                "same_quotient_expected_query_range_m": (
                    max(expected_queries) - min(expected_queries)
                ),
                "same_quotient_stiffness_decision_disagrees": (
                    len(set(stiffness_decisions)) > 1
                ),
                "same_quotient_max_l1": float(
                    entry["same_quotient_max_l1"]
                ),
            }
        )

    specimen_rows, metrics = aggregate_results(
        rows, case_summaries, protocol
    )
    primary = metrics["contrasts"][
        "jeffrey_i_projection_minus_prior_physical_belief"
    ]
    metrics.update(
        {
            "study_id": protocol["study_id"],
            "source_recordings": 32,
            "target_recordings": len(targets),
            "hypothesis_count": 9,
            "paper_claim_authorized": False,
            "fresh_confirmation_authorized": False,
            "decision": {
                "mechanism_contract_passed": metrics["contract"][
                    "same_quotient_contract_passed"
                ],
                "source_query_log_score_improved_vs_prior": (
                    primary["query_log_score_nats"]["mean_difference"]
                    < 0.0
                ),
                "source_query_brier_improved_vs_prior": (
                    primary["query_brier_score"]["mean_difference"]
                    < 0.0
                ),
                "jeffrey_trajectory_energy_improved_vs_prior": (
                    primary["trajectory_energy_score_mm"][
                        "mean_difference"
                    ]
                    < 0.0
                ),
                "paper_claim_authorized": False,
            },
        }
    )
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": object_digest(protocol),
        "prediction_seal_sha256": digest(
            destination / "prediction_seal.json"
        ),
        "metrics": metrics,
        "case_summaries": case_summaries,
        "information_boundary": {
            "public_real_measurements": True,
            "source_motion_only_for_belief_update": True,
            "target_predictions_sealed_before_free_marker_scoring": True,
            "raw_data_uploaded": False,
            "fresh_confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
    }
    validate_result(result)
    write_json(destination / "result.json", result)
    write_json(destination / "metrics.json", metrics)
    write_json(destination / "case_summaries.json", case_summaries)
    save_csv(destination / "case_arm_scores.csv", rows)
    save_csv(destination / "specimen_arm_scores.csv", specimen_rows)
    manifest = json.loads(
        (destination / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest.update(
        {
            "completed_at": now(),
            "stage": "completed",
            "target_numeric_outcomes_read": True,
            "result_sha256": digest(destination / "result.json"),
        }
    )
    write_json(destination / "run_manifest.json", manifest)
    write_report(destination, metrics)


def write_report(destination: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Tracking Cloth query-quotient real-data evaluation",
        "",
        "| Arm | Query log | Query Brier | Accuracy | Energy [mm] | "
        "RMSE [mm] | Specificity [nats] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ALL_ARMS:
        value = metrics["arms"][arm]
        lines.append(
            f"| {arm} | {value['query_log_score_nats']:.4f} | "
            f"{value['query_brier_score']:.4f} | "
            f"{100 * value['query_class_correct']:.2f}% | "
            f"{value['trajectory_energy_score_mm']:.4f} | "
            f"{value['trajectory_rmse_mm']:.4f} | "
            f"{value['unsupported_specificity_nats']:.6f} |"
        )
    primary = metrics["contrasts"][
        "jeffrey_i_projection_minus_prior_physical_belief"
    ]
    lines.extend(["", "## Jeffrey minus registered prior", ""])
    for metric in (
        "query_log_score_nats",
        "query_brier_score",
        "trajectory_energy_score_mm",
        "trajectory_rmse_mm",
    ):
        value = primary[metric]
        lines.append(
            f"- `{metric}`: `{value['mean_difference']:.6f}`, "
            f"95% specimen bootstrap "
            f"`{value['specimen_bootstrap_95_interval']}`, "
            f"W/T/L `{value['specimen_wins']}/"
            f"{value['specimen_ties']}/"
            f"{value['specimen_losses']}`."
        )
    contract = metrics["contract"]
    lines.extend(
        [
            "",
            "## Mechanism audit",
            "",
            f"- Same-quotient L1 mismatch: "
            f"`{contract['same_quotient_max_l1']:.3e}`.",
            f"- Same-quotient query-score spread: "
            f"`{contract['same_quotient_query_log_score_max_spread']:.3e}`.",
            f"- Jeffrey unsupported specificity: "
            f"`{contract['jeffrey_max_unsupported_specificity_nats']:.3e}`.",
            f"- Positive non-Jeffrey specificity: "
            f"`{contract['cases_with_positive_non_jeffrey_specificity']}/32`.",
            f"- Continuous-query expectation disagreement: "
            f"`{contract['cases_with_continuous_query_expectation_disagreement']}/32`.",
            f"- Stiffness-decision disagreement: "
            f"`{contract['cases_with_stiffness_decision_disagreement']}/32`.",
            f"- Contract passed: "
            f"`{contract['same_quotient_contract_passed']}`.",
            "",
            "All same-quotient lifts receive identical categorical-query scores. "
            "Their full-trajectory scores and latent expectations may differ. "
            "The Jeffrey lift is the zero-extra-specificity representative; it "
            "is not asserted to optimize held-out trajectory error.",
            "",
            "This is retrospective public-data evidence with eight specimen "
            "clusters and measured prescribed corner trajectories. It is not "
            "fresh confirmation, unique material identification, or a safety "
            "certificate. No paper claim is automatically authorized.",
        ]
    )
    (destination / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def validate_result(result: dict[str, Any]) -> None:
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("unexpected result schema")
    metrics = result.get("metrics", {})
    contract = metrics.get("contract", {})
    if metrics.get("target_recordings") != 32:
        raise ValueError("all 32 targets are required")
    if (
        contract.get("target_cases") != 32
        or contract.get("same_quotient_max_l1", 1.0) > 1e-12
        or contract.get(
            "same_quotient_query_log_score_max_spread", 1.0
        )
        > 1e-12
        or contract.get(
            "jeffrey_max_unsupported_specificity_nats", 1.0
        )
        > 1e-10
        or contract.get("same_quotient_contract_passed") is not True
    ):
        raise ValueError("query-quotient mechanism contract failed")
    boundary = result.get("information_boundary", {})
    if boundary.get("public_real_measurements") is not True:
        raise ValueError("result is not identified as public real data")
    for key in (
        "raw_data_uploaded",
        "fresh_confirmation_authorized",
        "paper_claim_authorized",
    ):
        if boundary.get(key) is not False:
            raise ValueError("result self-authorized or widened data handling")
    if metrics.get("paper_claim_authorized") is not False:
        raise ValueError("metrics self-authorized a paper claim")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "protocol.json"
    )
    parser.add_argument(
        "--stage", choices=("prepare", "score"), required=True
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    try:
        if args.stage == "prepare":
            prepare(
                args.dataset_root,
                args.output,
                args.protocol,
                args.workers,
            )
        else:
            score_run(args.dataset_root, args.output, args.protocol)
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
                    "scientific_decision": "incomplete; no paper claim",
                },
            )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
