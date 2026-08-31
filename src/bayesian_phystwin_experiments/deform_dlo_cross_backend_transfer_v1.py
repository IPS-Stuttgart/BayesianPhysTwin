"""Source-only no-refit coefficient transfer across DEFORM and PyElastica.

The evaluator tests whether local-residual models fitted against the DEFORM
physical backend improve sealed PyElastica source predictions without refitting
their coefficients.  It is deliberately source-only and retrospective: the
registered DLO3 source-test trajectories were opened in earlier studies.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

SCHEMA_VERSION = 1
CONTRACT = "deform-dlo3-cross-backend-coefficient-transfer-v1"
RESULT_CONTRACT = "deform-dlo3-cross-backend-coefficient-transfer-result-v1"


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_array(value: object, *, ndim: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {ndim}-D array")
    return array


def _identity(value: object, *, label: str) -> dict[str, object]:
    identity = _mapping(value, label=label)
    path = str(identity.get("path", ""))
    digest = str(identity.get("sha256", ""))
    size = int(cast(Any, identity.get("size_bytes", -1)))
    if not path or len(digest) != 64 or size <= 0:
        raise ValueError(f"{label} identity is invalid")
    return {"path": path, "sha256": digest, "size_bytes": size}


def load_cross_backend_transfer_protocol(path: str | Path) -> dict[str, object]:
    """Load and validate the frozen source-only transfer protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cross-backend protocol must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported cross-backend protocol schema")
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported cross-backend protocol contract")
    source_panel = _mapping(payload.get("source_panel"), label="source panel")
    if (
        source_panel.get("dataset") != "DEFORM-DLO3"
        or source_panel.get("partition") != "train/source_test"
        or int(cast(Any, source_panel.get("trajectory_count", -1))) != 8
        or int(cast(Any, source_panel.get("frame_count", -1))) != 500
        or int(cast(Any, source_panel.get("node_count", -1))) != 12
        or source_panel.get("official_evaluation_read") is not False
        or source_panel.get("source_outcomes_previously_opened") is not True
    ):
        raise ValueError("cross-backend source-panel boundary changed")
    artifacts = _mapping(payload.get("artifacts"), label="artifacts")
    _identity(artifacts.get("source_manifest"), label="source manifest")
    _identity(
        artifacts.get("pyelastica_source_predictions"),
        label="PyElastica source predictions",
    )
    raw_models = artifacts.get("deform_local_residual_models")
    if not isinstance(raw_models, Sequence) or isinstance(raw_models, (str, bytes)):
        raise ValueError("DEFORM residual models must be a sequence")
    models = [_mapping(value, label="DEFORM residual model") for value in raw_models]
    if [int(cast(Any, value.get("seed", -1))) for value in models] != [42, 43, 44]:
        raise ValueError("DEFORM transfer seeds changed")
    for value in models:
        _identity(value, label="DEFORM residual model")
    evaluation = _mapping(payload.get("evaluation"), label="evaluation")
    if (
        evaluation.get("primary_arm") != "equal-seed-no-refit-transfer"
        or evaluation.get("seed_aggregation") != "arithmetic-prediction-mean"
        or float(cast(Any, evaluation.get("shrinkage", math.nan))) != 0.25
        or evaluation.get("metric") != "mean-coordinate-l1-m-all-nodes"
        or int(cast(Any, evaluation.get("bootstrap_repetitions", -1))) != 10000
        or int(cast(Any, evaluation.get("bootstrap_seed", -1))) != 20260901
    ):
        raise ValueError("cross-backend evaluation contract changed")
    gate = _mapping(payload.get("promotion_gate"), label="promotion gate")
    minimum = float(cast(Any, gate.get("minimum_relative_improvement", math.nan)))
    wins = int(cast(Any, gate.get("minimum_case_wins", -1)))
    maximum = float(cast(Any, gate.get("maximum_case_ratio", math.nan)))
    stable = int(cast(Any, gate.get("minimum_improving_seed_models", -1)))
    if minimum != 0.01 or wins != 6 or maximum != 1.10 or stable != 2:
        raise ValueError("cross-backend promotion gate changed")
    boundary = _mapping(payload.get("information_boundary"), label="boundary")
    if (
        boundary.get("pyelastica_refit") is not False
        or boundary.get("deform_refit") is not False
        or boundary.get("target_side_selection") is not False
        or boundary.get("dlo3_official_evaluation_read") is not False
        or boundary.get("dlo4_or_dlo5_read") is not False
        or boundary.get("paper_claim_authorized") is not False
    ):
        raise ValueError("cross-backend information boundary changed")
    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> list[float]:
    """Return a paired object/trajectory bootstrap interval for a mean."""

    vector = _finite_array(values, ndim=1, label="bootstrap values")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if len(vector) == 1:
        return [float(vector[0]), float(vector[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(repetitions, len(vector)))
    draws = vector[indices].mean(axis=1)
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def _case_errors(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth shapes differ")
    return np.mean(np.abs(prediction - truth), axis=(1, 2, 3))


def paired_point_summary(
    candidate: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    names: Sequence[str],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    """Summarize a trajectory-paired lower-is-better point comparison."""

    candidate_array = _finite_array(candidate, ndim=4, label="candidate")
    baseline_array = _finite_array(baseline, ndim=4, label="baseline")
    truth_array = _finite_array(truth, ndim=4, label="truth")
    if (
        candidate_array.shape != baseline_array.shape
        or candidate_array.shape != truth_array.shape
        or len(names) != candidate_array.shape[0]
        or len(set(names)) != len(names)
    ):
        raise ValueError("paired point arrays or names do not align")
    candidate_error = _case_errors(candidate_array, truth_array)
    baseline_error = _case_errors(baseline_array, truth_array)
    differences = candidate_error - baseline_error
    tolerance = 1e-12
    ratios = candidate_error / np.maximum(baseline_error, np.finfo(float).tiny)
    cases = [
        {
            "name": str(name),
            "candidate_l1_m": float(candidate_error[index]),
            "baseline_l1_m": float(baseline_error[index]),
            "difference_m": float(differences[index]),
            "candidate_to_baseline_ratio": float(ratios[index]),
            "candidate_wins": bool(differences[index] < -tolerance),
        }
        for index, name in enumerate(names)
    ]
    candidate_mean = float(np.mean(candidate_error))
    baseline_mean = float(np.mean(baseline_error))
    return {
        "case_count": len(names),
        "candidate_mean_l1_m": candidate_mean,
        "baseline_mean_l1_m": baseline_mean,
        "mean_difference_m": float(np.mean(differences)),
        "relative_improvement": float(1.0 - candidate_mean / baseline_mean),
        "object_bootstrap_95_interval_m": bootstrap_mean_interval(
            differences,
            repetitions=repetitions,
            seed=seed,
        ),
        "wins": int(np.sum(differences < -tolerance)),
        "ties": int(np.sum(np.abs(differences) <= tolerance)),
        "losses": int(np.sum(differences > tolerance)),
        "maximum_case_ratio": float(np.max(ratios)),
        "cases": cases,
    }


def _gate(
    summary: Mapping[str, object],
    gate: Mapping[str, object],
) -> dict[str, object]:
    minimum = float(cast(Any, gate["minimum_relative_improvement"]))
    minimum_wins = int(cast(Any, gate["minimum_case_wins"]))
    maximum_ratio = float(cast(Any, gate["maximum_case_ratio"]))
    passed = (
        float(cast(Any, summary["relative_improvement"])) >= minimum
        and int(cast(Any, summary["wins"])) >= minimum_wins
        and float(cast(Any, summary["maximum_case_ratio"])) <= maximum_ratio
    )
    return {
        "passed": passed,
        "minimum_relative_improvement": minimum,
        "minimum_case_wins": minimum_wins,
        "maximum_case_ratio": maximum_ratio,
    }


def evaluate_cross_backend_transfer(
    *,
    names: Sequence[str],
    truth: np.ndarray,
    pyelastica_backend: np.ndarray,
    pyelastica_specific_candidate: np.ndarray,
    transferred_predictions: Mapping[int, np.ndarray],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the frozen equal-seed, no-refit coefficient-transfer arm."""

    source_panel = _mapping(protocol.get("source_panel"), label="source panel")
    expected_count = int(cast(Any, source_panel["trajectory_count"]))
    if len(names) != expected_count:
        raise ValueError("source trajectory count differs from protocol")
    seeds = sorted(transferred_predictions)
    if seeds != [42, 43, 44]:
        raise ValueError("transferred prediction seed set changed")
    truth_array = _finite_array(truth, ndim=4, label="truth")
    backend = _finite_array(
        pyelastica_backend,
        ndim=4,
        label="PyElastica backend",
    )
    backend_specific = _finite_array(
        pyelastica_specific_candidate,
        ndim=4,
        label="PyElastica-specific candidate",
    )
    seed_predictions = {
        seed: _finite_array(
            transferred_predictions[seed],
            ndim=4,
            label=f"seed-{seed} transferred prediction",
        )
        for seed in seeds
    }
    shapes = {
        truth_array.shape,
        backend.shape,
        backend_specific.shape,
        *(value.shape for value in seed_predictions.values()),
    }
    if len(shapes) != 1:
        raise ValueError("cross-backend prediction shapes differ")
    evaluation = _mapping(protocol.get("evaluation"), label="evaluation")
    repetitions = int(cast(Any, evaluation["bootstrap_repetitions"]))
    bootstrap_seed = int(cast(Any, evaluation["bootstrap_seed"]))
    ensemble = np.mean(np.stack(list(seed_predictions.values())), axis=0)
    primary = paired_point_summary(
        ensemble,
        backend,
        truth_array,
        names,
        repetitions=repetitions,
        seed=bootstrap_seed,
    )
    backend_specific_summary = paired_point_summary(
        backend_specific,
        backend,
        truth_array,
        names,
        repetitions=repetitions,
        seed=bootstrap_seed + 1,
    )
    seed_summaries = {
        str(seed): paired_point_summary(
            prediction,
            backend,
            truth_array,
            names,
            repetitions=repetitions,
            seed=bootstrap_seed + seed,
        )
        for seed, prediction in seed_predictions.items()
    }
    direct_vs_specific = paired_point_summary(
        ensemble,
        backend_specific,
        truth_array,
        names,
        repetitions=repetitions,
        seed=bootstrap_seed + 2,
    )
    gate_contract = _mapping(protocol.get("promotion_gate"), label="promotion gate")
    primary_gate = _gate(primary, gate_contract)
    improving_seeds = sum(
        float(cast(Any, summary["relative_improvement"])) > 0.0
        for summary in seed_summaries.values()
    )
    minimum_improving = int(cast(Any, gate_contract["minimum_improving_seed_models"]))
    seed_stability_passed = improving_seeds >= minimum_improving
    baseline_mean = float(cast(Any, primary["baseline_mean_l1_m"]))
    direct_mean = float(cast(Any, primary["candidate_mean_l1_m"]))
    specific_mean = float(cast(Any, backend_specific_summary["candidate_mean_l1_m"]))
    specific_gain = baseline_mean - specific_mean
    retention = (
        (baseline_mean - direct_mean) / specific_gain if specific_gain > 0.0 else None
    )
    supported = bool(primary_gate["passed"]) and seed_stability_passed
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": RESULT_CONTRACT,
        "decision": (
            "no-refit-cross-backend-transfer-supported"
            if supported
            else "no-refit-cross-backend-transfer-not-supported"
        ),
        "primary_arm": "equal-seed-no-refit-transfer",
        "source_trajectory_count": len(names),
        "methods": {
            "raw_pyelastica": baseline_mean,
            "pyelastica_specific_candidate": specific_mean,
            "deform_no_refit_equal_seed_transfer": direct_mean,
            **{
                f"deform_no_refit_seed_{seed}": float(
                    cast(Any, summary["candidate_mean_l1_m"])
                )
                for seed, summary in (
                    (int(key), value) for key, value in seed_summaries.items()
                )
            },
        },
        "primary_vs_raw_pyelastica": primary,
        "pyelastica_specific_vs_raw_pyelastica": backend_specific_summary,
        "direct_transfer_vs_pyelastica_specific": direct_vs_specific,
        "individual_seed_vs_raw_pyelastica": seed_summaries,
        "promotion_gate": {
            **primary_gate,
            "improving_seed_models": improving_seeds,
            "minimum_improving_seed_models": minimum_improving,
            "seed_stability_passed": seed_stability_passed,
            "supported": supported,
        },
        "backend_specific_gain_retained_fraction": (
            None if retention is None else float(retention)
        ),
        "information_boundary": {
            "source_outcomes_previously_opened": True,
            "pyelastica_refit": False,
            "deform_refit": False,
            "target_side_selection": False,
            "dlo3_official_evaluation_read": False,
            "dlo4_or_dlo5_read": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "Retrospective source-only DLO3 coefficient-transfer diagnostic. "
            "A positive decision supports no-refit transfer from DEFORM-fitted "
            "local residuals to sealed PyElastica source predictions; it does "
            "not establish target confirmation, arbitrary-backend transfer, "
            "zero-shot object generalization, safety, or state of the art."
        ),
    }
