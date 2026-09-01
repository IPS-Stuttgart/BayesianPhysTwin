"""DLO-stratified source-replication synthesis for the sealed DLO4/DLO5 run."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

SCHEMA_VERSION = 1
PROTOCOL_CONTRACT = "deform-dlo45-source-meta-analysis-protocol-v1"
RESULT_CONTRACT = "deform-dlo45-source-meta-analysis-result-v1"
DLOS = ("DLO4", "DLO5")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_vector(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a nonempty finite vector")
    return array


def _as_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def load_source_meta_protocol(path: str | Path) -> dict[str, object]:
    """Load and validate the frozen post-source, pre-target meta-analysis protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source meta-analysis protocol must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported source meta-analysis protocol schema")
    if payload.get("contract") != PROTOCOL_CONTRACT:
        raise ValueError("unsupported source meta-analysis protocol contract")
    if payload.get("status") != "frozen-after-source-results-before-target-outcome":
        raise ValueError("source meta-analysis protocol status changed")

    run = _mapping(payload.get("blocking_run"), label="blocking run")
    if (
        _as_int(run.get("run_id"), label="blocking run id") != 33361441865
        or run.get("head_sha") != "0376ece871d7c3d9355788f812a3c4cc1c9165b0"
        or run.get("source_artifact_id") != 9777311875
        or run.get("target_status_when_frozen")
        != "in-progress-refit-and-predict-before-joint-seal"
    ):
        raise ValueError("blocking run identity changed")

    raw_results = _mapping(payload.get("source_results"), label="source results")
    if set(raw_results) != set(DLOS):
        raise ValueError("source result roster changed")
    expected = {
        "DLO4": (
            "c811db44c6f680e31d70698d4eea2e740fce2fd1703abf27fccf873ec32950e1",
            8867,
        ),
        "DLO5": (
            "ce1cb3d1844516416ad53b5353cc9c0b55812ddb12d3d50b57a4da45698664d0",
            8879,
        ),
    }
    for dlo, (sha256, size_bytes) in expected.items():
        identity = _mapping(raw_results[dlo], label=f"{dlo} source result")
        if identity.get("sha256") != sha256 or identity.get("size_bytes") != size_bytes:
            raise ValueError(f"{dlo} source result identity changed")

    evaluation = _mapping(payload.get("evaluation"), label="evaluation")
    if (
        evaluation.get("statistical_unit") != "complete-source-test-trajectory"
        or evaluation.get("aggregation")
        != "resample-eight-trajectories-within-each-dlo-then-pool-equally"
        or _as_int(
            evaluation.get("bootstrap_repetitions"), label="bootstrap repetitions"
        )
        != 10000
        or _as_int(evaluation.get("bootstrap_seed"), label="bootstrap seed") != 20260901
        or evaluation.get("primary_metric")
        != "pooled-mean-coordinate-l1-relative-improvement"
        or evaluation.get("decision_rule") != "both-original-source-gates-passed"
    ):
        raise ValueError("source meta-analysis evaluation changed")

    boundary = _mapping(
        payload.get("information_boundary"), label="information boundary"
    )
    if (
        boundary.get("source_outcomes_already_opened") is not True
        or boundary.get("meta_analysis_frozen_before_target_outcome") is not True
        or boundary.get("target_scores_used") is not False
        or boundary.get("new_model_selection") is not False
        or boundary.get("paper_claim_authorized") is not False
    ):
        raise ValueError("source meta-analysis information boundary changed")

    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def exact_upper_sign_probability(*, wins: int, losses: int) -> float:
    """Return P[X >= wins] for X~Binomial(wins+losses, 0.5), excluding ties."""

    if wins < 0 or losses < 0 or wins + losses == 0:
        raise ValueError("sign-test counts must be nonnegative and nonempty")
    total = wins + losses
    numerator = sum(math.comb(total, k) for k in range(wins, total + 1))
    return float(numerator / (2**total))


def _validate_source_result(
    value: Mapping[str, object],
    *,
    dlo: str,
) -> dict[str, object]:
    if (
        value.get("schema_version") != 1
        or value.get("contract") != "deform-dlo45-source-result-v1"
        or value.get("dlo") != dlo
        or value.get("source_test_opened") is not True
        or value.get("target_eval_enumerated") is not False
        or value.get("target_eval_read") is not False
        or value.get("target_authorized") is not False
        or value.get("retry_authorized") is not False
        or value.get("prob4d_used") is not False
    ):
        raise ValueError(f"{dlo} source-result boundary changed")

    gate = _mapping(value.get("source_gate"), label=f"{dlo} source gate")
    names = gate.get("case_names")
    if (
        gate.get("metric") != "official-mean-coordinate-l1-all-nodes"
        or not isinstance(names, list)
        or len(names) != 8
        or len(set(names)) != 8
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError(f"{dlo} source-gate roster changed")

    baseline = _finite_vector(
        gate.get("baseline_case_l1_m"), label=f"{dlo} baseline cases"
    )
    candidate = _finite_vector(
        gate.get("candidate_case_l1_m"), label=f"{dlo} candidate cases"
    )
    if baseline.shape != candidate.shape or baseline.size != len(names):
        raise ValueError(f"{dlo} source-gate arrays differ")
    if np.any(baseline <= 0.0) or np.any(candidate < 0.0):
        raise ValueError(f"{dlo} source-gate errors must be nonnegative")

    differences = baseline - candidate
    tolerance = 1e-15
    wins = int(np.sum(differences > tolerance))
    ties = int(np.sum(np.abs(differences) <= tolerance))
    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate))
    relative_improvement = 1.0 - candidate_mean / baseline_mean
    maximum_ratio = float(np.max(candidate / baseline))
    reported = {
        "wins": wins,
        "ties": ties,
        "baseline_mean_l1_m": baseline_mean,
        "candidate_mean_l1_m": candidate_mean,
        "relative_improvement": relative_improvement,
        "worst_candidate_to_baseline_ratio": maximum_ratio,
    }
    for key, expected in reported.items():
        observed = gate.get(key)
        if isinstance(expected, int):
            if observed != expected:
                raise ValueError(f"{dlo} source-gate {key} does not reproduce")
        elif not math.isclose(
            float(cast(Any, observed)),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{dlo} source-gate {key} does not reproduce")

    free = _mapping(
        gate.get("free_node_diagnostic"), label=f"{dlo} free-node diagnostic"
    )
    free_baseline = _finite_vector(
        free.get("baseline_case_l1_m"), label=f"{dlo} free-node baseline"
    )
    free_candidate = _finite_vector(
        free.get("candidate_case_l1_m"), label=f"{dlo} free-node candidate"
    )
    if free_baseline.shape != baseline.shape or free_candidate.shape != baseline.shape:
        raise ValueError(f"{dlo} free-node arrays differ")

    distributions = _mapping(
        value.get("bayesian_distributions"), label=f"{dlo} distributions"
    )
    registered = _mapping(
        distributions.get("trajectory-clustered-full-coordinate-covariance-v1"),
        label=f"{dlo} registered covariance",
    )
    conformalized = _mapping(
        distributions.get("calibrated-full-coordinate-covariance-v1"),
        label=f"{dlo} conformalized covariance",
    )
    metric_keys = (
        "coordinate_coverage_90",
        "coordinate_nees",
        "multivariate_nees",
        "energy_score",
        "gaussian_nll",
        "interval_width_m",
    )
    for label, metrics in (
        ("registered", registered),
        ("conformalized", conformalized),
    ):
        for key in metric_keys:
            number = float(cast(Any, metrics.get(key)))
            if not math.isfinite(number):
                raise ValueError(f"{dlo} {label} covariance {key} is non-finite")

    return {
        "names": [str(name) for name in names],
        "baseline": baseline,
        "candidate": candidate,
        "free_baseline": free_baseline,
        "free_candidate": free_candidate,
        "source_gate_passed": gate.get("passed") is True,
        "registered_covariance": {
            key: float(cast(Any, registered[key])) for key in metric_keys
        },
        "conformalized_covariance": {
            key: float(cast(Any, conformalized[key])) for key in metric_keys
        },
    }


def _paired_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    differences = baseline - candidate
    tolerance = 1e-15
    wins = int(np.sum(differences > tolerance))
    ties = int(np.sum(np.abs(differences) <= tolerance))
    losses = int(np.sum(differences < -tolerance))
    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate))
    return {
        "case_count": int(baseline.size),
        "baseline_mean_l1_m": baseline_mean,
        "candidate_mean_l1_m": candidate_mean,
        "absolute_improvement_m": baseline_mean - candidate_mean,
        "relative_improvement": 1.0 - candidate_mean / baseline_mean,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "median_candidate_to_baseline_ratio": float(np.median(candidate / baseline)),
        "maximum_candidate_to_baseline_ratio": float(np.max(candidate / baseline)),
        "exact_upper_sign_probability": exact_upper_sign_probability(
            wins=wins,
            losses=losses,
        ),
    }


def _stratified_bootstrap(
    panels: Mapping[str, Mapping[str, object]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    absolute = np.empty(repetitions, dtype=np.float64)
    relative = np.empty(repetitions, dtype=np.float64)
    per_dlo_relative = {dlo: np.empty(repetitions, dtype=np.float64) for dlo in DLOS}
    for draw in range(repetitions):
        baselines = []
        candidates = []
        for dlo in DLOS:
            panel = panels[dlo]
            baseline = cast(np.ndarray, panel["baseline"])
            candidate = cast(np.ndarray, panel["candidate"])
            indices = rng.integers(0, baseline.size, baseline.size)
            sampled_baseline = baseline[indices]
            sampled_candidate = candidate[indices]
            baselines.append(sampled_baseline)
            candidates.append(sampled_candidate)
            per_dlo_relative[dlo][draw] = 1.0 - np.mean(sampled_candidate) / np.mean(
                sampled_baseline
            )
        pooled_baseline = np.concatenate(baselines)
        pooled_candidate = np.concatenate(candidates)
        absolute[draw] = np.mean(pooled_baseline) - np.mean(pooled_candidate)
        relative[draw] = 1.0 - np.mean(pooled_candidate) / np.mean(pooled_baseline)

    return {
        "repetitions": repetitions,
        "seed": seed,
        "interpretation": (
            "descriptive-resampling-of-complete-source-test-trajectories-"
            "within-each-dlo"
        ),
        "absolute_improvement_95_interval_m": [
            float(value) for value in np.quantile(absolute, [0.025, 0.975])
        ],
        "relative_improvement_95_interval": [
            float(value) for value in np.quantile(relative, [0.025, 0.975])
        ],
        "per_dlo_relative_improvement_95_intervals": {
            dlo: [
                float(value)
                for value in np.quantile(per_dlo_relative[dlo], [0.025, 0.975])
            ]
            for dlo in DLOS
        },
    }


def evaluate_source_meta_analysis(
    *,
    protocol: Mapping[str, object],
    source_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Synthesize the two sealed source panels at the complete-trajectory unit."""

    if set(source_results) != set(DLOS):
        raise ValueError("source-result roster must contain DLO4 and DLO5")
    panels = {
        dlo: _validate_source_result(source_results[dlo], dlo=dlo) for dlo in DLOS
    }

    pooled_baseline = np.concatenate(
        [cast(np.ndarray, panels[dlo]["baseline"]) for dlo in DLOS]
    )
    pooled_candidate = np.concatenate(
        [cast(np.ndarray, panels[dlo]["candidate"]) for dlo in DLOS]
    )
    pooled_free_baseline = np.concatenate(
        [cast(np.ndarray, panels[dlo]["free_baseline"]) for dlo in DLOS]
    )
    pooled_free_candidate = np.concatenate(
        [cast(np.ndarray, panels[dlo]["free_candidate"]) for dlo in DLOS]
    )

    evaluation = _mapping(protocol.get("evaluation"), label="evaluation")
    bootstrap = _stratified_bootstrap(
        panels,
        repetitions=_as_int(
            evaluation.get("bootstrap_repetitions"),
            label="bootstrap repetitions",
        ),
        seed=_as_int(evaluation.get("bootstrap_seed"), label="bootstrap seed"),
    )

    per_dlo = {}
    cases = []
    for dlo in DLOS:
        panel = panels[dlo]
        summary = _paired_summary(
            cast(np.ndarray, panel["baseline"]),
            cast(np.ndarray, panel["candidate"]),
        )
        free_summary = _paired_summary(
            cast(np.ndarray, panel["free_baseline"]),
            cast(np.ndarray, panel["free_candidate"]),
        )
        per_dlo[dlo] = {
            **summary,
            "source_gate_passed": bool(panel["source_gate_passed"]),
            "free_node": free_summary,
        }
        for index, name in enumerate(cast(Sequence[str], panel["names"])):
            baseline = cast(np.ndarray, panel["baseline"])
            candidate = cast(np.ndarray, panel["candidate"])
            cases.append(
                {
                    "dlo": dlo,
                    "name": name,
                    "baseline_l1_m": float(baseline[index]),
                    "candidate_l1_m": float(candidate[index]),
                    "absolute_improvement_m": float(baseline[index] - candidate[index]),
                    "candidate_to_baseline_ratio": float(
                        candidate[index] / baseline[index]
                    ),
                }
            )

    metric_keys = (
        "coordinate_coverage_90",
        "coordinate_nees",
        "multivariate_nees",
        "energy_score",
        "gaussian_nll",
        "interval_width_m",
    )
    uncertainty = {}
    for family, field in (
        ("registered_full_covariance", "registered_covariance"),
        ("conformalized_full_covariance", "conformalized_covariance"),
    ):
        uncertainty[family] = {
            "aggregation": "equal-dlo-mean; source-side diagnostic only",
            **{
                key: float(
                    np.mean(
                        [
                            cast(Mapping[str, float], panels[dlo][field])[key]
                            for dlo in DLOS
                        ]
                    )
                )
                for key in metric_keys
            },
        }

    both_original_gates_passed = all(
        bool(panels[dlo]["source_gate_passed"]) for dlo in DLOS
    )
    pooled = _paired_summary(pooled_baseline, pooled_candidate)
    pooled_free = _paired_summary(pooled_free_baseline, pooled_free_candidate)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": RESULT_CONTRACT,
        "decision": (
            "both-original-dlo45-source-gates-passed"
            if both_original_gates_passed
            else "one-or-more-original-dlo45-source-gates-failed"
        ),
        "both_original_source_gates_passed": both_original_gates_passed,
        "per_dlo": per_dlo,
        "pooled_equal_trajectory": pooled,
        "pooled_free_node": pooled_free,
        "dlo_stratified_bootstrap": bootstrap,
        "source_uncertainty_diagnostic": uncertainty,
        "cases": cases,
        "information_boundary": {
            "source_outcomes_already_opened": True,
            "meta_analysis_frozen_before_target_outcome": True,
            "target_scores_used": False,
            "new_model_selection": False,
            "new_hyperparameter_selection": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "This retrospective aggregation was frozen after the two source results "
            "were opened but before any DLO4/DLO5 target outcome. It strengthens the "
            "source-side replication statement by reporting the complete-trajectory "
            "joint effect and 16-case directional consistency. The sign probability "
            "uses a simple independent fair-direction null and is not an operator-level "
            "generalization p-value. This result is not held-out target evidence, "
            "cross-backend transfer, arbitrary-object generalization, or state of the art."
        ),
    }
