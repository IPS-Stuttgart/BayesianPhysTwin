#!/usr/bin/env python3
"""Controlled study of quantitative attribution and intervention planning."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from bayesian_phystwin_experiments.interventional_cause_estimation_v1 import (
    CauseQueryEstimateStatus,
    estimate_cause_queries,
    plan_diagnostic_interventions,
)
from bayesian_phystwin_experiments.interventional_cause_identifiability_v1 import (
    CauseResponseSignatureV1,
    InterventionalCauseIdentifiabilityCertificateV1,
    InterventionResponseBlockV1,
)

SHA: Final = "a" * 64
CAUSES: Final = (
    "observation_bias",
    "physical_parameter",
    "physical_state",
    "realized_intervention",
    "source_local_discrepancy",
)
PHYSICAL_CAUSES: Final = frozenset(
    {"physical_parameter", "physical_state", "realized_intervention"}
)
ACTIONS: Final = (
    "action-0-source",
    "action-1-view-change",
    "action-2-contact-change",
    "action-3-control-change",
    "action-4-redundant-decoy",
)
ACTION_COSTS: Final[Mapping[str, float]] = {
    "action-0-source": 0.0,
    "action-1-view-change": 1.0,
    "action-2-contact-change": 1.5,
    "action-3-control-change": 0.8,
    "action-4-redundant-decoy": 0.2,
}
EXPECTED_PLAN: Final = (
    "action-0-source",
    "action-1-view-change",
    "action-3-control-change",
)


def response_columns() -> dict[str, np.ndarray]:
    """Cause signatures with an intentionally uninformative factual action."""

    return {
        "observation_bias": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.2, 0.2, 0.2],
            ],
            dtype=np.float64,
        ),
        "physical_parameter": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.2],
                [0.2, 0.3, 1.0],
                [0.5, 1.0, -0.3],
                [0.2, 0.2, 0.2],
            ],
            dtype=np.float64,
        ),
        "physical_state": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, -0.5, 0.5],
                [1.0, 0.2, -0.6],
                [0.2, 0.2, 0.2],
            ],
            dtype=np.float64,
        ),
        "realized_intervention": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.5, -0.5, 1.0],
                [-0.5, 1.0, 0.1],
                [1.0, -1.0, 0.5],
                [0.2, 0.2, 0.2],
            ],
            dtype=np.float64,
        ),
        "source_local_discrepancy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.2, 0.2, 0.2],
            ],
            dtype=np.float64,
        ),
    }


def _cause(cause_id: str, values: np.ndarray) -> CauseResponseSignatureV1:
    return CauseResponseSignatureV1(
        cause_id=cause_id,
        latent_coordinates_id=SHA,
        cause_query_id=SHA,
        intervention_blocks=tuple(
            InterventionResponseBlockV1(
                intervention_id=action_id,
                response_signature_id=SHA,
                whitened_response_signature=values[index, :, None],
            )
            for index, action_id in enumerate(ACTIONS)
        ),
        cause_query_map=np.eye(1, dtype=np.float64),
    )


def build_certificate() -> InterventionalCauseIdentifiabilityCertificateV1:
    columns = response_columns()
    return InterventionalCauseIdentifiabilityCertificateV1(
        observation_whitening_id=SHA,
        declared_nuisance_id=SHA,
        cause_family_id=SHA,
        cause_signatures=tuple(_cause(name, columns[name]) for name in CAUSES),
        joint_whitened_nuisance_design=np.empty((15, 0), dtype=np.float64),
        metadata={"study": "why-is-the-twin-wrong-estimation-design-v1"},
    )


def _rows(intervention_ids: Sequence[str]) -> np.ndarray:
    offsets = np.arange(0, 3 * (len(ACTIONS) + 1), 3)
    selected = set(intervention_ids)
    return np.concatenate(
        [
            np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
            for index, action_id in enumerate(ACTIONS)
            if action_id in selected
        ]
    )


def _prepared_operators(
    certificate: InterventionalCauseIdentifiabilityCertificateV1,
    intervention_ids: tuple[str, ...],
    noise_variance: float,
) -> dict[str, Any]:
    bundle = estimate_cause_queries(
        certificate,
        np.zeros(15, dtype=np.float64),
        intervention_ids=intervention_ids,
        residual_noise_variance=noise_variance,
        confidence_level=0.95,
    )
    return {
        "intervention_ids": intervention_ids,
        "rows": _rows(intervention_ids),
        "fully_identified": all(
            item.status is CauseQueryEstimateStatus.IDENTIFIABLE
            for item in bundle.cause_estimates
        ),
        "factor": np.stack(
            [item.factor_operator[0] for item in bundle.cause_estimates],
            axis=0,
        ),
        "standard_error": np.asarray(
            [np.sqrt(item.covariance[0, 0]) for item in bundle.cause_estimates],
            dtype=np.float64,
        ),
        "bundle_id": bundle.artifact_id,
    }


def _method_metrics(
    residuals: np.ndarray,
    truth_indices: np.ndarray,
    amplitudes: np.ndarray,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    trial_count = len(residuals)
    if not prepared["fully_identified"]:
        return {
            "resolved_coverage": 0.0,
            "accuracy_among_resolved": 0.0,
            "false_physical_promotion_among_nonphysical": 0.0,
            "true_cause_query_rmse": None,
            "nominal_95_coverage": None,
            "resolved_trial_count": 0,
            "trial_count": trial_count,
        }
    rows = prepared["rows"]
    factor = prepared["factor"]
    standard_error = prepared["standard_error"]
    estimates = residuals[:, rows] @ factor.T
    standardized = np.abs(estimates) / np.maximum(standard_error, 1e-15)
    predicted = np.argmax(standardized, axis=1)
    selected = estimates[np.arange(trial_count), truth_indices]
    errors = selected - amplitudes
    nonphysical = np.asarray(
        [CAUSES[index] not in PHYSICAL_CAUSES for index in truth_indices],
        dtype=bool,
    )
    predicted_physical = np.asarray(
        [CAUSES[index] in PHYSICAL_CAUSES for index in predicted],
        dtype=bool,
    )
    false_promotion = (
        float(np.mean(predicted_physical[nonphysical])) if np.any(nonphysical) else 0.0
    )
    covered = np.abs(errors) <= 1.959963984540054 * standard_error[truth_indices]
    return {
        "resolved_coverage": 1.0,
        "accuracy_among_resolved": float(np.mean(predicted == truth_indices)),
        "false_physical_promotion_among_nonphysical": false_promotion,
        "true_cause_query_rmse": float(np.sqrt(np.mean(errors**2))),
        "nominal_95_coverage": float(np.mean(covered)),
        "resolved_trial_count": trial_count,
        "trial_count": trial_count,
    }


def _random_budget_metrics(
    residuals: np.ndarray,
    truth_indices: np.ndarray,
    amplitudes: np.ndarray,
    prepared_by_subset: Mapping[tuple[str, ...], Mapping[str, Any]],
    random_choices: np.ndarray,
    subsets: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    trial_count = len(residuals)
    resolved = np.zeros(trial_count, dtype=bool)
    predicted = np.full(trial_count, -1, dtype=np.int64)
    estimates_true = np.full(trial_count, np.nan, dtype=np.float64)
    covered = np.zeros(trial_count, dtype=bool)
    for subset_index, subset in enumerate(subsets):
        mask = random_choices == subset_index
        prepared = prepared_by_subset[subset]
        if not np.any(mask) or not prepared["fully_identified"]:
            continue
        rows = prepared["rows"]
        factor = prepared["factor"]
        standard_error = prepared["standard_error"]
        estimates = residuals[mask][:, rows] @ factor.T
        standardized = np.abs(estimates) / np.maximum(standard_error, 1e-15)
        predicted[mask] = np.argmax(standardized, axis=1)
        resolved[mask] = True
        local_truth = truth_indices[mask]
        local_estimates = estimates[np.arange(int(np.sum(mask))), local_truth]
        estimates_true[mask] = local_estimates
        covered[mask] = (
            np.abs(local_estimates - amplitudes[mask])
            <= 1.959963984540054 * standard_error[local_truth]
        )
    if not np.any(resolved):
        raise RuntimeError("random-budget benchmark produced no resolved trials")
    nonphysical = np.asarray(
        [CAUSES[index] not in PHYSICAL_CAUSES for index in truth_indices],
        dtype=bool,
    )
    relevant_nonphysical = resolved & nonphysical
    predicted_physical = np.asarray(
        [index >= 0 and CAUSES[index] in PHYSICAL_CAUSES for index in predicted],
        dtype=bool,
    )
    return {
        "resolved_coverage": float(np.mean(resolved)),
        "accuracy_among_resolved": float(
            np.mean(predicted[resolved] == truth_indices[resolved])
        ),
        "false_physical_promotion_among_nonphysical": float(
            np.mean(predicted_physical[relevant_nonphysical])
        ),
        "true_cause_query_rmse": float(
            np.sqrt(np.mean((estimates_true[resolved] - amplitudes[resolved]) ** 2))
        ),
        "nominal_95_coverage": float(np.mean(covered[resolved])),
        "resolved_trial_count": int(np.sum(resolved)),
        "trial_count": trial_count,
    }


def _wrong_relation_metrics(
    residuals: np.ndarray,
    truth_indices: np.ndarray,
    amplitudes: np.ndarray,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    rows = prepared["rows"]
    factor = prepared["factor"]
    standard_error = prepared["standard_error"]
    selected = residuals[:, rows].copy()
    # Keep the factual block fixed and swap the two diagnostic-action blocks.
    selected[:, 3:6], selected[:, 6:9] = (
        selected[:, 6:9].copy(),
        selected[:, 3:6].copy(),
    )
    estimates = selected @ factor.T
    predicted = np.argmax(
        np.abs(estimates) / np.maximum(standard_error, 1e-15),
        axis=1,
    )
    true_estimates = estimates[np.arange(len(residuals)), truth_indices]
    return {
        "resolved_coverage": 1.0,
        "accuracy_among_resolved": float(np.mean(predicted == truth_indices)),
        "true_cause_query_rmse": float(
            np.sqrt(np.mean((true_estimates - amplitudes) ** 2))
        ),
    }


def canonical_id(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_study(
    *,
    trial_count: int,
    seed: int,
    noise_standard_deviation: float,
) -> dict[str, Any]:
    if trial_count < 1000 or trial_count % len(CAUSES):
        raise ValueError("trial_count must be at least 1000 and divisible by five")
    if not np.isfinite(noise_standard_deviation) or noise_standard_deviation <= 0.0:
        raise ValueError("noise_standard_deviation must be positive")
    certificate = build_certificate()
    plan = plan_diagnostic_interventions(
        certificate,
        ACTION_COSTS,
        required_intervention_ids=(ACTIONS[0],),
        maximum_interventions=3,
        metadata={"study": "controlled-minimum-cost-diagnosis-v1"},
    )
    noise_variance = noise_standard_deviation**2
    planned = _prepared_operators(
        certificate,
        plan.selected_intervention_ids,
        noise_variance,
    )
    full = _prepared_operators(certificate, ACTIONS, noise_variance)
    source = _prepared_operators(certificate, (ACTIONS[0],), noise_variance)
    decoy = _prepared_operators(
        certificate,
        (ACTIONS[0], ACTIONS[1], ACTIONS[4]),
        noise_variance,
    )

    diagnostic_pairs = tuple(itertools.combinations(ACTIONS[1:], 2))
    random_subsets = tuple((ACTIONS[0], *pair) for pair in diagnostic_pairs)
    prepared_by_subset = {
        subset: _prepared_operators(certificate, subset, noise_variance)
        for subset in random_subsets
    }

    rng = np.random.default_rng(seed)
    truth_indices = np.arange(trial_count, dtype=np.int64) % len(CAUSES)
    rng.shuffle(truth_indices)
    amplitudes = rng.choice(np.asarray((-1.0, 1.0)), size=trial_count) * rng.uniform(
        0.5,
        1.5,
        size=trial_count,
    )
    columns = response_columns()
    residuals = np.empty((trial_count, 15), dtype=np.float64)
    for trial, (truth_index, amplitude) in enumerate(
        zip(truth_indices, amplitudes, strict=True)
    ):
        cause = CAUSES[int(truth_index)]
        residuals[trial] = np.concatenate(
            [
                amplitude * columns[cause][index]
                + rng.normal(0.0, noise_standard_deviation, size=3)
                for index in range(len(ACTIONS))
            ]
        )
    random_choices = rng.integers(0, len(random_subsets), size=trial_count)

    planned_metrics = _method_metrics(
        residuals,
        truth_indices,
        amplitudes,
        planned,
    )
    full_metrics = _method_metrics(
        residuals,
        truth_indices,
        amplitudes,
        full,
    )
    random_metrics = _random_budget_metrics(
        residuals,
        truth_indices,
        amplitudes,
        prepared_by_subset,
        random_choices,
        random_subsets,
    )
    wrong_relation = _wrong_relation_metrics(
        residuals,
        truth_indices,
        amplitudes,
        planned,
    )
    forced_physical_accuracy = float(
        np.mean(truth_indices == CAUSES.index("physical_parameter"))
    )
    nonphysical_count = int(
        np.sum([CAUSES[index] not in PHYSICAL_CAUSES for index in truth_indices])
    )
    factual_forced = {
        "resolved_coverage": 1.0,
        "accuracy_among_resolved": forced_physical_accuracy,
        "false_physical_promotion_among_nonphysical": 1.0,
        "nonphysical_trial_count": nonphysical_count,
        "rule": "break the factual-action tie in favor of physical_parameter",
    }
    source_certificate = _method_metrics(
        residuals,
        truth_indices,
        amplitudes,
        source,
    )
    decoy_certificate = _method_metrics(
        residuals,
        truth_indices,
        amplitudes,
        decoy,
    )

    planned_cost = plan.selected_total_cost
    full_cost = float(sum(ACTION_COSTS.values()))
    checks = {
        "minimum_cost_plan_is_expected": (
            plan.selected_intervention_ids == EXPECTED_PLAN
        ),
        "planned_portfolio_identifies_all_causes": (
            planned_metrics["resolved_coverage"] == 1.0
        ),
        "planned_accuracy_at_least_99_percent": (
            planned_metrics["accuracy_among_resolved"] >= 0.99
        ),
        "planned_interval_coverage_between_93_and_97_percent": (
            0.93 <= planned_metrics["nominal_95_coverage"] <= 0.97
        ),
        "planned_rmse_within_50_percent_of_full_roster": (
            planned_metrics["true_cause_query_rmse"]
            <= 1.5 * full_metrics["true_cause_query_rmse"]
        ),
        "planned_cost_at_most_60_percent_of_full_roster": (
            planned_cost <= 0.6 * full_cost
        ),
        "random_budget_resolves_no_more_than_60_percent": (
            random_metrics["resolved_coverage"] <= 0.60
        ),
        "random_budget_resolves_at_least_40_percent": (
            random_metrics["resolved_coverage"] >= 0.40
        ),
        "source_action_certificate_abstains": (
            source_certificate["resolved_coverage"] == 0.0
        ),
        "redundant_decoy_does_not_resolve_attribution": (
            decoy_certificate["resolved_coverage"] == 0.0
        ),
        "factual_forced_label_is_at_chance": (
            factual_forced["accuracy_among_resolved"] <= 0.25
        ),
        "factual_forced_label_promotes_every_nonphysical_case": (
            factual_forced["false_physical_promotion_among_nonphysical"] == 1.0
        ),
        "wrong_action_relation_reduces_accuracy": (
            wrong_relation["accuracy_among_resolved"] <= 0.75
        ),
        "wrong_action_relation_increases_rmse_fivefold": (
            wrong_relation["true_cause_query_rmse"]
            >= 5.0 * planned_metrics["true_cause_query_rmse"]
        ),
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.interventional-cause-estimation-controlled.v1",
        "schema_version": 1,
        "trial_count": trial_count,
        "seed": seed,
        "noise_standard_deviation": noise_standard_deviation,
        "cause_ids": list(CAUSES),
        "intervention_ids": list(ACTIONS),
        "intervention_costs": dict(ACTION_COSTS),
        "certificate_id": certificate.artifact_id,
        "plan": plan.to_record(),
        "methods": {
            "factual_forced_physical_label": factual_forced,
            "source_action_certificate": source_certificate,
            "redundant_decoy_certificate": decoy_certificate,
            "random_equal_count_portfolio": random_metrics,
            "minimum_cost_planned_portfolio": planned_metrics,
            "full_intervention_roster": full_metrics,
            "wrong_action_relation": wrong_relation,
        },
        "cost_summary": {
            "planned_cost": planned_cost,
            "full_roster_cost": full_cost,
            "planned_fraction_of_full_cost": planned_cost / full_cost,
            "planned_intervention_count": len(plan.selected_intervention_ids),
            "full_intervention_count": len(ACTIONS),
        },
        "checks": checks,
        "passed": passed,
        "decision": (
            "minimum-cost-interventions-recover-attribution-with-calibrated-uncertainty"
            if passed
            else "registered-attribution-design-criteria-not-met"
        ),
        "claim_boundary": (
            "Controlled exact-model mechanism evidence.  A pass shows that the "
            "registered finite planner can find a lower-cost intervention subset "
            "that identifies all registered cause queries and supports calibrated "
            "linear-Gaussian intervals in this study.  It does not establish "
            "natural real-data cause labels, cause-family completeness, nonlinear "
            "physical validity, unseen-object transfer, online control, deployment "
            "safety, or state of the art."
        ),
    }
    result["result_id"] = canonical_id(result)
    return result


def report(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    lines = [
        "# Interventional cause estimation and diagnostic design v1",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        "## Diagnostic intervention plan",
        "",
        f"- selected: `{', '.join(result['plan']['selected_intervention_ids'])}`",
        f"- selected cost: {result['cost_summary']['planned_cost']:.3f}",
        f"- full-roster cost: {result['cost_summary']['full_roster_cost']:.3f}",
        "- cost fraction: "
        f"{100 * result['cost_summary']['planned_fraction_of_full_cost']:.2f}%",
        "",
        "## Controlled results",
        "",
        "| Method | Resolved coverage | Accuracy | False physical promotion "
        "| Query RMSE | 95% coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "factual_forced_physical_label": "Factual-only forced physical label",
        "source_action_certificate": "Source-action certificate",
        "redundant_decoy_certificate": "Source + redundant decoy",
        "random_equal_count_portfolio": "Random equal-count portfolio",
        "minimum_cost_planned_portfolio": "Minimum-cost planned portfolio",
        "full_intervention_roster": "Full intervention roster",
        "wrong_action_relation": "Wrong-action relation",
    }
    for key, label in labels.items():
        item = methods[key]
        false_promotion = item.get("false_physical_promotion_among_nonphysical")
        rmse = item.get("true_cause_query_rmse")
        coverage = item.get("nominal_95_coverage")
        lines.append(
            "| "
            + label
            + f" | {100 * item['resolved_coverage']:.2f}%"
            + f" | {100 * item['accuracy_among_resolved']:.2f}%"
            + " | "
            + ("—" if false_promotion is None else f"{100 * false_promotion:.2f}%")
            + " | "
            + ("—" if rmse is None else f"{rmse:.4f}")
            + " | "
            + ("—" if coverage is None else f"{100 * coverage:.2f}%")
            + " |"
        )
    lines.extend(["", "## Registered checks", ""])
    for name, value in result["checks"].items():
        lines.append(f"- `{name}`: **{'pass' if value else 'fail'}**")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
            "",
            f"Result ID: `{result['result_id']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_902)
    parser.add_argument("--noise-standard-deviation", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_study(
        trial_count=args.trials,
        seed=args.seed,
        noise_standard_deviation=args.noise_standard_deviation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(result), encoding="utf-8")
    print(
        json.dumps(
            {"decision": result["decision"], "result_id": result["result_id"]},
            indent=2,
        )
    )
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
