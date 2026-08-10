"""Equal-group analysis and selection for discrepancy tournaments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, cast

import numpy as np

from ._discrepancy_tournament_contracts import (
    DISCREPANCY_TOURNAMENT_CLAIM_BOUNDARY,
    DISCREPANCY_TOURNAMENT_REPORT_CONTRACT,
    CandidateSpec,
    TournamentEvidence,
    TournamentRecord,
    _require,
    parse_discrepancy_candidate_tournament,
)
from .provider_failure_report_io import canonical_json_sha256


def _records_for(
    evidence: TournamentEvidence,
    candidate_id: str,
    groups: frozenset[str],
) -> tuple[TournamentRecord, ...]:
    return tuple(
        record
        for record in evidence.records
        if record.candidate_id == candidate_id and record.group_id in groups
    )


def _group_means(
    records: Sequence[TournamentRecord],
    attribute: str,
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for record in records:
        values.setdefault(record.group_id, []).append(float(getattr(record, attribute)))
    return {
        group: float(np.mean(group_values)) for group, group_values in values.items()
    }


def _bootstrap_difference_interval(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    differences = candidate - reference
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    estimates = np.mean(differences[indices], axis=1)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _candidate_summary(
    evidence: TournamentEvidence,
    candidate: CandidateSpec,
    groups: frozenset[str],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    config = evidence.config
    records = _records_for(evidence, candidate.candidate_id, groups)
    reference_records = _records_for(evidence, evidence.reference_candidate, groups)
    candidate_point = _group_means(records, "deployed_point_loss")
    candidate_proper = _group_means(records, "deployed_proper_score")
    reference_point = _group_means(reference_records, "deployed_point_loss")
    reference_proper = _group_means(reference_records, "deployed_proper_score")
    ordered_groups = tuple(sorted(groups))
    _require(
        set(candidate_point) == set(reference_point) == set(groups),
        "candidate and reference group rosters differ",
    )
    candidate_point_values = np.asarray(
        [candidate_point[group] for group in ordered_groups], dtype=np.float64
    )
    reference_point_values = np.asarray(
        [reference_point[group] for group in ordered_groups], dtype=np.float64
    )
    candidate_proper_values = np.asarray(
        [candidate_proper[group] for group in ordered_groups], dtype=np.float64
    )
    reference_proper_values = np.asarray(
        [reference_proper[group] for group in ordered_groups], dtype=np.float64
    )
    _require(
        np.all(reference_point_values > config.numerical_tolerance),
        "reference group point loss must be positive",
    )
    relative_changes = candidate_point_values / reference_point_values - 1.0
    accepted = np.asarray([record.accepted for record in records], dtype=bool)
    raw_losses = np.asarray([record.point_loss for record in records], dtype=np.float64)
    fallback_losses = np.asarray(
        [record.fallback_point_loss for record in records], dtype=np.float64
    )
    harmful = accepted & (raw_losses > fallback_losses + config.numerical_tolerance)
    coverage: float | None
    mean_width: float | None
    if records and records[0].interval_covered is not None:
        coverage_by_group = _group_means(records, "interval_covered")
        width_by_group = _group_means(records, "interval_width")
        coverage = float(
            np.mean([coverage_by_group[group] for group in ordered_groups])
        )
        mean_width = float(np.mean([width_by_group[group] for group in ordered_groups]))
    else:
        coverage = None
        mean_width = None

    point_difference = candidate_point_values - reference_point_values
    proper_difference = candidate_proper_values - reference_proper_values
    point_interval = _bootstrap_difference_interval(
        candidate_point_values,
        reference_point_values,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + 1009 * seed_offset,
    )
    proper_interval = _bootstrap_difference_interval(
        candidate_proper_values,
        reference_proper_values,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + 1009 * seed_offset + 1,
    )
    failures: list[str] = []
    if candidate.candidate_id in {
        evidence.reference_candidate,
        evidence.physical_fallback_candidate,
    }:
        failures.append("registered-baseline")
    improvement = float(-np.mean(relative_changes))
    worst_regression = float(max(0.0, np.max(relative_changes)))
    mean_proper_regression = float(np.mean(proper_difference))
    if improvement + config.numerical_tolerance < (
        config.minimum_relative_point_improvement
    ):
        failures.append("insufficient-point-improvement")
    if worst_regression > (
        config.maximum_worst_group_relative_regression + config.numerical_tolerance
    ):
        failures.append("worst-group-regression")
    harmful_count = int(np.sum(harmful))
    if harmful_count > config.maximum_harmful_accepted_count:
        failures.append("harmful-accepted-updates")
    if mean_proper_regression > (
        config.maximum_mean_proper_score_regression + config.numerical_tolerance
    ):
        failures.append("proper-score-regression")
    if (
        config.require_paired_point_upper_bound_nonpositive
        and point_interval[1] > config.numerical_tolerance
    ):
        failures.append("paired-point-upper-bound-positive")
    if config.nominal_interval_coverage is not None:
        assert coverage is not None
        minimum_coverage = (
            config.nominal_interval_coverage
            - config.maximum_interval_coverage_shortfall
        )
        if coverage + config.numerical_tolerance < minimum_coverage:
            failures.append("interval-undercoverage")

    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "group_count": len(ordered_groups),
        "unit_count": len(records),
        "accepted_count": int(np.sum(accepted)),
        "fallback_count": int(len(records) - np.sum(accepted)),
        "harmful_accepted_count": harmful_count,
        "mean_deployed_point_loss": float(np.mean(candidate_point_values)),
        "mean_deployed_proper_score": float(np.mean(candidate_proper_values)),
        "mean_relative_point_improvement_vs_reference": improvement,
        "worst_group_relative_regression_vs_reference": worst_regression,
        "mean_point_difference_vs_reference": float(np.mean(point_difference)),
        "paired_point_difference_ci95": point_interval,
        "mean_proper_score_difference_vs_reference": mean_proper_regression,
        "paired_proper_score_difference_ci95": proper_interval,
        "interval_coverage": coverage,
        "mean_interval_width": mean_width,
        "complexity": {
            "state_dimension": candidate.state_dimension,
            "parameter_count": candidate.parameter_count,
            "runtime_milliseconds": candidate.runtime_milliseconds,
            "covariance_bytes": candidate.covariance_bytes,
        },
        "eligible": not failures,
        "eligibility_failures": failures,
    }


def _select_candidate(
    evidence: TournamentEvidence,
    groups: frozenset[str],
    *,
    seed_offset: int,
) -> tuple[str, list[dict[str, Any]]]:
    summaries = [
        _candidate_summary(
            evidence,
            candidate,
            groups,
            seed_offset=seed_offset + index,
        )
        for index, candidate in enumerate(evidence.candidates)
    ]
    specs = {candidate.candidate_id: candidate for candidate in evidence.candidates}
    eligible = [summary for summary in summaries if summary["eligible"]]
    if not eligible:
        return evidence.reference_candidate, summaries
    selected = min(
        eligible,
        key=lambda summary: (
            float(summary["mean_deployed_proper_score"]),
            float(summary["mean_deployed_point_loss"]),
            (
                float("inf")
                if summary["mean_interval_width"] is None
                else float(summary["mean_interval_width"])
            ),
            specs[str(summary["candidate_id"])].complexity_key,
        ),
    )
    return str(selected["candidate_id"]), summaries


def analyze_discrepancy_candidate_tournament(
    payload: Mapping[str, object],
) -> dict[str, Any]:
    """Select at most one candidate under the frozen source-only gate."""

    evidence = parse_discrepancy_candidate_tournament(payload)
    groups = frozenset(record.group_id for record in evidence.records)
    selected, summaries = _select_candidate(evidence, groups, seed_offset=0)
    summary_by_id = {str(summary["candidate_id"]): summary for summary in summaries}

    folds: list[dict[str, Any]] = []
    stable_selection = True
    held_group_nonregression = True
    for fold_index, held_group in enumerate(sorted(groups)):
        training_groups = groups - {held_group}
        fold_selected, _ = _select_candidate(
            evidence,
            training_groups,
            seed_offset=1000 + 100 * fold_index,
        )
        stable_selection &= fold_selected == selected
        held_records = _records_for(evidence, fold_selected, frozenset({held_group}))
        held_reference = _records_for(
            evidence,
            evidence.reference_candidate,
            frozenset({held_group}),
        )
        held_point = float(
            np.mean([record.deployed_point_loss for record in held_records])
        )
        reference_point = float(
            np.mean([record.deployed_point_loss for record in held_reference])
        )
        _require(
            reference_point > evidence.config.numerical_tolerance,
            "reference loss is zero",
        )
        held_regression = held_point / reference_point - 1.0
        nonregression = held_regression <= (
            evidence.config.maximum_worst_group_relative_regression
            + evidence.config.numerical_tolerance
        )
        held_group_nonregression &= nonregression
        folds.append(
            {
                "held_group": held_group,
                "selected_candidate": fold_selected,
                "held_group_relative_regression_vs_reference": held_regression,
                "held_group_nonregression": nonregression,
            }
        )

    selected_summary = summary_by_id[selected]
    source_gate_passed = (
        selected != evidence.reference_candidate
        and bool(selected_summary["eligible"])
        and held_group_nonregression
        and (stable_selection or not evidence.config.require_crossfit_stability)
    )
    decision = (
        "advance-selected-candidate"
        if source_gate_passed
        else "retain-reference-candidate"
    )
    report: dict[str, Any] = {
        "contract": DISCREPANCY_TOURNAMENT_REPORT_CONTRACT,
        "schema_version": 1,
        "protocol_id": evidence.protocol_id,
        "evidence_id": canonical_json_sha256(cast(Mapping[str, Any], payload)),
        "statistical_unit": evidence.statistical_unit,
        "split": "source-only",
        "reference_candidate": evidence.reference_candidate,
        "physical_fallback_candidate": evidence.physical_fallback_candidate,
        "evaluation": asdict(evidence.evaluation),
        "candidates": [asdict(candidate) for candidate in evidence.candidates],
        "group_count": len(groups),
        "unit_count": len({record.unit_id for record in evidence.records}),
        "selection": asdict(evidence.config),
        "selected_candidate": selected,
        "selected_candidate_summary": selected_summary,
        "candidate_summaries": summaries,
        "cross_fitted": {
            "folds": folds,
            "stable_selection": stable_selection,
            "held_group_nonregression": held_group_nonregression,
        },
        "source_gate_passed": source_gate_passed,
        "decision": decision,
        "claim_authorized": False,
        "claim_boundary": DISCREPANCY_TOURNAMENT_CLAIM_BOUNDARY,
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


__all__ = ["analyze_discrepancy_candidate_tournament"]
