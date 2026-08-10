"""Source-only selection among matched discrepancy-belief candidates.

The tournament consumes already scored, target-blind candidate predictions. It
selects at most one candidate relative to a frozen reference while preserving
exact physical fallback, equal-group weighting, proper-score non-regression, and
leave-one-group-out stability. A passing source gate authorizes only a later
prospective protocol; it is not target or claim-bearing evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ._discrepancy_tournament_analysis import (
    analyze_discrepancy_candidate_tournament as _analyze_tournament,
)
from ._discrepancy_tournament_contracts import (
    DISCREPANCY_TOURNAMENT_CLAIM_BOUNDARY,
    DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
    DISCREPANCY_TOURNAMENT_REPORT_CONTRACT,
    CandidateSpec,
    TournamentEvaluationSpec,
    TournamentEvidence,
    TournamentRecord,
    TournamentSelectionConfig,
    _require,
    parse_discrepancy_candidate_tournament as _parse_tournament,
)
from .provider_failure_report_io import (
    DEFAULT_MAXIMUM_INPUT_BYTES,
    load_provider_failure_input,
    publish_provider_failure_report,
)

ALLOWED_DISCREPANCY_TOURNAMENT_STATISTICAL_UNITS: Final[frozenset[str]] = (
    frozenset(
        {
            "physical-object",
            "acquisition-session",
            "physical-object-session",
            "physical-object-or-session",
        }
    )
)
MAXIMUM_DISCREPANCY_TOURNAMENT_BOOTSTRAP_INDEX_CELLS: Final = 10_000_000


def _validate_deployment_semantics(
    evidence: TournamentEvidence,
) -> TournamentEvidence:
    _require(
        evidence.statistical_unit
        in ALLOWED_DISCREPANCY_TOURNAMENT_STATISTICAL_UNITS,
        "statistical_unit must identify a physical object or acquisition session",
    )
    groups = {record.group_id for record in evidence.records}
    bootstrap_cells = evidence.config.bootstrap_samples * len(groups)
    _require(
        bootstrap_cells <= MAXIMUM_DISCREPANCY_TOURNAMENT_BOOTSTRAP_INDEX_CELLS,
        "bootstrap allocation exceeds the tournament resource budget",
    )

    interval_presence = {
        record.interval_covered is not None for record in evidence.records
    }
    _require(
        len(interval_presence) == 1,
        "registered candidates differ in interval availability",
    )
    intervals_present = interval_presence == {True}
    nominal_coverage = evidence.config.nominal_interval_coverage
    if nominal_coverage is None:
        _require(
            not intervals_present,
            "interval records are forbidden when interval coverage is disabled",
        )
        _require(
            evidence.config.maximum_interval_coverage_shortfall == 0.0,
            "interval coverage shortfall must be zero when intervals are disabled",
        )
    else:
        _require(intervals_present, "registered interval records are missing")
        _require(
            evidence.config.maximum_interval_coverage_shortfall
            <= nominal_coverage,
            "interval coverage shortfall cannot exceed nominal coverage",
        )

    by_unit: dict[str, list[TournamentRecord]] = {}
    for record in evidence.records:
        by_unit.setdefault(record.unit_id, []).append(record)
    for unit_id, records in by_unit.items():
        fallback = next(
            record
            for record in records
            if record.candidate_id == evidence.physical_fallback_candidate
        )
        for record in records:
            if record.accepted:
                continue
            _require(
                record.interval_covered == fallback.interval_covered,
                f"{unit_id}/{record.candidate_id} rejected interval coverage "
                "violates exact fallback",
            )
            _require(
                record.interval_width == fallback.interval_width,
                f"{unit_id}/{record.candidate_id} rejected interval width "
                "violates exact fallback",
            )
    return evidence


def parse_discrepancy_candidate_tournament(
    payload: Mapping[str, object],
) -> TournamentEvidence:
    """Validate the base contract and deployment-specific tournament semantics."""

    return _validate_deployment_semantics(_parse_tournament(payload))


def analyze_discrepancy_candidate_tournament(
    payload: Mapping[str, object],
) -> dict[str, Any]:
    """Select at most one candidate after the complete public validation path."""

    parse_discrepancy_candidate_tournament(payload)
    return _analyze_tournament(payload)


def load_discrepancy_candidate_tournament_input(
    path: str | Path,
    *,
    maximum_input_bytes: int = DEFAULT_MAXIMUM_INPUT_BYTES,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Read one unchanged ordinary UTF-8 JSON tournament input."""

    return load_provider_failure_input(
        path,
        maximum_input_bytes=maximum_input_bytes,
    )


def publish_discrepancy_candidate_tournament_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    input_artifact: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Publish one tournament report atomically without silent replacement."""

    return publish_provider_failure_report(
        path,
        report,
        input_artifact=input_artifact,
        overwrite=overwrite,
    )


__all__ = [
    "ALLOWED_DISCREPANCY_TOURNAMENT_STATISTICAL_UNITS",
    "DEFAULT_MAXIMUM_INPUT_BYTES",
    "DISCREPANCY_TOURNAMENT_CLAIM_BOUNDARY",
    "DISCREPANCY_TOURNAMENT_INPUT_CONTRACT",
    "DISCREPANCY_TOURNAMENT_REPORT_CONTRACT",
    "MAXIMUM_DISCREPANCY_TOURNAMENT_BOOTSTRAP_INDEX_CELLS",
    "CandidateSpec",
    "TournamentEvaluationSpec",
    "TournamentEvidence",
    "TournamentRecord",
    "TournamentSelectionConfig",
    "analyze_discrepancy_candidate_tournament",
    "load_discrepancy_candidate_tournament_input",
    "parse_discrepancy_candidate_tournament",
    "publish_discrepancy_candidate_tournament_report",
]