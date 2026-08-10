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
from typing import Any

from ._discrepancy_tournament_analysis import (
    analyze_discrepancy_candidate_tournament,
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
    parse_discrepancy_candidate_tournament,
)
from .provider_failure_report_io import (
    DEFAULT_MAXIMUM_INPUT_BYTES,
    load_provider_failure_input,
    publish_provider_failure_report,
)


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
    "DEFAULT_MAXIMUM_INPUT_BYTES",
    "DISCREPANCY_TOURNAMENT_CLAIM_BOUNDARY",
    "DISCREPANCY_TOURNAMENT_INPUT_CONTRACT",
    "DISCREPANCY_TOURNAMENT_REPORT_CONTRACT",
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
