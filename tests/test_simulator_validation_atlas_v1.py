from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from bayesian_phystwin.query_competence_certificate_v1 import SimulatorQueryScopeV1
from bayesian_phystwin.simulator_validation_atlas_v1 import (
    STAGE_NAMES,
    SimulatorValidationAtlasV1,
    SimulatorValidationEntryV1,
    StageStatus,
    ValidationEvidenceReferenceV1,
    ValidationStageAssessmentV1,
    load_simulator_validation_atlas,
    save_simulator_validation_atlas,
    select_prospectively_validated_candidate,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _scope(label: str) -> SimulatorQueryScopeV1:
    return SimulatorQueryScopeV1(
        simulator_id=_digest(f"simulator-{label}"),
        task_id=_digest(f"task-{label}"),
        observation_policy_id=_digest(f"observation-{label}"),
        action_bank_id=_digest(f"action-{label}"),
        metric_id=_digest(f"metric-{label}"),
        world_distribution_id=_digest(f"world-{label}"),
        statistical_unit="public simulator world",
        metadata={"label": label},
    )


def _reference(label: str) -> ValidationEvidenceReferenceV1:
    return ValidationEvidenceReferenceV1(
        repository="IPS-Stuttgart/BayesianPhysTwin",
        commit="1" * 40,
        path=f"results/source/{label}.json",
        file_sha256=_digest(f"file-{label}"),
        artifact_id=_digest(f"artifact-{label}"),
    )


def _assessment(label: str, status: str) -> ValidationStageAssessmentV1:
    checked = cast(StageStatus, status)
    if checked in {"passed", "failed"}:
        return ValidationStageAssessmentV1(
            status=checked,
            reason=f"{label}-{status}",
            evidence=(_reference(f"{label}-{status}"),),
        )
    return ValidationStageAssessmentV1(status=checked, reason=f"{label}-{status}")


def _entry(
    label: str,
    statuses: tuple[str, str, str, str, str, str] = (
        "passed",
        "passed",
        "passed",
        "passed",
        "passed",
        "passed",
    ),
) -> SimulatorValidationEntryV1:
    return SimulatorValidationEntryV1(
        backend_key=f"backend-{label}",
        display_name=f"Backend {label}",
        dataset="PublicData",
        query_scope=_scope(label),
        independent_group_count=12,
        stages={
            name: _assessment(f"{label}-{name}", status)
            for name, status in zip(STAGE_NAMES, statuses, strict=True)
        },
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        protected_target_data_read=False,
        new_recording_used=False,
        terminal_reason=f"terminal-{label}",
    )


def test_stage_order_rejects_pass_after_unmet_prerequisite() -> None:
    with pytest.raises(ValueError, match="unmet prerequisite"):
        _entry(
            "invalid",
            ("passed", "failed", "passed", "not_applicable", "passed", "passed"),
        )


def test_entry_rejects_missing_independent_group_denominator() -> None:
    valid = _entry("denominator")
    with pytest.raises(ValueError, match="independent_group_count"):
        SimulatorValidationEntryV1(
            backend_key=valid.backend_key,
            display_name=valid.display_name,
            dataset=valid.dataset,
            query_scope=valid.query_scope,
            independent_group_count=0,
            stages=valid.stages,
            exact_fallback_retained=True,
            protocol_frozen_before_outcomes=True,
            protected_target_data_read=False,
            new_recording_used=False,
            terminal_reason=valid.terminal_reason,
        )


def test_not_applicable_stage_does_not_block_later_source_evidence() -> None:
    entry = _entry(
        "source",
        (
            "passed",
            "passed",
            "passed",
            "not_applicable",
            "passed",
            "not_evaluated",
        ),
    )
    assert entry.decision == "source_supported"
    assert entry.first_failed_stage is None
    assert entry.furthest_evaluated_stage == "source_value"


def test_failed_or_uncustodied_entries_cannot_be_promoted() -> None:
    failed = _entry(
        "failed",
        (
            "passed",
            "passed",
            "passed",
            "not_applicable",
            "failed",
            "not_evaluated",
        ),
    )
    assert failed.decision == "rejected"
    valid = _entry("valid")
    uncustodied = SimulatorValidationEntryV1(
        backend_key=valid.backend_key,
        display_name=valid.display_name,
        dataset=valid.dataset,
        query_scope=valid.query_scope,
        independent_group_count=valid.independent_group_count,
        stages=valid.stages,
        exact_fallback_retained=False,
        protocol_frozen_before_outcomes=True,
        protected_target_data_read=False,
        new_recording_used=False,
        terminal_reason="fallback-not-retained",
    )
    assert uncustodied.decision == "rejected"


def test_atlas_roundtrip_rederives_counts_and_claim_boundaries(
    tmp_path: Path,
) -> None:
    certified = _entry("certified")
    rejected = _entry(
        "rejected",
        (
            "passed",
            "passed",
            "failed",
            "not_applicable",
            "not_evaluated",
            "not_evaluated",
        ),
    )
    atlas = SimulatorValidationAtlasV1(entries=(rejected, certified))
    output = tmp_path / "atlas.json"
    save_simulator_validation_atlas(output, atlas)
    loaded = load_simulator_validation_atlas(output)
    assert loaded.to_record() == atlas.to_record()
    assert loaded.decision_counts == {
        "prospective_certified": 1,
        "rejected": 1,
    }
    assert loaded.stage_counts["full_horizon_qualification"]["failed"] == 1
    assert loaded.stage_counts["full_horizon_qualification"]["passed"] == 1
    record = loaded.to_record()
    record["backend_wide_competence_claim"] = True
    with pytest.raises(ValueError, match="cannot assert"):
        SimulatorValidationAtlasV1.from_mapping(record)


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


def test_selection_requires_exact_prospective_certificate_and_preserves_identity() -> (
    None
):
    certified = _entry("certified")
    source_only = _entry(
        "source-only",
        (
            "passed",
            "passed",
            "passed",
            "not_applicable",
            "passed",
            "not_evaluated",
        ),
    )
    atlas = SimulatorValidationAtlasV1(entries=(certified, source_only))
    baseline = _Belief(_digest("baseline"))
    candidate = _Belief(_digest("candidate"))

    selected, receipt = select_prospectively_validated_candidate(
        baseline,
        candidate,
        atlas,
        query_id=str(certified.query_scope.query_id),
        inference_admissible=True,
    )
    assert selected is candidate
    assert receipt["reason"] == "prospective-query-certified"

    selected, receipt = select_prospectively_validated_candidate(
        baseline,
        candidate,
        atlas,
        query_id=str(source_only.query_scope.query_id),
        inference_admissible=True,
    )
    assert selected is baseline
    assert receipt["reason"] == "query-not-prospectively-certified"

    selected, receipt = select_prospectively_validated_candidate(
        baseline,
        candidate,
        atlas,
        query_id=_digest("unknown"),
        inference_admissible=True,
    )
    assert selected is baseline
    assert receipt["reason"] == "unknown-query"


def test_evidence_reference_rejects_noncanonical_paths() -> None:
    for path in (
        "/absolute.json",
        "../outside.json",
        "inside/../alias.json",
        "bad\\path.json",
    ):
        with pytest.raises(ValueError, match="path"):
            ValidationEvidenceReferenceV1(
                repository="IPS-Stuttgart/BayesianPhysTwin",
                commit="1" * 40,
                path=path,
                file_sha256=_digest("file"),
            )
