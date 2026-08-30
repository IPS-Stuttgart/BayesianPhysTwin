from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from bayesian_phystwin.simulator_validation_atlas_v1 import (
    STAGE_NAMES,
    load_simulator_validation_atlas,
    select_prospectively_validated_candidate,
)
from scripts.build_cross_backend_validation_atlas_v2 import (
    NATIVE_EVIDENCE,
    NATIVE_EVIDENCE_SHA256,
    build_atlas,
)

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "results/source/cross_backend_validation_atlas_v2/atlas.json"
EXPECTED_ARTIFACT_ID = (
    "d1faadce843d1077e47594c17ce452acf9bdac36ce76218b4973d791a4ed7240"
)
EXPECTED_FILE_SHA256 = (
    "ee6f1b90c1517dbb81ca5e73fcbfd73c0db7fe55dc271dc71207ea4858e4e421"
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_native_continuum_evidence_bytes_remain_exact() -> None:
    for name, relative in NATIVE_EVIDENCE.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == (
            NATIVE_EVIDENCE_SHA256[name]
        )


def test_committed_v2_atlas_is_exact_builder_output() -> None:
    committed = load_simulator_validation_atlas(ATLAS)
    rebuilt = build_atlas()
    assert committed.to_record() == rebuilt.to_record()
    assert committed.artifact_id == EXPECTED_ARTIFACT_ID
    assert hashlib.sha256(ATLAS.read_bytes()).hexdigest() == EXPECTED_FILE_SHA256
    assert len(committed.entries) == 12
    assert committed.to_record()["backend_count"] == 8
    assert committed.to_record()["dataset_count"] == 3
    assert committed.metadata["parent_atlas_artifact_id"] == (
        "a04edd702cc95ed1cd89fe05f3a209b036c6d1e22406161b130d89c6c56cded4"
    )
    assert committed.metadata["new_outcomes_read"] is False
    assert committed.metadata["native_continuum_source_outcomes_read"] is False


def test_native_continuum_failures_are_mapped_to_their_exact_stage() -> None:
    atlas = load_simulator_validation_atlas(ATLAS)
    entries = {entry.display_name: entry for entry in atlas.entries}
    expected = {
        "MuJoCo Flex": (
            "passed",
            "failed",
            "not_evaluated",
            "not_applicable",
            "not_evaluated",
            "not_evaluated",
        ),
        "JAX-FEM v2": (
            "passed",
            "passed",
            "failed",
            "not_applicable",
            "not_evaluated",
            "not_evaluated",
        ),
        "SOFA FEM v3": (
            "passed",
            "passed",
            "failed",
            "not_applicable",
            "not_evaluated",
            "not_evaluated",
        ),
    }
    for display_name, statuses in expected.items():
        entry = entries[display_name]
        assert tuple(entry.stages[name].status for name in STAGE_NAMES) == statuses
        assert entry.decision == "rejected"
        assert entry.exact_fallback_retained is True
        assert entry.protocol_frozen_before_outcomes is True
        assert entry.protected_target_data_read is False
        assert entry.new_recording_used is False
        assert entry.metadata["public_source_outcome_opened"] is False
        assert entry.metadata["target_outcome_opened"] is False


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


def test_native_continuum_entries_cannot_escape_exact_fallback() -> None:
    atlas = load_simulator_validation_atlas(ATLAS)
    entries = {
        entry.display_name: entry
        for entry in atlas.entries
        if entry.display_name in {"MuJoCo Flex", "JAX-FEM v2", "SOFA FEM v3"}
    }
    for display_name, entry in entries.items():
        baseline = _Belief(_digest(f"{display_name}-baseline"))
        candidate = _Belief(_digest(f"{display_name}-candidate"))
        selected, receipt = select_prospectively_validated_candidate(
            baseline,
            candidate,
            atlas,
            query_id=str(entry.query_scope.query_id),
            inference_admissible=True,
        )
        assert selected is baseline
        assert receipt["selected_candidate"] is False
        assert receipt["reason"] == "query-not-prospectively-certified"
