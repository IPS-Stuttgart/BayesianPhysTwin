from __future__ import annotations

import hashlib
from pathlib import Path

from bayesian_phystwin.simulator_validation_atlas_v1 import (
    STAGE_NAMES,
    SimulatorValidationAtlasV1,
    SimulatorValidationEntryV1,
    load_simulator_validation_atlas,
)
from scripts.build_cross_backend_validation_atlas_v1 import (
    BOUND_FILES,
    build_atlas,
)

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "results/source/cross_backend_validation_atlas_v1/atlas.json"


def _by_name() -> tuple[
    SimulatorValidationAtlasV1, dict[str, SimulatorValidationEntryV1]
]:
    atlas = load_simulator_validation_atlas(ATLAS)
    return atlas, {entry.display_name: entry for entry in atlas.entries}


def test_bound_evidence_capsules_are_byte_identical_to_frozen_sources() -> None:
    for relative, expected in BOUND_FILES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_committed_atlas_is_exact_builder_output() -> None:
    committed = load_simulator_validation_atlas(ATLAS)
    rebuilt = build_atlas()
    assert committed.to_record() == rebuilt.to_record()
    assert hashlib.sha256(ATLAS.read_bytes()).hexdigest() == (
        "efb3c6070e76ba4f71b59025406ce7728798a540e614db6f5802e94f953f6097"
    )
    assert committed.artifact_id == (
        "a04edd702cc95ed1cd89fe05f3a209b036c6d1e22406161b130d89c6c56cded4"
    )
    assert len(committed.entries) == 9
    assert committed.to_record()["backend_count"] == 5
    assert committed.to_record()["dataset_count"] == 2
    assert committed.decision_counts == {
        "prospective_certified": 1,
        "rejected": 8,
    }


def test_exact_stage_roster_preserves_positive_and_negative_evidence() -> None:
    atlas, entries = _by_name()
    assert set(entries) == {
        "DLO-Lab wrapping",
        "DLO-Lab slingshot",
        "DLO-Lab coiling",
        "DLO-Lab separation",
        "DLO-Lab unknotting",
        "ARCSim Dirichlet",
        "Codim-IPC",
        "LibuIPC ensemble",
        "MatPhys pinned runtime",
    }
    expected = {
        "DLO-Lab wrapping": ("passed",) * 6,
        "DLO-Lab slingshot": (
            "passed",
            "passed",
            "passed",
            "failed",
            "failed",
            "failed",
        ),
        "DLO-Lab coiling": (
            "passed",
            "passed",
            "passed",
            "failed",
            "failed",
            "not_evaluated",
        ),
        "DLO-Lab separation": (
            "passed",
            "failed",
            "not_evaluated",
            "not_evaluated",
            "not_evaluated",
            "not_evaluated",
        ),
        "DLO-Lab unknotting": (
            "passed",
            "failed",
            "not_evaluated",
            "not_evaluated",
            "not_evaluated",
            "not_evaluated",
        ),
        "ARCSim Dirichlet": (
            "passed",
            "passed",
            "passed",
            "not_applicable",
            "failed",
            "not_evaluated",
        ),
        "Codim-IPC": (
            "passed",
            "passed",
            "failed",
            "not_applicable",
            "not_evaluated",
            "not_evaluated",
        ),
        "LibuIPC ensemble": (
            "passed",
            "passed",
            "failed",
            "not_applicable",
            "not_evaluated",
            "not_evaluated",
        ),
        "MatPhys pinned runtime": (
            "failed",
            "not_evaluated",
            "not_evaluated",
            "not_applicable",
            "not_evaluated",
            "not_evaluated",
        ),
    }
    for name, statuses in expected.items():
        assert tuple(entries[name].stages[stage].status for stage in STAGE_NAMES) == (
            statuses
        )
    assert entries["DLO-Lab wrapping"].decision == "prospective_certified"
    assert entries["DLO-Lab wrapping"].independent_group_count == 288
    assert all(
        entries[name].independent_group_count == 1
        for name in (
            "ARCSim Dirichlet",
            "Codim-IPC",
            "LibuIPC ensemble",
            "MatPhys pinned runtime",
        )
    )
    assert len(atlas.prospectively_certified_query_ids) == 1


def test_arcsim_gain_over_weak_prior_does_not_override_stronger_comparators() -> None:
    _, entries = _by_name()
    stage = entries["ARCSim Dirichlet"].stages["source_value"]
    assert stage.status == "failed"
    assert stage.metrics["candidate_real_to_sim_l1_m"] == 0.05322464662233837
    comparators = stage.metrics["comparators"]
    assert comparators["physical_real_to_sim_l1_m"] == 0.05939356004993469
    assert comparators["selected_dynamic_real_to_sim_l1_m"] == 0.04964315282920389
    assert comparators["published_garment_dynamics_real_to_sim_l1_m"] == 0.0419
    improvements = stage.metrics["relative_improvements"]
    assert improvements["physical_real_to_sim_l1_m"] > 0.10
    assert improvements["selected_dynamic_real_to_sim_l1_m"] < 0.0
    assert improvements["published_garment_dynamics_real_to_sim_l1_m"] < 0.0


def test_atlas_contains_no_new_or_protected_outcome_claim() -> None:
    atlas, entries = _by_name()
    assert atlas.metadata["new_outcomes_read"] is False
    assert atlas.metadata["new_recordings"] is False
    assert atlas.metadata["protected_target_data_read"] is False
    assert atlas.to_record()["backend_wide_competence_claim"] is False
    assert atlas.to_record()["cross_backend_ranking_claim"] is False
    assert atlas.to_record()["official_benchmark_claim"] is False
    assert all(not entry.protected_target_data_read for entry in entries.values())
    assert all(not entry.new_recording_used for entry in entries.values())
