import copy
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_bias_aware_prospective_v2_protocol import (
    CANONICAL_CONFIG_SHA256,
    EXPECTED_FRESH_CALIBRATION,
    build_bias_aware_prospective_v2_protocol,
    load_bias_aware_prospective_v2_protocol,
    metadata_ranked_episode_ids,
    metadata_ranked_fresh_filament_objects,
    validate_bias_aware_prospective_v2_protocol,
)


ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "deform360_bias_aware_guarded_belief_prospective_v2.json"
)


def _objects(cohort: dict[str, list[dict[str, object]]]) -> set[str]:
    return {
        str(record["object_id"])
        for records in cohort.values()
        for record in records
    }


def test_committed_v2_protocol_matches_canonical_builder() -> None:
    committed = load_bias_aware_prospective_v2_protocol(PROTOCOL, root=ROOT)

    assert committed == build_bias_aware_prospective_v2_protocol()
    assert committed["config_sha256"] == CANONICAL_CONFIG_SHA256
    assert committed["config"]["status"] == (
        "locked-before-fresh-calibration-download-or-media-access"
    )


def test_fresh_filament_selection_is_name_only_and_deterministic() -> None:
    payload = load_bias_aware_prospective_v2_protocol(PROTOCOL)
    repair = payload["config"]["repair"]

    assert metadata_ranked_fresh_filament_objects()[:3] == (
        "078-fishing-line",
        "161-tube",
        "088-snake",
    )
    assert {
        record["object_id"]: tuple(record["episode_ids"])
        for record in repair["fresh_calibration"]
    } == EXPECTED_FRESH_CALIBRATION
    for object_id, episode_ids in EXPECTED_FRESH_CALIBRATION.items():
        assert metadata_ranked_episode_ids(object_id)[:1] == episode_ids
    assert all(
        audit["selected_object_matching_path_count"] == 0
        for audit in repair["prelock_server_path_audit"]
    )


def test_v2_preserves_reserved_targets_method_and_support_gates() -> None:
    v2 = load_bias_aware_prospective_v2_protocol(PROTOCOL)["config"]
    v1 = json.loads(
        (
            ROOT
            / "configs"
            / "sota"
            / "deform360_bias_aware_guarded_belief_prospective_v1.json"
        ).read_text(encoding="utf-8")
    )["config"]

    assert v2["target_cohort"] == v1["target_cohort"]
    assert _objects(v2["target_cohort"]).isdisjoint(
        _objects(v2["calibration_cohort"])
    )
    assert len(_objects(v2["target_cohort"])) == 12
    assert v2["repair"]["method_family_changed"] is False
    assert v2["repair"]["candidate_threshold_changed"] is False
    assert v2["repair"]["calibration_gate_changed"] is False
    assert v2["calibration_support_gate"]["minimum_automatic_twin_objects"] == (
        v1["calibration_gate"]["minimum_evaluable_objects"]
    )
    assert v2["calibration_support_gate"][
        "minimum_automatic_twin_objects_per_stratum"
    ] == v1["calibration_gate"]["minimum_evaluable_objects_per_stratum"]
    assert v2["method"]["source_lock_sha256"] == v1["method"][
        "source_lock_sha256"
    ]


def test_v2_fallback_cannot_manufacture_calibration_support() -> None:
    fallback = load_bias_aware_prospective_v2_protocol(PROTOCOL)["config"][
        "reconstruction_fallback"
    ]

    assert fallback["physical_policy"] == "persistence_only"
    assert fallback["candidate_and_baseline_bit_exact"] is True
    assert fallback["eligible_for_absolute_accuracy_or_calibration"] is False
    assert fallback["counts_as_paired_non_regression_tie"] is True


def test_v2_rejects_target_or_gate_mutation() -> None:
    payload = build_bias_aware_prospective_v2_protocol()
    mutated_target = copy.deepcopy(payload)
    mutated_target["config"]["target_cohort"]["filament"][0]["episode_ids"] = [0]
    mutated_target["config_sha256"] = CANONICAL_CONFIG_SHA256
    with pytest.raises(ValueError, match="checksum changed"):
        validate_bias_aware_prospective_v2_protocol(mutated_target)

    mutated_gate = copy.deepcopy(payload)
    mutated_gate["config"]["calibration_support_gate"][
        "minimum_automatic_twin_objects"
    ] = 5
    mutated_gate["config_sha256"] = CANONICAL_CONFIG_SHA256
    with pytest.raises(ValueError, match="checksum changed"):
        validate_bias_aware_prospective_v2_protocol(mutated_gate)
