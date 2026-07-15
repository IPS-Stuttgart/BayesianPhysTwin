from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal4d_public.pokeflex import (
    PokeFlexReadinessConfig,
    preflight_pokeflex_dataset,
    write_synthetic_pokeflex_fixture,
)
from causal4d_public.pokeflex_source_qa import (
    PokeFlexSourceQaConfig,
    audit_pokeflex_source_take,
    load_source_qa_policy,
    run_pokeflex_source_qa,
    source_qa_artifact_sha256,
    validate_source_qa_artifact,
)


def _policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "pokeflex_source_qa_v1.json"
    )


def _align_tool_with_fixture_surface(robot_path: Path) -> None:
    records = json.loads(robot_path.read_text(encoding="utf-8"))
    for record in records:
        record["T_WT"][0][3] = 0.0
        record["T_WT"][1][3] = 0.0
        record["T_WT"][2][3] = 0.1
    robot_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def test_canonical_pokeflex_source_qa_policy_is_locked() -> None:
    config = load_source_qa_policy(_policy_path())

    assert config.expected_object_id == "3dPrintedBunny"
    assert len(config.expected_development_take_ids) == 5
    assert config.maximum_shared_graph_rigid_chamfer_mm == 5.0


def test_aligned_fixture_take_passes_pose_wrench_and_surface_gates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pokeflex"
    write_synthetic_pokeflex_fixture(root, takes_per_object=1, frame_count=16)
    take_root = root / "fixture_plush_00" / "poke_000"
    _align_tool_with_fixture_surface(take_root / "robot_data.json")

    record, first_surface = audit_pokeflex_source_take(take_root)

    assert first_surface.shape == (4, 3)
    assert record["contact_alignment"]["passed"] is True
    assert record["gates"]["surface_graph_geometry_ready"] is True
    assert record["gates"]["material_vertex_identity_ready"] is False


def test_source_qa_does_not_open_calibration_or_target_takes(tmp_path: Path) -> None:
    root = tmp_path / "pokeflex"
    write_synthetic_pokeflex_fixture(root, takes_per_object=7, frame_count=16)
    preflight = preflight_pokeflex_dataset(root, PokeFlexReadinessConfig())
    assignments = preflight["metadata_only_split"]["assignments"]
    development = sorted(
        value["take_id"] for value in assignments if value["split"] == "development"
    )
    unopened = sorted(
        value["take_id"] for value in assignments if value["split"] != "development"
    )
    for take_id in development:
        _align_tool_with_fixture_surface(
            root / "fixture_plush_00" / take_id / "robot_data.json"
        )
    for take_id in unopened:
        (root / "fixture_plush_00" / take_id / "robot_data.json").write_text(
            "{forbidden}\n", encoding="utf-8"
        )
    config = PokeFlexSourceQaConfig(
        expected_preflight_result_sha256=preflight["result_sha256"],
        expected_object_id="fixture_plush_00",
        expected_development_take_ids=tuple(development),
    )

    result = run_pokeflex_source_qa(root, preflight, config)

    assert result["source_qa_passed"] is True
    assert result["information_boundary"]["opened_take_ids"] == development
    assert result["information_boundary"]["unopened_take_ids"] == unopened
    assert result["information_boundary"]["target_take_data_read"] is False


def test_source_qa_artifact_rejects_tampering() -> None:
    payload = {
        "schema_version": 1,
        "artifact_kind": "PublicPokeFlexSourceQa",
        "source_qa_passed": True,
        "information_boundary": {
            "opened_take_ids": ["source"],
            "development_data_only": True,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
            "prediction_metrics_computed": False,
            "model_parameters_fitted": False,
        },
    }
    payload["result_sha256"] = source_qa_artifact_sha256(payload)

    assert validate_source_qa_artifact(payload)["passed"] is True
    payload["source_qa_passed"] = False
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_source_qa_artifact(payload)
