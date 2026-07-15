from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.pokeflex import write_synthetic_pokeflex_fixture
from causal4d_public.pokeflex_warp_source import (
    PokeFlexWarpCandidate,
    PokeFlexWarpSourceConfig,
    build_pokeflex_warp_case,
    load_warp_policy,
    summarize_pooling_controls,
    validate_warp_artifact,
    warp_artifact_sha256,
    warp_candidates,
)


def _policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "pokeflex_warp_source_v1.json"
    )


def _surface_grid() -> str:
    lines = []
    for row in range(4):
        for column in range(4):
            lines.append(f"v {20 * column} {20 * row} {2 * (row + column)}")
    return "\n".join(lines) + "\n"


def _prepare_fixture_take(take_root: Path) -> None:
    for mesh_path in (take_root / "meshes").glob("*.obj"):
        mesh_path.write_text(_surface_grid(), encoding="utf-8")
    records = json.loads((take_root / "robot_data.json").read_text(encoding="utf-8"))
    for record in records:
        record["T_WT"][0][3] = 0.0
        record["T_WT"][1][3] = 0.0
        record["T_WT"][2][3] = 0.0
    (take_root / "robot_data.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


def test_canonical_pokeflex_warp_policy_is_locked() -> None:
    config = load_warp_policy(_policy_path())

    assert len(warp_candidates(config)) == 50
    assert config.graph_node_count == 128
    assert config.substeps == 64


def test_build_case_uses_take_specific_connected_surface_graph(tmp_path: Path) -> None:
    root = tmp_path / "pokeflex"
    write_synthetic_pokeflex_fixture(root, takes_per_object=1, frame_count=16)
    take_root = root / "fixture_plush_00" / "poke_000"
    _prepare_fixture_take(take_root)
    config = PokeFlexWarpSourceConfig(
        graph_node_count=16,
        graph_knn=4,
        controller_patch_size=2,
        evaluation_surface_point_count=16,
        substeps=1,
    )

    case = build_pokeflex_warp_case(take_root, config)

    assert case.graph_connected is True
    assert case.rest_positions_m.shape == (16, 3)
    assert case.controller_positions_m.shape == (11, 2, 3)
    assert case.object_spring_count > 16
    assert case.input_summary["material_identity_used"] is False


def test_pooling_control_reports_direct_single_source_comparison() -> None:
    candidates = (
        PokeFlexWarpCandidate(100.0, 100.0, 0.0),
        PokeFlexWarpCandidate(300.0, 300.0, 0.0),
        PokeFlexWarpCandidate(1000.0, 1000.0, 0.0),
    )
    scores = np.asarray(
        [
            [0.010, 0.040, 0.030],
            [0.020, 0.018, 0.019],
            [0.050, 0.010, 0.040],
        ]
    )
    persistence = np.asarray([0.030, 0.030, 0.030])

    result = summarize_pooling_controls(
        candidates, ["a", "b", "c"], scores, persistence
    )

    assert result["pooled_candidate_index"] == 1
    assert len(result["leave_one_out"]) == 3
    assert all("single_source_median_chamfer_m" in fold for fold in result["leave_one_out"])


def test_warp_artifact_rejects_tampering() -> None:
    payload = {
        "schema_version": 1,
        "artifact_kind": "PublicPokeFlexWarpSourceBackend",
        "source_backend_admitted": False,
        "information_boundary": {
            "development_data_only": True,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
            "material_identity_metrics_computed": False,
        },
    }
    payload["result_sha256"] = warp_artifact_sha256(payload)

    assert validate_warp_artifact(payload)["passed"] is True
    payload["source_backend_admitted"] = True
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_warp_artifact(payload)
