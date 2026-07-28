from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_active_query_feasibility import (
    ARCHIVE_FILENAME,
    PROTOCOL_ID,
    ActiveQueryFeasibilityConfig,
    build_active_query_feasibility_audit,
    readout_modes_to_node_basis,
    validate_active_query_feasibility_artifacts,
    write_active_query_feasibility_artifacts,
)


def _synthetic_inputs(
    *,
    supported_camera_count: int = 4,
) -> tuple[np.ndarray, ...]:
    frame_count = 4
    node_count = 10
    camera_count = 4
    height = width = 96
    frame_zero = np.column_stack(
        (
            np.linspace(-0.18, 0.18, node_count),
            np.linspace(-0.08, 0.08, node_count),
            np.full(node_count, 2.0),
        )
    )
    rollout = np.repeat(frame_zero[None], frame_count, axis=0)
    rollout[:, :, 1] += np.linspace(0.0, 0.009, frame_count)[:, None]
    graph_basis = np.zeros((node_count, 3, 8), dtype=np.float64)
    for mode in range(8):
        graph_basis[:, mode % 3, mode] = (
            np.arange(node_count, dtype=np.float64) + mode + 1.0
        )
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 60.0
    intrinsics[:, 1, 1] = 60.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    poses = np.repeat(np.eye(4)[None], camera_count, axis=0)
    poses[:, :3, 3] = np.asarray(
        [
            [-0.12, 0.00, 0.00],
            [0.12, 0.00, 0.00],
            [0.00, -0.12, 0.00],
            [0.00, 0.12, 0.00],
        ]
    )
    shapes = np.repeat(
        np.asarray([[height, width]], dtype=np.int64),
        camera_count,
        axis=0,
    )
    depth = np.full((camera_count, height, width), 2.0, dtype=np.float64)
    masks = np.zeros_like(depth, dtype=bool)
    masks[:supported_camera_count] = True
    names = np.asarray([f"camera_{index}" for index in range(camera_count)])
    return (
        rollout,
        graph_basis,
        intrinsics,
        poses,
        shapes,
        names,
        depth,
        masks,
    )


def test_complete_frame_zero_budget_is_admitted_deterministically() -> None:
    inputs = _synthetic_inputs()

    first = build_active_query_feasibility_audit(*inputs)
    second = build_active_query_feasibility_audit(*inputs)

    assert first.admitted
    assert first.plan.initial_query_count == 8
    assert len(first.candidate_entity_ids) == 10
    assert np.all(first.candidate_support_count == 4)
    assert np.all(np.sum(first.plan.camera_mask, axis=1) == 4)
    assert first.artifact_sha256 == second.artifact_sha256
    np.testing.assert_array_equal(first.plan.node_ids, second.plan.node_ids)
    assert not first.candidate_entity_ids.flags.writeable


def test_two_view_association_gate_abstains_without_relaxation() -> None:
    inputs = _synthetic_inputs(supported_camera_count=1)

    audit = build_active_query_feasibility_audit(*inputs)

    assert not audit.admitted
    assert len(audit.candidate_entity_ids) == 0
    assert audit.plan.initial_query_count == 0
    assert audit.plan.minimum_camera_support == 2


def test_readout_mode_conversion_rejects_an_invalid_vector_basis() -> None:
    with pytest.raises(ValueError, match="graph_basis"):
        readout_modes_to_node_basis(
            np.ones((4, 2)),
            node_count=3,
            rank=2,
        )


def test_sealed_artifact_validates_and_detects_archive_tampering(
    tmp_path: Path,
) -> None:
    audit = build_active_query_feasibility_audit(*_synthetic_inputs())
    protocol = tmp_path / "protocol.json"
    physical_manifest = tmp_path / "physical.json"
    physical_archive = tmp_path / "physical.npz"
    protocol.write_text("{}\n", encoding="utf-8")
    physical_manifest.write_text("{}\n", encoding="utf-8")
    physical_archive.write_bytes(b"sealed-physical-input")
    output = tmp_path / "result"

    report = write_active_query_feasibility_artifacts(
        output,
        audit,
        case_id="source-case",
        repository_revision="a" * 40,
        protocol_path=protocol,
        physical_manifest_path=physical_manifest,
        physical_archive_path=physical_archive,
        camera_certificate_sha256="b" * 64,
    )
    validated, arrays = validate_active_query_feasibility_artifacts(output)

    assert validated["result_sha256"] == report["result_sha256"]
    assert validated["status"] == "admitted"
    assert len(arrays["plan_node_ids"]) == 8
    with (output / ARCHIVE_FILENAME).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="archive checksum"):
        validate_active_query_feasibility_artifacts(output)


def test_v10_source_protocol_matches_typed_defaults_and_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "configs/sota/deform360_active_query_feasibility_source_v10.json"
        ).read_text(encoding="utf-8")
    )
    cases = payload["cases"]

    assert payload["protocol_id"] == PROTOCOL_ID
    assert ActiveQueryFeasibilityConfig(**payload["feasibility"]) == (
        ActiveQueryFeasibilityConfig()
    )
    assert len(cases) == payload["source_gate"]["locked_case_count"] == 8
    assert len({record["case"] for record in cases}) == 8
    assert all(len(record["physical_archive_sha256"]) == 64 for record in cases)
    assert payload["source_gate"]["minimum_admitted_case_count"] == 6
    assert payload["source_gate"]["tracker_execution_allowed_during_gate"] is False
    assert (
        payload["source_gate"][
            "candidate_state_updates_allowed_during_gate"
        ]
        is False
    )
    boundary = payload["information_boundary"]
    assert boundary["object_observation_frames_used"] == [0]
    assert boundary["future_object_depth_or_mask_read"] is False
    assert boundary["future_metric_read"] is False
    assert boundary["v1_sealed_target_read"] is False
    assert boundary["held_v8_read"] is False
