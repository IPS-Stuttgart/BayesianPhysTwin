from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.genesis_mpm_source_qualification_v1 import (
    attachment_targets_m,
    file_sha256,
    load_genesis_source_inputs_v1,
    load_genesis_source_physics_protocol_v1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/genesis_mpm_zebra_source_physics_v1.json"


def test_frozen_protocol_loads_and_binds_two_independent_groups() -> None:
    protocol = load_genesis_source_physics_protocol_v1(PROTOCOL)

    assert protocol.canonical_profile_id == "genesis-mpm-v1"
    assert protocol.producer_profile_id == "genesis-mpm-v1"
    assert protocol.transport == "material-trajectory-v1"
    assert (
        protocol.runtime_id
        == "aecd2a170f974a166495da0c8692631acebf09d7b605c4ec0f9621f49434132a"
    )
    assert [group.group_id for group in protocol.source_groups] == [
        "double_lift_zebra",
        "double_stretch_zebra",
    ]
    assert protocol.protocol_sha256 == file_sha256(PROTOCOL)
    assert protocol.simulation["base_substeps"] == 64
    assert protocol.simulation["refined_substeps"] == 128
    assert protocol.simulation["domain_padding_m"] == 0.15
    assert protocol.simulation["grid_aligned_translation_m"] == [
        0.15625,
        -0.09375,
        0.09375,
    ]
    assert (
        protocol.simulation["controller_boundary_policy"]
        == "frame-boundary-position-velocity-overwrite-free-particles-v1"
    )
    assert (
        protocol.value["information_boundary"]["source_object_outcomes_allowed"]
        is False
    )


def test_attachment_targets_preserve_frame_zero_and_candidate_residuals() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float64)
    controller = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]],
            [[0.01, 0.0, 0.0], [0.0, 0.03, 0.0]],
        ],
        dtype=np.float64,
    )
    indices = np.array([0, 1], dtype=np.int64)
    weights = np.array([[1.0, 0.0], [0.25, 0.75]], dtype=np.float64)

    targets = attachment_targets_m(points, controller, indices, weights)

    np.testing.assert_array_equal(targets[0], points)
    np.testing.assert_allclose(targets[1, 0], [0.01, 0.0, 0.0])
    np.testing.assert_allclose(targets[1, 1], [0.0125, 0.015, 0.0])


def _source_archive(
    path: Path,
    *,
    material_count: int,
    frame_count: int,
    controller_count: int,
    attached: int,
) -> None:
    points = np.zeros((material_count, 3), dtype=np.float32)
    controller = np.zeros((frame_count, controller_count, 3), dtype=np.float32)
    indices = np.arange(attached, dtype=np.int32)
    weights = np.zeros((attached, controller_count), dtype=np.float32)
    weights[:, 0] = 1.0
    write_deterministic_npz(
        path,
        {
            "frame_zero_points_m": points,
            "controller_points_m": controller,
            "attachment_indices": indices,
            "attachment_weights": weights,
            "action_support": np.zeros(material_count, dtype=np.float32),
        },
    )


def test_source_loader_rejects_digest_or_roster_mutation(tmp_path: Path) -> None:
    protocol = load_genesis_source_physics_protocol_v1(PROTOCOL)
    original = protocol.source_groups[0]
    source = tmp_path / "source.npz"
    _source_archive(
        source,
        material_count=original.material_particle_count,
        frame_count=original.frame_count,
        controller_count=original.controller_point_count,
        attached=original.attached_particle_count,
    )
    group = original.__class__(
        group_id=original.group_id,
        source_inputs_relative_path=original.source_inputs_relative_path,
        source_inputs_sha256=file_sha256(source),
        incumbent_relative_path=original.incumbent_relative_path,
        incumbent_sha256=original.incumbent_sha256,
        frame_count=original.frame_count,
        material_particle_count=original.material_particle_count,
        controller_point_count=original.controller_point_count,
        attached_particle_count=original.attached_particle_count,
    )
    arrays = load_genesis_source_inputs_v1(source, group=group)
    assert arrays["frame_zero_points_m"].shape == (original.material_particle_count, 3)

    source.write_bytes(source.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        load_genesis_source_inputs_v1(source, group=group)


def test_protocol_rejects_information_boundary_or_backend_mutation(
    tmp_path: Path,
) -> None:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["information_boundary"]["source_object_outcomes_allowed"] = True
    path = tmp_path / "bad-boundary.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="information boundary"):
        load_genesis_source_physics_protocol_v1(path)

    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["backend"]["transport"] = "lagrangian-export-v1"
    path = tmp_path / "bad-backend.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="transport"):
        load_genesis_source_physics_protocol_v1(path)
