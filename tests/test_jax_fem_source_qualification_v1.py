from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import bayesian_phystwin.jax_fem_source_qualification_v1 as qualification_module
from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    _NativeReplay,
    attachment_targets_m,
    build_tetrahedral_cells_v1,
    contact_patch_local_indices_v1,
    deformation_determinants_v1,
    file_sha256,
    load_jax_fem_source_inputs_v1,
    load_jax_fem_source_physics_protocol_v1,
    mesh_component_count_v1,
    rigid_contact_projection_v1,
    rigid_transform_v1,
    run_jax_fem_source_qualification_v1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/jax_fem_zebra_source_physics_v1.json"
RUNNER = ROOT / "scripts/remote/run_jax_fem_source_qualification_v1.py"


def test_frozen_protocol_loads_and_binds_two_source_groups() -> None:
    protocol = load_jax_fem_source_physics_protocol_v1(PROTOCOL)

    assert protocol.canonical_profile_id == "jax-fem-quasistatic-v1"
    assert protocol.producer_profile_id == "jax-fem-quasistatic-v1"
    assert protocol.transport == "lagrangian-export-v1"
    assert (
        protocol.runtime_id
        == "20c46dfa402712247416730e82289d4d4cd46096cab8c15b49ddb84a69d02a81"
    )
    assert [group.group_id for group in protocol.source_groups] == [
        "double_lift_zebra",
        "double_stretch_zebra",
    ]
    assert protocol.source_groups[0].expected_contact_patch_sizes == (40, 67)
    assert protocol.source_groups[1].expected_base_cell_count == 23140
    assert protocol.simulation["base_frame_indices"] == [0, 3, 6, 9]
    assert protocol.simulation["refined_frame_indices"] == list(range(10))
    assert protocol.protocol_sha256 == file_sha256(PROTOCOL)
    assert (
        protocol.value["information_boundary"]["source_object_outcomes_allowed"]
        is False
    )


def test_attachment_targets_are_displacements_from_frame_zero() -> None:
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


def test_tetrahedral_mesh_is_deterministic_oriented_and_connected() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
            [0.01, 0.01, 0.01],
        ],
        dtype=np.float64,
    )

    first = build_tetrahedral_cells_v1(
        points,
        maximum_edge_m=0.03,
        minimum_shape_ratio=1.0e-4,
    )
    second = build_tetrahedral_cells_v1(
        points,
        maximum_edge_m=0.03,
        minimum_shape_ratio=1.0e-4,
    )

    np.testing.assert_array_equal(first, second)
    assert set(np.unique(first)) == set(range(len(points)))
    assert mesh_component_count_v1(first, node_count=len(points)) == 1
    vertices = points[first]
    signed = np.linalg.det(
        np.stack(
            (
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 3] - vertices[:, 0],
            ),
            axis=2,
        )
    )
    assert np.all(signed > 0.0)


def _two_patch_points() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    tetrahedron = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.0, 0.002],
        ],
        dtype=np.float64,
    )
    points = np.concatenate((tetrahedron, tetrahedron + [0.03, 0.0, 0.0]))
    return points, np.arange(8, dtype=np.int64)


def test_rigid_contact_projection_recovers_two_independent_patches() -> None:
    points, indices = _two_patch_points()
    patches = contact_patch_local_indices_v1(points, indices, radius_m=0.015)
    assert [len(patch) for patch in patches] == [4, 4]
    first_rotation = rigid_transform_v1([1.0, 2.0, 3.0], 0.1)
    second_rotation = rigid_transform_v1([3.0, 2.0, 1.0], 0.2)
    targets = np.repeat(points[None], 2, axis=0)
    targets[1, patches[0]] = points[patches[0]] @ first_rotation.T + [
        0.001,
        0.002,
        0.003,
    ]
    targets[1, patches[1]] = points[patches[1]] @ second_rotation.T + [
        -0.002,
        0.001,
        0.0,
    ]

    projection = rigid_contact_projection_v1(points, indices, targets, patches)

    assert projection.patch_ranks == (3, 3)
    np.testing.assert_array_equal(projection.projected_targets_m[0], points)
    np.testing.assert_allclose(
        projection.projected_targets_m[1], targets[1], atol=1e-15
    )
    assert all(np.linalg.det(rotation) > 0.0 for rotation in projection.rotations[1])


def test_deformation_determinants_report_uniform_scaling() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
    deformed = np.stack((points, points * 2.0))

    determinants = deformation_determinants_v1(points, cells, deformed)

    np.testing.assert_allclose(determinants[:, 0], [1.0, 8.0])


def test_protocol_rejects_boundary_or_mesh_policy_mutation(tmp_path: Path) -> None:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["information_boundary"]["source_object_outcomes_allowed"] = True
    bad_boundary = tmp_path / "bad-boundary.json"
    bad_boundary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="information boundary"):
        load_jax_fem_source_physics_protocol_v1(bad_boundary)

    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["simulation"]["mesh_policy"] = "unregistered"
    bad_mesh = tmp_path / "bad-mesh.json"
    bad_mesh.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="mesh policy"):
        load_jax_fem_source_physics_protocol_v1(bad_mesh)


def _source_archive(path: Path, *, points: npt.NDArray[np.float32]) -> None:
    frame_count = 10
    controller = np.zeros((frame_count, 2, 3), dtype=np.float32)
    controller[:, 0, 0] = np.linspace(0.0, 0.002, frame_count)
    controller[:, 1, 1] = np.linspace(0.0, 0.001, frame_count)
    weights = np.zeros((len(points), 2), dtype=np.float32)
    weights[:4, 0] = 1.0
    weights[4:, 1] = 1.0
    write_deterministic_npz(
        path,
        {
            "frame_zero_points_m": points,
            "controller_points_m": controller,
            "attachment_indices": np.arange(len(points), dtype=np.int32),
            "attachment_weights": weights,
            "action_support": np.ones(len(points), dtype=np.float32),
        },
    )


def _physical_archive(
    path: Path,
    *,
    points: npt.NDArray[np.float32],
) -> Path:
    prediction = np.repeat(points[None], 10, axis=0).astype(np.float32)
    return cast(
        Path,
        write_deterministic_npz(
            path,
            {
                "action_support": np.ones(len(points), dtype=np.float32),
                "driven_readout_m": prediction,
                "frame_zero_points_m": prediction[0],
                "persistence_m": prediction.copy(),
                "prediction_m": prediction,
                "zero_action_readout_m": prediction.copy(),
            },
        ),
    )


def _synthetic_protocol_and_roots(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    points64, _ = _two_patch_points()
    points = points64.astype(np.float32)
    roots: dict[str, Path] = {}
    for raw_group in value["source_groups"]:
        group_id = str(raw_group["group_id"])
        root = tmp_path / "source" / group_id
        root.mkdir(parents=True)
        roots[group_id] = root
        source = root / "source-inputs.npz"
        _source_archive(source, points=points)
        incumbent = _physical_archive(root / "incumbent.npz", points=points)
        raw_group.update(
            {
                "source_inputs_relative_path": source.name,
                "source_inputs_sha256": file_sha256(source),
                "incumbent_relative_path": incumbent.name,
                "incumbent_sha256": file_sha256(incumbent),
                "frame_count": 10,
                "material_node_count": len(points),
                "controller_point_count": 2,
                "attached_node_count": len(points),
                "expected_contact_patch_sizes": [4, 4],
                "expected_base_cell_count": 3,
                "expected_coarse_cell_count": 3,
            }
        )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(value), encoding="utf-8")
    return protocol, roots


def _fake_native_replay(**kwargs: Any) -> _NativeReplay:
    points = np.asarray(kwargs["points_m"], dtype=np.float64)
    contact = kwargs["contact"]
    frame_indices = tuple(kwargs["frame_indices"])
    poisson = float(kwargs["poisson_ratio"])
    driven = bool(kwargs["driven"])
    positions: list[npt.NDArray[np.float64]] = []
    for frame in frame_indices:
        if not driven:
            displacement = np.zeros(3, dtype=np.float64)
        else:
            attached_reference = points[np.concatenate(contact.patch_local_indices)]
            attached_target = contact.projected_targets_m[
                frame, np.concatenate(contact.patch_local_indices)
            ]
            displacement = np.mean(attached_target - attached_reference, axis=0)
            displacement *= 1.0 + 0.3 * (poisson - 0.35)
        positions.append(points + displacement)
    trajectory = np.ascontiguousarray(np.stack(positions))
    return _NativeReplay(
        frame_indices=frame_indices,
        positions_m=trajectory,
        deformation_determinants=np.ones(
            (len(frame_indices), len(kwargs["cells"])), dtype=np.float64
        ),
    )


def test_source_qualification_passes_and_copies_exact_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, roots = _synthetic_protocol_and_roots(tmp_path)
    synthetic_cells = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]], dtype=np.int32
    )
    monkeypatch.setattr(
        qualification_module,
        "_git_provenance",
        lambda _: {
            "git_head": "1" * 40,
            "git_worktree_clean": True,
            "source_files": {"synthetic.py": "2" * 64},
        },
    )
    monkeypatch.setattr(
        qualification_module,
        "_load_native_modules",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(
        qualification_module,
        "build_tetrahedral_cells_v1",
        lambda *_, **__: synthetic_cells.copy(),
    )
    monkeypatch.setattr(
        qualification_module,
        "_run_native_replay",
        _fake_native_replay,
    )

    output = tmp_path / "result"
    result = run_jax_fem_source_qualification_v1(
        protocol_path=protocol,
        group_roots=roots,
        output_dir=output,
        repo_root=ROOT,
    )

    assert result["qualified"] is True
    assert result["source_value_scoring_authorized"] is True
    assert result["failure_reasons"] == []
    assert result["information_boundary"]["source_object_outcomes_read"] is False
    for raw_group in json.loads(protocol.read_text(encoding="utf-8"))["source_groups"]:
        group_id = raw_group["group_id"]
        incumbent = roots[group_id] / raw_group["incumbent_relative_path"]
        fallback = output / group_id / "exact-incumbent-fallback.npz"
        assert fallback.read_bytes() == incumbent.read_bytes()


def test_source_loader_rejects_digest_mutation(tmp_path: Path) -> None:
    protocol = load_jax_fem_source_physics_protocol_v1(PROTOCOL)
    original = protocol.source_groups[0]
    points = np.zeros((original.material_node_count, 3), dtype=np.float32)
    source = tmp_path / "source.npz"
    frame_count = original.frame_count
    controller = np.zeros(
        (frame_count, original.controller_point_count, 3), dtype=np.float32
    )
    weights = np.zeros(
        (original.attached_node_count, original.controller_point_count),
        dtype=np.float32,
    )
    weights[:, 0] = 1.0
    write_deterministic_npz(
        source,
        {
            "frame_zero_points_m": points,
            "controller_points_m": controller,
            "attachment_indices": np.arange(
                original.attached_node_count, dtype=np.int32
            ),
            "attachment_weights": weights,
            "action_support": np.zeros(original.material_node_count, dtype=np.float32),
        },
    )
    group = original.__class__(
        **{
            **{
                field: getattr(original, field)
                for field in original.__dataclass_fields__
            },
            "source_inputs_sha256": file_sha256(source),
        }
    )
    arrays = load_jax_fem_source_inputs_v1(source, group=group)
    assert arrays["frame_zero_points_m"].shape == (original.material_node_count, 3)
    source.write_bytes(source.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        load_jax_fem_source_inputs_v1(source, group=group)


def test_runner_help_imports_without_native_jax_fem() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--group-root" in result.stdout
