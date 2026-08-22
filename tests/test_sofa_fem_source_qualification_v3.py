from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import bayesian_phystwin.sofa_fem_source_qualification_v3 as qualification_module
from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    attachment_targets_m,
    rigid_contact_projection_v1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz
from bayesian_phystwin.sofa_fem_canonical_source_v3 import (
    canonicalize_sofa_source_v3,
)
from bayesian_phystwin.sofa_fem_source_qualification_v3 import (
    FALLBACK_FILENAME,
    file_sha256,
    load_prepared_sofa_source_v3,
    load_sofa_source_inputs_v3,
    load_sofa_source_physics_protocol_v3,
    run_sofa_fem_source_qualification_v3,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/sofa_fem_zebra_source_physics_v3.json"


def test_frozen_protocol_binds_canonical_sofa_runtime_and_source_groups() -> None:
    protocol = load_sofa_source_physics_protocol_v3(PROTOCOL)

    assert protocol.canonical_profile_id == "sofa-fem-v1"
    assert protocol.producer_profile_id == "sofa-fem-v1"
    assert protocol.transport == "material-trajectory-v1"
    assert (
        protocol.runtime_id
        == "f46e53707317bc652499e2f3af5b860a330f9dbf06014364acd19ffaf8acca8e"
    )
    assert [group.group_id for group in protocol.source_groups] == [
        "double_lift_zebra",
        "double_stretch_zebra",
    ]
    assert protocol.protocol_sha256 == file_sha256(PROTOCOL)
    assert protocol.protocol_sha256 == (
        "4a9a72210787314e727b742795bb8c35af99ee6e75419d73435db2f1083eea73"
    )
    assert protocol.simulation["base_interval_substeps"] == 32
    assert protocol.simulation["refined_interval_substeps"] == 64
    assert protocol.simulation["qualification_frame_count"] == 2
    assert protocol.simulation["canonical_rounding_m"] == 1.0e-11
    assert protocol.gates["maximum_rigid_equivariance_error_m"] == 1.0e-12
    assert protocol.backend["predecessor_result_id"] == (
        "1f6871d2841e638bd666fb1d8bdb19abd6f7a1813f09335441dbd19d63d9cc2e"
    )
    assert (
        protocol.value["information_boundary"]["incumbent_prediction_arrays_allowed"]
        is False
    )
    assert (
        protocol.value["information_boundary"]["source_object_outcomes_allowed"]
        is False
    )


def test_frozen_protocol_ancestry_is_hash_verified() -> None:
    protocol = load_sofa_source_physics_protocol_v3(PROTOCOL)
    records = qualification_module._verify_protocol_ancestry(
        protocol,
        repo_root=ROOT,
    )

    assert records["native_smoke"]["sha256"] == (
        "1785b151adc66bd6b52850336d7ed1c633746a378cb7a466ec23b36a8d9ba442"
    )
    assert records["predecessor_protocol"]["sha256"] == (
        "76f2934082fec366b3a11c0c62d0f62802864dfde1e134f8c4143d9a285a8117"
    )
    assert records["predecessor_result"]["sha256"] == (
        "1508bd4f6f043825a8ad720a346e9cae0904da883e12ace4a2ba7e48a806084b"
    )


def _source_points() -> npt.NDArray[np.float32]:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.014, 0.0, 0.0],
            [0.0, 0.009, 0.0],
            [0.0, 0.0, 0.006],
            [0.002, 0.003, 0.001],
        ],
        dtype=np.float32,
    )


def _write_source(path: Path) -> None:
    points = _source_points()
    controller = np.repeat(points[:4][None], 2, axis=0)
    controller[1, :, 0] += 0.002
    write_deterministic_npz(
        path,
        {
            "frame_zero_points_m": points,
            "controller_points_m": controller,
            "attachment_indices": np.arange(4, dtype=np.int32),
            "attachment_weights": np.eye(4, dtype=np.float32),
            "action_support": np.ones(len(points), dtype=np.float32),
        },
    )


def _write_prepared(path: Path, source_path: Path) -> None:
    with np.load(source_path, allow_pickle=False) as source:
        points = np.asarray(source["frame_zero_points_m"], dtype=np.float64)
        indices = np.asarray(source["attachment_indices"], dtype=np.int64)
        raw_targets = attachment_targets_m(
            points,
            source["controller_points_m"],
            indices,
            source["attachment_weights"],
        )
    patches = (np.arange(4, dtype=np.int64),)
    contact = rigid_contact_projection_v1(points, indices, raw_targets, patches)
    write_deterministic_npz(
        path,
        {
            "points": points,
            "cells": np.asarray([[0, 1, 2, 4], [0, 1, 4, 3]], dtype=np.int32),
            "attachment_indices": indices,
            "projected_targets": contact.projected_targets_m,
            "rotations": contact.rotations,
            "translations": contact.translations_m,
            "patch_flat": np.arange(4, dtype=np.int64),
            "patch_offsets": np.asarray([0, 4], dtype=np.int64),
            "patch_ranks": np.asarray([3], dtype=np.int64),
        },
    )


def _synthetic_protocol_and_roots(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["simulation"]["base_interval_substeps"] = 2
    value["simulation"]["refined_interval_substeps"] = 4
    roots: dict[str, Path] = {}
    for raw_group in value["source_groups"]:
        group_id = str(raw_group["group_id"])
        root = tmp_path / "source" / group_id
        root.mkdir(parents=True)
        roots[group_id] = root
        source = root / "source-inputs.npz"
        prepared = root / "prepared.npz"
        incumbent = root / "incumbent.npz"
        _write_source(source)
        _write_prepared(prepared, source)
        incumbent.write_bytes(f"opaque-{group_id}".encode())
        raw_group.update(
            {
                "source_inputs_relative_path": source.name,
                "source_inputs_sha256": file_sha256(source),
                "prepared_archive_relative_path": prepared.name,
                "prepared_archive_sha256": file_sha256(prepared),
                "incumbent_relative_path": incumbent.name,
                "incumbent_sha256": file_sha256(incumbent),
                "frame_count": 2,
                "material_node_count": 5,
                "controller_point_count": 4,
                "attached_node_count": 4,
                "tetrahedron_count": 2,
                "expected_contact_patch_sizes": [4],
            }
        )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(value), encoding="utf-8")
    return protocol, roots


def _fake_native_replay(**kwargs: Any) -> SimpleNamespace:
    prepared = kwargs["prepared"]
    points = np.asarray(kwargs["points_m"], dtype=np.float64)
    contact = kwargs["contact"]
    driven = bool(kwargs["driven"])
    modulus = float(kwargs["young_modulus_pa"])
    substeps = int(kwargs["interval_substeps"])
    simulation = kwargs["simulation"]
    gauge = canonicalize_sofa_source_v3(
        points_m=points,
        cells=prepared.cells,
        attachment_indices=prepared.attachment_indices,
        contact=contact,
        canonical_rounding_m=float(simulation["canonical_rounding_m"]),
        minimum_relative_eigengap=float(simulation["minimum_relative_eigengap"]),
    )
    canonical_positions = np.repeat(gauge.canonical_points_m[None], 2, axis=0)
    if driven:
        canonical_positions[-1, prepared.attachment_indices] = (
            gauge.canonical_contact.projected_targets_m[-1]
        )
        if modulus <= 25_000.0:
            scale = 0.8
        elif modulus >= 500_000.0:
            scale = 1.2
        elif substeps >= 4:
            scale = 0.995
        else:
            scale = 1.0
        canonical_positions[-1, 4, 0] += scale * 0.001
    positions = np.ascontiguousarray(
        canonical_positions @ gauge.world_from_canonical.T + gauge.center_m
    )
    expected_targets = (
        contact.projected_targets_m
        if driven
        else np.repeat(contact.projected_targets_m[:1], 2, axis=0)
    )
    world_attachment = float(
        np.max(
            np.linalg.norm(
                positions[:, prepared.attachment_indices] - expected_targets,
                axis=2,
            )
        )
    )
    identity = f"{gauge.gauge_sha256}:{driven}:{substeps}:{modulus}".encode()
    digest = hashlib.sha256(identity).hexdigest()
    return SimpleNamespace(
        positions_m=positions,
        deformation_determinants=np.ones((2, 2), dtype=np.float64),
        minimum_continuation_deformation_determinant=1.0,
        maximum_attachment_error_m=0.0,
        maximum_world_attachment_approximation_error_m=world_attachment,
        native_step_count=substeps,
        scene_sha256=digest,
        schedule_sha256=hashlib.sha256(identity + b":schedule").hexdigest(),
        gauge_sha256=gauge.gauge_sha256,
        material_vertex_count=5,
        tetrahedron_count=2,
        attachment_count=4,
        total_reference_mass_kg=1.0,
        maximum_point_quantization_error_m=(gauge.maximum_point_quantization_error_m),
        maximum_target_quantization_error_m=(gauge.maximum_target_quantization_error_m),
        maximum_contact_reprojection_error_m=(
            gauge.maximum_contact_reprojection_error_m
        ),
    )


def test_protocol_rejects_boundary_or_backend_mutation(tmp_path: Path) -> None:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["unexpected"] = True
    path = tmp_path / "fields.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fields changed"):
        load_sofa_source_physics_protocol_v3(path)

    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["information_boundary"]["source_object_outcomes_allowed"] = True
    path = tmp_path / "boundary.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="information boundary"):
        load_sofa_source_physics_protocol_v3(path)

    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["backend"]["transport"] = "lagrangian-export-v1"
    path = tmp_path / "backend.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="transport"):
        load_sofa_source_physics_protocol_v3(path)


def test_source_and_prepared_loaders_replay_locked_contact(tmp_path: Path) -> None:
    protocol_path, roots = _synthetic_protocol_and_roots(tmp_path)
    protocol = load_sofa_source_physics_protocol_v3(protocol_path)
    group = protocol.source_groups[0]
    root = roots[group.group_id]
    source = root / group.source_inputs_relative_path.as_posix()
    prepared_path = root / group.prepared_archive_relative_path.as_posix()

    arrays = load_sofa_source_inputs_v3(source, group=group)
    prepared = load_prepared_sofa_source_v3(
        prepared_path,
        group=group,
        source_inputs=arrays,
        qualification_frame_count=2,
    )
    assert prepared.points_m.shape == (5, 3)
    assert prepared.cells.shape == (2, 4)
    assert prepared.contact.patch_ranks == (3,)

    prepared_path.write_bytes(prepared_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        load_prepared_sofa_source_v3(
            prepared_path,
            group=group,
            source_inputs=arrays,
            qualification_frame_count=2,
        )


def test_orchestrator_passes_without_opening_incumbent_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, roots = _synthetic_protocol_and_roots(tmp_path)
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
        "load_native_sofa_fem_modules_v1",
        lambda **_: SimpleNamespace(
            installed_records={"synthetic": {"sha256": "3" * 64}}
        ),
    )
    monkeypatch.setattr(
        qualification_module,
        "_verify_protocol_ancestry",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(qualification_module, "_native_replay", _fake_native_replay)

    output = tmp_path / "result"
    result = run_sofa_fem_source_qualification_v3(
        protocol_path=protocol,
        group_roots=roots,
        output_dir=output,
        repo_root=tmp_path,
        distribution_archive=tmp_path / "sofa.zip",
        sofa_root=tmp_path / "sofa",
    )

    assert result["qualified"] is True
    assert result["source_value_scoring_authorized"] is True
    boundary = result["information_boundary"]
    assert boundary["incumbent_bytes_read_for_exact_fallback"] is True
    assert boundary["incumbent_prediction_arrays_read"] is False
    assert boundary["source_object_outcomes_read"] is False
    assert boundary["target_or_held_out_artifact_read"] is False
    assert len(result["source_groups"]) == 2
    for record in result["source_groups"]:
        assert record["deterministic_replay_valid"] is True
        assert record["canonical_gauge_identity_under_rigid_pose"] is True
        assert record["canonical_scene_identity_under_rigid_pose"] is True
        assert record["canonical_schedule_identity_under_rigid_pose"] is True
        assert record["maximum_world_point_approximation_error_m"] <= 2.0e-11
        assert record["maximum_world_attachment_approximation_error_m"] <= 2.0e-11
        assert record["topology_identity_preserved"] is True
        assert record["exact_fallback_verified"] is True
        fallback = output / record["group_id"] / FALLBACK_FILENAME
        incumbent = roots[record["group_id"]] / "incumbent.npz"
        assert fallback.read_bytes() == incumbent.read_bytes()

    with pytest.raises(FileExistsError):
        run_sofa_fem_source_qualification_v3(
            protocol_path=protocol,
            group_roots=roots,
            output_dir=output,
            repo_root=tmp_path,
            distribution_archive=tmp_path / "sofa.zip",
            sofa_root=tmp_path / "sofa",
        )
    with pytest.raises(ValueError, match="complete frozen source roster"):
        run_sofa_fem_source_qualification_v3(
            protocol_path=protocol,
            group_roots={next(iter(roots)): next(iter(roots.values()))},
            output_dir=tmp_path / "incomplete",
            repo_root=tmp_path,
            distribution_archive=tmp_path / "sofa.zip",
            sofa_root=tmp_path / "sofa",
        )
