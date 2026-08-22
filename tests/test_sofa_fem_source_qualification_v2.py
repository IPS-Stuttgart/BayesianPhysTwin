from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import bayesian_phystwin.sofa_fem_source_qualification_v2 as qualification_module
from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    attachment_targets_m,
    rigid_contact_projection_v1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz
from bayesian_phystwin.sofa_fem_source_qualification_v2 import (
    FALLBACK_FILENAME,
    file_sha256,
    load_prepared_sofa_source_v2,
    load_sofa_source_inputs_v2,
    load_sofa_source_physics_protocol_v2,
    run_sofa_fem_source_qualification_v2,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/sofa_fem_zebra_source_physics_v2.json"


def test_frozen_protocol_binds_exact_sofa_runtime_and_two_source_groups() -> None:
    protocol = load_sofa_source_physics_protocol_v2(PROTOCOL)

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
    assert protocol.simulation["base_interval_substeps"] == 32
    assert protocol.simulation["refined_interval_substeps"] == 64
    assert protocol.simulation["qualification_frame_count"] == 2
    assert (
        protocol.value["information_boundary"]["incumbent_prediction_arrays_allowed"]
        is False
    )
    assert (
        protocol.value["information_boundary"]["source_object_outcomes_allowed"]
        is False
    )


def _source_points() -> npt.NDArray[np.float32]:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
            [0.002, 0.002, 0.002],
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
    positions = np.repeat(points[None], 2, axis=0)
    if driven:
        action = np.mean(
            contact.projected_targets_m[-1] - contact.projected_targets_m[0],
            axis=0,
        )
        positions[-1, prepared.attachment_indices] = contact.projected_targets_m[-1]
        if modulus <= 25_000.0:
            scale = 0.8
        elif modulus >= 500_000.0:
            scale = 1.2
        elif substeps >= 4:
            scale = 0.995
        else:
            scale = 1.0
        positions[-1, 4] += scale * action
    digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
    return SimpleNamespace(
        positions_m=np.ascontiguousarray(positions),
        deformation_determinants=np.ones((2, 2), dtype=np.float64),
        minimum_continuation_deformation_determinant=1.0,
        maximum_attachment_error_m=0.0,
        native_step_count=substeps,
        scene_sha256=digest,
        schedule_sha256=hashlib.sha256(
            np.ascontiguousarray(contact.projected_targets_m).tobytes()
        ).hexdigest(),
        material_vertex_count=5,
        tetrahedron_count=2,
        attachment_count=4,
        total_reference_mass_kg=1.0,
    )


def test_protocol_rejects_boundary_or_backend_mutation(tmp_path: Path) -> None:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["unexpected"] = True
    path = tmp_path / "fields.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fields changed"):
        load_sofa_source_physics_protocol_v2(path)

    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["information_boundary"]["source_object_outcomes_allowed"] = True
    path = tmp_path / "boundary.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="information boundary"):
        load_sofa_source_physics_protocol_v2(path)

    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["backend"]["transport"] = "lagrangian-export-v1"
    path = tmp_path / "backend.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="transport"):
        load_sofa_source_physics_protocol_v2(path)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: qualification_module._mapping([], name="value"), "mapping"),
        (
            lambda: qualification_module._exact_fields(
                {"extra": 1}, frozenset({"required"}), "value"
            ),
            "fields changed",
        ),
        (
            lambda: qualification_module._canonical_string(" padded ", name="value"),
            "canonical",
        ),
        (lambda: qualification_module._sha256("0" * 63, name="value"), "SHA-256"),
        (
            lambda: qualification_module._git_revision("g" * 40, name="value"),
            "Git revision",
        ),
        (lambda: qualification_module._positive_int(0, name="value"), "positive"),
        (
            lambda: qualification_module._nonnegative_int(-1, name="value"),
            "nonnegative",
        ),
        (lambda: qualification_module._finite(True, name="value"), "finite"),
        (
            lambda: qualification_module._finite(float("nan"), name="value"),
            "finite",
        ),
        (lambda: qualification_module._vector3([1.0], name="value"), "three-element"),
        (
            lambda: qualification_module._positive_int_tuple([], name="value"),
            "nonempty",
        ),
        (
            lambda: qualification_module._canonical_relative_path(
                "../value", name="value"
            ),
            "canonical",
        ),
    ],
)
def test_protocol_helpers_reject_noncanonical_values(
    operation: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_source_and_prepared_loaders_replay_locked_contact(tmp_path: Path) -> None:
    protocol_path, roots = _synthetic_protocol_and_roots(tmp_path)
    protocol = load_sofa_source_physics_protocol_v2(protocol_path)
    group = protocol.source_groups[0]
    root = roots[group.group_id]
    source = root / group.source_inputs_relative_path.as_posix()
    prepared_path = root / group.prepared_archive_relative_path.as_posix()

    arrays = load_sofa_source_inputs_v2(source, group=group)
    prepared = load_prepared_sofa_source_v2(
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
        load_prepared_sofa_source_v2(
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
    monkeypatch.setattr(qualification_module, "_native_replay", _fake_native_replay)

    output = tmp_path / "result"
    result = run_sofa_fem_source_qualification_v2(
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
        assert record["topology_identity_preserved"] is True
        assert record["exact_fallback_verified"] is True
        fallback = output / record["group_id"] / FALLBACK_FILENAME
        incumbent = roots[record["group_id"]] / "incumbent.npz"
        assert fallback.read_bytes() == incumbent.read_bytes()

    with pytest.raises(FileExistsError):
        run_sofa_fem_source_qualification_v2(
            protocol_path=protocol,
            group_roots=roots,
            output_dir=output,
            repo_root=tmp_path,
            distribution_archive=tmp_path / "sofa.zip",
            sofa_root=tmp_path / "sofa",
        )
    with pytest.raises(ValueError, match="complete frozen source roster"):
        run_sofa_fem_source_qualification_v2(
            protocol_path=protocol,
            group_roots={next(iter(roots)): next(iter(roots.values()))},
            output_dir=tmp_path / "incomplete",
            repo_root=tmp_path,
            distribution_archive=tmp_path / "sofa.zip",
            sofa_root=tmp_path / "sofa",
        )
