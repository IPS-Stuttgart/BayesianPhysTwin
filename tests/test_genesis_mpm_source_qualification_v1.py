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

import bayesian_phystwin.genesis_mpm_source_qualification_v1 as qualification_module
from bayesian_phystwin.genesis_mpm_source_qualification_v1 import (
    attachment_targets_m,
    file_sha256,
    load_genesis_source_inputs_v1,
    load_genesis_source_physics_protocol_v1,
    run_genesis_mpm_source_qualification_v1,
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
    points: npt.NDArray[np.float32] = np.zeros((material_count, 3), dtype=np.float32)
    controller: npt.NDArray[np.float32] = np.zeros(
        (frame_count, controller_count, 3), dtype=np.float32
    )
    indices: npt.NDArray[np.int32] = np.arange(attached, dtype=np.int32)
    weights: npt.NDArray[np.float32] = np.zeros(
        (attached, controller_count), dtype=np.float32
    )
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


def _physical_archive(
    path: Path,
    *,
    points: npt.NDArray[np.float32],
    frame_count: int,
) -> Path:
    prediction = np.repeat(points[None], frame_count, axis=0).astype(np.float32)
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
    frame_count = 10
    material_count = 4
    controller_count = 2
    attached_count = 1
    roots: dict[str, Path] = {}
    for raw_group in value["source_groups"]:
        group_id = str(raw_group["group_id"])
        root = tmp_path / "source" / group_id
        root.mkdir(parents=True)
        roots[group_id] = root
        source = root / "source-inputs.npz"
        _source_archive(
            source,
            material_count=material_count,
            frame_count=frame_count,
            controller_count=controller_count,
            attached=attached_count,
        )
        points: npt.NDArray[np.float32] = np.zeros(
            (material_count, 3), dtype=np.float32
        )
        incumbent = _physical_archive(
            root / "incumbent.npz",
            points=points,
            frame_count=frame_count,
        )
        raw_group.update(
            {
                "source_inputs_relative_path": source.name,
                "source_inputs_sha256": file_sha256(source),
                "incumbent_relative_path": incumbent.name,
                "incumbent_sha256": file_sha256(incumbent),
                "frame_count": frame_count,
                "material_particle_count": material_count,
                "controller_point_count": controller_count,
                "attached_particle_count": attached_count,
            }
        )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(value), encoding="utf-8")
    return protocol, roots


def _fake_native_replay(**kwargs: Any) -> SimpleNamespace:
    points = np.asarray(kwargs["points_m"], dtype=np.float64)
    targets = np.asarray(kwargs["targets_m"], dtype=np.float64)
    translation = kwargs.get("translation_m")
    shift = (
        np.zeros(3, dtype=np.float64)
        if translation is None
        else np.asarray(translation, dtype=np.float64)
    )
    driven = bool(kwargs["driven"])
    modulus = float(kwargs["young_modulus_pa"])
    substeps = int(kwargs["substeps"])
    if not driven:
        final_displacement = 0.0
    elif modulus <= 25_000.0:
        final_displacement = 0.0008
    elif modulus >= 500_000.0:
        final_displacement = 0.0012
    else:
        final_displacement = 0.00099 if substeps >= 128 else 0.001
    fractions = np.linspace(0.0, 1.0, len(targets), dtype=np.float64)
    positions = np.repeat((points + shift)[None], len(targets), axis=0)
    positions[:, :, 0] += fractions[:, None] * final_displacement
    return SimpleNamespace(
        positions_m=np.ascontiguousarray(positions),
        active=np.ones(positions.shape[:2], dtype=np.bool_),
        deformation_determinants=np.ones(positions.shape[:2], dtype=np.float64),
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
    value["unexpected"] = True
    path = tmp_path / "bad-fields.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fields changed"):
        load_genesis_source_physics_protocol_v1(path)

    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["backend"] = "not-a-mapping"
    path = tmp_path / "bad-mapping.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_genesis_source_physics_protocol_v1(path)

    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
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


def test_source_qualification_orchestrator_passes_with_exact_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, roots = _synthetic_protocol_and_roots(tmp_path)
    fake_genesis = SimpleNamespace(cpu=object(), init=lambda **_: None)
    fake_torch = SimpleNamespace(
        set_num_threads=lambda _: None,
        use_deterministic_algorithms=lambda _: None,
    )
    monkeypatch.setitem(sys.modules, "genesis", fake_genesis)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
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
        "_run_native_replay",
        _fake_native_replay,
    )

    output = tmp_path / "result"
    result = run_genesis_mpm_source_qualification_v1(
        protocol_path=protocol,
        group_roots=roots,
        output_dir=output,
        repo_root=tmp_path,
    )

    assert result["qualified"] is True
    assert result["source_value_scoring_authorized"] is True
    assert result["information_boundary"]["source_object_outcomes_read"] is False
    assert len(result["source_groups"]) == 2
    for record in result["source_groups"]:
        assert record["deterministic_replay_valid"] is True
        assert record["topology_identity_preserved"] is True
        assert record["exact_fallback_verified"] is True
        root = roots[record["group_id"]]
        fallback = output / record["group_id"] / qualification_module.FALLBACK_FILENAME
        assert fallback.read_bytes() == (root / "incumbent.npz").read_bytes()

    with pytest.raises(FileExistsError):
        run_genesis_mpm_source_qualification_v1(
            protocol_path=protocol,
            group_roots=roots,
            output_dir=output,
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="complete frozen source roster"):
        run_genesis_mpm_source_qualification_v1(
            protocol_path=protocol,
            group_roots={next(iter(roots)): next(iter(roots.values()))},
            output_dir=tmp_path / "incomplete",
            repo_root=tmp_path,
        )


def test_git_provenance_binds_clean_source_files(tmp_path: Path) -> None:
    source = tmp_path / "src/bayesian_phystwin"
    runner = tmp_path / "scripts/remote"
    source.mkdir(parents=True)
    runner.mkdir(parents=True)
    (source / "genesis_mpm_source_qualification_v1.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (runner / "run_genesis_mpm_source_qualification_v1.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp_path, check=True)

    provenance = qualification_module._git_provenance(tmp_path)
    assert provenance["git_worktree_clean"] is True
    assert len(provenance["git_head"]) == 40
    assert set(provenance["source_files"]) == {
        "src/bayesian_phystwin/genesis_mpm_source_qualification_v1.py",
        "scripts/remote/run_genesis_mpm_source_qualification_v1.py",
    }

    (source / "genesis_mpm_source_qualification_v1.py").write_text(
        "VALUE = 3\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="clean Git worktree"):
        qualification_module._git_provenance(tmp_path)


def test_native_array_bridge_and_domain_bounds() -> None:
    class FakeTensor:
        def __init__(self, value: npt.NDArray[np.float64]) -> None:
            self.value = value

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def numpy(self) -> npt.NDArray[np.float64]:
            return self.value

    values = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    np.testing.assert_array_equal(
        qualification_module._host(FakeTensor(values)), values
    )
    np.testing.assert_array_equal(qualification_module._host(values), values)

    lower, upper = qualification_module._domain_bounds(
        values,
        values[None],
        padding_m=0.01,
    )
    np.testing.assert_allclose(np.subtract(upper, lower), [0.1, 0.1, 0.1])
