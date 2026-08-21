from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.jax_fem_hyperelastic_source_value_v2 as module
import bayesian_phystwin.jax_fem_source_value_v1 as value_module
from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin.jax_fem_hyperelastic_source_value_v2 import (
    generate_jax_fem_hyperelastic_source_value_predictions_v2,
)
from bayesian_phystwin.jax_fem_hyperelastic_v2 import HyperelasticReplayV2
from bayesian_phystwin.jax_fem_source_qualification_v1 import file_sha256
from bayesian_phystwin.jax_fem_source_value_v1 import (
    GRID_FILENAME,
    load_jax_fem_source_value_protocol_v1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/jax_fem_zebra_source_value_v2.json"
RUNNER = ROOT / "scripts/remote/run_jax_fem_hyperelastic_source_value_v2.py"
FAILURE_RECEIPT = (
    ROOT
    / "results/sota/diagnostics/jax_fem_zebra_source_value_v2/failure.json"
)
V2_SOURCE_FILES = {
    "src/bayesian_phystwin/jax_fem_source_qualification_v1.py",
    "src/bayesian_phystwin/jax_fem_hyperelastic_v2.py",
    "src/bayesian_phystwin/jax_fem_hyperelastic_source_qualification_v2.py",
    "src/bayesian_phystwin/jax_fem_source_value_v1.py",
    "src/bayesian_phystwin/jax_fem_hyperelastic_source_value_v2.py",
    "scripts/remote/run_jax_fem_hyperelastic_source_value_v2.py",
}


def test_frozen_v2_value_protocol_reuses_the_v1_scientific_roster() -> None:
    protocol = load_jax_fem_source_value_protocol_v1(PROTOCOL)

    assert protocol.value["protocol_label"] == "jax-fem-zebra-source-value-v2"
    assert (
        protocol.runtime_id
        == "0c1a24a70c805eb6ade62d176fafd574b3fc1c07fef8ca80592db6bb9ad23d15"
    )
    assert (
        protocol.qualification_artifact_id
        == "820df616afcd911af2999aa3b208f8d2da1e2acbe62521bc9d1980fc317aba50"
    )
    assert protocol.poisson_ratios == (0.2, 0.35, 0.45)
    assert protocol.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert protocol.gates["minimum_full_horizon_deformation_determinant"] == 0.5
    assert protocol.value["information_boundary"]["no_replacement"] is True


def test_frozen_v2_source_physical_failure_closes_outcome_access() -> None:
    receipt = json.loads(FAILURE_RECEIPT.read_text(encoding="utf-8"))

    assert receipt["schema"] == (
        "bayesian-phystwin.jax-fem-hyperelastic-source-value-failure-v2"
    )
    assert receipt["implementation"]["protocol_sha256"] == file_sha256(PROTOCOL)
    assert receipt["qualification"] == {
        "source_physics_result_sha256": (
            "10c2bd94436b3b4414f30becd859667ddab88c0445aa17c950186fc6e1f434e3"
        ),
        "artifact_sha256": (
            "e2f0797d0778b6143a076debb4b2596baffd430477e55b0499b45d1b68d51ef6"
        ),
        "artifact_id": (
            "820df616afcd911af2999aa3b208f8d2da1e2acbe62521bc9d1980fc317aba50"
        ),
        "qualified": True,
    }
    assert receipt["execution"]["launch_count"] == 1
    assert receipt["execution"]["completed_native_solve_count"] == 217
    assert receipt["execution"]["expected_native_solve_count"] == 768
    assert receipt["execution"]["prediction_grid_published"] is False
    assert receipt["failure"] == {
        "stage": "frozen-native-prediction-grid",
        "exception_type": "ValueError",
        "message": (
            "JAX-FEM v2 continuation violated its hard orientation threshold"
        ),
        "source_independent_runtime_failure": False,
        "source_physical_admission_failure": True,
    }
    assert len(receipt["partial_artifacts"]) == 4
    assert all(
        artifact["relative_path"].startswith("double_lift_zebra/")
        for artifact in receipt["partial_artifacts"]
    )
    assert receipt["information_boundary"] == {
        "source_inputs_read": True,
        "incumbent_and_matphys_predictions_read": True,
        "prefix_outcomes_read": False,
        "future_outcomes_read": False,
        "target_or_held_out_artifact_read": False,
        "dlo4_dlo5_access": False,
        "held_v8_access": False,
    }
    assert receipt["decision"] == {
        "candidate_admitted": False,
        "source_value_passed": False,
        "prefix_scoring_authorized": False,
        "future_scoring_authorized": False,
        "independent_untouched_evaluation_authorized": False,
        "exact_incumbent_fallback_retained": True,
        "retry_authorized": False,
        "method_change_authorized_from_this_run": False,
    }


def _source_archive(path: Path, *, frame_count: int) -> Path:
    points = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32)
    controller = np.zeros((frame_count, 1, 3), dtype=np.float32)
    controller[:, 0, 0] = np.arange(frame_count, dtype=np.float32) * 0.001
    return cast(
        Path,
        write_deterministic_npz(
            path,
            {
                "frame_zero_points_m": points,
                "controller_points_m": controller,
                "attachment_indices": np.asarray([0], dtype=np.int32),
                "attachment_weights": np.asarray([[1.0]], dtype=np.float32),
                "action_support": np.ones(2, dtype=np.float32),
            },
        ),
    )


def _physical_archive(
    path: Path,
    *,
    frame_count: int,
    slope_m: float,
) -> Path:
    frame_zero = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32)
    prediction = np.repeat(frame_zero[None], frame_count, axis=0)
    prediction[:, :, 0] += np.arange(frame_count, dtype=np.float32)[:, None] * slope_m
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    return cast(
        Path,
        write_deterministic_npz(
            path,
            {
                "action_support": np.ones(2, dtype=np.float32),
                "driven_readout_m": prediction,
                "frame_zero_points_m": frame_zero,
                "persistence_m": persistence,
                "prediction_m": prediction,
                "zero_action_readout_m": persistence.copy(),
            },
        ),
    )


def _synthetic_inputs(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    dict[str, Path],
    dict[str, Path],
    Any,
]:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frame_count = 8
    roots: dict[str, Path] = {}
    matphys_paths: dict[str, Path] = {}
    physics_groups: list[Any] = []
    for raw_group in value["source_groups"]:
        group_id = str(raw_group["group_id"])
        root = tmp_path / "source" / group_id
        root.mkdir(parents=True)
        roots[group_id] = root
        source = _source_archive(root / "source.npz", frame_count=frame_count)
        incumbent = _physical_archive(
            root / "incumbent.npz",
            frame_count=frame_count,
            slope_m=0.0007,
        )
        matphys = _physical_archive(
            root / "matphys.npz",
            frame_count=frame_count,
            slope_m=0.0008,
        )
        matphys_paths[group_id] = matphys
        raw_group.update(
            {
                "source_inputs_relative_path": source.name,
                "source_inputs_sha256": file_sha256(source),
                "incumbent_relative_path": incumbent.name,
                "incumbent_sha256": file_sha256(incumbent),
                "matphys_sha256": file_sha256(matphys),
                "frame_count": frame_count,
                "material_particle_count": 2,
                "controller_point_count": 1,
                "attached_particle_count": 1,
                "base_cell_count": 1,
                "contact_patch_sizes": [1],
            }
        )
        physics_groups.append(
            SimpleNamespace(
                group_id=group_id,
                source_inputs_relative_path=PurePosixPath(source.name),
                source_inputs_sha256=file_sha256(source),
                incumbent_relative_path=PurePosixPath(incumbent.name),
                incumbent_sha256=file_sha256(incumbent),
                frame_count=frame_count,
                material_node_count=2,
                controller_point_count=1,
                attached_node_count=1,
                expected_base_cell_count=1,
                expected_contact_patch_sizes=(1,),
            )
        )
    physics_protocol = tmp_path / "physics-protocol.json"
    physics_result = tmp_path / "physics-result.json"
    qualification = tmp_path / "qualification.json"
    physics_protocol.write_text("{}\n", encoding="utf-8")
    physics_result.write_text("{}\n", encoding="utf-8")
    qualification.write_text("{}\n", encoding="utf-8")
    artifact_id = "a" * 64
    value["qualification"].update(
        {
            "source_physics_protocol_sha256": file_sha256(physics_protocol),
            "source_physics_result_sha256": file_sha256(physics_result),
            "qualification_artifact_sha256": file_sha256(qualification),
            "qualification_artifact_id": artifact_id,
        }
    )
    protocol = tmp_path / "protocol.json"
    write_atomic_json(value, protocol, overwrite=False)
    fake_physics = SimpleNamespace(
        runtime_id=value["qualification"]["runtime_id"],
        backend={"runtime_versions": {}, "installed_source_sha256": {}},
        simulation={
            "low_poisson_ratio": 0.2,
            "base_poisson_ratio": 0.35,
            "high_poisson_ratio": 0.45,
            "young_modulus_pa": 100000.0,
            "base_interval_substeps": 1,
            "newton_absolute_tolerance": 1e-8,
            "newton_relative_tolerance": 1e-10,
            "hard_minimum_deformation_determinant": 0.05,
        },
        base_protocol=SimpleNamespace(
            source_groups=tuple(physics_groups),
            simulation={
                "contact_cluster_radius_m": 0.015,
                "base_mesh_max_edge_m": 0.025,
                "minimum_tetrahedron_shape_ratio": 0.0001,
            },
        ),
    )
    return (
        protocol,
        physics_protocol,
        physics_result,
        qualification,
        roots,
        matphys_paths,
        fake_physics,
    )


def _fake_replay(**kwargs: Any) -> HyperelasticReplayV2:
    points = np.asarray(kwargs["points_m"], dtype=np.float64)
    frame_count = len(kwargs["contact"].rotations)
    poisson = float(kwargs["poisson_ratio"])
    slope = {0.2: 0.0009, 0.35: 0.001, 0.45: 0.0011}[poisson]
    positions = np.repeat(points[None], frame_count, axis=0)
    positions[:, :, 0] += np.arange(frame_count, dtype=np.float64)[:, None] * slope
    return HyperelasticReplayV2(
        positions_m=np.ascontiguousarray(positions),
        deformation_determinants=np.ones((frame_count, 1), dtype=np.float64),
        minimum_continuation_deformation_determinant=1.0,
        native_solve_count=frame_count,
    )


def test_v2_generator_seals_all_members_before_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        protocol,
        physics_protocol,
        physics_result,
        qualification,
        roots,
        matphys_paths,
        fake_physics,
    ) = _synthetic_inputs(tmp_path)
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        module,
        "load_material_backend_qualification_v1",
        lambda _: SimpleNamespace(
            artifact_id=protocol_value["qualification"]["qualification_artifact_id"]
        ),
    )
    monkeypatch.setattr(
        module,
        "require_qualified_material_backend_runtime",
        lambda **_: None,
    )
    monkeypatch.setattr(
        module,
        "load_jax_fem_hyperelastic_source_protocol_v2",
        lambda _: fake_physics,
    )
    monkeypatch.setattr(
        module,
        "load_native_jax_fem_modules_v2",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        module,
        "contact_patch_local_indices_v1",
        lambda *_args, **_kwargs: (np.asarray([0], dtype=np.int64),),
    )
    monkeypatch.setattr(
        module,
        "rigid_contact_projection_v1",
        lambda _points, _indices, targets, patches: SimpleNamespace(
            projected_targets_m=np.asarray(targets),
            rotations=np.repeat(np.eye(3)[None, None], len(targets), axis=0),
            translations_m=np.zeros((len(targets), 1, 3), dtype=np.float64),
            patch_local_indices=patches,
        ),
    )
    monkeypatch.setattr(
        module,
        "build_tetrahedral_cells_v1",
        lambda *_args, **_kwargs: np.asarray([[0, 0, 0, 0]], dtype=np.int32),
    )
    monkeypatch.setattr(module, "run_hyperelastic_replay_v2", _fake_replay)
    monkeypatch.setattr(
        module,
        "_git_provenance",
        lambda *_args, **_kwargs: {
            "git_head": "1" * 40,
            "git_worktree_clean": True,
            "source_files": {path: "2" * 64 for path in V2_SOURCE_FILES},
        },
    )

    output = tmp_path / "grid"
    grid = generate_jax_fem_hyperelastic_source_value_predictions_v2(
        protocol_path=protocol,
        physics_protocol_path=physics_protocol,
        physics_result_path=physics_result,
        qualification_path=qualification,
        group_roots=roots,
        matphys_paths=matphys_paths,
        output_dir=output,
        repo_root=tmp_path,
    )

    assert grid["successful_candidate_count_per_group"] == 3
    assert grid["information_boundary"] == {
        "source_inputs_read": True,
        "incumbent_and_matphys_predictions_read": True,
        "prefix_outcomes_read": False,
        "future_outcomes_read": False,
        "target_or_held_out_artifact_read": False,
    }
    loaded_protocol = load_jax_fem_source_value_protocol_v1(protocol)
    loaded = value_module._load_grid(
        output / GRID_FILENAME,
        protocol=loaded_protocol,
    )
    assert loaded["grid_id"] == grid["grid_id"]
    assert all(len(group["members"]) == 3 for group in grid["groups"])
    with pytest.raises(FileExistsError):
        generate_jax_fem_hyperelastic_source_value_predictions_v2(
            protocol_path=protocol,
            physics_protocol_path=physics_protocol,
            physics_result_path=physics_result,
            qualification_path=qualification,
            group_roots=roots,
            matphys_paths=matphys_paths,
            output_dir=output,
            repo_root=tmp_path,
        )


def test_v2_runner_help_imports_without_native_jax_fem() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "finalize-pre-prefix" in completed.stdout
