from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.rgbench_arcsim import (
    ARCSimClothParameters,
    ARCSimRollout,
    load_arcsim_vertices,
    write_arcsim_isotropic_material,
    write_arcsim_scene,
)
from bayesian_phystwin.rgbench_libuipc import (
    FlingPinController,
    PositionTrajectory,
)
from bayesian_phystwin.rgbench_online_belief import sha256_file
from scripts.held.run_rgbbench_arcsim_competence_v8 import (
    ARTIFACT_KIND,
    PROTOCOL_ID,
    SOURCE_DIGEST_KEYS,
    SUPPORTED_PROTOCOLS,
    _load_protocol,
    _parameters,
)


def _parameters_for_test(**changes: object) -> ARCSimClothParameters:
    values: dict[str, object] = {
        "timestep_s": 0.01,
        "youngs_modulus_pa": 230000.0,
        "poisson_ratio": 0.35,
        "volume_density_kg_m3": 220.0,
        "thickness_m": 0.001,
        "damping_s": 0.0,
        "handle_stiffness": 1e8,
        "gravity_m_s2": (0.0, 0.0, -9.81),
    }
    values.update(changes)
    return ARCSimClothParameters(**values)  # type: ignore[arg-type]


def _controller() -> FlingPinController:
    times = np.asarray([0.0, 1.0])
    left = PositionTrajectory(
        times_s=times,
        positions_m=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    )
    right = PositionTrajectory(
        times_s=times,
        positions_m=np.asarray([[1.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
    )
    return FlingPinController(
        pin_indices=(0, 2),
        initial_positions_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        left=left,
        right=right,
        prepare_time_s=0.0,
        wait_time_s=0.0,
    )


def test_arcsim_parameters_reject_nonphysical_values() -> None:
    with pytest.raises(ValueError, match="poisson_ratio"):
        _parameters_for_test(poisson_ratio=0.5)
    with pytest.raises(ValueError, match="damping_s"):
        _parameters_for_test(damping_s=-1.0)
    with pytest.raises(ValueError, match="handle_stiffness"):
        _parameters_for_test(handle_stiffness=0.0)


def test_arcsim_metric_shell_conversion() -> None:
    parameters = _parameters_for_test(
        youngs_modulus_pa=1200.0,
        poisson_ratio=0.2,
        volume_density_kg_m3=100.0,
        thickness_m=0.01,
    )
    assert parameters.areal_density_kg_m2 == pytest.approx(1.0)
    assert parameters.membrane_coefficients_n_m == pytest.approx(
        (12.5, 2.5, 12.5, 20.0)
    )
    assert parameters.bending_stiffness_n_m == pytest.approx(
        1200.0 * 0.01**3 / (12.0 * (1.0 - 0.2**2))
    )


def test_material_writer_uses_arcsim_tables_without_overwrite(
    tmp_path: Path,
) -> None:
    parameters = _parameters_for_test()
    destination = tmp_path / "material.json"
    write_arcsim_isotropic_material(destination, parameters)
    payload = json.loads(destination.read_text(encoding="ascii"))
    assert payload["density"] == pytest.approx(parameters.areal_density_kg_m2)
    assert len(payload["stretching"]) == 6
    assert len(payload["bending"]) == 3
    assert all(len(row) == 4 for row in payload["stretching"])
    assert all(len(row) == 5 for row in payload["bending"])
    assert payload["stretching"][0] == pytest.approx(
        [0.5 * value for value in parameters.membrane_coefficients_n_m]
    )
    with pytest.raises(ValueError, match="overwrite"):
        write_arcsim_isotropic_material(destination, parameters)


def test_scene_binds_fixed_topology_and_two_known_pin_trajectories(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "mesh.obj"
    mesh.write_text(
        "v 0 0 0\nv 0.5 0 0\nv 1 0 0\nf 1 2 3\n",
        encoding="ascii",
    )
    material = tmp_path / "material.json"
    material.write_text("{}\n", encoding="ascii")
    destination = tmp_path / "scene.json"
    step_count = write_arcsim_scene(
        destination,
        mesh_path=mesh,
        material_path=material,
        controller=_controller(),
        parameters=_parameters_for_test(),
        duration_s=0.1,
        initial_pose_xyz_wxyz=(0.1, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0),
    )
    payload = json.loads(destination.read_text(encoding="ascii"))
    assert step_count == 10
    assert payload["handles"] == [
        {"motion": 0, "nodes": [0]},
        {"motion": 1, "nodes": [2]},
    ]
    assert payload["cloths"][0]["transform"]["translate"] == [0.1, 0.0, 0.01]
    assert len(payload["motions"]) == 2
    assert len(payload["motions"][0]) == 11
    assert payload["motions"][0][-1]["transform"]["translate"] == pytest.approx(
        [0.01, 0.0, 0.0]
    )
    assert {
        "collision",
        "remeshing",
        "plasticity",
        "strainlimiting",
    } <= set(payload["disable"])


def test_scene_adds_kinematic_handle_flag_only_when_requested(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "mesh.obj"
    mesh.write_text(
        "v 0 0 0\nv 0.5 0 0\nv 1 0 0\nf 1 2 3\n",
        encoding="ascii",
    )
    material = tmp_path / "material.json"
    material.write_text("{}\n", encoding="ascii")
    destination = tmp_path / "scene.json"
    write_arcsim_scene(
        destination,
        mesh_path=mesh,
        material_path=material,
        controller=_controller(),
        parameters=_parameters_for_test(kinematic_handles=True),
        duration_s=0.1,
        initial_pose_xyz_wxyz=(0.1, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0),
    )
    payload = json.loads(destination.read_text(encoding="ascii"))
    assert payload["handles"] == [
        {"kinematic": True, "motion": 0, "nodes": [0]},
        {"kinematic": True, "motion": 1, "nodes": [2]},
    ]


def test_scene_rejects_nonidentity_rotation(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
    material = tmp_path / "material.json"
    material.write_text("{}\n", encoding="ascii")
    with pytest.raises(ValueError, match="identity rotation"):
        write_arcsim_scene(
            tmp_path / "scene.json",
            mesh_path=mesh,
            material_path=material,
            controller=_controller(),
            parameters=_parameters_for_test(),
            duration_s=0.1,
            initial_pose_xyz_wxyz=(0.0, 0.0, 0.0, 0.999, 0.0, 0.0, 0.0),
        )


def test_arcsim_obj_loader_preserves_vertex_order(tmp_path: Path) -> None:
    source = tmp_path / "frame.obj"
    source.write_text(
        "v 0.125 0 0\nvt 0 0\nv 0 0.25 0\nv 0 0 0.5\nf 1 2 3\n",
        encoding="ascii",
    )
    vertices = load_arcsim_vertices(source)
    np.testing.assert_array_equal(
        vertices,
        np.asarray([[0.125, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.5]]),
    )


def test_arcsim_rollout_makes_final_vertices_read_only() -> None:
    rollout = ARCSimRollout(
        final_vertices_m=np.asarray(
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]]
        ),
        maximum_pin_target_error_m=1e-8,
        step_count=10,
        elapsed_s=0.5,
    )
    assert not rollout.final_vertices_m.flags.writeable


def test_arcsim_protocol_binds_source_build_and_information_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "configs" / "sota" / "rgbbench_arcsim_competence_v8.json"
    protocol = _load_protocol(protocol_path)
    upstream = protocol["upstream"]
    case = protocol["competence_case"]
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["artifact_kind"] == ARTIFACT_KIND
    assert set(SOURCE_DIGEST_KEYS.values()) <= set(case)
    patch_path = root / "third_party" / "patches" / "arcsim_rgbbench_compat_v8.patch"
    assert (
        sha256_file(patch_path)
        == upstream["implementation_artifact_sha256s"][
            "third_party/patches/arcsim_rgbbench_compat_v8.patch"
        ]
    )
    assert upstream["arcsim_archive_sha256"] == (
        "053239c4fbc566228d3f46e8afd3428dc2ffa1c2d18d348af7b1094cd8f5a26e"
    )
    assert len(upstream["arcsim_executable_sha256"]) == 64
    assert protocol["competence_gate"]["expected_vertex_count"] == 9865
    assert protocol["competence_gate"]["require_byte_identical_final_vertices"]
    assert protocol["information_boundary"]["forbidden"]


def test_protocol_parameters_preserve_frozen_mechanics() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = _load_protocol(
        root / "configs" / "sota" / "rgbbench_arcsim_competence_v8.json"
    )
    parameters = _parameters(protocol)
    assert parameters.timestep_s == 0.01
    assert parameters.handle_stiffness == 1e8
    assert parameters.gravity_m_s2 == (0.0, 0.0, -9.81)
    assert not parameters.kinematic_handles


def test_v9_changes_only_provenance_paths_and_predecessor() -> None:
    root = Path(__file__).resolve().parents[1]
    v8 = _load_protocol(
        root / "configs" / "sota" / "rgbbench_arcsim_competence_v8.json"
    )
    v9 = _load_protocol(
        root / "configs" / "sota" / "rgbbench_arcsim_competence_v9.json"
    )
    assert set(SUPPORTED_PROTOCOLS) == {
        "rgbbench-arcsim-competence-v8",
        "rgbbench-arcsim-competence-v9",
        "rgbbench-arcsim-dirichlet-competence-v10",
        "rgbbench-arcsim-dirichlet-competence-v11",
    }
    assert v9["predecessor"]["protocol_id"] == v8["protocol_id"]
    assert v9["predecessor"]["status"] == "technical_failure_before_simulation"
    assert v9["competence_case"] == v8["competence_case"]
    assert v9["physics"] == v8["physics"]
    assert v9["competence_gate"] == v8["competence_gate"]
    assert v9["threshold_basis"] == v8["threshold_basis"]
    assert v9["information_boundary"] == v8["information_boundary"]
    assert set(v9["upstream"]["arcsim_source_sha256s"]) >= {
        "dependencies/lib/libjson.a",
        "dependencies/lib/libalglib.a",
    }
    assert all(
        len(expected_sha256) == 64
        for expected_sha256 in v9["upstream"][
            "implementation_artifact_sha256s"
        ].values()
    )


def test_v10_changes_only_the_control_semantics_and_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    v9 = _load_protocol(
        root / "configs" / "sota" / "rgbbench_arcsim_competence_v9.json"
    )
    v10 = _load_protocol(
        root / "configs" / "sota" / "rgbbench_arcsim_dirichlet_competence_v10.json"
    )
    assert v10["predecessor"]["protocol_id"] == v9["protocol_id"]
    assert v10["predecessor"]["status"] == "control_contract_failed"
    assert v10["competence_case"] == v9["competence_case"]
    assert v10["competence_gate"] == v9["competence_gate"]
    assert v10["physics"] == {
        **v9["physics"],
        "kinematic_handles": True,
    }
    assert (
        v10["information_boundary"]["forbidden"]
        == v9["information_boundary"]["forbidden"]
    )
    assert v10["method_change"] == {
        "changed": "enforce the two known actuator nodes as time-varying "
        "Dirichlet boundary conditions after every ARCSim substep",
        "unchanged": [
            "source case",
            "mesh and material parameters",
            "timestep and horizon",
            "native ARCSim penalty forces during each implicit solve",
            "disabled collision, remeshing, strain limiting, and plasticity",
            "competence thresholds",
            "information boundary",
        ],
        "selection_evidence": "v9 target-free pin error only; no point-cloud "
        "filename, coordinate, or accuracy outcome was read",
    }
    assert all(
        len(expected_sha256) == 64
        for expected_sha256 in v10["upstream"][
            "implementation_artifact_sha256s"
        ].values()
    )


def test_v11_changes_only_dirichlet_reference_initialization() -> None:
    root = Path(__file__).resolve().parents[1]
    v10 = _load_protocol(
        root / "configs" / "sota" / "rgbbench_arcsim_dirichlet_competence_v10.json"
    )
    v11 = _load_protocol(
        root / "configs" / "sota" / "rgbbench_arcsim_dirichlet_competence_v11.json"
    )
    assert v11["predecessor"]["protocol_id"] == v10["protocol_id"]
    assert (
        v11["predecessor"]["status"] == "control_reference_initialized_after_relaxation"
    )
    assert v11["competence_case"] == v10["competence_case"]
    assert v11["physics"] == v10["physics"]
    assert v11["competence_gate"] == v10["competence_gate"]
    assert (
        v11["information_boundary"]["forbidden"]
        == v10["information_boundary"]["forbidden"]
    )
    assert v11["method_change"]["changed"].startswith(
        "initialize and enforce the two declared Dirichlet handles before"
    )
    for relative_path, expected_sha256 in v11["upstream"][
        "implementation_artifact_sha256s"
    ].items():
        assert sha256_file(root / relative_path) == expected_sha256
