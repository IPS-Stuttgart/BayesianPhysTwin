from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin.rgbench_codim_ipc import (
    CodimIPCClothParameters,
    CodimIPCRollout,
    _load_runtime,
    write_obj_triangles,
)
from bayesian_phystwin.rgbench_online_belief import sha256_file
from scripts.held.run_rgbbench_codim_ipc_competence_v5 import (
    SOURCE_DIGEST_KEYS,
)


def _parameters(**changes: object) -> CodimIPCClothParameters:
    values: dict[str, object] = {
        "timestep_s": 0.01,
        "youngs_modulus_pa": 230000.0,
        "poisson_ratio": 0.35,
        "volume_density_kg_m3": 220.0,
        "thickness_m": 0.001,
        "bending_stiffness_multiplier": 1.0,
        "newton_tolerance": 1e-6,
        "contact_thickness_m": 0.001,
        "collision_enabled": False,
    }
    values.update(changes)
    return CodimIPCClothParameters(**values)  # type: ignore[arg-type]


def test_codim_parameters_reject_nonphysical_values() -> None:
    with pytest.raises(ValueError, match="poisson_ratio"):
        _parameters(poisson_ratio=0.5)
    with pytest.raises(ValueError, match="newton_tolerance"):
        _parameters(newton_tolerance=0.0)
    with pytest.raises(ValueError, match="collision_enabled"):
        _parameters(collision_enabled=1)


def test_codim_rollout_preserves_metric_vertices_read_only() -> None:
    rollout = CodimIPCRollout(
        final_vertices_m=np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]),
        maximum_pin_target_error_m=1e-12,
        step_count=10,
        total_newton_iterations=42,
    )
    assert not rollout.final_vertices_m.flags.writeable
    assert rollout.maximum_pin_target_error_m == pytest.approx(1e-12)


def test_obj_writer_preserves_vertex_order_and_one_indexes_faces(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "mesh.obj"
    write_obj_triangles(
        destination,
        np.asarray(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.5, 0.0]]
        ),
        np.asarray([[0, 1, 2]]),
    )
    assert destination.read_text(encoding="ascii") == (
        "v 0 0 0\n"
        "v 0.25 0 0\n"
        "v 0 0.5 0\n"
        "f 1 2 3\n"
    )
    with pytest.raises(ValueError, match="overwrite"):
        write_obj_triangles(
            destination,
            np.zeros((3, 3)),
            np.asarray([[0, 1, 2]]),
        )


def test_codim_patch_binds_exact_node_and_target_operations() -> None:
    root = Path(__file__).resolve().parents[1]
    patch = (
        root / "third_party" / "patches" / "codim_ipc_rgbbench_v5.patch"
    ).read_text(encoding="utf-8")
    for symbol in (
        "Init_Dirichlet_Nodes",
        "Set_Dirichlet_Targets",
        "Node_Positions",
        "Dirichlet_Max_Error",
    ):
        assert symbol in patch


def test_codim_runtime_rejects_wrong_compiled_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_root = tmp_path / "module"
    python_root = tmp_path / "python"
    module_root.mkdir()
    python_root.mkdir()
    fem = SimpleNamespace(
        Init_Dirichlet_Nodes=object(),
        Set_Dirichlet_Targets=object(),
        Node_Positions=object(),
        Dirichlet_Max_Error=object(),
    )
    jgsl = SimpleNamespace(FEM=fem, linear_solver_backend="CHOLMOD")
    drivers = SimpleNamespace()
    monkeypatch.setattr(
        "bayesian_phystwin.rgbench_codim_ipc.importlib.import_module",
        lambda name: jgsl if name == "JGSL" else drivers,
    )
    loaded, _ = _load_runtime(
        module_root=module_root,
        python_root=python_root,
        expected_linear_solver_backend="CHOLMOD",
    )
    assert loaded is jgsl
    with pytest.raises(RuntimeError, match="linear solver changed"):
        _load_runtime(
            module_root=module_root,
            python_root=python_root,
            expected_linear_solver_backend="EIGEN",
        )


def test_cholmod_build_patch_uses_bound_system_libraries_and_marker() -> None:
    root = Path(__file__).resolve().parents[1]
    patch = (
        root
        / "third_party"
        / "patches"
        / "codim_ipc_cholmod_system_blas_v6.patch"
    ).read_text(encoding="utf-8")
    assert "JGSL_OUTPUT_DIRECTORY" in patch
    assert 'm.attr("linear_solver_backend") = "CHOLMOD"' in patch
    assert "SYSTEM_BLAS_LIBRARY" in patch
    assert "SYSTEM_LAPACK_LIBRARY" in patch
    assert "target_link_libraries(cholmod PUBLIC blas lapack)" in patch
    assert "include(mkl)" in patch
    assert patch.count("-    include(mkl)") == 1


def test_codim_protocol_binds_patch_and_every_source_digest() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root / "configs" / "sota" / "rgbbench_codim_ipc_competence_v5.json"
    )
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    case = payload["competence_case"]
    assert set(SOURCE_DIGEST_KEYS.values()) <= set(case)
    patch_path = root / payload["upstream"]["codim_patch_relative_path"]
    assert sha256_file(patch_path) == payload["upstream"]["codim_patch_sha256"]
    assert payload["competence_gate"]["require_byte_identical_final_vertices"]
    assert payload["information_boundary"]["forbidden"]
