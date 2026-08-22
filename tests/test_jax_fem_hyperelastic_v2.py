from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.jax_fem_hyperelastic_v2 as module
from bayesian_phystwin.jax_fem_hyperelastic_v2 import (
    NativeJaxFemModulesV2,
    interpolate_rotation_v2,
    lame_parameters_v2,
    load_native_jax_fem_modules_v2,
    normalized_objectivity_errors_v2,
    stable_neo_hookean_energy_v2,
)
from bayesian_phystwin.jax_fem_source_qualification_v1 import file_sha256


def _runtime_versions() -> dict[str, str]:
    return {
        "python": "3.12.13",
        "jax": "0.9.0.1",
        "jaxlib": "0.9.0.1",
        "jax_fem": "0.0.10",
        "numpy": "2.4.2",
        "scipy": "1.17.1",
        "petsc4py": "3.24.5",
        "gmsh": "4.15.1",
        "meshio": "5.3.5",
    }


def test_native_loader_verifies_versions_sources_and_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "site" / "jax_fem"
    package_root.mkdir(parents=True)
    package_file = package_root / "__init__.py"
    package_file.write_text("# pinned source\n", encoding="utf-8")
    updates: list[tuple[str, bool]] = []
    jax = SimpleNamespace(
        config=SimpleNamespace(update=lambda key, value: updates.append((key, value))),
        devices=lambda: [SimpleNamespace(platform="cpu")],
    )
    mesh_type = type("Mesh", (), {})
    problem_type = type("Problem", (), {})

    def solver(*_: object, **__: object) -> list[object]:
        return []

    modules: dict[str, object] = {
        "jax": jax,
        "jax.numpy": np,
        "jax_fem": SimpleNamespace(__file__=str(package_file)),
        "jax_fem.generate_mesh": SimpleNamespace(Mesh=mesh_type),
        "jax_fem.problem": SimpleNamespace(Problem=problem_type),
        "jax_fem.solver": SimpleNamespace(solver=solver),
    }
    versions = _runtime_versions()
    monkeypatch.setattr(module.platform, "python_version", lambda: versions["python"])
    monkeypatch.setattr(
        module.importlib.metadata,
        "version",
        lambda name: versions[name.replace("-", "_")],
    )
    monkeypatch.setattr(module.importlib, "import_module", modules.__getitem__)

    native = load_native_jax_fem_modules_v2(
        runtime_versions=versions,
        installed_source_sha256={"jax_fem/__init__.py": file_sha256(package_file)},
    )

    assert native.Mesh is mesh_type
    assert native.Problem is problem_type
    assert native.solver is solver
    assert native.package_root == package_root
    assert updates == [("jax_enable_x64", True)]

    with pytest.raises(ValueError, match="version roster changed"):
        load_native_jax_fem_modules_v2(
            runtime_versions={"python": versions["python"]},
            installed_source_sha256={},
        )
    with pytest.raises(ValueError, match="source changed"):
        load_native_jax_fem_modules_v2(
            runtime_versions=versions,
            installed_source_sha256={"jax_fem/__init__.py": "0" * 64},
        )
    jax.devices = lambda: [SimpleNamespace(platform="gpu")]
    with pytest.raises(ValueError, match="frozen CPU runtime"):
        load_native_jax_fem_modules_v2(
            runtime_versions=versions,
            installed_source_sha256={},
        )


@pytest.mark.parametrize(
    ("young", "poisson", "message"),
    [
        (0.0, 0.3, "young_modulus_pa"),
        (True, 0.3, "young_modulus_pa"),
        (1.0, float("nan"), "poisson_ratio"),
        (1.0, 0.0, "must lie"),
        (1.0, 0.5, "must lie"),
    ],
)
def test_lame_parameters_reject_invalid_materials(
    young: Any,
    poisson: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        lame_parameters_v2(young, poisson)


def test_constitutive_energy_and_objectivity_probe_use_frozen_law() -> None:
    native = NativeJaxFemModulesV2(
        jax=SimpleNamespace(
            grad=lambda _: lambda value: np.zeros_like(value, dtype=np.float64)
        ),
        jnp=np,
        Mesh=object,
        Problem=object,
        solver=lambda *_args, **_kwargs: [],
        package_root=Path("/pinned/jax_fem"),
    )
    shear, first_lame, alpha = lame_parameters_v2(12_000.0, 0.35)
    assert shear > 0.0
    assert first_lame > 0.0
    assert alpha > 1.0
    energy = stable_neo_hookean_energy_v2(
        native,
        young_modulus_pa=12_000.0,
        poisson_ratio=0.35,
    )
    assert np.isfinite(float(energy(np.eye(3))))
    assert normalized_objectivity_errors_v2(
        native,
        young_modulus_pa=12_000.0,
        poisson_ratio=0.35,
    ) == (0.0, 0.0)


def test_rotation_interpolation_validates_inputs_and_repairs_reflection() -> None:
    reflection = np.diag([-1.0, 1.0, 1.0])
    repaired = interpolate_rotation_v2(np.eye(3), reflection, 0.75)
    np.testing.assert_allclose(repaired.T @ repaired, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(repaired) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="rotations changed"):
        interpolate_rotation_v2(np.eye(2), np.eye(3), 0.5)
    with pytest.raises(ValueError, match="fraction must lie"):
        interpolate_rotation_v2(np.eye(3), np.eye(3), -0.1)
    with pytest.raises(ValueError, match="fraction must be finite"):
        interpolate_rotation_v2(np.eye(3), np.eye(3), True)


def test_boundary_condition_factories_bind_nodes_and_components() -> None:
    location = module._location_factory(np, np.asarray([1, 3]))
    assert bool(location(np.zeros(3), np.asarray(3))) is True
    assert bool(location(np.zeros(3), np.asarray(2))) is False

    value = module._value_factory(np.eye(3), np.asarray([1.0, 2.0, 3.0]), 1)
    assert float(value(np.asarray([4.0, 5.0, 6.0]))) == pytest.approx(2.0)
