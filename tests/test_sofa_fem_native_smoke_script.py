from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "run_sofa_fem_native_smoke.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bpt_sofa_fem_native_smoke",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_import_does_not_require_sofa() -> None:
    module = _load_script()
    assert module.SOFA_VERSION == "26.06.00"
    assert module.SOFA_REVISION == "7c18e95d5c5f2839079892c69e7d89a313c79603"
    assert module.SOFA_ARCHIVE_SHA256 == (
        "129211fd01781bdd5ba3f28f1c3617a2f3792a71b62dc609cf866eec4ac745e2"
    )
    assert "Sofa.Component.SolidMechanics.FEM.HyperElastic" in (
        module.SOFA_REQUIRED_PLUGINS
    )


def test_mesh_is_positive_fixed_identity_tetrahedral_block() -> None:
    module = _load_script()
    points, cells = module._mesh()
    assert points.shape == (27, 3)
    assert cells.shape == (48, 4)
    assert len(np.unique(cells)) == 27
    determinants = np.array(
        [np.linalg.det((points[cell[1:]] - points[cell[0]]).T) for cell in cells]
    )
    assert np.all(determinants > 0.0)
    assert module._topology_descriptor()["tetrahedron_count"] == 48


def test_lame_parameter_conversion_is_physical() -> None:
    module = _load_script()
    shear, first_lame = module._lame_parameters(1000.0, 0.3)
    assert shear == pytest.approx(384.6153846153846)
    assert first_lame == pytest.approx(576.9230769230769)


def test_deformation_determinants_recover_identity() -> None:
    module = _load_script()
    points, cells = module._mesh()
    observed = module._deformation_determinants(points, points.copy(), cells)
    np.testing.assert_array_equal(observed, np.ones(len(cells)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 1}, "frame_count"),
        ({"output_time_step_s": 0.0}, "output_time_step_s"),
        ({"integrator_time_step_s": 0.0}, "integrator_time_step_s"),
        ({"total_force_n": 0.0}, "total_force_n"),
        ({"young_modulus_pa": 0.0}, "young_modulus_pa"),
        ({"poisson_ratio": 0.5}, "poisson_ratio"),
        ({"total_mass_kg": 0.0}, "total_mass_kg"),
        (
            {"output_time_step_s": 0.025, "integrator_time_step_s": 0.003},
            "exact integrator-step multiple",
        ),
    ],
)
def test_invalid_arguments_fail_before_native_import(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    module = _load_script()
    output = tmp_path / "result"
    with pytest.raises(ValueError, match=message):
        module.run_smoke(
            output,
            distribution_archive=tmp_path / "missing.zip",
            sofa_root=tmp_path / "missing-root",
            **kwargs,
        )
    assert not output.exists()
