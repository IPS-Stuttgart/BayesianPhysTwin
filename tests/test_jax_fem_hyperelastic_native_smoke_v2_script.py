from __future__ import annotations

import hashlib
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
    / "run_jax_fem_hyperelastic_native_smoke_v2.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bpt_jax_fem_hyperelastic_native_smoke_v2",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_import_does_not_require_jax_fem() -> None:
    module = _load_script()
    assert module.JAX_FEM_VERSION == "0.0.12"
    assert module.JAX_FEM_REVISION == "82c6993c16704e38611f9cb91a5b70f1c690daee"
    assert module.RUNTIME_VERSIONS["jax"] == "0.4.38"
    assert module.JAX_FEM_SOURCE_BLOBS["solver.py"] == (
        "f0f64cb629e202f2d179710b745ea4d682f1ace2"
    )


def test_git_blob_sha1_uses_git_object_identity(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source.py"
    source.write_bytes(b"abc\n")
    expected = hashlib.sha1(b"blob 4\0abc\n").hexdigest()
    assert module._git_blob_sha1(source) == expected


def test_mesh_has_fixed_positive_orientation_identity() -> None:
    module = _load_script()
    points, cells, tetrahedra = module._mesh()
    assert points.shape == (27, 3)
    assert cells.shape == (8, 8)
    assert tetrahedra.shape == (48, 4)
    determinants = module._deformation_determinants(points, points, tetrahedra)
    np.testing.assert_array_equal(determinants, np.ones(48))


def test_stable_neo_hookean_parameters_have_rest_stability_shift() -> None:
    module = _load_script()
    shear, first_lame, alpha = module._lame_parameters(100_000.0, 0.35)
    assert shear > 0.0
    assert first_lame > 0.0
    assert alpha == pytest.approx(1.0 + shear / first_lame)


def test_right_face_control_is_rigid_rotation_plus_axial_extension() -> None:
    module = _load_script()
    point = np.array([0.04, 0.02, 0.01])
    displacement = np.asarray(
        module._right_face_displacement(
            point,
            jnp=np,
            load_fraction=1.0,
            twist_angle_rad=np.pi / 2.0,
            axial_displacement_m=0.005,
        )
    )
    np.testing.assert_allclose(displacement, [0.005, -0.01, 0.01], atol=1e-15)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 2}, "frame_count"),
        ({"continuation_substeps": 0}, "continuation_substeps"),
        ({"young_modulus_pa": 0.0}, "young_modulus_pa"),
        ({"poisson_ratio": 0.0}, "poisson_ratio"),
        ({"poisson_ratio": 0.5}, "poisson_ratio"),
        ({"twist_angle_rad": 0.0}, "twist_angle_rad"),
        ({"axial_displacement_m": 0.0}, "axial_displacement_m"),
        (
            {"minimum_deformation_determinant": 1.0},
            "minimum_deformation_determinant",
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
        module.run_smoke(output, **kwargs)
    assert not output.exists()
