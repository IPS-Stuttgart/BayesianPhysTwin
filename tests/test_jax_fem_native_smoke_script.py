from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "run_jax_fem_native_smoke.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bpt_jax_fem_native_smoke", SCRIPT_PATH
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
    assert module.JAX_FEM_SOURCE_BLOBS == {
        "problem.py": "8a20d24fc2e98aa33d4bd76e543f00c471740551",
        "solver.py": "f0f64cb629e202f2d179710b745ea4d682f1ace2",
        "generate_mesh.py": "bd564c8f4a049ae28bc3592e21d9547a5f509629",
    }


def test_git_blob_sha1_uses_git_object_identity(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source.py"
    source.write_bytes(b"abc\n")
    expected = hashlib.sha1(b"blob 4\0abc\n").hexdigest()
    assert module._git_blob_sha1(source) == expected


def test_content_id_is_mapping_order_independent() -> None:
    module = _load_script()
    assert module._content_id({"b": 2, "a": 1}) == module._content_id({"a": 1, "b": 2})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 1}, "frame_count"),
        ({"maximum_displacement_m": 0.0}, "maximum_displacement_m"),
        ({"young_modulus_pa": 0.0}, "young_modulus_pa"),
        ({"poisson_ratio": 0.5}, "poisson_ratio"),
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
