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
    / "run_genesis_mpm_native_smoke.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bpt_genesis_mpm_native_smoke",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_import_does_not_require_genesis() -> None:
    module = _load_script()
    assert module.GENESIS_VERSION == "1.3.3"
    assert module.GENESIS_REVISION == "0796d27667087d0087fe09d903f8aadf7fa9adeb"
    assert module.GENESIS_SOURCE_BLOBS == {
        "__init__.py": "6313cf06d94a8203ecc77810eea5121bbeae9d99",
        "engine/entities/mpm_entity.py": ("f700601b4abb37985d4b256d54661dbd6dc1f525"),
        "engine/solvers/mpm_solver.py": ("4cf9df95858d5af114ed428d4bf302b81b4daceb"),
        "engine/materials/MPM/elastic.py": ("98ad7b8e0f19aadb1bfaf6b3ec4bb98a94fefc39"),
        "options/solvers.py": "0ef3c50de61ae3754384329949ea3a0f6a077916",
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
        ({"time_step_s": 0.0}, "time_step_s"),
        ({"substeps": 0}, "substeps"),
        ({"grid_density": 1}, "grid_density"),
        ({"velocity_m_s": 0.0}, "velocity_m_s"),
        ({"young_modulus_pa": 0.0}, "young_modulus_pa"),
        ({"poisson_ratio": 0.5}, "poisson_ratio"),
        ({"density_kg_m3": 0.0}, "density_kg_m3"),
        ({"seed": -1}, "seed"),
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
