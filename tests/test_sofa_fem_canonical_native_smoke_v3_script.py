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
    / "run_sofa_fem_canonical_native_smoke_v3.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bpt_sofa_fem_canonical_native_smoke_v3",
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
    assert module.RESULT_SCHEMA.endswith("canonical-native-smoke-v3")
    assert module.CANONICAL_ROUNDING_M == 1.0e-11
    assert module.SOFA_VERSION == "26.06.00"
    assert module.SOFA_REVISION == "7c18e95d5c5f2839079892c69e7d89a313c79603"


def test_synthetic_source_has_positive_tetrahedra_and_full_rank_contact() -> None:
    module = _load_script()
    points, cells, indices, contact = module._synthetic_source()
    determinants = np.asarray(
        [np.linalg.det((points[cell[1:]] - points[cell[0]]).T) for cell in cells]
    )

    assert points.shape == (5, 3)
    assert cells.shape == (2, 4)
    assert np.all(determinants > 0.0)
    np.testing.assert_array_equal(indices, np.arange(4, dtype=np.int64))
    assert contact.patch_ranks == (3,)


def test_preexisting_output_is_rejected_before_any_native_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    output = tmp_path / "existing"
    output.mkdir()

    def unexpected_native_access(**_: object) -> object:
        raise AssertionError("native access must not occur")

    monkeypatch.setattr(
        module, "load_native_sofa_fem_modules_v1", unexpected_native_access
    )
    with pytest.raises(FileExistsError):
        module.run_sofa_fem_canonical_native_smoke_v3(
            distribution_archive=tmp_path / "missing.zip",
            sofa_root=tmp_path / "missing-sofa",
            repo_root=tmp_path / "missing-repo",
            output_dir=output,
        )
