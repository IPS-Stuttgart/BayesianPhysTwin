from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "run_sofa_fem_source_qualification_v3.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bpt_sofa_fem_source_qualification_v3",
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
    assert "pose-canonical SOFA FEM v3" in module.__doc__


def test_group_root_requires_unique_canonical_absolute_binding() -> None:
    module = _load_script()
    assert module._group_root("lift=/tmp/lift") == ("lift", Path("/tmp/lift"))
    with pytest.raises(argparse.ArgumentTypeError, match="GROUP_ID=/absolute/path"):
        module._group_root("lift")
    with pytest.raises(argparse.ArgumentTypeError, match="must be absolute"):
        module._group_root("lift=relative")
