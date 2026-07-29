from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "remote" / (
    "stage_deform360_causal_response_direct_depth_v14_window.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("v14_window_stage", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_requires_absolute_executable_before_case_work(
    tmp_path: Path,
) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="path must be absolute"):
        module._resolve_required_executable(Path("ffmpeg"), name="ffmpeg")

    with pytest.raises(ValueError, match="executable is unavailable"):
        module._resolve_required_executable(
            tmp_path / "missing-ffmpeg",
            name="ffmpeg",
        )


def test_stage_accepts_pinned_executable(tmp_path: Path) -> None:
    module = _load_script()
    executable = tmp_path / "ffmpeg"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | 0o111)

    resolved = module._resolve_required_executable(executable, name="ffmpeg")

    assert resolved == executable.resolve()
    assert os.access(resolved, os.X_OK)
