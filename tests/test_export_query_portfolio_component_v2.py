from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapping_export_uses_configured_method_arm_order() -> None:
    wrapper = _load(
        ROOT / "scripts/remote/run_dlolab_wrapping_portfolio_replication_v1.py",
        "test_portfolio_wrapping_wrapper",
    )
    runner = wrapper._load_runner()

    assert not hasattr(runner, "ARM_NAMES")
    assert wrapper.method.ARM_NAMES.index("posterior_975_guard") == 2
