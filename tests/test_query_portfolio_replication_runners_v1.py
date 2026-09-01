from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from bayesian_phystwin.query_portfolio_replication_v1 import WORLD_COUNT

ROOT = Path(__file__).resolve().parents[1]


def _module(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapping_wrapper_registers_fresh_roster_and_self_workers() -> None:
    module = _module(
        "scripts/remote/run_dlolab_wrapping_portfolio_replication_v1.py",
        "wrapping_portfolio_test",
    )
    runner = module._load_runner()
    assert runner.WORLD_COUNT == WORLD_COUNT
    assert runner.PREFIX_BATCH_COUNT == 36
    assert Path(runner.__file__).resolve() == Path(module.__file__).resolve()
    assert runner.OUTPUT == module.OUTPUT


def test_slingshot_wrapper_keeps_disjoint_calibration_and_self_workers() -> None:
    module = _module(
        "scripts/remote/run_dlolab_slingshot_portfolio_replication_v1.py",
        "slingshot_portfolio_test",
    )
    wrapper = module._load_runner()
    assert wrapper.method.COUNTS == {"calibration": 128, "evaluation": WORLD_COUNT}
    assert wrapper.runner.WORKER_RUNNER_PATH == Path(module.__file__).resolve()
    assert wrapper.runner.OUTPUT_ROOT == module.OUTPUT
