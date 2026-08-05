from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "run_deform360_tactile_metric_gauge_provider.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_deform360_tactile_metric_gauge_provider",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_provider_uses_inspected_model_set_adapter_factory() -> None:
    config = object()
    adapter_factory = object()

    class ModelSet:
        def adapter_factory(self) -> object:
            return adapter_factory

    class Runner:
        def __init__(self, supplied_config: object, **kwargs: object) -> None:
            self.config = supplied_config
            self.kwargs = kwargs

    runner = MODULE._build_pinned_runner(ModelSet(), Runner, config)

    assert runner.config is config
    assert runner.kwargs == {"adapter_factory": adapter_factory}
