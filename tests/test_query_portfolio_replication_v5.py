from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin.query_portfolio_replication_v5 import protocol

ROOT = Path(__file__).resolve().parents[1]


def _module(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_binds_terminal_v4_and_fresh_roster() -> None:
    value = protocol()
    assert value["version"] == 5
    assert value["outcomes_opened"] is False
    assert value["recovery"]["evaluation_futures_sealed"] == 0
    assert value["recovery"]["evaluation_outcomes_opened"] is False
    assert value["recovery"]["worlds_reused"] is False
    assert value["world_seeds"]["dlolab_slingshot_v4"]["evaluation"] == 263_502


def test_v5_runner_binds_transitive_helpers_to_320_worlds() -> None:
    module = _module(
        "scripts/remote/run_dlolab_slingshot_portfolio_replication_v1.py",
        "slingshot_portfolio_v5_transitive_test",
    )
    module._configure_methods()
    for helper_name in ("guarded_decisions", "pre_future_checks"):
        helper = getattr(module.method_v4, helper_name)
        assert helper.__globals__["COUNTS"]["evaluation"] == 320
    assert module.method_v2.BOOTSTRAP_REPLICATES == 100_000


def test_v5_committed_lock_matches_protocol() -> None:
    path = ROOT / "configs/experiments/query_portfolio_replication_v5.json"
    assert json.loads(path.read_text(encoding="utf-8")) == protocol()
