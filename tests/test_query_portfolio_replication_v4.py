from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin.query_portfolio_replication_v4 import protocol

ROOT = Path(__file__).resolve().parents[1]


def test_v4_binds_fresh_roster_and_complete_upstream() -> None:
    value = protocol()
    assert value["version"] == 4
    assert value["outcomes_opened"] is False
    assert value["recovery"]["ordinary_prefix_seals"] == 0
    assert value["recovery"]["scene_construction_preflight_passed"] is True
    assert value["world_seeds"]["dlolab_slingshot_v4"]["evaluation"] == 263_402


def test_v4_committed_lock_matches_protocol() -> None:
    path = ROOT / "configs/experiments/query_portfolio_replication_v4.json"
    assert json.loads(path.read_text(encoding="utf-8")) == protocol()
