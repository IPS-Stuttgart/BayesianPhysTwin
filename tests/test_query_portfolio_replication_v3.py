from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin.query_portfolio_replication_v3 import protocol

ROOT = Path(__file__).resolve().parents[1]


def test_v3_is_outcome_blind_and_uses_new_roster() -> None:
    value = protocol()
    assert value["version"] == 3
    assert value["outcomes_opened"] is False
    assert value["recovery"]["ordinary_prefix_seals"] == 0
    assert value["recovery"]["worker_import_preflight_required"] is True
    assert value["world_seeds"]["dlolab_slingshot_v4"] == {
        "calibration": 263_301,
        "evaluation": 263_302,
    }


def test_v3_committed_lock_matches_protocol() -> None:
    path = ROOT / "configs/experiments/query_portfolio_replication_v3.json"
    assert json.loads(path.read_text(encoding="utf-8")) == protocol()
