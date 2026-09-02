from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/active-decision-acquisition-study-v1.json"
RESULT = (
    ROOT
    / "results/science/active_decision_acquisition_v1/controlled-v1/result.json"
)
SCRIPT = ROOT / "experiments/active_decision_acquisition_v1/run.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("active_acquisition_study", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load active acquisition study")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_is_content_addressed_and_bounded() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id")
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    assert hashlib.sha256(encoded).hexdigest() == protocol_id
    assert protocol["study"]["hypothesis_count"] == 24
    assert protocol["study"]["action_group_sizes"] == [16, 4, 4]
    assert "Controlled finite-hypothesis mechanism evidence only" in protocol[
        "claim_boundary"
    ]
    assert "deployment safety" in protocol["claim_boundary"]


def test_committed_result_matches_a_fresh_deterministic_run() -> None:
    module = _load_script()
    protocol = module._load_protocol(PROTOCOL)
    expected = json.loads(RESULT.read_text(encoding="utf-8"))

    first = module.run(protocol)
    second = module.run(protocol)

    assert first == second == expected
    assert expected["decision"] == "controlled-active-decision-acquisition-passed"
    assert all(expected["checks"].values())
    assert expected["active_policy"]["root_worst_case_cost"] == pytest.approx(2.0)
    assert expected["active_policy"]["uniform_expected_cost"] == pytest.approx(
        4.0 / 3.0
    )
    assert expected["entropy_greedy"]["uniform_expected_cost"] == pytest.approx(
        10.0 / 3.0
    )
    assert expected["full_state_probe_set"]["total_cost"] == pytest.approx(6.0)
    assert not expected["unresolvable_control"]["feasible"]
