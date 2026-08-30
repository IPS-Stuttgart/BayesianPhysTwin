from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "results/source/dlolab_slingshot_policy_certificate_development_v2/summary.json"
)


def _builder() -> ModuleType:
    path = ROOT / "scripts/audit_dlolab_slingshot_policy_certificate_development_v2.py"
    spec = importlib.util.spec_from_file_location("policy_certificate_v2_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_development_artifact_reproduces() -> None:
    expected = _builder().build()
    actual = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert actual == expected
    assert actual["selected_model"] == "combined_distance_k7"
    assert actual["advancement_gate_passed"] is True
    assert actual["selected_prefix_capacity"]["accepted_prefix_count"] == 44
    assert actual["prefix_panel_outcomes_read"] is False
    assert actual["prospective_coverage_claim"] is False
