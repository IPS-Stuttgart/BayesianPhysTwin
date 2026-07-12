from __future__ import annotations

import json

import pytest

from bayesian_phystwin.structural_protocol import (
    audit_structural_protocol_readiness,
    build_structural_protocol_amendment,
    locked_action_design_sha256,
    scaffold_structural_protocol_amendment,
    validate_structural_protocol_amendment,
)
from causal4d.real_protocol import build_same_object_real_protocol


def test_structural_amendment_is_measurement_only():
    protocol = build_same_object_real_protocol()
    before = locked_action_design_sha256(protocol)
    amendment = build_structural_protocol_amendment(protocol)
    validation = validate_structural_protocol_amendment(protocol, amendment)
    assert validation["passed"] is True
    assert amendment["locked_action_design_sha256"] == before
    assert amendment["design_change"] == {
        "action_profiles_changed": False,
        "realization_conditions_changed": False,
        "execution_order_changed": False,
        "analysis_splits_changed": False,
        "outcomes_changed": False,
        "measurement_contract_only": True,
    }


def test_structural_amendment_scaffolds_before_acquisition(tmp_path):
    protocol = build_same_object_real_protocol()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    (tmp_path / "executions").mkdir()
    result = scaffold_structural_protocol_amendment(protocol_path, tmp_path)
    assert result["execution_template_count"] == 36
    assert result["session_template_count"] == len(protocol["sessions"])
    readiness = audit_structural_protocol_readiness(protocol_path, tmp_path)
    assert readiness["ready"] is True
    assert readiness["status"] == "structural_measurements_locked_awaiting_acquisition"


def test_structural_amendment_refuses_after_acquisition(tmp_path):
    protocol = build_same_object_real_protocol()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    execution = tmp_path / "executions" / protocol["executions"][0]["execution_id"]
    execution.mkdir(parents=True)
    (execution / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="after acquisition began"):
        scaffold_structural_protocol_amendment(protocol_path, tmp_path)
