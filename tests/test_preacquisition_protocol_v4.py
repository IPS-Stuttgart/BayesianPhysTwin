from __future__ import annotations

from copy import deepcopy

import pytest

import causal4d.preacquisition_protocol_v4 as protocol_v4_module
from causal4d.mechanism_gate_controls import mechanism_gate_control_sha256
from causal4d.preacquisition_protocol import build_preacquisition_amendment
from causal4d.preacquisition_protocol_v3 import build_preacquisition_v3
from causal4d.preacquisition_protocol_v4 import (
    build_preacquisition_v4,
    preacquisition_v4_sha256,
    validate_preacquisition_v4,
)
from causal4d.real_protocol import build_same_object_real_protocol


def _passing_control_evidence() -> dict:
    evidence = {
        "schema_version": 1,
        "artifact_kind": "MechanismGateControlEvidence",
        "claim_boundary": "controlled only",
        "design": {
            "placebo": "matched scalar placebo",
            "positive_control": "known scalar mechanism",
        },
        "config": {
            "simulation_count": 512,
            "minimum_shrinkage_fraction": 0.10,
            "minimum_positive_sessions": 8,
        },
        "arms": {
            "placebo_null": {
                "full_eligibility_pass_count": 0,
                "full_eligibility_pass_rate": 0.0,
                "full_eligibility_wilson_95": [0.0, 0.008],
            },
            "positive_control": {
                "full_eligibility_pass_count": 485,
                "full_eligibility_pass_rate": 485 / 512,
                "full_eligibility_wilson_95": [0.92, 0.97],
            },
        },
        "acceptance_checks": {
            "placebo_null_full_gate_upper_below_5_percent": True,
            "positive_control_full_gate_lower_above_80_percent": True,
            "wrong_family_on_positive_upper_below_5_percent": True,
        },
        "frozen_v3_gate_supported_in_controlled_benchmark": True,
    }
    evidence["result_sha256"] = mechanism_gate_control_sha256(evidence)
    return evidence


def test_v4_preserves_v3_design_and_locks_controls(monkeypatch) -> None:
    monkeypatch.setattr(protocol_v4_module, "_CANONICAL_V4_SHA256", "")
    protocol = build_same_object_real_protocol()
    v2 = build_preacquisition_amendment(protocol)
    v3 = build_preacquisition_v3(protocol, v2)
    evidence = _passing_control_evidence()
    v4 = build_preacquisition_v4(protocol, v2, v3, evidence)

    result = validate_preacquisition_v4(v4, protocol, v2, v3, evidence)
    assert result["passed"] is True
    assert result["physical_execution_count_changed"] is False
    assert result["mechanism_gate_threshold_changed"] is False
    assert result["contact_registration_schema"] == 3
    assert v4["prospective_mode0_reset_crosscheck"]["status"].startswith(
        "preregistered"
    )


def test_v4_rejects_a_post_control_threshold_change(monkeypatch) -> None:
    monkeypatch.setattr(protocol_v4_module, "_CANONICAL_V4_SHA256", "")
    protocol = build_same_object_real_protocol()
    v2 = build_preacquisition_amendment(protocol)
    v3 = build_preacquisition_v3(protocol, v2)
    evidence = _passing_control_evidence()
    v4 = build_preacquisition_v4(protocol, v2, v3, evidence)
    mutated = deepcopy(v4)
    mutated["mechanism_gate_control_lock"]["frozen_threshold"][
        "minimum_geometric_mean_shrinkage_fraction"
    ] = 0.05
    mutated["amendment_sha256"] = preacquisition_v4_sha256(mutated)

    with pytest.raises(ValueError, match="controlled v3 mechanism gate"):
        validate_preacquisition_v4(mutated, protocol, v2, v3, evidence)
