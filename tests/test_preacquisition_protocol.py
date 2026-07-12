from __future__ import annotations

from copy import deepcopy

import pytest

from causal4d.preacquisition_analysis import audit_base_protocol_power
from causal4d.preacquisition_protocol import (
    build_preacquisition_amendment,
    validate_preacquisition_amendment,
)
from causal4d.real_protocol import build_same_object_real_protocol


def test_preacquisition_amendment_preserves_targets_and_adds_power() -> None:
    protocol = build_same_object_real_protocol()
    amendment = build_preacquisition_amendment(protocol)
    summary = validate_preacquisition_amendment(amendment, protocol)

    assert summary["passed"] is True
    assert summary["signature_execution_count"] == 12
    assert summary["confirmatory_execution_count"] == 36
    assert summary["calibration_sessions_per_fold"] == 9
    assert all(
        fold["calibration_plan"]["finite_without_infinite_sentinel"]
        for fold in amendment["amended_cross_action_calibration_folds"]
    )


def test_base_protocol_power_audit_rejects_current_replication() -> None:
    audit = audit_base_protocol_power(build_same_object_real_protocol())
    assert audit["exact_replication"]["cells_with_at_least_three_repeats"] == 0
    assert audit["signature_contrasts"]["independent_speed_contrast"] is False
    assert audit["signature_contrasts"]["independent_hold_contrast"] is False
    assert audit["calibration"]["minimum_independent_sessions_per_fold"] == 2
    assert audit["passed_for_first_locked_execution"] is False


def test_preacquisition_amendment_rejects_target_mutation() -> None:
    protocol = build_same_object_real_protocol()
    amendment = build_preacquisition_amendment(protocol)
    mutated = deepcopy(amendment)
    mutated["amended_cross_action_calibration_folds"][0]["target_execution_ids"][0] = (
        mutated["amended_cross_action_calibration_folds"][1]["target_execution_ids"][0]
    )
    from causal4d.preacquisition_protocol import amendment_sha256

    mutated["amendment_sha256"] = amendment_sha256(mutated)
    with pytest.raises(ValueError, match="locked canonical design"):
        validate_preacquisition_amendment(mutated, protocol)
