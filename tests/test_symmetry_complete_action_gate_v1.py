from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from bayesian_phystwin.symmetry_complete_action_gate_v1 import (
    STATUS_BOUNDED,
    STATUS_EXACT,
    STATUS_INVALID,
    STATUS_REJECT,
    act_or_fallback_symmetry_complete_v1,
    verify_causal4d_intervention_receipt_v1,
)


ACTION_NAMES = ("track-shared-frame", "orthogonal-shared-frame", "fallback")
STRUCTURAL_PAIRWISE = np.array(
    [
        [0.0, -2.0, -1.0],
        [2.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ]
)


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rotation(index: int) -> list[list[float]]:
    angle = index * math.pi / 2.0
    return [
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(row[column] * vector[column] for column in range(len(vector)))
        for row in matrix
    ]


def _receipt(radius: float = 0.0) -> dict[str, object]:
    templates = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    rotations = [_rotation(index) for index in range(4)]
    commanded = [
        [_matvec(rotation, template) for template in templates]
        for rotation in rotations
    ]
    realized = json.loads(json.dumps(commanded))
    for group_row in realized:
        group_row[0][0] += radius
    one_action = [radius, 0.0, 0.0]
    pairwise = [
        [0.0, radius, radius],
        [radius, 0.0, 0.0],
        [radius, 0.0, 0.0],
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_id": "a" * 64,
        "transform_instance_id": "frame-instance-17",
        "state_evidence_id": "state-evidence-v1",
        "action_template_id": "action-bank-v1",
        "commanded_intervention_id": "command-v1",
        "realized_intervention_id": "realization-v1",
        "loss_id": "registered-loss-v1",
        "fallback_id": "nominal-controller-v1",
        "radius_provenance_id": "fixture-metrology-v1",
        "radius_scope": "deterministic-complete",
        "group_element_ids": ["r0", "r90", "r180", "r270"],
        "action_templates": templates,
        "commanded_action_orbit": commanded,
        "realized_action_orbit": realized,
        "observed_realization_radius_by_action": [radius, 0.0, 0.0],
        "declared_realization_radius_by_action": [radius, 0.0, 0.0],
        "action_loss_lipschitz_by_action": [1.0, 1.0, 1.0],
        "action_realization_loss_margin": one_action,
        "pairwise_realization_margin": pairwise,
        "verification_tolerance": 1e-12,
    }
    payload["receipt_id"] = _canonical_digest(payload)
    return payload


def _gate(
    *,
    radius: float = 0.0,
    tolerance: float = 0.0,
    receipt: dict[str, object] | None = None,
    **overrides,
):
    arguments = dict(
        action_names=ACTION_NAMES,
        structural_pairwise_upper_bound=STRUCTURAL_PAIRWISE,
        structural_certificate_id="prob4d-structural-certificate-v1",
        state_evidence_id="state-evidence-v1",
        action_template_id="action-bank-v1",
        loss_id="registered-loss-v1",
        fallback_action=2,
        fallback_id="nominal-controller-v1",
        causal4d_receipt=_receipt(radius) if receipt is None else receipt,
        regret_tolerance=tolerance,
        require_radius_scope="deterministic-complete",
    )
    arguments.update(overrides)
    return act_or_fallback_symmetry_complete_v1(**arguments)


def test_verifies_portable_causal4d_receipt() -> None:
    verified = verify_causal4d_intervention_receipt_v1(_receipt(0.2))

    assert verified.action_count == 3
    assert verified.state_evidence_id == "state-evidence-v1"
    assert verified.radius_scope == "deterministic-complete"
    assert verified.action_realization_loss_margin == pytest.approx([0.2, 0.0, 0.0])
    assert verified.pairwise_realization_margin[0] == pytest.approx([0.0, 0.2, 0.2])


def test_exact_action_is_executed_without_gauge_completion() -> None:
    decision = _gate(radius=0.2)

    assert decision.valid_evidence
    assert decision.status == STATUS_EXACT
    assert decision.robustly_optimal.tolist() == [True, False, False]
    assert decision.worst_case_regret_upper_bound[0] == pytest.approx(0.0)
    assert decision.minimax_action == 0
    assert decision.selected_action == 0
    assert decision.selected_action_name == "track-shared-frame"
    assert decision.admitted
    assert not decision.exact_fallback


def test_large_realization_margin_rejects_and_returns_exact_fallback() -> None:
    decision = _gate(radius=1.2)

    assert decision.valid_evidence
    assert decision.status == STATUS_REJECT
    assert decision.worst_case_regret_upper_bound[0] == pytest.approx(0.2)
    assert decision.minimax_action == 0
    assert not decision.admitted
    assert decision.selected_action == 2
    assert decision.selected_action_name == "fallback"
    assert decision.exact_fallback


def test_registered_tolerance_admits_explicit_bounded_regret() -> None:
    decision = _gate(radius=1.2, tolerance=0.25)

    assert decision.valid_evidence
    assert decision.status == STATUS_BOUNDED
    assert not decision.robustly_optimal[0]
    assert decision.epsilon_admissible[0]
    assert decision.admitted
    assert decision.selected_action == 0
    assert not decision.exact_fallback


def test_transport_margin_is_separate_and_can_force_fallback() -> None:
    transport = np.array(
        [
            [0.0, 0.0, 1.1],
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ]
    )
    decision = _gate(
        radius=0.0,
        transport_pairwise_margin=transport,
        transport_certificate_id="object-level-calibration-v1",
    )

    assert decision.valid_evidence
    assert decision.transport_certificate_id == "object-level-calibration-v1"
    assert decision.worst_case_regret_upper_bound[0] == pytest.approx(0.1)
    assert decision.status == STATUS_REJECT
    assert decision.exact_fallback


def test_tampered_receipt_fails_closed_to_exact_fallback() -> None:
    receipt = _receipt(0.2)
    receipt["declared_realization_radius_by_action"] = [0.3, 0.0, 0.0]
    decision = _gate(receipt=receipt)

    assert not decision.valid_evidence
    assert decision.status == STATUS_INVALID
    assert "inconsistent" in decision.invalid_reasons[0] or "receipt_id" in decision.invalid_reasons[0]
    assert decision.selected_action == 2
    assert decision.exact_fallback
    assert decision.minimax_action is None


def test_cross_context_identifiers_must_match() -> None:
    for name, value in (
        ("state_evidence_id", "other-state"),
        ("action_template_id", "other-bank"),
        ("loss_id", "other-loss"),
        ("fallback_id", "other-fallback"),
    ):
        decision = _gate(**{name: value})
        assert not decision.valid_evidence
        assert decision.status == STATUS_INVALID
        assert name in decision.invalid_reasons[0]
        assert decision.exact_fallback


def test_radius_scope_can_be_required_by_policy() -> None:
    receipt = _receipt(0.0)
    payload = {key: value for key, value in receipt.items() if key != "receipt_id"}
    payload["radius_scope"] = "registered-group-nodes-only"
    payload["receipt_id"] = _canonical_digest(payload)
    decision = _gate(receipt=payload)

    assert not decision.valid_evidence
    assert decision.status == STATUS_INVALID
    assert "radius scope" in decision.invalid_reasons[0]
    assert decision.exact_fallback


def test_transport_margin_requires_provenance_identity() -> None:
    decision = _gate(
        transport_pairwise_margin=np.zeros((3, 3)),
        transport_certificate_id=None,
    )
    assert not decision.valid_evidence
    assert decision.status == STATUS_INVALID
    assert "transport_certificate_id" in decision.invalid_reasons[0]
    assert decision.exact_fallback


def test_decision_record_is_deterministic_and_content_sensitive() -> None:
    first = _gate(radius=0.2)
    second = _gate(radius=0.2)
    changed = _gate(radius=0.3)

    assert first.decision_record_id == second.decision_record_id
    assert first.decision_record_id != changed.decision_record_id
    assert len(first.decision_record_id) == 64


def test_outputs_are_immutable() -> None:
    decision = _gate(radius=0.2)

    with pytest.raises(ValueError):
        decision.total_pairwise_upper_bound[0, 1] = 0.0
    with pytest.raises(ValueError):
        decision.robustly_optimal[0] = False


def test_invalid_top_level_contracts_raise_before_a_fallback_can_be_defined() -> None:
    with pytest.raises(ValueError, match="at least two"):
        act_or_fallback_symmetry_complete_v1(
            action_names=("only",),
            structural_pairwise_upper_bound=[[0.0]],
            structural_certificate_id="certificate",
            state_evidence_id="state",
            action_template_id="bank",
            loss_id="loss",
            fallback_action=0,
            fallback_id="fallback",
            causal4d_receipt=_receipt(),
        )
    with pytest.raises(ValueError, match="fallback_action"):
        _gate(fallback_action=9)
