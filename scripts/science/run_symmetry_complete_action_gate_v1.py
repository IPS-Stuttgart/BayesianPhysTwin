#!/usr/bin/env python3
"""Deterministic end-to-end symmetry-complete act-or-fallback study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.symmetry_complete_action_gate_v1 import (
    act_or_fallback_symmetry_complete_v1,
)

SCHEMA = "bayesian-phystwin.symmetry-complete-action-gate-study"
SCHEMA_VERSION = 1
ACTION_NAMES = ("track-shared-frame", "orthogonal-shared-frame", "fallback")
STRUCTURAL_PAIRWISE = np.array(
    [
        [0.0, -2.0, -1.0],
        [2.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
REALIZATION_RADII = (0.0, 0.2, 0.8, 1.0, 1.2)
CLAIM_BOUNDARY = (
    "Controlled cross-repository contract evidence. The Prob4D structural gap, "
    "Causal4D receipt contents and optional transport margin are supplied. This "
    "does not validate the physical symmetry, learned provider, target exchangeability, "
    "robot actuator, loss constants, deployment safety, or state of the art."
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


def _receipt(radius: float) -> dict[str, object]:
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


def _decision(
    radius: float,
    *,
    tolerance: float = 0.0,
    transport_margin: np.ndarray | None = None,
    receipt: dict[str, object] | None = None,
) -> dict[str, Any]:
    result = act_or_fallback_symmetry_complete_v1(
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
        transport_pairwise_margin=transport_margin,
        transport_certificate_id=(
            None if transport_margin is None else "object-calibration-v1"
        ),
        require_radius_scope="deterministic-complete",
    )
    return {
        "radius": radius,
        "regret_tolerance": tolerance,
        "status": result.status,
        "valid_evidence": result.valid_evidence,
        "invalid_reasons": list(result.invalid_reasons),
        "structural_pairwise_upper_bound": (
            result.structural_pairwise_upper_bound.tolist()
        ),
        "realization_pairwise_margin": (
            result.realization_pairwise_margin.tolist()
        ),
        "transport_pairwise_margin": result.transport_pairwise_margin.tolist(),
        "total_pairwise_upper_bound": result.total_pairwise_upper_bound.tolist(),
        "worst_case_regret_upper_bound": (
            result.worst_case_regret_upper_bound.tolist()
        ),
        "robustly_optimal": result.robustly_optimal.tolist(),
        "epsilon_admissible": result.epsilon_admissible.tolist(),
        "minimax_action": result.minimax_action,
        "fallback_action": result.fallback_action,
        "selected_action": result.selected_action,
        "selected_action_name": result.selected_action_name,
        "admitted": result.admitted,
        "exact_fallback": result.exact_fallback,
        "intervention_receipt_id": result.intervention_receipt_id,
        "transport_certificate_id": result.transport_certificate_id,
        "decision_record_id": result.decision_record_id,
    }


def build_result() -> dict[str, Any]:
    sweep = [_decision(radius) for radius in REALIZATION_RADII]
    bounded = _decision(1.2, tolerance=0.25)
    transport = np.array(
        [
            [0.0, 0.0, 1.1],
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ]
    )
    transported = _decision(0.0, transport_margin=transport)
    tampered_receipt = _receipt(0.2)
    tampered_receipt["declared_realization_radius_by_action"] = [0.3, 0.0, 0.0]
    tampered = _decision(0.2, receipt=tampered_receipt)
    mismatched_receipt = _receipt(0.0)
    payload = {
        key: value
        for key, value in mismatched_receipt.items()
        if key != "receipt_id"
    }
    payload["state_evidence_id"] = "other-state-evidence"
    payload["receipt_id"] = _canonical_digest(payload)
    mismatched = _decision(0.0, receipt=payload)

    checks = {
        "small_realization_radii_execute_exact_action": all(
            row["status"] == "verified-exact-optimal"
            and row["selected_action"] == 0
            and row["admitted"]
            for row in sweep[:-1]
        ),
        "large_realization_radius_rejects_to_exact_fallback": bool(
            sweep[-1]["status"] == "verified-reject-exact-fallback"
            and sweep[-1]["selected_action"] == 2
            and sweep[-1]["exact_fallback"]
            and abs(sweep[-1]["worst_case_regret_upper_bound"][0] - 0.2)
            < 1e-12
        ),
        "registered_tolerance_admits_bounded_regret": bool(
            bounded["status"] == "verified-bounded-regret"
            and bounded["selected_action"] == 0
            and bounded["admitted"]
        ),
        "transport_margin_is_not_hidden": bool(
            transported["transport_certificate_id"] == "object-calibration-v1"
            and abs(transported["worst_case_regret_upper_bound"][0] - 0.1)
            < 1e-12
            and transported["exact_fallback"]
        ),
        "tampered_receipt_fails_closed": bool(
            tampered["status"] == "invalid-fail-closed"
            and not tampered["valid_evidence"]
            and tampered["selected_action"] == 2
            and tampered["exact_fallback"]
        ),
        "cross_context_receipt_fails_closed": bool(
            mismatched["status"] == "invalid-fail-closed"
            and not mismatched["valid_evidence"]
            and mismatched["selected_action"] == 2
            and mismatched["exact_fallback"]
        ),
        "all_decision_records_are_content_addressed": all(
            len(row["decision_record_id"]) == 64
            for row in [*sweep, bounded, transported, tampered, mismatched]
        ),
    }
    decision = (
        "controlled-symmetry-complete-action-gate-passed"
        if all(checks.values())
        else "controlled-symmetry-complete-action-gate-failed"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "claim_boundary": CLAIM_BOUNDARY,
        "action_names": list(ACTION_NAMES),
        "structural_pairwise_upper_bound": STRUCTURAL_PAIRWISE.tolist(),
        "realization_radii": list(REALIZATION_RADII),
        "zero_tolerance_sweep": sweep,
        "bounded_regret_case": bounded,
        "transport_margin_case": transported,
        "tampered_receipt_case": tampered,
        "cross_context_receipt_case": mismatched,
        "checks": checks,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["result_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    result = build_result()
    if result["decision"] != "controlled-symmetry-complete-action-gate-passed":
        raise SystemExit(json.dumps(result["checks"], indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
