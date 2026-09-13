#!/usr/bin/env python3
"""Verify the Prob4D -> Causal4D -> BayesianPhysTwin action chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.symmetry_complete_action_gate_v1 import (
    act_or_fallback_symmetry_complete_v1,
)
from prob4d.equivariant_decision import certify_gauge_coupled_actions

SCHEMA = "symmetry-complete-three-repository-chain"
SCHEMA_VERSION = 1
ACTION_NAMES = ("track-shared-frame", "orthogonal-shared-frame", "fallback")
CLAIM_BOUNDARY = (
    "Controlled cross-repository interface evidence only. It does not validate "
    "the physical symmetry, provider, target transport, actuator, loss bounds, "
    "deployment safety, or state of the art."
)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def build_result(
    causal_result: dict[str, Any],
    *,
    prob4d_head: str,
    causal4d_head: str,
    bayesian_phystwin_head: str,
) -> dict[str, Any]:
    if causal_result.get("decision") != "controlled-shared-gauge-receipt-passed":
        raise ValueError("Causal4D controlled receipt result did not pass")
    causal_checks = causal_result.get("checks")
    if not isinstance(causal_checks, dict) or not all(causal_checks.values()):
        raise ValueError("Causal4D controlled receipt checks did not all pass")
    receipt = causal_result.get("portable_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("Causal4D result does not contain a portable receipt")

    losses = np.empty((1, 4, 3), dtype=np.float64)
    losses[0, :, :] = np.array([0.0, 2.0, 1.0])
    structural_certificate = certify_gauge_coupled_actions(
        losses,
        [1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        fallback_action=2,
        regret_tolerance=0.0,
    )
    structural = structural_certificate.pairwise_upper_bound
    decision = act_or_fallback_symmetry_complete_v1(
        action_names=ACTION_NAMES,
        structural_pairwise_upper_bound=structural,
        structural_certificate_id="prob4d-crossrepo-c4-v1",
        state_evidence_id=receipt["state_evidence_id"],
        action_template_id=receipt["action_template_id"],
        loss_id=receipt["loss_id"],
        fallback_action=2,
        fallback_id=receipt["fallback_id"],
        causal4d_receipt=receipt,
        regret_tolerance=0.0,
        require_radius_scope="deterministic-complete",
    )

    tampered = json.loads(json.dumps(receipt))
    tampered["pairwise_realization_margin"][0][2] = 0.0
    invalid = act_or_fallback_symmetry_complete_v1(
        action_names=ACTION_NAMES,
        structural_pairwise_upper_bound=structural,
        structural_certificate_id="prob4d-crossrepo-c4-v1",
        state_evidence_id=receipt["state_evidence_id"],
        action_template_id=receipt["action_template_id"],
        loss_id=receipt["loss_id"],
        fallback_action=2,
        fallback_id=receipt["fallback_id"],
        causal4d_receipt=tampered,
        regret_tolerance=0.0,
        require_radius_scope="deterministic-complete",
    )

    checks = {
        "prob4d_structural_action_exact": bool(
            structural_certificate.robustly_optimal[0]
            and structural_certificate.worst_case_regret_upper_bound[0] < 1e-12
        ),
        "causal4d_portable_receipt_passed": True,
        "bayesian_phystwin_executes_action_0": bool(
            decision.valid_evidence
            and decision.status == "verified-exact-optimal"
            and decision.selected_action == 0
            and decision.admitted
        ),
        "realization_margin_consumed_exactly": bool(
            np.array_equal(
                decision.realization_pairwise_margin,
                np.asarray(receipt["pairwise_realization_margin"]),
            )
        ),
        "tampered_crossrepo_receipt_fails_closed": bool(
            not invalid.valid_evidence
            and invalid.status == "invalid-fail-closed"
            and invalid.selected_action == 2
            and invalid.exact_fallback
        ),
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": (
            "three-repository-chain-passed"
            if all(checks.values())
            else "three-repository-chain-failed"
        ),
        "repository_heads": {
            "Prob4D": prob4d_head,
            "Causal4D": causal4d_head,
            "BayesianPhysTwin": bayesian_phystwin_head,
        },
        "prob4d": {
            "structural_pairwise_upper_bound": structural.tolist(),
            "worst_case_regret_upper_bound": (
                structural_certificate.worst_case_regret_upper_bound.tolist()
            ),
            "selected_action": structural_certificate.selected_action,
        },
        "causal4d": {
            "contract_id": receipt["contract_id"],
            "receipt_id": receipt["receipt_id"],
            "pairwise_realization_margin": receipt["pairwise_realization_margin"],
            "radius_scope": receipt["radius_scope"],
        },
        "bayesian_phystwin": {
            "status": decision.status,
            "total_pairwise_upper_bound": (
                decision.total_pairwise_upper_bound.tolist()
            ),
            "worst_case_regret_upper_bound": (
                decision.worst_case_regret_upper_bound.tolist()
            ),
            "selected_action": decision.selected_action,
            "selected_action_name": decision.selected_action_name,
            "admitted": decision.admitted,
            "decision_record_id": decision.decision_record_id,
        },
        "tampered_receipt_control": {
            "status": invalid.status,
            "selected_action": invalid.selected_action,
            "exact_fallback": invalid.exact_fallback,
            "invalid_reasons": list(invalid.invalid_reasons),
        },
        "checks": checks,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    canonical = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    result["result_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-result", type=Path, required=True)
    parser.add_argument("--prob4d-head", required=True)
    parser.add_argument("--causal4d-head", required=True)
    parser.add_argument("--bayesian-phystwin-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    result = build_result(
        _read_json_object(args.causal_result),
        prob4d_head=args.prob4d_head,
        causal4d_head=args.causal4d_head,
        bayesian_phystwin_head=args.bayesian_phystwin_head,
    )
    if result["decision"] != "three-repository-chain-passed":
        raise SystemExit(json.dumps(result["checks"], indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
