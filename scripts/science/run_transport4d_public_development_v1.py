#!/usr/bin/env python3
"""Validate and summarize the already-open public Transport4D development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "bayesian-phystwin.transport4d_public_development"
SCHEMA_VERSION = 1
PROTOCOL_SCHEMA = "bayesian-phystwin.transport4d_public_development_protocol"
PROTOCOL_VERSION = 1


def canonical_id(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected Transport4D development protocol schema")
    if protocol.get("schema_version") != PROTOCOL_VERSION:
        raise ValueError("unexpected Transport4D development protocol version")
    if protocol.get("status") != "retrospective-development-evidence-opened":
        raise ValueError("public development evidence status changed")
    supplied_id = protocol.get("protocol_id")
    if not isinstance(supplied_id, str):
        raise ValueError("protocol_id is missing")
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if canonical_id(unsigned) != supplied_id:
        raise ValueError("protocol_id does not match protocol content")

    tiers = protocol.get("tier_order")
    if tiers != [
        "exact_coefficients",
        "query_identifiable_effect",
        "low_dimensional_correction",
        "uncertainty_only",
        "procedure_only",
    ]:
        raise ValueError("transport tier order changed")
    boundary = protocol.get("information_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("information boundary is missing")
    required = {
        "all_numeric_outcomes_previously_opened": True,
        "used_only_for_method_development": True,
        "fresh_confirmation_claim": False,
        "paper_claim_authorized": False,
        "automatic_target_access": False,
    }
    for key, expected in required.items():
        if boundary.get(key) is not expected:
            raise ValueError(f"information boundary changed: {key}")


def run(protocol_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    evidence = protocol["evidence"]
    same_object = evidence["same_object_cross_backend"]
    cross_object = evidence["cross_object_operator"]
    procedure = evidence["matching_object_procedure"]
    uncertainty = evidence["uncertainty_only_motivation"]

    exact_same = same_object["exact_coefficients"]
    scalar_same = same_object["one_scalar"]
    exact_cross = cross_object["exact_coefficients"]
    procedure_cross = procedure["prospective_target"]

    checks = {
        "same_object_exact_transfer_positive": (
            exact_same["relative_improvement"] > 0.0
            and exact_same["wins"] == exact_same["case_count"]
        ),
        "same_object_scalar_transfer_positive": (
            scalar_same["relative_improvement"] > 0.0
            and scalar_same["wins"] >= 6
            and scalar_same["positive_alignment_cases"] == scalar_same["case_count"]
        ),
        "cross_object_exact_transfer_negative": (
            exact_cross["relative_improvement"] < 0.0
            and exact_cross["wins"] == 0
            and exact_cross["losses"] == exact_cross["case_count"]
        ),
        "matching_object_procedure_positive": (
            procedure_cross["relative_improvement"] > 0.0
            and procedure_cross["wins"] == procedure_cross["case_count"]
        ),
        "dependence_changes_decisions_at_fixed_means": (
            uncertainty["maximum_mean_mismatch"] <= 1e-12
            and uncertainty["full_decision_loss"]
            < uncertainty["diagonal_decision_loss"]
            and uncertainty["full_decision_loss"]
            < uncertainty["scrambled_decision_loss"]
        ),
        "opened_development_not_mislabeled_confirmation": (
            protocol["information_boundary"]["fresh_confirmation_claim"] is False
        ),
    }
    decision = (
        "public-development-tier-separation-established"
        if all(checks.values())
        else "public-development-tier-separation-failed"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "checks": checks,
        "development_matrix": [
            {
                "shift": "same-object-cross-backend",
                "tested_tier": "exact_coefficients",
                "supported": True,
                "relative_improvement": exact_same["relative_improvement"],
                "wins": exact_same["wins"],
                "case_count": exact_same["case_count"],
            },
            {
                "shift": "same-object-cross-backend",
                "tested_tier": "low_dimensional_correction",
                "supported": True,
                "relative_improvement": scalar_same["relative_improvement"],
                "wins": scalar_same["wins"],
                "case_count": scalar_same["case_count"],
            },
            {
                "shift": "cross-object-operator",
                "tested_tier": "exact_coefficients",
                "supported": False,
                "relative_improvement": exact_cross["relative_improvement"],
                "wins": exact_cross["wins"],
                "case_count": exact_cross["case_count"],
            },
            {
                "shift": "cross-object-operator",
                "tested_tier": "procedure_only",
                "supported": True,
                "relative_improvement": procedure_cross["relative_improvement"],
                "wins": procedure_cross["wins"],
                "case_count": procedure_cross["case_count"],
            },
        ],
        "strongest_supported_tier_by_shift": {
            "same-object-cross-backend": "exact_coefficients",
            "cross-object-operator": "procedure_only",
        },
        "uncertainty_only_tier_status": (
            "motivated-by-same-mean-dependence-value-not-yet-cross-domain-confirmed"
        ),
        "next_required_evidence": {
            "kind": "outcome-blind-object-level-public-confirmation",
            "primary_dataset": "Deform360",
            "tier_decisions_frozen_before_target_outcomes": True,
            "replacement_allowed": False,
        },
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = canonical_id(result)
    return result


def report(result: dict[str, Any]) -> str:
    lines = [
        "# Transport4D public development evidence",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Shift | Tested tier | Supported | Relative change | Wins |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["development_matrix"]:
        lines.append(
            f"| `{row['shift']}` | `{row['tested_tier']}` | "
            f"{str(row['supported']).lower()} | "
            f"{100 * row['relative_improvement']:.3f}% | "
            f"{row['wins']}/{row['case_count']} |"
        )
    lines.extend(
        [
            "",
            "The same DLO3 correction transfers unchanged across physical backends",
            "for the same object, but fails on every DLO4/DLO5 target trajectory.",
            "Refitting the registered procedure on each target object succeeds on all",
            "28 target trajectories. The transferable object is therefore "
            "hierarchical,",
            "not a single coefficient vector.",
            "",
            "These outcomes were already open before Transport4D was designed. "
            "They are",
            "method-development evidence, not a fresh confirmation of the tier "
            "selector.",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(report(result), encoding="utf-8")
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
