#!/usr/bin/env python3
"""Generate the controlled Transport4D tier-separation result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.transport4d_tiered_certificate_v1 import (
    TRANSPORT4D_CLAIM_BOUNDARY,
    TieredTransportCertificateV1,
    TransportCandidateV1,
    TransportTier,
)

SCHEMA = "bayesian-phystwin/transport4d-tiered-controlled-result-v1"
SCHEMA_VERSION = 1


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def canonical_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def candidate(
    case: str,
    tier: TransportTier,
    *,
    eligible: bool,
    effect: float | None = 1.0,
    radius: float | None = 0.1,
) -> TransportCandidateV1:
    adaptation = 0
    mean = True
    uncertainty = True
    if tier is TransportTier.LOW_DIMENSIONAL_CORRECTION:
        adaptation = 1
    elif tier is TransportTier.UNCERTAINTY_ONLY:
        mean = False
        uncertainty = True
        effect = None
        radius = None
    elif tier is TransportTier.PROCEDURE_ONLY:
        adaptation = 8
        mean = False
        uncertainty = False
        effect = None
        radius = None
    return TransportCandidateV1(
        tier=tier,
        evidence_id=digest(f"{case}:{tier.value}:evidence"),
        checks={
            "family_or_transform_adequate": eligible,
            "source_gate": eligible,
            "target_query_supported": eligible,
        },
        target_outcome_blind=True,
        adaptation_dimension=adaptation,
        transports_mean=mean,
        transports_uncertainty=uncertainty,
        query_effect=(
            np.asarray([effect], dtype=np.float64) if effect is not None else None
        ),
        query_error_radius=radius,
        metadata={"controlled_case": case},
    )


def certificate(
    case: str,
    candidates: tuple[TransportCandidateV1, ...],
) -> TieredTransportCertificateV1:
    return TieredTransportCertificateV1(
        query_id=f"{case}-benefit",
        query_contract_id=digest(f"{case}:query"),
        baseline_belief_id=digest(f"{case}:baseline"),
        action_portfolio_id=digest("hold-or-execute-affine-v1"),
        baseline_query=np.asarray([0.0]),
        action_names=("hold", "execute"),
        action_weights=np.asarray([[0.0], [-1.0]]),
        action_offsets=np.asarray([0.0, 0.0]),
        fallback_action_name="hold",
        candidates=candidates,
        regret_tolerance=0.0,
        metadata={"controlled_case": case},
    )


def run() -> dict[str, Any]:
    exact = TransportTier.EXACT_COEFFICIENTS
    query_effect = TransportTier.QUERY_IDENTIFIABLE_EFFECT
    low_dimensional = TransportTier.LOW_DIMENSIONAL_CORRECTION
    uncertainty_only = TransportTier.UNCERTAINTY_ONLY
    procedure_only = TransportTier.PROCEDURE_ONLY

    cases = [
        (
            "same-object-cross-backend",
            exact,
            (
                candidate("same-object-cross-backend", exact, eligible=True),
                candidate(
                    "same-object-cross-backend",
                    low_dimensional,
                    eligible=True,
                ),
                candidate(
                    "same-object-cross-backend",
                    procedure_only,
                    eligible=True,
                ),
            ),
        ),
        (
            "known-coordinate-pushforward",
            query_effect,
            (
                candidate(
                    "known-coordinate-pushforward",
                    exact,
                    eligible=False,
                ),
                candidate(
                    "known-coordinate-pushforward",
                    query_effect,
                    eligible=True,
                    effect=0.9,
                ),
                candidate(
                    "known-coordinate-pushforward",
                    procedure_only,
                    eligible=True,
                ),
            ),
        ),
        (
            "amplitude-recalibration",
            low_dimensional,
            (
                candidate("amplitude-recalibration", exact, eligible=False),
                candidate(
                    "amplitude-recalibration",
                    query_effect,
                    eligible=False,
                ),
                candidate(
                    "amplitude-recalibration",
                    low_dimensional,
                    eligible=True,
                    effect=0.8,
                    radius=0.15,
                ),
                candidate(
                    "amplitude-recalibration",
                    procedure_only,
                    eligible=True,
                ),
            ),
        ),
        (
            "mean-unsupported-dependence-retained",
            uncertainty_only,
            (
                candidate(
                    "mean-unsupported-dependence-retained",
                    exact,
                    eligible=False,
                ),
                candidate(
                    "mean-unsupported-dependence-retained",
                    low_dimensional,
                    eligible=False,
                ),
                candidate(
                    "mean-unsupported-dependence-retained",
                    uncertainty_only,
                    eligible=True,
                ),
                candidate(
                    "mean-unsupported-dependence-retained",
                    procedure_only,
                    eligible=True,
                ),
            ),
        ),
        (
            "cross-object-coefficient-failure",
            procedure_only,
            (
                candidate(
                    "cross-object-coefficient-failure",
                    exact,
                    eligible=False,
                ),
                candidate(
                    "cross-object-coefficient-failure",
                    low_dimensional,
                    eligible=False,
                ),
                candidate(
                    "cross-object-coefficient-failure",
                    uncertainty_only,
                    eligible=False,
                ),
                candidate(
                    "cross-object-coefficient-failure",
                    procedure_only,
                    eligible=True,
                ),
            ),
        ),
        (
            "query-conditional-descent",
            query_effect,
            (
                candidate(
                    "query-conditional-descent",
                    exact,
                    eligible=True,
                    effect=0.05,
                    radius=0.20,
                ),
                candidate(
                    "query-conditional-descent",
                    query_effect,
                    eligible=True,
                    effect=0.8,
                    radius=0.1,
                ),
            ),
        ),
        (
            "no-supported-transport",
            None,
            (
                candidate("no-supported-transport", exact, eligible=False),
                candidate(
                    "no-supported-transport",
                    procedure_only,
                    eligible=False,
                ),
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    exact_fallback_cases = 0
    expected_selection_count = 0
    for name, expected_tier, candidates in cases:
        result = certificate(name, candidates)
        selected = result.selected_tier
        expected_selection_count += int(selected is expected_tier)
        should_fallback = expected_tier in {
            None,
            uncertainty_only,
            procedure_only,
        }
        exact_fallback_cases += int(
            (not should_fallback)
            or (
                result.used_exact_fallback
                and result.selected_action_name is result.fallback_action_name
            )
        )
        rows.append(
            {
                "case": name,
                "expected_tier": (
                    expected_tier.value if expected_tier is not None else None
                ),
                "selected_tier": selected.value if selected is not None else None,
                "selected_action": result.selected_action_name,
                "used_exact_fallback": result.used_exact_fallback,
                "belief_transport_only": result.belief_transport_only,
                "certificate_id": result.artifact_id,
                "tier_evaluations": [
                    evaluation.to_record() for evaluation in result.tier_evaluations
                ],
            }
        )

    checks = {
        "all_expected_tiers_selected": expected_selection_count == len(cases),
        "belief_and_procedure_cases_return_exact_fallback": (
            exact_fallback_cases == len(cases)
        ),
        "query_conditional_descent_rejects_uncertain_exact_tier": (
            rows[5]["selected_tier"] == query_effect.value
            and rows[5]["tier_evaluations"][0]["reason_code"]
            == "regret-budget-exceeded"
        ),
        "unsupported_case_selects_no_transport_tier": (
            rows[6]["selected_tier"] is None and rows[6]["used_exact_fallback"] is True
        ),
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": (
            "controlled-tier-separation-passed"
            if all(checks.values())
            else "controlled-tier-separation-failed"
        ),
        "case_count": len(cases),
        "expected_selection_count": expected_selection_count,
        "checks": checks,
        "cases": rows,
        "claim": (
            "A physical twin can select the strongest source-justified "
            "query-conditional correction tier, descend when a more specific "
            "tier does not certify the pending action, and preserve exact "
            "fallback for uncertainty-only, procedure-only, and unsupported cases."
        ),
        "claim_boundary": TRANSPORT4D_CLAIM_BOUNDARY,
        "public_data_result": False,
        "target_outcomes_used": False,
    }
    result["result_id"] = canonical_id(result)
    return result


def report(result: dict[str, Any]) -> str:
    lines = [
        "# Controlled Transport4D tier separation",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Case | Expected | Selected | Action | Exact fallback |",
        "|---|---|---|---|---:|",
    ]
    for row in result["cases"]:
        lines.append(
            f"| `{row['case']}` | `{row['expected_tier']}` | "
            f"`{row['selected_tier']}` | `{row['selected_action']}` | "
            f"{str(row['used_exact_fallback']).lower()} |"
        )
    lines.extend(
        [
            "",
            "The strict hierarchy is exact coefficients, a query-identifiable",
            "effect, low-dimensional correction, uncertainty only, and",
            "procedure only. Deterministic mean tiers additionally need a unique",
            "finite-action decision inside the registered robust-regret budget.",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    if not all(result["checks"].values()):
        raise SystemExit("controlled Transport4D checks failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
