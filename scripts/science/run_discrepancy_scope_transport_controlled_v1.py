#!/usr/bin/env python3
"""Generate controlled and public-development discrepancy-scope evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin_experiments.discrepancy_scope_transport_v1 import (
    DiscrepancyScopeTransportV1,
    EvidenceDisposition,
    OperationalDisposition,
    PortableTargetDiagnosisV1,
    ScopeHypothesis,
    ScopeStatus,
    TransferAxis,
    TransferEvidenceV1,
    TransportTier,
)

FALLBACK_ID = "f" * 64
DIAGNOSIS_ID = "d" * 64
CONTROL_EVIDENCE_ID = "c" * 64
PUBLIC_DEFORM_ARTIFACT_SHA256 = (
    "bb3db2474a7f91e0bf1df869cb48f1105913f7b56aca7c98b6557323dde588b0"
)
PUBLIC_DEFORM_WORKFLOW_RUN = 33536420739
PUBLIC_DEFORM_ARTIFACT_ID = 9811886089
RESERVE_ID = "0fba6a0cda4ac23fc8f900ca4632d73a22ae0559d42e48cc2ec8c5ea79030dc4"
SUPPORT_ID = "57aaea3215ad1cdf76accfc455c173bb5d84a926bfed013fa64b6ec3625befed"


def canonical_id(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def diagnosis(
    disposition: str = "transport_without_cause",
) -> PortableTargetDiagnosisV1:
    transporting = disposition in {"transport_without_cause", "explain_and_transport"}
    none = disposition == "none_of_the_above"
    return PortableTargetDiagnosisV1(
        pipeline_artifact_id=DIAGNOSIS_ID,
        target_id="held-intervention-query",
        disposition=disposition,
        adequacy_status=("unmodeled_cause" if none else "adequate_set_valued"),
        transport_permitted=transporting,
        fallback_required_now=not transporting,
        none_of_the_above=none,
    )


def controlled_evidence(
    *,
    same_object_new_backend: bool,
    new_object_same_backend: bool,
    new_object_new_backend: bool,
    procedure_on_double_shift: bool = False,
) -> list[TransferEvidenceV1]:
    records = []
    for axis, passed in (
        (
            TransferAxis.SAME_OBJECT_NEW_BACKEND,
            same_object_new_backend,
        ),
        (
            TransferAxis.NEW_OBJECT_SAME_BACKEND,
            new_object_same_backend,
        ),
        (
            TransferAxis.NEW_OBJECT_NEW_BACKEND,
            new_object_new_backend,
        ),
    ):
        records.append(
            TransferEvidenceV1(
                axis=axis,
                tier=TransportTier.EXACT_COEFFICIENTS,
                disposition=(
                    EvidenceDisposition.SUPPORTED
                    if passed
                    else EvidenceDisposition.REJECTED
                ),
                evidence_id=CONTROL_EVIDENCE_ID,
                relative_improvement=0.08 if passed else -0.08,
                wins=8 if passed else 0,
                total=8,
                description="deterministic controlled transfer-axis outcome",
            )
        )
    if procedure_on_double_shift:
        records.append(
            TransferEvidenceV1(
                axis=TransferAxis.NEW_OBJECT_NEW_BACKEND,
                tier=TransportTier.PROCEDURE_ONLY,
                disposition=EvidenceDisposition.SUPPORTED,
                evidence_id=CONTROL_EVIDENCE_ID,
                relative_improvement=0.04,
                wins=8,
                total=8,
                description="controlled procedure-only recovery",
            )
        )
    return records


def controlled_cases() -> dict[str, DiscrepancyScopeTransportV1]:
    cases: dict[str, DiscrepancyScopeTransportV1] = {}
    patterns = {
        "shared-physics": (True, True, True),
        "object-specific-backend-stable": (True, False, False),
        "backend-specific-object-stable": (False, True, False),
        "object-backend-local": (False, False, False),
    }
    for name, pattern in patterns.items():
        requested_axis = (
            TransferAxis.NEW_OBJECT_NEW_BACKEND
            if name == "shared-physics"
            else TransferAxis.SAME_OBJECT_NEW_BACKEND
        )
        cases[name] = DiscrepancyScopeTransportV1(
            requested_axis=requested_axis,
            fallback_id=FALLBACK_ID,
            evidence=controlled_evidence(
                same_object_new_backend=pattern[0],
                new_object_same_backend=pattern[1],
                new_object_new_backend=pattern[2],
                procedure_on_double_shift=name != "shared-physics",
            ),
            diagnosis=diagnosis(),
            metadata={"case": name},
        )

    object_records = controlled_evidence(
        same_object_new_backend=True,
        new_object_same_backend=False,
        new_object_new_backend=False,
    )
    object_records.append(
        TransferEvidenceV1(
            axis=TransferAxis.NEW_OBJECT_SAME_BACKEND,
            tier=TransportTier.PROCEDURE_ONLY,
            disposition=EvidenceDisposition.SUPPORTED,
            evidence_id=CONTROL_EVIDENCE_ID,
            relative_improvement=0.04,
            wins=8,
            total=8,
            description="procedure transfers although exact coefficients do not",
        )
    )
    cases["object-shift-procedure-only"] = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_SAME_BACKEND,
        fallback_id=FALLBACK_ID,
        evidence=object_records,
        diagnosis=diagnosis(),
        metadata={"case": "object-shift-procedure-only"},
    )
    cases["unmodeled-cause"] = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK_ID,
        evidence=controlled_evidence(
            same_object_new_backend=True,
            new_object_same_backend=True,
            new_object_new_backend=True,
        ),
        diagnosis=diagnosis("none_of_the_above"),
        metadata={"case": "unmodeled-cause"},
    )
    cases["incompatible-scope-family"] = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK_ID,
        evidence=controlled_evidence(
            same_object_new_backend=False,
            new_object_same_backend=False,
            new_object_new_backend=True,
        ),
        diagnosis=diagnosis(),
        metadata={"case": "incompatible-scope-family"},
    )
    return cases


def public_deform_evidence() -> list[TransferEvidenceV1]:
    common_metadata = {
        "workflow_run_id": PUBLIC_DEFORM_WORKFLOW_RUN,
        "artifact_id": PUBLIC_DEFORM_ARTIFACT_ID,
        "artifact_sha256": PUBLIC_DEFORM_ARTIFACT_SHA256,
        "evidence_role": "retrospective-public-development-matrix",
    }
    return [
        TransferEvidenceV1(
            axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
            tier=TransportTier.EXACT_COEFFICIENTS,
            disposition=EvidenceDisposition.SUPPORTED,
            evidence_id=PUBLIC_DEFORM_ARTIFACT_SHA256,
            relative_improvement=0.02985,
            wins=8,
            total=8,
            description=(
                "DLO3 coefficients fitted against DEFORM improve PyElastica "
                "without coefficient refitting"
            ),
            metadata=common_metadata,
        ),
        TransferEvidenceV1(
            axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
            tier=TransportTier.SCALAR_AMPLITUDE,
            disposition=EvidenceDisposition.SUPPORTED,
            evidence_id=PUBLIC_DEFORM_ARTIFACT_SHA256,
            relative_improvement=0.02250,
            wins=6,
            total=8,
            description=(
                "DLO3 residual direction with one source-fitted scalar improves "
                "the PyElastica backend"
            ),
            metadata=common_metadata,
        ),
        TransferEvidenceV1(
            axis=TransferAxis.NEW_OBJECT_SAME_BACKEND,
            tier=TransportTier.EXACT_COEFFICIENTS,
            disposition=EvidenceDisposition.REJECTED,
            evidence_id=PUBLIC_DEFORM_ARTIFACT_SHA256,
            relative_improvement=-0.16590,
            wins=0,
            total=28,
            description=(
                "Unchanged DLO3 coefficients are worse on DLO4/DLO5 under the "
                "DEFORM backend"
            ),
            metadata=common_metadata,
        ),
        TransferEvidenceV1(
            axis=TransferAxis.NEW_OBJECT_SAME_BACKEND,
            tier=TransportTier.PROCEDURE_ONLY,
            disposition=EvidenceDisposition.SUPPORTED,
            evidence_id=PUBLIC_DEFORM_ARTIFACT_SHA256,
            relative_improvement=0.06803,
            wins=28,
            total=28,
            description=(
                "The frozen fitting procedure, refitted on the matching DLO, "
                "improves every DLO4/DLO5 source trajectory"
            ),
            metadata=common_metadata,
        ),
        TransferEvidenceV1(
            axis=TransferAxis.NEW_OBJECT_NEW_BACKEND,
            tier=TransportTier.EXACT_COEFFICIENTS,
            disposition=EvidenceDisposition.UNAVAILABLE,
            evidence_id=None,
            description="No direct cross-object plus cross-backend coefficient test",
            frozen_before_outcome=True,
            target_selection_free=True,
        ),
    ]


def build_result() -> dict[str, Any]:
    cases = controlled_cases()
    expected_scopes = {
        "shared-physics": ScopeHypothesis.SHARED_PHYSICS,
        "object-specific-backend-stable": (
            ScopeHypothesis.OBJECT_SPECIFIC_BACKEND_STABLE
        ),
        "backend-specific-object-stable": (
            ScopeHypothesis.BACKEND_SPECIFIC_OBJECT_STABLE
        ),
        "object-backend-local": ScopeHypothesis.OBJECT_BACKEND_LOCAL,
    }
    checks = {
        "all_four_registered_scopes_separated": all(
            cases[name].scope_status is ScopeStatus.UNIQUE
            and cases[name].compatible_scopes == (scope,)
            for name, scope in expected_scopes.items()
        ),
        "shared_physics_authorizes_double_shift_coefficients": (
            cases["shared-physics"].operational_disposition
            is OperationalDisposition.TRANSPORT_EXACT_COEFFICIENTS
            and not cases["shared-physics"].fallback_required_now
        ),
        "object_shift_descends_to_procedure_only": (
            cases["object-shift-procedure-only"].operational_disposition
            is OperationalDisposition.PROCEDURE_ONLY_REFIT_REQUIRED
            and cases["object-shift-procedure-only"].fallback_required_now
        ),
        "unmodeled_cause_overrides_positive_transfer": (
            cases["unmodeled-cause"].operational_disposition
            is OperationalDisposition.NONE_OF_THE_ABOVE
            and cases["unmodeled-cause"].fallback_required_now
        ),
        "unregistered_transfer_pattern_fails_closed": (
            cases["incompatible-scope-family"].scope_status
            is ScopeStatus.NONE_OF_THE_ABOVE
            and cases["incompatible-scope-family"].fallback_required_now
        ),
    }

    public_records = public_deform_evidence()
    public_backend = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK_ID,
        evidence=public_records,
        diagnosis=None,
        metadata={
            "study": "DEFORM-DLO3-DLO4-DLO5-public-development",
            "reserve_id": RESERVE_ID,
            "support_id": SUPPORT_ID,
        },
    )
    public_object = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_SAME_BACKEND,
        fallback_id=FALLBACK_ID,
        evidence=public_records,
        diagnosis=None,
        metadata={
            "study": "DEFORM-DLO3-DLO4-DLO5-public-development",
            "reserve_id": RESERVE_ID,
            "support_id": SUPPORT_ID,
        },
    )
    public_double = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK_ID,
        evidence=public_records,
        diagnosis=None,
        metadata={
            "study": "DEFORM-DLO3-DLO4-DLO5-public-development",
            "reserve_id": RESERVE_ID,
            "support_id": SUPPORT_ID,
        },
    )
    public_checks = {
        "exact_scope_is_object_specific_backend_stable_on_tested_axes": (
            public_backend.scope_status is ScopeStatus.UNIQUE
            and public_backend.compatible_scopes
            == (ScopeHypothesis.OBJECT_SPECIFIC_BACKEND_STABLE,)
        ),
        "same_object_new_backend_strongest_tier_is_exact_coefficients": (
            public_backend.strongest_directly_supported_tier
            is TransportTier.EXACT_COEFFICIENTS
        ),
        "new_object_same_backend_strongest_tier_is_procedure_only": (
            public_object.strongest_directly_supported_tier
            is TransportTier.PROCEDURE_ONLY
        ),
        "new_object_new_backend_remains_unsupported": (
            public_double.strongest_directly_supported_tier is None
        ),
        "no_real_transport_is_authorized_without_cause_diagnosis": all(
            certificate.operational_disposition is OperationalDisposition.EVIDENCE_ONLY
            and certificate.fallback_required_now
            for certificate in (public_backend, public_object, public_double)
        ),
    }
    checks.update(public_checks)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"registered checks failed: {failed}")

    result: dict[str, Any] = {
        "schema": "bayesian_phystwin.discrepancy-scope-transport-study",
        "schema_version": 1,
        "decision": "controlled-separation-and-bounded-public-scope-diagnosis",
        "checks": checks,
        "controlled_cases": {
            name: certificate.to_record() for name, certificate in sorted(cases.items())
        },
        "public_deform_development": {
            "classification": ("object-specific-backend-stable-on-tested-DLO3-axes"),
            "strongest_supported_tier": {
                TransferAxis.SAME_OBJECT_NEW_BACKEND.value: (
                    TransportTier.EXACT_COEFFICIENTS.value
                ),
                TransferAxis.NEW_OBJECT_SAME_BACKEND.value: (
                    TransportTier.PROCEDURE_ONLY.value
                ),
                TransferAxis.NEW_OBJECT_NEW_BACKEND.value: None,
            },
            "certificates": {
                "same-object-new-backend": public_backend.to_record(),
                "new-object-same-backend": public_object.to_record(),
                "new-object-new-backend": public_double.to_record(),
            },
            "development_evidence": {
                "workflow_run_id": PUBLIC_DEFORM_WORKFLOW_RUN,
                "artifact_id": PUBLIC_DEFORM_ARTIFACT_ID,
                "artifact_sha256": PUBLIC_DEFORM_ARTIFACT_SHA256,
                "same_object_new_backend_exact_improvement": 0.02985,
                "same_object_new_backend_exact_wins": "8/8",
                "new_object_same_backend_exact_improvement": -0.16590,
                "new_object_same_backend_exact_wins": "0/28",
                "new_object_same_backend_procedure_improvement": 0.06803,
                "new_object_same_backend_procedure_wins": "28/28",
            },
            "fresh_confirmation_reserve": {
                "reserve_id": RESERVE_ID,
                "support_id": SUPPORT_ID,
                "supported_calibration_objects": 15,
                "supported_confirmation_objects": 33,
                "numerical_confirmation_authorized": False,
            },
            "interpretation": (
                "The current DLO evidence rejects a shared exact correction: "
                "the DLO3 coefficients survive a backend change on the same "
                "object but fail across DLO4/DLO5. The fitting procedure, not "
                "the coefficient vector, is the strongest supported "
                "cross-object tier. This is retrospective development evidence "
                "and no real cause diagnosis is bound, so no correction is "
                "operationally authorized by this study."
            ),
        },
        "claim_boundary": (
            "The controlled cases verify the finite scope/tier logic. The public "
            "classification is a retrospective diagnosis over one DLO3 "
            "cross-backend test and DLO3-to-DLO4/DLO5 same-backend tests. It "
            "does not prove a natural cause, a universal object-specific law, "
            "fresh confirmation, unseen-object plus unseen-backend transport, "
            "or deployment safety. Numerical access to the frozen 33-object "
            "confirmation cohort remains unauthorized."
        ),
    }
    unsigned = dict(result)
    result["result_id"] = canonical_id(unsigned)
    return result


def render_report(result: dict[str, Any]) -> str:
    public = result["public_deform_development"]
    evidence = public["development_evidence"]
    lines = [
        "# Diagnose--Decompose--Transport controlled and public-development result",
        "",
        f"Decision: **`{result['decision']}`**",
        "",
        "## Combined contribution",
        "",
        "The certificate first consumes a cause-family diagnosis, then identifies "
        "the set of transfer scopes compatible with frozen exact-coefficient "
        "tests, and finally selects the strongest directly supported transport "
        "tier for the requested domain shift. A missing or inadequate diagnosis, "
        "an unregistered transfer pattern, or procedure-only evidence returns "
        "the exact fallback.",
        "",
        "## Controlled mechanism",
        "",
        "The deterministic study separates shared, object-specific/backend-stable, "
        "backend-specific/object-stable, and object--backend-local corrections. "
        "It additionally verifies that `none_of_the_above` overrides apparently "
        "positive transfer evidence and that procedure replication never becomes "
        "coefficient deployment.",
        "",
        "## Existing public DEFORM evidence",
        "",
        "| Shift | Tested reusable object | Result | Diagnosis |",
        "|---|---|---:|---|",
        (
            "| Same DLO3, DEFORM→PyElastica | unchanged coefficients | "
            f"**{100 * evidence['same_object_new_backend_exact_improvement']:.3f}%**, "
            f"{evidence['same_object_new_backend_exact_wins']} wins | "
            "exact coefficients supported on this tested backend shift |"
        ),
        (
            "| DLO3→DLO4/DLO5, DEFORM | unchanged coefficients | "
            f"**{100 * evidence['new_object_same_backend_exact_improvement']:.3f}%**, "
            f"{evidence['new_object_same_backend_exact_wins']} wins | "
            "cross-object coefficient transport rejected |"
        ),
        (
            "| Matching DLO4/DLO5 source refit | frozen fitting procedure | "
            f"**{100 * evidence['new_object_same_backend_procedure_improvement']:.3f}%**, "
            f"{evidence['new_object_same_backend_procedure_wins']} wins | "
            "procedure-only cross-object transfer supported |"
        ),
        "",
        "The resulting finite-family classification is:",
        "",
        "> **Object-specific but backend-stable on the tested DLO3 axes; "
        "procedure-only across DLO identities.**",
        "",
        "This explicitly rejects the stronger shared-exact-physics interpretation. "
        "Because no real interventional cause certificate is bound to these old "
        "DLO results, the public section remains evidence-only and authorizes no "
        "deployed transport.",
        "",
        "## Fresh next stage",
        "",
        "The already frozen Deform360 reserve contains 15 carrier-supported "
        "calibration objects and 33 fixed confirmation objects. The next numerical "
        "protocol must learn cause signatures and all tier gates on the 15 "
        "calibration objects, then jointly seal every confirmation prediction "
        "before opening the 33 confirmation outcomes. Unsupported objects cannot "
        "be replaced.",
        "",
        f"Result ID: `{result['result_id']}`.",
        "",
        "## Boundary",
        "",
        result["claim_boundary"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = build_result()
    report = render_report(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "result_id": result["result_id"],
                "public_classification": result["public_deform_development"][
                    "classification"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
