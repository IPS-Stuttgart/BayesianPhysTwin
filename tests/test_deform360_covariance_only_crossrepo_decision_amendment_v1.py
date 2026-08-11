from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_covariance_only_crossrepo_decision_amendment_v1.json"
)
CODE_PROTOCOL = ROOT / (
    "protocols/locks/deform360_covariance_only_independent_validation_v1.json"
)
CROSSREPO_BINDING = ROOT / (
    "protocols/locks/"
    "deform360_covariance_only_crossrepo_preregistration_binding_v1.json"
)
EXPECTED_AMENDMENT_ID = (
    "efacabe4ceb6e1d3c4cd523e0959bdf16f8ff4253f9800e8de11574734623802"
)
PAPER_PARENT_ID = "fa16c105e6d535d1e229ccf086fd69d05b2be74592b5c4e3f6c5289b8915fee3"
PAPER_AMENDMENT_ID = "c78868d0397988d4ca4f438ba93ef0b02c6d07d031251dda9d8058eef4403bcc"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _content_id(payload: dict[str, Any], identity_field: str) -> str:
    body = {key: value for key, value in payload.items() if key != identity_field}
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_decision_amendment_is_content_addressed_and_target_closed() -> None:
    amendment = _load(AMENDMENT)

    assert amendment["schema"] == (
        "bayesian-phystwin.deform360-covariance-only-crossrepo-decision-amendment"
    )
    assert amendment["schema_version"] == 1
    assert amendment["status"] == "target-closed"
    assert amendment["amendment_id"] == EXPECTED_AMENDMENT_ID
    assert _content_id(amendment, "amendment_id") == EXPECTED_AMENDMENT_ID
    assert amendment["information_boundary"] == {
        "development_suffix_opened_by_amendment": False,
        "confirmation_payload_opened_by_amendment": False,
        "confirmation_prediction_opened_by_amendment": False,
        "confirmation_outcome_opened_by_amendment": False,
        "target_execution_authorized_by_amendment_alone": False,
        "claim_authorized_by_amendment_alone": False,
        "deployment_authorized": False,
    }


def test_parent_code_and_binding_identities_are_exact() -> None:
    amendment = _load(AMENDMENT)
    protocol = _load(CODE_PROTOCOL)
    binding = _load(CROSSREPO_BINDING)

    assert amendment["parent_code_protocol"] == {
        "repository": "IPS-Stuttgart/BayesianPhysTwin",
        "merge_revision": "d337c5209c639430abe801a9688cc2788faa2aaf",
        "path": (
            "protocols/locks/deform360_covariance_only_independent_validation_v1.json"
        ),
        "protocol_id": protocol["protocol_id"],
    }
    assert amendment["parent_crossrepo_binding"] == {
        "path": (
            "protocols/locks/deform360_covariance_only_"
            "crossrepo_preregistration_binding_v1.json"
        ),
        "binding_id": binding["binding_id"],
    }


def test_merged_paper_decision_amendment_has_analysis_precedence() -> None:
    amendment = _load(AMENDMENT)
    paper = amendment["paper_analysis_authority"]
    precedence = amendment["precedence"]

    assert paper == {
        "repository": "FlorianPfaff/BayesianPhysTwin-Paper",
        "parent_protocol_merge_revision": ("7951467b1a24ac428a2ffc81dd0ce8bd0d622ae5"),
        "parent_protocol_path": (
            "preregistrations/deform360_covariance_only_confirmation_v1.json"
        ),
        "parent_protocol_id": PAPER_PARENT_ID,
        "decision_amendment_merge_revision": (
            "4e448ce7628b3826658fdffd8590cb680c500a88"
        ),
        "decision_amendment_path": (
            "preregistrations/deform360_covariance_only_"
            "confirmation_v1_decision_amendment.json"
        ),
        "decision_amendment_id": PAPER_AMENDMENT_ID,
    }
    assert precedence == {
        "prediction_custody_authority": "parent_code_protocol",
        "statistical_analysis_authority": (
            "paper-parent-protocol-plus-decision-amendment"
        ),
        "claim_wording_authority": ("paper-parent-protocol-plus-decision-amendment"),
        "conflict_rule": "fail-closed-no-target-opening",
        "target_execution_requires_all_bindings": True,
        "target_result_promotion_requires_all_bindings": True,
    }


def test_missing_outcomes_are_not_zero_effect_ties() -> None:
    semantics = _load(AMENDMENT)["corrected_software_semantics"]

    assert semantics["parent_missing_unit_imputation_superseded"] is True
    assert (
        semantics["missing_or_unscorable_primary_target_outcome_imputation_allowed"]
        is False
    )
    assert semantics["missing_or_unscorable_primary_target_outcome_consequence"] == (
        "confirmatory-analysis-incomplete-and-claim-ineligible"
    )
    assert semantics["candidate_dependent_row_or_unit_deletion_allowed"] is False
    assert semantics["unit_replacement_allowed"] is False
    assert semantics["observed_target_exact_fallback_zero_effect_allowed"] is True
    assert semantics["observed_target_exact_fallback_zero_effect_requires"] == [
        "primary-target-outcome-present-and-scorable",
        "candidate-and-b1-use-the-same-registered-fallback-distribution-byte-for-byte",
    ]
    assert (
        semantics["incomplete_analysis_is_not_negative_or_equivalence_evidence"] is True
    )


def test_exact_sign_flip_and_harmful_stratum_gates_cannot_be_dropped() -> None:
    semantics = _load(AMENDMENT)["corrected_software_semantics"]
    mechanism = semantics["mechanism_confirmation_requires"]
    physical = semantics["physical_fallback_improvement_additionally_requires"]

    assert (
        "holm-adjusted-exact-2^12-sign-flip-pvalue-candidate-minus-b1-below-0.05"
    ) in mechanism
    assert "sheet-mean-candidate-minus-b1-nll-nonpositive" in mechanism
    assert "volumetric-mean-candidate-minus-b1-nll-nonpositive" in mechanism
    assert "complete-ordered-twelve-unit-primary-score-table" in mechanism
    assert (
        "holm-adjusted-exact-2^12-sign-flip-pvalue-candidate-minus-b0-below-0.05"
    ) in physical
    assert semantics["target_retuning_allowed"] is False
    assert (
        semantics[
            "negative_or_inconclusive_complete_only_for_complete_ordered_target_cohort"
        ]
        is True
    )
