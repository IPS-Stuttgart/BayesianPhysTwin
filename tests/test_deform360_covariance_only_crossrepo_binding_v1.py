from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BINDING = (
    ROOT / "protocols/locks/"
    "deform360_covariance_only_crossrepo_preregistration_binding_v1.json"
)
CODE_PROTOCOL = (
    ROOT / "protocols/locks/deform360_covariance_only_independent_validation_v1.json"
)
EXPECTED_BINDING_ID = "531123205959a3d3d0549d9256b6ec222dca636198bc1e93f1b468d1a77c8f33"
PAPER_PROTOCOL_ID = "fa16c105e6d535d1e229ccf086fd69d05b2be74592b5c4e3f6c5289b8915fee3"
PAPER_MERGE_REVISION = "7951467b1a24ac428a2ffc81dd0ce8bd0d622ae5"


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


def test_crossrepo_binding_is_content_addressed_and_target_closed() -> None:
    binding = _load(BINDING)

    assert binding["schema"] == (
        "bayesian-phystwin.deform360-covariance-only-crossrepo-preregistration-binding"
    )
    assert binding["schema_version"] == 1
    assert binding["status"] == "target-closed"
    assert binding["binding_id"] == EXPECTED_BINDING_ID
    assert _content_id(binding, "binding_id") == EXPECTED_BINDING_ID

    assert binding["information_boundary"] == {
        "development_suffix_opened_by_binding": False,
        "confirmation_payload_opened_by_binding": False,
        "confirmation_prediction_opened_by_binding": False,
        "confirmation_outcome_opened_by_binding": False,
        "target_execution_authorized_by_binding_alone": False,
        "claim_authorized_by_binding_alone": False,
    }


def test_code_and_paper_protocol_roles_are_explicit_and_fail_closed() -> None:
    binding = _load(BINDING)
    code = binding["code_protocol"]
    paper = binding["paper_protocol"]
    precedence = binding["precedence"]

    assert code == {
        "repository": "IPS-Stuttgart/BayesianPhysTwin",
        "path": (
            "protocols/locks/deform360_covariance_only_independent_validation_v1.json"
        ),
        "protocol_id": (
            "0f13d7a1f1610588ca9e7119f94814c99940fb31050419de16fa9cae06f683cc"
        ),
        "role": "prediction-custody-and-runtime-lock",
    }
    assert paper == {
        "repository": "FlorianPfaff/BayesianPhysTwin-Paper",
        "merge_revision": PAPER_MERGE_REVISION,
        "path": ("preregistrations/deform360_covariance_only_confirmation_v1.json"),
        "protocol_id": PAPER_PROTOCOL_ID,
        "role": "analysis-and-claim-authority",
    }
    assert precedence == {
        "prediction_custody_authority": "code_protocol",
        "statistical_analysis_authority": "paper_protocol",
        "claim_wording_authority": "paper_protocol",
        "conflict_rule": "fail-closed-no-target-opening",
        "target_execution_requires_both": True,
        "target_result_promotion_requires_both": True,
    }


def test_binding_matches_the_local_code_protocol_and_frozen_values() -> None:
    binding = _load(BINDING)
    protocol = _load(CODE_PROTOCOL)
    frozen = binding["aligned_frozen_values"]

    assert binding["code_protocol"]["protocol_id"] == protocol["protocol_id"]
    candidate = protocol["frozen_candidate"]
    barrier = protocol["prediction_barrier"]
    cohort = protocol["cohort"]
    inference = protocol["inference"]
    decision = protocol["claim_decision"]

    assert frozen["mean_predictor"] == candidate["reference_mean"]
    assert frozen["covariance_donor"] == candidate["covariance_donor_id"]
    assert (
        frozen["covariance_scales"] == candidate["early_middle_late_covariance_scales"]
    )
    assert frozen["observation_std_m"] == candidate["observation_std_m"]
    assert (
        frozen["development_object_session_count"]
        == cohort["development_object_session_count"]
    )
    assert (
        frozen["confirmation_object_session_count"]
        == cohort["target_object_session_count"]
    )
    assert (
        frozen["source_prediction_record_count"]
        == barrier["source_prediction_seal_count_required"]
    )
    assert frozen["bootstrap_replicates"] == inference["bootstrap_replicates"]
    assert frozen["bootstrap_seed"] == inference["bootstrap_seed"]
    assert frozen["independent_statistical_unit"] == cohort["statistical_unit"]
    assert (
        frozen["negative_result_complete"]
        == decision["negative_or_inconclusive_result_is_complete"]
    )
    assert frozen["target_retuning_allowed"] is False
    assert candidate["target_scale_retuning_allowed"] is False
