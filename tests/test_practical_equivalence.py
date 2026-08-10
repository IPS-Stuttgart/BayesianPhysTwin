from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bayesian_phystwin.cli.practical_equivalence import main as equivalence_main
from bayesian_phystwin.practical_equivalence import (
    PRACTICAL_EQUIVALENCE_POLICY_CONTRACT,
    PRACTICAL_EQUIVALENCE_REPORT_CONTRACT,
    assess_practical_equivalence,
    parse_practical_equivalence_policy,
)


def _record(
    *,
    unit_id: str,
    group_id: str,
    method: str,
    loss: float,
    metric: str = "track_error_m",
    fallback_loss: float = 2.0,
    accepted: bool = True,
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "group_id": group_id,
        "metric": metric,
        "method": method,
        "loss": loss,
        "fallback_loss": fallback_loss,
        "risk_score": 0.1,
        "accepted": accepted,
        "deployed_loss": loss if accepted else fallback_loss,
        "horizon": "late",
        "reliability": 0.8,
        "identifiable_rank": 2,
        "intervals": [],
    }


def _payload(
    differences: tuple[float, ...] = (0.05, 0.05, 0.05),
    *,
    metric: str = "track_error_m",
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for index, difference in enumerate(differences):
        group_id = f"object-{index}"
        records.extend(
            [
                _record(
                    unit_id=f"unit-{index}",
                    group_id=group_id,
                    method="candidate",
                    loss=1.0 + difference,
                    metric=metric,
                ),
                _record(
                    unit_id=f"unit-{index}",
                    group_id=group_id,
                    method="reference",
                    loss=1.0,
                    metric=metric,
                ),
            ]
        )
    return {
        "schema_version": 1,
        "contract": "bayesian-phystwin-decisive-evidence-v1",
        "protocol_id": "practical-equivalence-test-v1",
        "statistical_unit": "physical-object-session",
        "claim_boundary": "unit test only",
        "reference_method": "reference",
        "records": records,
    }


def _policy(
    *,
    margin: float = 0.1,
    metric: str = "track_error_m",
    stream: str = "deployed",
    minimum_independent_groups: int = 3,
) -> dict[str, object]:
    return {
        "contract": PRACTICAL_EQUIVALENCE_POLICY_CONTRACT,
        "schema_version": 1,
        "protocol_id": "practical-equivalence-test-v1",
        "statistical_unit": "physical-object-session",
        "candidate_method": "candidate",
        "reference_method": "reference",
        "bootstrap": {
            "replicates": 1000,
            "seed": 17,
            "confidence": 0.9,
        },
        "minimum_independent_groups": minimum_independent_groups,
        "information_boundary": {
            "margins_frozen_before_outcomes": True,
            "outcomes_used_for_margin_selection": False,
            "groups_independent": True,
        },
        "targets": [
            {
                "metric": metric,
                "stream": stream,
                "margin": margin,
                "unit": "m",
                "margin_basis": "predeclared measurement resolution",
            }
        ],
        "claim_boundary": "unit test only",
    }


def test_small_regression_is_practically_equivalent() -> None:
    report = assess_practical_equivalence(_payload(), _policy())
    assert report["contract"] == PRACTICAL_EQUIVALENCE_REPORT_CONTRACT
    decision = report["metric_decisions"][0]
    assert decision["equal_group_mean_difference"] == pytest.approx(0.05)
    assert decision["candidate_better_group_count"] == 0
    assert decision["candidate_worse_group_count"] == 3
    assert decision["statistical_decision"] == "practically_equivalent"
    assert decision["decision"] == "practically_equivalent"
    assert decision["noninferiority_pass"] is True
    assert decision["practical_equivalence_pass"] is True
    assert report["summary"]["overall_decision"] == "practically_equivalent"
    assert report["claim_authorized"] is False
    assert report["promotion_authorized"] is False
    assert len(report["policy_id"]) == 64
    assert len(report["source_evidence_id"]) == 64
    assert len(report["report_id"]) == 64


def test_clear_improvement_is_superior_not_equivalent() -> None:
    report = assess_practical_equivalence(
        _payload((-0.3, -0.3, -0.3)),
        _policy(),
    )
    decision = report["metric_decisions"][0]
    assert decision["statistical_decision"] == "superior"
    assert decision["superiority_pass"] is True
    assert decision["practical_equivalence_pass"] is False
    assert report["summary"]["overall_decision"] == "superior"


def test_clear_regression_beyond_margin_fails() -> None:
    report = assess_practical_equivalence(
        _payload((0.3, 0.3, 0.3)),
        _policy(),
    )
    decision = report["metric_decisions"][0]
    assert decision["statistical_decision"] == "inferior_beyond_margin"
    assert decision["inferior_beyond_margin"] is True
    assert report["summary"]["overall_decision"] == "failed_inferiority"


def test_wide_interval_is_inconclusive() -> None:
    report = assess_practical_equivalence(
        _payload((-0.3, 0.0, 0.3)),
        _policy(),
    )
    decision = report["metric_decisions"][0]
    assert decision["statistical_decision"] == "inconclusive"
    assert decision["noninferiority_pass"] is False
    assert decision["inferior_beyond_margin"] is False
    assert report["summary"]["overall_decision"] == "inconclusive"


def test_raw_and_deployed_streams_remain_separate() -> None:
    payload = _payload((0.3, 0.3, 0.3))
    for record in payload["records"]:
        if record["method"] == "candidate":
            record["accepted"] = False
            record["fallback_loss"] = 1.0
            record["deployed_loss"] = 1.0
        else:
            record["fallback_loss"] = 1.0
    policy = _policy()
    policy["targets"] = [
        {
            "metric": "track_error_m",
            "stream": "deployed",
            "margin": 0.1,
            "unit": "m",
            "margin_basis": "predeclared measurement resolution",
        },
        {
            "metric": "track_error_m",
            "stream": "raw",
            "margin": 0.1,
            "unit": "m",
            "margin_basis": "predeclared measurement resolution",
        },
    ]
    report = assess_practical_equivalence(payload, policy)
    deployed, raw = report["metric_decisions"]
    assert deployed["statistical_decision"] == "practically_equivalent"
    assert raw["statistical_decision"] == "inferior_beyond_margin"
    assert report["summary"]["overall_decision"] == "failed_inferiority"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("margins_frozen_before_outcomes", False),
        ("outcomes_used_for_margin_selection", True),
        ("groups_independent", False),
    ],
)
def test_nonprospective_policy_is_diagnostic_only(field: str, value: bool) -> None:
    policy = _policy()
    policy["information_boundary"][field] = value
    report = assess_practical_equivalence(_payload(), policy)
    decision = report["metric_decisions"][0]
    assert decision["statistical_decision"] == "practically_equivalent"
    assert decision["decision"] == "diagnostic_only"
    assert decision["decision_authorized"] is False
    assert report["summary"]["overall_decision"] == "diagnostic_only"


def test_registered_minimum_group_count_is_enforced() -> None:
    report = assess_practical_equivalence(
        _payload(),
        _policy(minimum_independent_groups=4),
    )
    decision = report["metric_decisions"][0]
    assert decision["bootstrap_status"] == "complete"
    assert decision["minimum_independent_group_requirement_met"] is False
    assert decision["decision"] == "diagnostic_only"
    assert report["summary"]["overall_decision"] == "diagnostic_only"


def test_single_group_remains_insufficient() -> None:
    report = assess_practical_equivalence(
        _payload((0.0,)),
        _policy(minimum_independent_groups=2),
    )
    decision = report["metric_decisions"][0]
    assert decision["bootstrap_status"] == "insufficient_independent_groups"
    assert decision["confidence_interval"] is None
    assert decision["statistical_decision"] == "insufficient_independent_groups"
    assert decision["decision"] == "diagnostic_only"


def test_report_is_invariant_to_evidence_row_order() -> None:
    payload = _payload((-0.1, 0.0, 0.1))
    forward = assess_practical_equivalence(payload, _policy())
    reversed_payload = copy.deepcopy(payload)
    reversed_payload["records"].reverse()
    reverse = assess_practical_equivalence(reversed_payload, _policy())
    assert reverse == forward


def test_policy_id_uses_normalized_values() -> None:
    integer_margin = _policy(margin=0)
    float_margin = _policy(margin=0.0)
    first = assess_practical_equivalence(_payload((0.0, 0.0, 0.0)), integer_margin)
    second = assess_practical_equivalence(_payload((0.0, 0.0, 0.0)), float_margin)
    assert first["policy_id"] == second["policy_id"]
    assert first["report_id"] == second["report_id"]


def test_policy_and_evidence_contracts_fail_closed() -> None:
    invalid = _policy()
    invalid["candidate_method"] = "reference"
    with pytest.raises(ValueError, match="must differ"):
        parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["bootstrap"]["replicates"] = 999
    with pytest.raises(ValueError, match="1000"):
        parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["targets"][0]["margin"] = -0.1
    with pytest.raises(ValueError, match="at least 0.0"):
        parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["targets"] = [invalid["targets"][0], invalid["targets"][0]]
    with pytest.raises(ValueError, match="unique and sorted"):
        parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["protocol_id"] = "another-protocol"
    with pytest.raises(ValueError, match="protocol_id"):
        assess_practical_equivalence(_payload(), invalid)

    invalid = _policy(metric="missing_metric")
    with pytest.raises(ValueError, match="absent evidence metrics"):
        assess_practical_equivalence(_payload(), invalid)


def test_shipped_policy_template_is_valid() -> None:
    template = json.loads(
        Path("protocols/templates/practical_equivalence_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )
    policy = parse_practical_equivalence_policy(template)
    assert policy.candidate_method == "bayesian_full_guarded"
    assert policy.reference_method == "last_residual"
    assert policy.minimum_independent_groups == 10


def test_cli_publishes_both_input_identities_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.json"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "report.json"
    evidence_path.write_text(json.dumps(_payload()), encoding="utf-8")
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
    arguments = [str(evidence_path), str(policy_path), str(report_path)]

    assert equivalence_main(arguments) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["input_artifact"]["evidence"]["bytes"] > 0
    assert report["input_artifact"]["policy"]["bytes"] > 0
    assert len(report["status_sha256"]) == 64
    assert report["summary"]["overall_decision"] == "practically_equivalent"

    with pytest.raises(FileExistsError):
        equivalence_main(arguments)
    assert equivalence_main([*arguments, "--overwrite"]) == 0
