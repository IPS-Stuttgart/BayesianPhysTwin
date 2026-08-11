from __future__ import annotations

import copy
import json
import runpy
import sys
from pathlib import Path

import pytest

from bayesian_phystwin import practical_equivalence as pe
from bayesian_phystwin.cli import practical_equivalence as equivalence_cli
from bayesian_phystwin.decisive_evidence import EvidenceBundle, EvidenceRecord


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
        "contract": pe.PRACTICAL_EQUIVALENCE_POLICY_CONTRACT,
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


def _parsed_record(*, method: str, group_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        unit_id=f"{method}-{group_id}",
        group_id=group_id,
        metric="track_error_m",
        method=method,
        loss=1.0,
        fallback_loss=1.0,
        risk_score=0.0,
        accepted=True,
        deployed_loss=1.0,
        horizon="late",
        reliability=None,
        identifiable_rank=None,
        intervals=(),
    )


def _bundle(*records: EvidenceRecord) -> EvidenceBundle:
    return EvidenceBundle(
        protocol_id="practical-equivalence-test-v1",
        statistical_unit="physical-object-session",
        claim_boundary="unit test only",
        reference_method="reference",
        records=records,
    )


def _bootstrap_output(
    *,
    lower: float = 0.05,
    upper: float = 0.05,
    observed: float = 0.05,
    probability: object = 0.5,
) -> dict[str, object]:
    comparison = {
        "status": "complete",
        "observed": {"mean_loss_difference": observed},
        "mean_loss_difference_interval": {
            "lower": lower,
            "upper": upper,
        },
        "bootstrap_probability_candidate_better": probability,
    }
    return {
        "contract": "test-bootstrap-v1",
        "metrics": {
            "track_error_m": {
                "methods": {
                    "candidate": {
                        "deployed_vs_reference_method": comparison,
                        "raw_vs_reference_method": comparison,
                    }
                }
            }
        },
    }


def _write_cli_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    evidence_path = tmp_path / "evidence.json"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "report.json"
    evidence_path.write_text(json.dumps(_payload()), encoding="utf-8")
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
    return evidence_path, policy_path, report_path


def test_small_regression_is_practically_equivalent() -> None:
    report = pe.assess_practical_equivalence(_payload(), _policy())
    assert report["contract"] == pe.PRACTICAL_EQUIVALENCE_REPORT_CONTRACT
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
    report = pe.assess_practical_equivalence(
        _payload((-0.3, -0.3, -0.3)),
        _policy(),
    )
    decision = report["metric_decisions"][0]
    assert decision["statistical_decision"] == "superior"
    assert decision["superiority_pass"] is True
    assert decision["practical_equivalence_pass"] is False
    assert report["summary"]["overall_decision"] == "superior"


def test_clear_regression_beyond_margin_fails() -> None:
    report = pe.assess_practical_equivalence(
        _payload((0.3, 0.3, 0.3)),
        _policy(),
    )
    decision = report["metric_decisions"][0]
    assert decision["statistical_decision"] == "inferior_beyond_margin"
    assert decision["inferior_beyond_margin"] is True
    assert report["summary"]["overall_decision"] == "failed_inferiority"


def test_wide_interval_is_inconclusive() -> None:
    report = pe.assess_practical_equivalence(
        _payload((-0.3, 0.0, 0.3)),
        _policy(),
    )
    decision = report["metric_decisions"][0]
    assert decision["statistical_decision"] == "inconclusive"
    assert decision["noninferiority_pass"] is False
    assert decision["inferior_beyond_margin"] is False
    assert report["summary"]["overall_decision"] == "inconclusive"


def test_noninferior_classifier_and_mixed_overall_decision() -> None:
    classified = pe._classify_interval(lower=-0.2, upper=0.05, margin=0.1)
    assert classified["statistical_decision"] == "noninferior"
    assert classified["noninferiority_pass"] is True
    assert classified["practical_equivalence_pass"] is False

    equivalent = {
        "decision_authorized": True,
        "inferior_beyond_margin": False,
        "noninferiority_pass": True,
        "practical_equivalence_pass": True,
        "superiority_pass": False,
    }
    superior = {
        "decision_authorized": True,
        "inferior_beyond_margin": False,
        "noninferiority_pass": True,
        "practical_equivalence_pass": False,
        "superiority_pass": True,
    }
    assert pe._overall_decision(()) == "diagnostic_only"
    assert pe._overall_decision((equivalent, superior)) == "noninferior_or_better"


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
    report = pe.assess_practical_equivalence(payload, policy)
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
    report = pe.assess_practical_equivalence(_payload(), policy)
    decision = report["metric_decisions"][0]
    assert decision["statistical_decision"] == "practically_equivalent"
    assert decision["decision"] == "diagnostic_only"
    assert decision["decision_authorized"] is False
    assert report["summary"]["overall_decision"] == "diagnostic_only"


def test_registered_minimum_group_count_is_enforced() -> None:
    report = pe.assess_practical_equivalence(
        _payload(),
        _policy(minimum_independent_groups=4),
    )
    decision = report["metric_decisions"][0]
    assert decision["bootstrap_status"] == "complete"
    assert decision["minimum_independent_group_requirement_met"] is False
    assert decision["decision"] == "diagnostic_only"
    assert report["summary"]["overall_decision"] == "diagnostic_only"


def test_single_group_remains_insufficient() -> None:
    report = pe.assess_practical_equivalence(
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
    forward = pe.assess_practical_equivalence(payload, _policy())
    reversed_payload = copy.deepcopy(payload)
    reversed_payload["records"].reverse()
    reverse = pe.assess_practical_equivalence(reversed_payload, _policy())
    assert reverse == forward


def test_policy_id_uses_normalized_values() -> None:
    integer_margin = _policy(margin=0)
    float_margin = _policy(margin=0.0)
    first = pe.assess_practical_equivalence(_payload((0.0, 0.0, 0.0)), integer_margin)
    second = pe.assess_practical_equivalence(_payload((0.0, 0.0, 0.0)), float_margin)
    assert first["policy_id"] == second["policy_id"]
    assert first["report_id"] == second["report_id"]


def test_policy_requires_object_exact_fields_and_array_targets() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        pe.parse_practical_equivalence_policy([])

    invalid = _policy()
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="fields changed"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["targets"] = "not-an-array"
    with pytest.raises(ValueError, match="JSON array"):
        pe.parse_practical_equivalence_policy(invalid)


def test_policy_text_boolean_and_target_validation_fail_closed() -> None:
    invalid = _policy()
    invalid["protocol_id"] = " padded "
    with pytest.raises(ValueError, match="literal string"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["claim_boundary"] = "two\nlines"
    with pytest.raises(ValueError, match="single line"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["information_boundary"]["groups_independent"] = 1
    with pytest.raises(ValueError, match="must be a bool"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["targets"] = [1]
    with pytest.raises(ValueError, match="JSON object"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["targets"][0]["stream"] = "unknown"
    with pytest.raises(ValueError, match="stream must be one of"):
        pe.parse_practical_equivalence_policy(invalid)


def test_policy_number_and_version_validation_fail_closed() -> None:
    invalid = _policy()
    invalid["targets"][0]["margin"] = True
    with pytest.raises(ValueError, match="finite number"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["targets"][0]["margin"] = float("nan")
    with pytest.raises(ValueError, match="finite number"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["contract"] = "another-contract"
    with pytest.raises(ValueError, match="contract must be"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["schema_version"] = True
    with pytest.raises(ValueError, match="integer 1"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["targets"] = []
    with pytest.raises(ValueError, match="must not be empty"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["bootstrap"]["confidence"] = 0.5
    with pytest.raises(ValueError, match="greater than"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["bootstrap"]["confidence"] = 1.0
    with pytest.raises(ValueError, match="strictly below"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["bootstrap"]["confidence"] = 1.1
    with pytest.raises(ValueError, match="at most"):
        pe.parse_practical_equivalence_policy(invalid)


def test_policy_and_evidence_contracts_fail_closed() -> None:
    invalid = _policy()
    invalid["candidate_method"] = "reference"
    with pytest.raises(ValueError, match="must differ"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["bootstrap"]["replicates"] = 999
    with pytest.raises(ValueError, match="1000"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["targets"][0]["margin"] = -0.1
    with pytest.raises(ValueError, match="at least 0.0"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["targets"] = [invalid["targets"][0], invalid["targets"][0]]
    with pytest.raises(ValueError, match="unique and sorted"):
        pe.parse_practical_equivalence_policy(invalid)

    invalid = _policy()
    invalid["protocol_id"] = "another-protocol"
    with pytest.raises(ValueError, match="protocol_id"):
        pe.assess_practical_equivalence(_payload(), invalid)

    invalid = _policy()
    invalid["statistical_unit"] = "another-unit"
    with pytest.raises(ValueError, match="statistical_unit"):
        pe.assess_practical_equivalence(_payload(), invalid)

    invalid = _policy(metric="missing_metric")
    with pytest.raises(ValueError, match="absent evidence metrics"):
        pe.assess_practical_equivalence(_payload(), invalid)


def test_group_comparison_requires_both_methods_and_matching_groups() -> None:
    reference = _parsed_record(method="reference", group_id="group-a")
    candidate = _parsed_record(method="candidate", group_id="group-a")

    with pytest.raises(ValueError, match="candidate method"):
        pe._group_difference_summary(
            _bundle(reference),
            metric="track_error_m",
            stream="deployed",
            candidate_method="candidate",
            reference_method="reference",
        )

    with pytest.raises(ValueError, match="reference method"):
        pe._group_difference_summary(
            _bundle(candidate),
            metric="track_error_m",
            stream="deployed",
            candidate_method="candidate",
            reference_method="reference",
        )

    mismatched_reference = _parsed_record(
        method="reference",
        group_id="group-b",
    )
    with pytest.raises(ValueError, match="group sets differ"):
        pe._group_difference_summary(
            _bundle(candidate, mismatched_reference),
            metric="track_error_m",
            stream="deployed",
            candidate_method="candidate",
            reference_method="reference",
        )


def test_internal_shape_and_range_guards_fail_closed() -> None:
    with pytest.raises(AssertionError, match="changed type"):
        pe._required_mapping([], name="malformed bootstrap")
    with pytest.raises(ValueError, match="at most"):
        pe._number(2.0, name="probability", maximum=1.0)


def test_bootstrap_probability_none_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pe,
        "group_clustered_paired_bootstrap",
        lambda *_args, **_kwargs: _bootstrap_output(probability=None),
    )
    report = pe.assess_practical_equivalence(_payload(), _policy())
    decision = report["metric_decisions"][0]
    assert decision["bootstrap_probability_candidate_better"] is None


def test_bootstrap_interval_and_observed_difference_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pe,
        "group_clustered_paired_bootstrap",
        lambda *_args, **_kwargs: _bootstrap_output(lower=0.2, upper=0.1),
    )
    with pytest.raises(AssertionError, match="bounds changed order"):
        pe.assess_practical_equivalence(_payload(), _policy())

    monkeypatch.setattr(
        pe,
        "group_clustered_paired_bootstrap",
        lambda *_args, **_kwargs: _bootstrap_output(observed=0.0),
    )
    with pytest.raises(AssertionError, match="mean differences diverged"):
        pe.assess_practical_equivalence(_payload(), _policy())


def test_shipped_policy_template_is_valid() -> None:
    template = json.loads(
        Path("protocols/templates/practical_equivalence_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )
    policy = pe.parse_practical_equivalence_policy(template)
    assert policy.candidate_method == "bayesian_full_guarded"
    assert policy.reference_method == "last_residual"
    assert policy.minimum_independent_groups == 10


def test_cli_publishes_both_input_identities_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    evidence_path, policy_path, report_path = _write_cli_inputs(tmp_path)
    arguments = [str(evidence_path), str(policy_path), str(report_path)]

    assert equivalence_cli.main(arguments) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["input_artifact"]["evidence"]["bytes"] > 0
    assert report["input_artifact"]["policy"]["bytes"] > 0
    assert len(report["status_sha256"]) == 64
    assert report["summary"]["overall_decision"] == "practically_equivalent"

    with pytest.raises(FileExistsError):
        equivalence_cli.main(arguments)
    assert equivalence_cli.main([*arguments, "--overwrite"]) == 0


def test_cli_rejects_changed_published_report_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.json"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "report.json"
    evidence_path.write_text("{}", encoding="utf-8")
    policy_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        equivalence_cli,
        "assess_practical_equivalence",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        equivalence_cli,
        "publish_json_report",
        lambda *_args, **_kwargs: {"summary": []},
    )

    with pytest.raises(AssertionError, match="summary changed type"):
        equivalence_cli.main([str(evidence_path), str(policy_path), str(report_path)])


def test_cli_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence_path, policy_path, report_path = _write_cli_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bayesian_phystwin.cli.practical_equivalence",
            str(evidence_path),
            str(policy_path),
            str(report_path),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(
            "bayesian_phystwin.cli.practical_equivalence",
            run_name="__main__",
        )
    assert exc_info.value.code == 0
