"""Prospective practical-equivalence decisions for matched physical losses.

The analysis consumes an already validated ``DecisiveEvidenceV1`` bundle and a
separately frozen margin policy. Independent ``group_id`` values are the paired
resampling units. The resulting report is diagnostic infrastructure: it never
authorizes a scientific claim, provider promotion, or deployment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from ._canonical_contracts import plain_json
from ._portable_contracts import content_id
from .decisive_evidence import (
    DECISIVE_EVIDENCE_INPUT_CONTRACT,
    EvidenceBundle,
    EvidenceRecord,
    parse_decisive_evidence,
)
from .decisive_evidence_bootstrap import group_clustered_paired_bootstrap

PRACTICAL_EQUIVALENCE_POLICY_CONTRACT: Final = (
    "bayesian-phystwin-practical-equivalence-policy-v1"
)
PRACTICAL_EQUIVALENCE_REPORT_CONTRACT: Final = (
    "bayesian-phystwin-practical-equivalence-report-v1"
)
PRACTICAL_EQUIVALENCE_IMPLEMENTATION: Final = (
    "equal-group-bootstrap-practical-equivalence-v1"
)
PRACTICAL_EQUIVALENCE_CLAIM_BOUNDARY: Final = (
    "Diagnostic paired-loss analysis only. A favorable result does not establish "
    "fresh-object transfer, calibrated uncertainty, physical-state identification, "
    "provider competence, deployment safety, or state of the art."
)
RAW_STREAM: Final = "raw"
DEPLOYED_STREAM: Final = "deployed"
_ALLOWED_STREAMS: Final = frozenset({RAW_STREAM, DEPLOYED_STREAM})
_MINIMUM_BOOTSTRAP_REPLICATES: Final = 1000


@dataclass(frozen=True, slots=True)
class PracticalEquivalenceTargetV1:
    metric: str
    stream: str
    margin: float
    unit: str
    margin_basis: str


@dataclass(frozen=True, slots=True)
class PracticalEquivalencePolicyV1:
    protocol_id: str
    statistical_unit: str
    candidate_method: str
    reference_method: str
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_confidence: float
    minimum_independent_groups: int
    margins_frozen_before_outcomes: bool
    outcomes_used_for_margin_selection: bool
    groups_independent: bool
    targets: tuple[PracticalEquivalenceTargetV1, ...]
    claim_boundary: str


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _fields(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    name: str,
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty literal string")
    if any(character in value for character in "\r\n"):
        raise ValueError(f"{name} must be a single line")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _integer(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    minimum_exclusive: bool = False,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result <= minimum if minimum_exclusive else result < minimum
        if invalid:
            relation = "greater than" if minimum_exclusive else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _parse_target(value: object, *, index: int) -> PracticalEquivalenceTargetV1:
    name = f"targets[{index}]"
    target = _mapping(value, name=name)
    _fields(
        target,
        required=frozenset(
            {"metric", "stream", "margin", "unit", "margin_basis"}
        ),
        name=name,
    )
    stream = _text(target["stream"], name=f"{name}.stream")
    if stream not in _ALLOWED_STREAMS:
        raise ValueError(f"{name}.stream must be one of {sorted(_ALLOWED_STREAMS)}")
    return PracticalEquivalenceTargetV1(
        metric=_text(target["metric"], name=f"{name}.metric"),
        stream=stream,
        margin=_number(target["margin"], name=f"{name}.margin", minimum=0.0),
        unit=_text(target["unit"], name=f"{name}.unit"),
        margin_basis=_text(
            target["margin_basis"],
            name=f"{name}.margin_basis",
        ),
    )


def parse_practical_equivalence_policy(
    payload: object,
) -> PracticalEquivalencePolicyV1:
    """Validate one prospective practical-equivalence margin policy."""

    root = _mapping(payload, name="policy")
    _fields(
        root,
        required=frozenset(
            {
                "contract",
                "schema_version",
                "protocol_id",
                "statistical_unit",
                "candidate_method",
                "reference_method",
                "bootstrap",
                "minimum_independent_groups",
                "information_boundary",
                "targets",
                "claim_boundary",
            }
        ),
        name="policy",
    )
    if root["contract"] != PRACTICAL_EQUIVALENCE_POLICY_CONTRACT:
        raise ValueError(
            "contract must be " f"{PRACTICAL_EQUIVALENCE_POLICY_CONTRACT!r}"
        )
    if isinstance(root["schema_version"], bool) or root["schema_version"] != 1:
        raise ValueError("schema_version must be the integer 1")

    bootstrap = _mapping(root["bootstrap"], name="policy.bootstrap")
    _fields(
        bootstrap,
        required=frozenset({"replicates", "seed", "confidence"}),
        name="policy.bootstrap",
    )
    information = _mapping(
        root["information_boundary"],
        name="policy.information_boundary",
    )
    _fields(
        information,
        required=frozenset(
            {
                "margins_frozen_before_outcomes",
                "outcomes_used_for_margin_selection",
                "groups_independent",
            }
        ),
        name="policy.information_boundary",
    )

    targets = tuple(
        _parse_target(raw_target, index=index)
        for index, raw_target in enumerate(
            _sequence(root["targets"], name="policy.targets")
        )
    )
    if not targets:
        raise ValueError("policy.targets must not be empty")
    target_keys = tuple((target.metric, target.stream) for target in targets)
    if target_keys != tuple(sorted(target_keys)) or len(set(target_keys)) != len(
        target_keys
    ):
        raise ValueError("policy.targets must be unique and sorted by metric/stream")

    candidate_method = _text(
        root["candidate_method"],
        name="policy.candidate_method",
    )
    reference_method = _text(
        root["reference_method"],
        name="policy.reference_method",
    )
    if candidate_method == reference_method:
        raise ValueError("candidate_method must differ from reference_method")

    confidence = _number(
        bootstrap["confidence"],
        name="policy.bootstrap.confidence",
        minimum=0.5,
        minimum_exclusive=True,
        maximum=1.0,
    )
    if confidence >= 1.0:
        raise ValueError("policy.bootstrap.confidence must lie strictly below 1")

    return PracticalEquivalencePolicyV1(
        protocol_id=_text(root["protocol_id"], name="policy.protocol_id"),
        statistical_unit=_text(
            root["statistical_unit"],
            name="policy.statistical_unit",
        ),
        candidate_method=candidate_method,
        reference_method=reference_method,
        bootstrap_replicates=_integer(
            bootstrap["replicates"],
            name="policy.bootstrap.replicates",
            minimum=_MINIMUM_BOOTSTRAP_REPLICATES,
        ),
        bootstrap_seed=_integer(
            bootstrap["seed"],
            name="policy.bootstrap.seed",
            minimum=0,
        ),
        bootstrap_confidence=confidence,
        minimum_independent_groups=_integer(
            root["minimum_independent_groups"],
            name="policy.minimum_independent_groups",
            minimum=2,
        ),
        margins_frozen_before_outcomes=_boolean(
            information["margins_frozen_before_outcomes"],
            name=(
                "policy.information_boundary."
                "margins_frozen_before_outcomes"
            ),
        ),
        outcomes_used_for_margin_selection=_boolean(
            information["outcomes_used_for_margin_selection"],
            name=(
                "policy.information_boundary."
                "outcomes_used_for_margin_selection"
            ),
        ),
        groups_independent=_boolean(
            information["groups_independent"],
            name="policy.information_boundary.groups_independent",
        ),
        targets=targets,
        claim_boundary=_text(
            root["claim_boundary"],
            name="policy.claim_boundary",
        ),
    )


def _policy_json(policy: PracticalEquivalencePolicyV1) -> dict[str, object]:
    return {
        "contract": PRACTICAL_EQUIVALENCE_POLICY_CONTRACT,
        "schema_version": 1,
        "protocol_id": policy.protocol_id,
        "statistical_unit": policy.statistical_unit,
        "candidate_method": policy.candidate_method,
        "reference_method": policy.reference_method,
        "bootstrap": {
            "replicates": policy.bootstrap_replicates,
            "seed": policy.bootstrap_seed,
            "confidence": policy.bootstrap_confidence,
        },
        "minimum_independent_groups": policy.minimum_independent_groups,
        "information_boundary": {
            "margins_frozen_before_outcomes": (
                policy.margins_frozen_before_outcomes
            ),
            "outcomes_used_for_margin_selection": (
                policy.outcomes_used_for_margin_selection
            ),
            "groups_independent": policy.groups_independent,
        },
        "targets": [
            {
                "metric": target.metric,
                "stream": target.stream,
                "margin": target.margin,
                "unit": target.unit,
                "margin_basis": target.margin_basis,
            }
            for target in policy.targets
        ],
        "claim_boundary": policy.claim_boundary,
    }


def _evidence_json(bundle: EvidenceBundle) -> dict[str, object]:
    records = []
    for record in sorted(
        bundle.records,
        key=lambda item: (item.metric, item.unit_id, item.method),
    ):
        records.append(
            {
                "unit_id": record.unit_id,
                "group_id": record.group_id,
                "metric": record.metric,
                "method": record.method,
                "loss": record.loss,
                "fallback_loss": record.fallback_loss,
                "risk_score": record.risk_score,
                "accepted": record.accepted,
                "deployed_loss": record.deployed_loss,
                "horizon": record.horizon,
                "reliability": record.reliability,
                "identifiable_rank": record.identifiable_rank,
                "intervals": [
                    {
                        "nominal_coverage": interval.nominal_coverage,
                        "covered": interval.covered,
                        "width": interval.width,
                    }
                    for interval in record.intervals
                ],
            }
        )
    return {
        "contract": DECISIVE_EVIDENCE_INPUT_CONTRACT,
        "schema_version": 1,
        "protocol_id": bundle.protocol_id,
        "statistical_unit": bundle.statistical_unit,
        "claim_boundary": bundle.claim_boundary,
        "reference_method": bundle.reference_method,
        "records": records,
    }


def _group_means(
    records: Sequence[EvidenceRecord],
    *,
    method: str,
    stream: str,
) -> dict[str, float]:
    by_group: dict[str, list[float]] = {}
    for record in records:
        if record.method != method:
            continue
        value = record.loss if stream == RAW_STREAM else record.deployed_loss
        by_group.setdefault(record.group_id, []).append(value)
    return {
        group_id: float(np.mean(values))
        for group_id, values in sorted(by_group.items())
    }


def _group_difference_summary(
    bundle: EvidenceBundle,
    *,
    metric: str,
    stream: str,
    candidate_method: str,
    reference_method: str,
) -> dict[str, object]:
    metric_records = tuple(
        record for record in bundle.records if record.metric == metric
    )
    candidate = _group_means(
        metric_records,
        method=candidate_method,
        stream=stream,
    )
    reference = _group_means(
        metric_records,
        method=reference_method,
        stream=stream,
    )
    if not candidate:
        raise ValueError(
            f"candidate method {candidate_method!r} is absent for metric {metric!r}"
        )
    if not reference:
        raise ValueError(
            f"reference method {reference_method!r} is absent for metric {metric!r}"
        )
    if tuple(candidate) != tuple(reference):
        raise ValueError(
            f"candidate/reference group sets differ for metric {metric!r}"
        )
    differences = {
        group_id: candidate[group_id] - reference[group_id] for group_id in candidate
    }
    values = np.asarray(tuple(differences.values()), dtype=np.float64)
    worst_group_id = max(differences, key=differences.__getitem__)
    best_group_id = min(differences, key=differences.__getitem__)
    return {
        "group_count": len(differences),
        "equal_group_mean_difference": float(np.mean(values)),
        "candidate_better_group_count": int(np.sum(values < 0.0)),
        "exact_tie_group_count": int(np.sum(values == 0.0)),
        "candidate_worse_group_count": int(np.sum(values > 0.0)),
        "worst_group_id": worst_group_id,
        "worst_group_difference": differences[worst_group_id],
        "best_group_id": best_group_id,
        "best_group_difference": differences[best_group_id],
    }


def _required_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{name} changed type")
    return value


def _classify_interval(
    *,
    lower: float,
    upper: float,
    margin: float,
) -> dict[str, object]:
    superiority = upper < 0.0
    noninferiority = upper <= margin
    equivalence = lower >= -margin and upper <= margin
    inferior_beyond_margin = lower > margin
    if equivalence:
        decision = "practically_equivalent"
    elif superiority:
        decision = "superior"
    elif noninferiority:
        decision = "noninferior"
    elif inferior_beyond_margin:
        decision = "inferior_beyond_margin"
    else:
        decision = "inconclusive"
    return {
        "statistical_decision": decision,
        "superiority_pass": superiority,
        "noninferiority_pass": noninferiority,
        "practical_equivalence_pass": equivalence,
        "inferior_beyond_margin": inferior_beyond_margin,
    }


def _overall_decision(results: Sequence[Mapping[str, object]]) -> str:
    if not results or not all(
        result["decision_authorized"] is True for result in results
    ):
        return "diagnostic_only"
    if any(result["inferior_beyond_margin"] is True for result in results):
        return "failed_inferiority"
    if any(result["noninferiority_pass"] is not True for result in results):
        return "inconclusive"
    if all(result["practical_equivalence_pass"] is True for result in results):
        return "practically_equivalent"
    if all(result["superiority_pass"] is True for result in results):
        return "superior"
    return "noninferior_or_better"


def assess_practical_equivalence(
    evidence_payload: Mapping[str, object],
    policy_payload: object,
) -> dict[str, object]:
    """Evaluate a frozen margin policy with paired equal-group bootstrap intervals."""

    bundle = parse_decisive_evidence(evidence_payload)
    policy = parse_practical_equivalence_policy(policy_payload)
    if policy.protocol_id != bundle.protocol_id:
        raise ValueError("policy protocol_id does not match the evidence")
    if policy.statistical_unit != bundle.statistical_unit:
        raise ValueError("policy statistical_unit does not match the evidence")

    available_metrics = {record.metric for record in bundle.records}
    missing_metrics = sorted(
        {target.metric for target in policy.targets} - available_metrics
    )
    if missing_metrics:
        raise ValueError(f"policy targets absent evidence metrics {missing_metrics}")

    bootstrap = group_clustered_paired_bootstrap(
        evidence_payload,
        replicates=policy.bootstrap_replicates,
        seed=policy.bootstrap_seed,
        confidence=policy.bootstrap_confidence,
        reference_method=policy.reference_method,
    )
    bootstrap_metrics = _required_mapping(
        bootstrap.get("metrics"),
        name="bootstrap metrics",
    )
    prospective_policy = (
        policy.margins_frozen_before_outcomes
        and not policy.outcomes_used_for_margin_selection
        and policy.groups_independent
    )

    decisions: list[dict[str, object]] = []
    for target in policy.targets:
        group_summary = _group_difference_summary(
            bundle,
            metric=target.metric,
            stream=target.stream,
            candidate_method=policy.candidate_method,
            reference_method=policy.reference_method,
        )
        metric_summary = _required_mapping(
            bootstrap_metrics.get(target.metric),
            name=f"bootstrap metric {target.metric!r}",
        )
        methods = _required_mapping(
            metric_summary.get("methods"),
            name=f"bootstrap methods for {target.metric!r}",
        )
        candidate = _required_mapping(
            methods.get(policy.candidate_method),
            name=(
                f"bootstrap candidate {policy.candidate_method!r} for "
                f"{target.metric!r}"
            ),
        )
        comparison = _required_mapping(
            candidate.get(f"{target.stream}_vs_reference_method"),
            name=f"bootstrap comparison for {target.metric!r}/{target.stream}",
        )
        status = comparison.get("status")
        interval_value = comparison.get("mean_loss_difference_interval")
        interval = (
            None
            if interval_value is None
            else _required_mapping(
                interval_value,
                name=f"bootstrap interval for {target.metric!r}/{target.stream}",
            )
        )
        complete = status == "complete" and interval is not None
        group_count = int(group_summary["group_count"])
        group_requirement_met = group_count >= policy.minimum_independent_groups
        decision_authorized = (
            prospective_policy and group_requirement_met and complete
        )
        result: dict[str, object] = {
            "metric": target.metric,
            "stream": target.stream,
            "unit": target.unit,
            "margin": target.margin,
            "margin_basis": target.margin_basis,
            "difference_semantics": "candidate_minus_reference; lower_is_better",
            "bootstrap_status": status,
            "minimum_independent_groups": policy.minimum_independent_groups,
            "minimum_independent_group_requirement_met": group_requirement_met,
            "decision_authorized": decision_authorized,
            **group_summary,
        }
        if not complete or interval is None:
            result.update(
                {
                    "confidence_interval": None,
                    "bootstrap_probability_candidate_better": None,
                    "statistical_decision": "insufficient_independent_groups",
                    "superiority_pass": False,
                    "noninferiority_pass": False,
                    "practical_equivalence_pass": False,
                    "inferior_beyond_margin": False,
                    "decision": "diagnostic_only",
                }
            )
        else:
            lower = _number(interval.get("lower"), name="interval lower")
            upper = _number(interval.get("upper"), name="interval upper")
            if lower > upper:
                raise AssertionError("bootstrap interval bounds changed order")
            observed = _required_mapping(
                comparison.get("observed"),
                name="bootstrap observed comparison",
            )
            observed_difference = _number(
                observed.get("mean_loss_difference"),
                name="bootstrap observed mean difference",
            )
            group_difference = _number(
                group_summary["equal_group_mean_difference"],
                name="equal-group observed mean difference",
            )
            if not np.isclose(
                observed_difference,
                group_difference,
                rtol=1e-12,
                atol=1e-15,
            ):
                raise AssertionError(
                    "bootstrap and direct equal-group mean differences diverged"
                )
            classification = _classify_interval(
                lower=lower,
                upper=upper,
                margin=target.margin,
            )
            probability = comparison.get(
                "bootstrap_probability_candidate_better"
            )
            result.update(
                {
                    "confidence_interval": {
                        "confidence": policy.bootstrap_confidence,
                        "lower": lower,
                        "upper": upper,
                    },
                    "bootstrap_probability_candidate_better": (
                        None
                        if probability is None
                        else _number(
                            probability,
                            name="bootstrap probability candidate better",
                            minimum=0.0,
                            maximum=1.0,
                        )
                    ),
                    **classification,
                    "decision": (
                        classification["statistical_decision"]
                        if decision_authorized
                        else "diagnostic_only"
                    ),
                }
            )
        decisions.append(result)

    normalized_policy = _policy_json(policy)
    summary = {
        "target_count": len(decisions),
        "authorized_target_count": sum(
            decision["decision_authorized"] is True for decision in decisions
        ),
        "all_noninferior": all(
            decision["noninferiority_pass"] is True for decision in decisions
        ),
        "all_practically_equivalent": all(
            decision["practical_equivalence_pass"] is True
            for decision in decisions
        ),
        "any_inferior_beyond_margin": any(
            decision["inferior_beyond_margin"] is True for decision in decisions
        ),
        "overall_decision": _overall_decision(decisions),
    }
    report = cast(
        dict[str, object],
        plain_json(
            {
                "contract": PRACTICAL_EQUIVALENCE_REPORT_CONTRACT,
                "schema_version": 1,
                "implementation": PRACTICAL_EQUIVALENCE_IMPLEMENTATION,
                "policy_id": content_id(normalized_policy),
                "source_evidence_id": content_id(_evidence_json(bundle)),
                "protocol_id": bundle.protocol_id,
                "statistical_unit": bundle.statistical_unit,
                "candidate_method": policy.candidate_method,
                "reference_method": policy.reference_method,
                "bootstrap": {
                    "contract": bootstrap["contract"],
                    "replicates": policy.bootstrap_replicates,
                    "seed": policy.bootstrap_seed,
                    "confidence": policy.bootstrap_confidence,
                    "resampling_unit": "group_id",
                    "group_weighting": "equal",
                    "within_group_aggregation": "mean_over_registered_units",
                },
                "information_boundary": {
                    "margins_frozen_before_outcomes": (
                        policy.margins_frozen_before_outcomes
                    ),
                    "outcomes_used_for_margin_selection": (
                        policy.outcomes_used_for_margin_selection
                    ),
                    "groups_independent": policy.groups_independent,
                    "prospective_policy": prospective_policy,
                },
                "source_evidence_claim_boundary": bundle.claim_boundary,
                "policy_claim_boundary": policy.claim_boundary,
                "metric_decisions": decisions,
                "summary": summary,
                "promotion_authorized": False,
                "claim_authorized": False,
                "scientific_boundary": PRACTICAL_EQUIVALENCE_CLAIM_BOUNDARY,
            }
        ),
    )
    report["report_id"] = content_id(report)
    return report


__all__ = [
    "DEPLOYED_STREAM",
    "PRACTICAL_EQUIVALENCE_CLAIM_BOUNDARY",
    "PRACTICAL_EQUIVALENCE_IMPLEMENTATION",
    "PRACTICAL_EQUIVALENCE_POLICY_CONTRACT",
    "PRACTICAL_EQUIVALENCE_REPORT_CONTRACT",
    "PracticalEquivalencePolicyV1",
    "PracticalEquivalenceTargetV1",
    "RAW_STREAM",
    "assess_practical_equivalence",
    "parse_practical_equivalence_policy",
]
