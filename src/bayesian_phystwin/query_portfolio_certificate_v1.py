"""Simultaneous finite-family certificate for a query competence atlas.

The certificate promotes no backend-wide notion of competence.  It combines a
finite, already enumerated query atlas with query-level prospective evidence,
charges the familywise error budget to every query that reached the final risk
stage, and leaves every rejected or unknown query on the exact fallback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .guard_harm_risk import one_sided_binomial_upper_bound

QUERY_PORTFOLIO_MEMBER_SCHEMA = "bayesian_phystwin.query_portfolio_member"
QUERY_PORTFOLIO_MEMBER_VERSION = 1
QUERY_PORTFOLIO_CERTIFICATE_SCHEMA = "bayesian_phystwin.query_portfolio_certificate"
QUERY_PORTFOLIO_CERTIFICATE_VERSION = 1

Decision = Literal["certified", "rejected"]
_DECISIONS = frozenset({"certified", "rejected"})


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _open_probability(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


def _optional_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
) -> int | None:
    if value is None:
        return None
    return cast(int, genuine_integer(value, name=name, minimum=minimum))


def _optional_real(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_real(value, name=name)


def _optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return cast(str, sha256_digest(value, name=name))


@dataclass(frozen=True, slots=True)
class QueryPortfolioMemberV1:
    """One exact query's deployed disposition and frozen evidence."""

    query_id: str
    decision: Decision
    prospective_risk_evaluated: bool
    candidate_deployed: bool
    exact_fallback_selected: bool
    evidence_artifact_id: str
    evidence_file_sha256: str
    independent_groups: int | None = None
    harmful_groups: int | None = None
    unguarded_harmful_groups: int | None = None
    mean_gain_over_fallback: float | None = None
    familywise_gain_lower: float | None = None
    familywise_harm_upper: float | None = None
    gain_vector_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query_id", sha256_digest(self.query_id, name="query_id")
        )
        if type(self.decision) is not str or self.decision not in _DECISIONS:
            raise ValueError("decision must be certified or rejected")
        for name in (
            "prospective_risk_evaluated",
            "candidate_deployed",
            "exact_fallback_selected",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        for name in ("evidence_artifact_id", "evidence_file_sha256"):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        for name in (
            "independent_groups",
            "harmful_groups",
            "unguarded_harmful_groups",
        ):
            object.__setattr__(
                self,
                name,
                _optional_integer(getattr(self, name), name=name),
            )
        for name in (
            "mean_gain_over_fallback",
            "familywise_gain_lower",
            "familywise_harm_upper",
        ):
            object.__setattr__(
                self,
                name,
                _optional_real(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "gain_vector_sha256",
            _optional_digest(self.gain_vector_sha256, name="gain_vector_sha256"),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="member metadata"),
        )
        self._validate_disposition()
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match member content")
        object.__setattr__(self, "artifact_id", expected_id)

    def _validate_disposition(self) -> None:
        statistical = (
            self.independent_groups,
            self.harmful_groups,
            self.unguarded_harmful_groups,
            self.mean_gain_over_fallback,
            self.familywise_gain_lower,
            self.familywise_harm_upper,
            self.gain_vector_sha256,
        )
        if self.candidate_deployed:
            if self.decision != "certified" or not self.prospective_risk_evaluated:
                raise ValueError("only a risk-evaluated certified query may deploy")
            if self.exact_fallback_selected or any(
                value is None for value in statistical
            ):
                raise ValueError(
                    "deployed query requires complete nonfallback evidence"
                )
            assert self.independent_groups is not None
            assert self.harmful_groups is not None
            assert self.unguarded_harmful_groups is not None
            assert self.mean_gain_over_fallback is not None
            assert self.familywise_gain_lower is not None
            assert self.familywise_harm_upper is not None
            if self.independent_groups < 1:
                raise ValueError("deployed query requires independent groups")
            if self.harmful_groups > self.independent_groups:
                raise ValueError("harmful_groups exceeds independent_groups")
            if self.unguarded_harmful_groups > self.independent_groups:
                raise ValueError("unguarded harms exceeds independent groups")
            if self.familywise_gain_lower > self.mean_gain_over_fallback:
                raise ValueError("gain lower bound exceeds observed mean gain")
            if not 0.0 <= self.familywise_harm_upper <= 1.0:
                raise ValueError("harm upper bound must be a probability")
            return
        if not self.exact_fallback_selected:
            raise ValueError("every nondeployed query must select exact fallback")
        if self.decision != "rejected":
            raise ValueError("a certified query cannot be silently dropped")
        if any(value is not None for value in statistical):
            raise ValueError("fallback-only member cannot carry deployed statistics")

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_PORTFOLIO_MEMBER_SCHEMA,
            "schema_version": QUERY_PORTFOLIO_MEMBER_VERSION,
            "query_id": self.query_id,
            "decision": self.decision,
            "prospective_risk_evaluated": self.prospective_risk_evaluated,
            "candidate_deployed": self.candidate_deployed,
            "exact_fallback_selected": self.exact_fallback_selected,
            "evidence_artifact_id": self.evidence_artifact_id,
            "evidence_file_sha256": self.evidence_file_sha256,
            "independent_groups": self.independent_groups,
            "harmful_groups": self.harmful_groups,
            "unguarded_harmful_groups": self.unguarded_harmful_groups,
            "mean_gain_over_fallback": self.mean_gain_over_fallback,
            "familywise_gain_lower": self.familywise_gain_lower,
            "familywise_harm_upper": self.familywise_harm_upper,
            "gain_vector_sha256": self.gain_vector_sha256,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "query portfolio member",
    ) -> QueryPortfolioMemberV1:
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "query_id",
                "decision",
                "prospective_risk_evaluated",
                "candidate_deployed",
                "exact_fallback_selected",
                "evidence_artifact_id",
                "evidence_file_sha256",
                "independent_groups",
                "harmful_groups",
                "unguarded_harmful_groups",
                "mean_gain_over_fallback",
                "familywise_gain_lower",
                "familywise_harm_upper",
                "gain_vector_sha256",
                "metadata",
                "artifact_id",
            }
        )
        require_exact_fields(value, expected=expected, name=name)
        if value["schema"] != QUERY_PORTFOLIO_MEMBER_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != QUERY_PORTFOLIO_MEMBER_VERSION:
            raise ValueError(f"{name} schema version changed")
        return cls(
            query_id=cast(str, value["query_id"]),
            decision=cast(Decision, value["decision"]),
            prospective_risk_evaluated=cast(bool, value["prospective_risk_evaluated"]),
            candidate_deployed=cast(bool, value["candidate_deployed"]),
            exact_fallback_selected=cast(bool, value["exact_fallback_selected"]),
            evidence_artifact_id=cast(str, value["evidence_artifact_id"]),
            evidence_file_sha256=cast(str, value["evidence_file_sha256"]),
            independent_groups=cast(int | None, value["independent_groups"]),
            harmful_groups=cast(int | None, value["harmful_groups"]),
            unguarded_harmful_groups=cast(
                int | None, value["unguarded_harmful_groups"]
            ),
            mean_gain_over_fallback=cast(
                float | None, value["mean_gain_over_fallback"]
            ),
            familywise_gain_lower=cast(float | None, value["familywise_gain_lower"]),
            familywise_harm_upper=cast(float | None, value["familywise_harm_upper"]),
            gain_vector_sha256=cast(str | None, value["gain_vector_sha256"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )


@dataclass(frozen=True, slots=True)
class QueryPortfolioCertificateV1:
    """Familywise certificate for every deployed query in one frozen atlas."""

    atlas_id: str
    atlas_file_sha256: str
    members: Sequence[QueryPortfolioMemberV1]
    familywise_confidence: float
    harm_risk_budget: float
    component_trials_prospective: bool
    portfolio_synthesis_posthoc: bool
    selector_must_be_outcome_independent: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("atlas_id", "atlas_file_sha256"):
            object.__setattr__(
                self, name, sha256_digest(getattr(self, name), name=name)
            )
        members = tuple(self.members)
        if not members or any(
            not isinstance(member, QueryPortfolioMemberV1) for member in members
        ):
            raise ValueError("portfolio requires query members")
        members = tuple(sorted(members, key=lambda member: member.query_id))
        if len({member.query_id for member in members}) != len(members):
            raise ValueError("portfolio query IDs must be unique")
        object.__setattr__(self, "members", members)
        object.__setattr__(
            self,
            "familywise_confidence",
            _open_probability(self.familywise_confidence, name="familywise_confidence"),
        )
        object.__setattr__(
            self,
            "harm_risk_budget",
            _open_probability(self.harm_risk_budget, name="harm_risk_budget"),
        )
        for name in (
            "component_trials_prospective",
            "portfolio_synthesis_posthoc",
            "selector_must_be_outcome_independent",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        if not self.component_trials_prospective:
            raise ValueError("portfolio requires prospective component trials")
        if not self.portfolio_synthesis_posthoc:
            raise ValueError("v1 must disclose post-hoc portfolio synthesis")
        if not self.selector_must_be_outcome_independent:
            raise ValueError(
                "portfolio guarantee requires outcome-independent selection"
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="portfolio metadata"),
        )
        self._validate_adjusted_bounds()
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match portfolio content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def risk_evaluable_count(self) -> int:
        return sum(member.prospective_risk_evaluated for member in self.members)

    @property
    def deployed_members(self) -> tuple[QueryPortfolioMemberV1, ...]:
        return tuple(member for member in self.members if member.candidate_deployed)

    @property
    def fallback_members(self) -> tuple[QueryPortfolioMemberV1, ...]:
        return tuple(
            member for member in self.members if member.exact_fallback_selected
        )

    @property
    def per_query_confidence(self) -> float:
        if self.risk_evaluable_count < 1:
            raise ValueError("portfolio requires at least one final-risk query")
        alpha = 1.0 - self.familywise_confidence
        return 1.0 - alpha / self.risk_evaluable_count

    @property
    def simultaneous_harm_passed(self) -> bool:
        return bool(
            self.deployed_members
            and all(
                cast(float, member.familywise_harm_upper) <= self.harm_risk_budget
                for member in self.deployed_members
            )
        )

    @property
    def simultaneous_positive_gain_passed(self) -> bool:
        return bool(
            self.deployed_members
            and all(
                cast(float, member.familywise_gain_lower) > 0.0
                for member in self.deployed_members
            )
        )

    @property
    def maximum_deployed_harm_upper(self) -> float:
        if not self.deployed_members:
            return 0.0
        return max(
            cast(float, member.familywise_harm_upper)
            for member in self.deployed_members
        )

    @property
    def joint_value_and_harm_confidence_lower(self) -> float:
        return max(0.0, 2.0 * self.familywise_confidence - 1.0)

    def _validate_adjusted_bounds(self) -> None:
        if self.risk_evaluable_count < len(self.deployed_members):
            raise ValueError("deployed roster exceeds final-risk family")
        if len(self.deployed_members) + len(self.fallback_members) != len(self.members):
            raise ValueError("each query must deploy or use exact fallback")
        confidence = self.per_query_confidence
        for member in self.deployed_members:
            assert member.harmful_groups is not None
            assert member.independent_groups is not None
            assert member.familywise_harm_upper is not None
            expected = one_sided_binomial_upper_bound(
                member.harmful_groups,
                member.independent_groups,
                confidence,
            )
            if not math.isclose(
                member.familywise_harm_upper,
                expected,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("familywise harm bound does not reproduce")

    def descriptor(self) -> dict[str, object]:
        guarded_harms = sum(
            cast(int, member.harmful_groups) for member in self.deployed_members
        )
        unguarded_harms = sum(
            cast(int, member.unguarded_harmful_groups)
            for member in self.deployed_members
        )
        aggregate_worlds = sum(
            cast(int, member.independent_groups) for member in self.deployed_members
        )
        return {
            "schema": QUERY_PORTFOLIO_CERTIFICATE_SCHEMA,
            "schema_version": QUERY_PORTFOLIO_CERTIFICATE_VERSION,
            "atlas_id": self.atlas_id,
            "atlas_file_sha256": self.atlas_file_sha256,
            "members": [member.to_record() for member in self.members],
            "query_count": len(self.members),
            "risk_evaluable_query_count": self.risk_evaluable_count,
            "deployed_query_ids": [member.query_id for member in self.deployed_members],
            "fallback_query_ids": [member.query_id for member in self.fallback_members],
            "familywise_confidence": self.familywise_confidence,
            "multiplicity_method": "bonferroni-over-all-final-risk-queries-v1",
            "per_query_confidence": self.per_query_confidence,
            "harm_risk_budget": self.harm_risk_budget,
            "simultaneous_harm_passed": self.simultaneous_harm_passed,
            "simultaneous_positive_gain_passed": self.simultaneous_positive_gain_passed,
            "maximum_deployed_harm_upper": self.maximum_deployed_harm_upper,
            "joint_value_and_harm_confidence_lower": (
                self.joint_value_and_harm_confidence_lower
            ),
            "descriptive_equal_query_aggregate": {
                "evaluation_worlds": aggregate_worlds,
                "guarded_harmful_worlds": guarded_harms,
                "unguarded_harmful_worlds": unguarded_harms,
                "harmful_world_reduction": unguarded_harms - guarded_harms,
                "harmful_world_reduction_fraction": (
                    (unguarded_harms - guarded_harms) / unguarded_harms
                    if unguarded_harms
                    else 0.0
                ),
                "cross_task_reward_gains_pooled": False,
            },
            "component_trials_prospective": self.component_trials_prospective,
            "portfolio_synthesis_posthoc": self.portfolio_synthesis_posthoc,
            "selector_must_be_outcome_independent": (
                self.selector_must_be_outcome_independent
            ),
            "backend_wide_competence_claim": False,
            "physical_safety_claim": False,
            "official_benchmark_or_sota_claim": False,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "query portfolio certificate",
    ) -> QueryPortfolioCertificateV1:
        derived = frozenset(
            {
                "query_count",
                "risk_evaluable_query_count",
                "deployed_query_ids",
                "fallback_query_ids",
                "multiplicity_method",
                "per_query_confidence",
                "simultaneous_harm_passed",
                "simultaneous_positive_gain_passed",
                "maximum_deployed_harm_upper",
                "joint_value_and_harm_confidence_lower",
                "descriptive_equal_query_aggregate",
                "backend_wide_competence_claim",
                "physical_safety_claim",
                "official_benchmark_or_sota_claim",
            }
        )
        expected = derived | frozenset(
            {
                "schema",
                "schema_version",
                "atlas_id",
                "atlas_file_sha256",
                "members",
                "familywise_confidence",
                "harm_risk_budget",
                "component_trials_prospective",
                "portfolio_synthesis_posthoc",
                "selector_must_be_outcome_independent",
                "metadata",
                "artifact_id",
            }
        )
        require_exact_fields(value, expected=expected, name=name)
        if value["schema"] != QUERY_PORTFOLIO_CERTIFICATE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != QUERY_PORTFOLIO_CERTIFICATE_VERSION:
            raise ValueError(f"{name} schema version changed")
        raw_members = value["members"]
        if isinstance(raw_members, (str, bytes)) or not isinstance(
            raw_members, Sequence
        ):
            raise ValueError(f"{name} members must be a sequence")
        result = cls(
            atlas_id=cast(str, value["atlas_id"]),
            atlas_file_sha256=cast(str, value["atlas_file_sha256"]),
            members=tuple(
                QueryPortfolioMemberV1.from_mapping(
                    cast(Mapping[str, object], member),
                    name=f"{name} member {index}",
                )
                for index, member in enumerate(raw_members)
            ),
            familywise_confidence=cast(float, value["familywise_confidence"]),
            harm_risk_budget=cast(float, value["harm_risk_budget"]),
            component_trials_prospective=cast(
                bool, value["component_trials_prospective"]
            ),
            portfolio_synthesis_posthoc=cast(
                bool, value["portfolio_synthesis_posthoc"]
            ),
            selector_must_be_outcome_independent=cast(
                bool, value["selector_must_be_outcome_independent"]
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )
        record = result.to_record()
        for key in derived:
            if value[key] != record[key]:
                raise ValueError(f"{name} derived field {key!r} changed")
        return result


def save_query_portfolio_certificate(
    path: Path,
    certificate: QueryPortfolioCertificateV1,
) -> None:
    write_atomic_json(certificate.to_record(), path, overwrite=False)


def load_query_portfolio_certificate(path: Path) -> QueryPortfolioCertificateV1:
    return QueryPortfolioCertificateV1.from_mapping(
        load_strict_json_object(path, label="query portfolio certificate")
    )
