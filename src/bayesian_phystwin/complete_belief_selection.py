"""Guarded selection of complete content-addressed beliefs.

Array-level fallback is useful for local numerical tests, but deployment must
select the whole belief so state, parameters, particle weights, discrepancy,
nuisance moments, and provenance remain consistent. Rejection therefore returns
the exact baseline object rather than reconstructing it from zero correction
coefficients.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)


class ArtifactBelief(Protocol):
    @property
    def artifact_id(self) -> str: ...


BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _content_id(values: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(values),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CompleteBeliefGuardDecisionV1:
    """Bind numerical admissibility and a regret certificate to two beliefs."""

    baseline_belief_id: str
    candidate_belief_id: str
    common_domain_id: str
    certificate_id: str
    inference_admissible: bool
    regret_guard_accepted: bool
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("baseline_belief_id", self.baseline_belief_id),
            ("candidate_belief_id", self.candidate_belief_id),
            ("common_domain_id", self.common_domain_id),
            ("certificate_id", self.certificate_id),
        ):
            _validate_sha256(value, name=name)
        inference_admissible = genuine_boolean(
            self.inference_admissible,
            name="inference_admissible",
        )
        regret_guard_accepted = genuine_boolean(
            self.regret_guard_accepted,
            name="regret_guard_accepted",
        )
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("guard decision reason must be nonempty")
        if regret_guard_accepted and not inference_admissible:
            raise ValueError(
                "regret_guard_accepted requires inference_admissible"
            )
        object.__setattr__(self, "inference_admissible", inference_admissible)
        object.__setattr__(self, "regret_guard_accepted", regret_guard_accepted)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata),
        )

    @property
    def decision_id(self) -> str:
        return _content_id(
            {
                "schema": "bayesian_phystwin.complete_belief_guard",
                "schema_version": 1,
                "baseline_belief_id": self.baseline_belief_id,
                "candidate_belief_id": self.candidate_belief_id,
                "common_domain_id": self.common_domain_id,
                "certificate_id": self.certificate_id,
                "inference_admissible": self.inference_admissible,
                "regret_guard_accepted": self.regret_guard_accepted,
                "reason": self.reason,
                "metadata": plain_json(self.metadata),
            }
        )


@dataclass(frozen=True)
class CompleteBeliefSelectionV1:
    """Content-addressed routing record for a complete belief."""

    baseline_belief_id: str
    candidate_belief_id: str
    common_domain_id: str
    guard_decision_id: str
    selected_belief_id: str
    selected_candidate: bool
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("baseline_belief_id", self.baseline_belief_id),
            ("candidate_belief_id", self.candidate_belief_id),
            ("common_domain_id", self.common_domain_id),
            ("guard_decision_id", self.guard_decision_id),
            ("selected_belief_id", self.selected_belief_id),
        ):
            _validate_sha256(value, name=name)
        selected_candidate = genuine_boolean(
            self.selected_candidate,
            name="selected_candidate",
        )
        expected = (
            self.candidate_belief_id
            if selected_candidate
            else self.baseline_belief_id
        )
        if self.selected_belief_id != expected:
            raise ValueError("selected belief ID contradicts routing decision")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("selection reason must be nonempty")
        object.__setattr__(self, "selected_candidate", selected_candidate)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata),
        )

    @property
    def selection_id(self) -> str:
        return _content_id(
            {
                "schema": "bayesian_phystwin.complete_belief_selection",
                "schema_version": 1,
                "baseline_belief_id": self.baseline_belief_id,
                "candidate_belief_id": self.candidate_belief_id,
                "common_domain_id": self.common_domain_id,
                "guard_decision_id": self.guard_decision_id,
                "selected_belief_id": self.selected_belief_id,
                "selected_candidate": self.selected_candidate,
                "reason": self.reason,
                "metadata": plain_json(self.metadata),
            }
        )


def select_complete_belief(
    baseline: BeliefT,
    candidate: BeliefT,
    decision: CompleteBeliefGuardDecisionV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BeliefT, CompleteBeliefSelectionV1]:
    """Select a candidate or return the exact baseline belief object."""

    if baseline.artifact_id != decision.baseline_belief_id:
        raise ValueError("guard decision does not bind the baseline belief")
    if candidate.artifact_id != decision.candidate_belief_id:
        raise ValueError("guard decision does not bind the candidate belief")
    selected_candidate = (
        decision.inference_admissible and decision.regret_guard_accepted
    )
    selected = candidate if selected_candidate else baseline
    if not selected_candidate and selected is not baseline:
        raise AssertionError("rejected routing did not reuse the baseline object")
    reason = (
        "guard-accepted"
        if selected_candidate
        else (
            "inference-rejected"
            if not decision.inference_admissible
            else "regret-guard-rejected"
        )
    )
    selection = CompleteBeliefSelectionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=decision.common_domain_id,
        guard_decision_id=decision.decision_id,
        selected_belief_id=selected.artifact_id,
        selected_candidate=selected_candidate,
        reason=reason,
        metadata=metadata or {},
    )
    return selected, selection


__all__ = [
    "ArtifactBelief",
    "CompleteBeliefGuardDecisionV1",
    "CompleteBeliefSelectionV1",
    "select_complete_belief",
]
