"""Content-bound candidate construction and guarded complete-belief selection.

The numerical candidate identity, the complete candidate belief, and the guard
selection are distinct artifacts.  This module binds them explicitly so a
downstream repository cannot combine inference A with belief B or bypass a
complete-belief guard while still presenting an inference-admissible update.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from ._validation import lowercase_sha256
from .complete_belief_selection import ArtifactBelief
from .inference.v1 import GuardedUpdateResultV1

CANDIDATE_CONSTRUCTION_SCHEMA = "bayesian_phystwin.candidate_belief_construction"
CANDIDATE_CONSTRUCTION_SCHEMA_VERSION = 1
GUARDED_SELECTION_SCHEMA = "bayesian_phystwin.guarded_belief_selection"
GUARDED_SELECTION_SCHEMA_VERSION = 2
DEFAULT_CONSTRUCTION_METHOD = "prob4d-complete-belief-construction-v1"
DEFAULT_GUARD_KIND = "complete-belief-regret-guard-v1"


def _content_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_exact_fields(
    values: object,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(values, Mapping) or any(type(key) is not str for key in values):
        raise ValueError(f"{name} must be a string-keyed mapping")
    missing = sorted(required - set(values))
    unexpected = sorted(set(values) - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return values


def _belief_id(value: object, *, name: str) -> str:
    try:
        artifact_id = value.artifact_id  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError(f"{name} must expose artifact_id") from error
    return lowercase_sha256(artifact_id, name=f"{name}.artifact_id")


@runtime_checkable
class ClaimBearingCandidateInference(Protocol):
    """Minimum inference lineage required by the construction receipt."""

    @property
    def candidate_id(self) -> str: ...

    @property
    def update_id(self) -> str: ...

    @property
    def admission_id(self) -> str: ...

    @property
    def observation_artifact_id(self) -> str: ...

    @property
    def linearization_artifact_id(self) -> str: ...

    @property
    def inference_admissible(self) -> bool: ...


_CONSTRUCTION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "inference_candidate_id",
        "update_id",
        "admission_id",
        "observation_artifact_id",
        "linearization_artifact_id",
        "baseline_belief_id",
        "candidate_belief_id",
        "common_domain_id",
        "construction_method",
        "inference_admissible",
        "metadata",
        "receipt_id",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateBeliefConstructionReceiptV1:
    """Bind one complete candidate belief to one admitted inference artifact."""

    inference_candidate_id: str
    update_id: str
    admission_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    baseline_belief_id: str
    candidate_belief_id: str
    common_domain_id: str
    construction_method: str
    inference_admissible: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "inference_candidate_id",
            "update_id",
            "admission_id",
            "observation_artifact_id",
            "linearization_artifact_id",
            "baseline_belief_id",
            "candidate_belief_id",
            "common_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                lowercase_sha256(getattr(self, name), name=name),
            )
        if self.inference_candidate_id != self.update_id:
            raise ValueError("inference candidate identity must equal update identity")
        object.__setattr__(
            self,
            "construction_method",
            _require_string(
                self.construction_method,
                name="construction_method",
            ),
        )
        object.__setattr__(
            self,
            "inference_admissible",
            genuine_boolean(
                self.inference_admissible,
                name="inference_admissible",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="candidate construction metadata",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": CANDIDATE_CONSTRUCTION_SCHEMA,
            "schema_version": CANDIDATE_CONSTRUCTION_SCHEMA_VERSION,
            "inference_candidate_id": self.inference_candidate_id,
            "update_id": self.update_id,
            "admission_id": self.admission_id,
            "observation_artifact_id": self.observation_artifact_id,
            "linearization_artifact_id": self.linearization_artifact_id,
            "baseline_belief_id": self.baseline_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "common_domain_id": self.common_domain_id,
            "construction_method": self.construction_method,
            "inference_admissible": self.inference_admissible,
            "metadata": plain_json(self.metadata),
        }

    @property
    def receipt_id(self) -> str:
        return _content_id(self._identity_payload())

    @property
    def artifact_id(self) -> str:
        return self.receipt_id

    def to_record(self) -> dict[str, object]:
        return {**self._identity_payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_record(
        cls,
        values: Mapping[str, Any],
    ) -> CandidateBeliefConstructionReceiptV1:
        fields = _require_exact_fields(
            values,
            name="candidate construction receipt",
            required=_CONSTRUCTION_FIELDS,
        )
        if fields["schema"] != CANDIDATE_CONSTRUCTION_SCHEMA:
            raise ValueError("unsupported candidate construction schema")
        if fields["schema_version"] != CANDIDATE_CONSTRUCTION_SCHEMA_VERSION:
            raise ValueError("unsupported candidate construction schema version")
        receipt = cls(
            inference_candidate_id=fields["inference_candidate_id"],
            update_id=fields["update_id"],
            admission_id=fields["admission_id"],
            observation_artifact_id=fields["observation_artifact_id"],
            linearization_artifact_id=fields["linearization_artifact_id"],
            baseline_belief_id=fields["baseline_belief_id"],
            candidate_belief_id=fields["candidate_belief_id"],
            common_domain_id=fields["common_domain_id"],
            construction_method=fields["construction_method"],
            inference_admissible=fields["inference_admissible"],
            metadata=fields["metadata"],
        )
        if fields["receipt_id"] != receipt.receipt_id:
            raise ValueError("candidate construction receipt identity changed")
        return receipt


_SELECTION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "candidate_construction",
        "guard_kind",
        "guard_certificate_id",
        "guard_decision_id",
        "selection_id",
        "selected_belief_id",
        "selected_candidate",
        "exact_fallback",
        "metadata",
        "receipt_id",
    }
)


@dataclass(frozen=True, slots=True)
class GuardedBeliefSelectionReceiptV2:
    """Self-contained receipt for construction, guard, and complete selection."""

    candidate_construction: CandidateBeliefConstructionReceiptV1
    guard_kind: str
    guard_certificate_id: str
    guard_decision_id: str
    selection_id: str
    selected_belief_id: str
    selected_candidate: bool
    exact_fallback: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.candidate_construction,
            CandidateBeliefConstructionReceiptV1,
        ):
            raise TypeError(
                "candidate_construction must be CandidateBeliefConstructionReceiptV1"
            )
        construction = CandidateBeliefConstructionReceiptV1.from_record(
            self.candidate_construction.to_record()
        )
        object.__setattr__(self, "candidate_construction", construction)
        object.__setattr__(
            self,
            "guard_kind",
            _require_string(self.guard_kind, name="guard_kind"),
        )
        for name in (
            "guard_certificate_id",
            "guard_decision_id",
            "selection_id",
            "selected_belief_id",
        ):
            object.__setattr__(
                self,
                name,
                lowercase_sha256(getattr(self, name), name=name),
            )
        selected_candidate = genuine_boolean(
            self.selected_candidate,
            name="selected_candidate",
        )
        exact_fallback = genuine_boolean(
            self.exact_fallback,
            name="exact_fallback",
        )
        expected = (
            construction.candidate_belief_id
            if selected_candidate
            else construction.baseline_belief_id
        )
        if self.selected_belief_id != expected:
            raise ValueError("selected belief contradicts construction identities")
        if exact_fallback == selected_candidate:
            raise ValueError(
                "exact_fallback must be the complement of selected_candidate"
            )
        if selected_candidate and not construction.inference_admissible:
            raise ValueError("selected candidate requires admissible inference")
        object.__setattr__(self, "selected_candidate", selected_candidate)
        object.__setattr__(self, "exact_fallback", exact_fallback)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="guarded selection metadata",
            ),
        )

    @property
    def candidate_construction_receipt_id(self) -> str:
        return self.candidate_construction.receipt_id

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": GUARDED_SELECTION_SCHEMA,
            "schema_version": GUARDED_SELECTION_SCHEMA_VERSION,
            "candidate_construction": self.candidate_construction.to_record(),
            "guard_kind": self.guard_kind,
            "guard_certificate_id": self.guard_certificate_id,
            "guard_decision_id": self.guard_decision_id,
            "selection_id": self.selection_id,
            "selected_belief_id": self.selected_belief_id,
            "selected_candidate": self.selected_candidate,
            "exact_fallback": self.exact_fallback,
            "metadata": plain_json(self.metadata),
        }

    @property
    def receipt_id(self) -> str:
        return _content_id(self._identity_payload())

    @property
    def artifact_id(self) -> str:
        return self.receipt_id

    def to_record(self) -> dict[str, object]:
        return {**self._identity_payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_record(
        cls,
        values: Mapping[str, Any],
    ) -> GuardedBeliefSelectionReceiptV2:
        fields = _require_exact_fields(
            values,
            name="guarded belief selection receipt",
            required=_SELECTION_FIELDS,
        )
        if fields["schema"] != GUARDED_SELECTION_SCHEMA:
            raise ValueError("unsupported guarded selection schema")
        if fields["schema_version"] != GUARDED_SELECTION_SCHEMA_VERSION:
            raise ValueError("unsupported guarded selection schema version")
        construction_raw = fields["candidate_construction"]
        if not isinstance(construction_raw, Mapping):
            raise ValueError("candidate_construction must be a mapping")
        receipt = cls(
            candidate_construction=(
                CandidateBeliefConstructionReceiptV1.from_record(construction_raw)
            ),
            guard_kind=fields["guard_kind"],
            guard_certificate_id=fields["guard_certificate_id"],
            guard_decision_id=fields["guard_decision_id"],
            selection_id=fields["selection_id"],
            selected_belief_id=fields["selected_belief_id"],
            selected_candidate=fields["selected_candidate"],
            exact_fallback=fields["exact_fallback"],
            metadata=fields["metadata"],
        )
        if fields["receipt_id"] != receipt.receipt_id:
            raise ValueError("guarded selection receipt identity changed")
        return receipt


def build_candidate_belief_construction_receipt(
    inference: ClaimBearingCandidateInference,
    baseline_belief: ArtifactBelief,
    candidate_belief: ArtifactBelief,
    *,
    common_domain_id: str,
    construction_method: str = DEFAULT_CONSTRUCTION_METHOD,
    metadata: Mapping[str, Any] | None = None,
) -> CandidateBeliefConstructionReceiptV1:
    """Bind a complete candidate belief to the exact inference lineage."""

    if not isinstance(inference, ClaimBearingCandidateInference):
        raise TypeError("inference lacks claim-bearing construction lineage")
    candidate_id = lowercase_sha256(
        inference.candidate_id,
        name="inference.candidate_id",
    )
    update_id = lowercase_sha256(inference.update_id, name="inference.update_id")
    if candidate_id != update_id:
        raise ValueError("inference candidate identity differs from update identity")
    return CandidateBeliefConstructionReceiptV1(
        inference_candidate_id=candidate_id,
        update_id=update_id,
        admission_id=inference.admission_id,
        observation_artifact_id=inference.observation_artifact_id,
        linearization_artifact_id=inference.linearization_artifact_id,
        baseline_belief_id=_belief_id(baseline_belief, name="baseline_belief"),
        candidate_belief_id=_belief_id(candidate_belief, name="candidate_belief"),
        common_domain_id=common_domain_id,
        construction_method=construction_method,
        inference_admissible=inference.inference_admissible,
        metadata=metadata or {},
    )


def bind_guarded_belief_selection_receipt(
    inference: ClaimBearingCandidateInference,
    guarded_update: GuardedUpdateResultV1[ArtifactBelief],
    candidate_construction: CandidateBeliefConstructionReceiptV1,
    *,
    guard_kind: str = DEFAULT_GUARD_KIND,
    metadata: Mapping[str, Any] | None = None,
) -> GuardedBeliefSelectionReceiptV2:
    """Bind construction, guard certificate, and selected complete belief."""

    if not isinstance(inference, ClaimBearingCandidateInference):
        raise TypeError("inference lacks claim-bearing construction lineage")
    if not isinstance(guarded_update, GuardedUpdateResultV1):
        raise TypeError("guarded_update must be GuardedUpdateResultV1")
    if not isinstance(
        candidate_construction,
        CandidateBeliefConstructionReceiptV1,
    ):
        raise TypeError(
            "candidate_construction must be CandidateBeliefConstructionReceiptV1"
        )
    construction = CandidateBeliefConstructionReceiptV1.from_record(
        candidate_construction.to_record()
    )
    if construction.inference_candidate_id != guarded_update.inference_candidate_id:
        raise ValueError("construction binds a different inference candidate")
    if construction.update_id != inference.update_id:
        raise ValueError("construction binds a different update")
    if construction.admission_id != inference.admission_id:
        raise ValueError("construction binds a different admission")
    if construction.observation_artifact_id != inference.observation_artifact_id:
        raise ValueError("construction binds different observation evidence")
    if construction.linearization_artifact_id != inference.linearization_artifact_id:
        raise ValueError("construction binds a different linearization")
    if construction.inference_admissible != guarded_update.inference_admissible:
        raise ValueError("construction changed inference admissibility")
    if construction.baseline_belief_id != guarded_update.baseline_belief.artifact_id:
        raise ValueError("construction does not bind the guarded baseline belief")
    if construction.candidate_belief_id != guarded_update.candidate_belief.artifact_id:
        raise ValueError("construction does not bind the guarded candidate belief")
    decision = guarded_update.guard_decision
    selection = guarded_update.selection
    if construction.common_domain_id != decision.common_domain_id:
        raise ValueError("construction and guard use different common domains")
    if selection.common_domain_id != decision.common_domain_id:
        raise ValueError("selection and guard use different common domains")
    if selection.guard_decision_id != decision.decision_id:
        raise ValueError("selection does not bind the guard decision")
    if selection.selected_belief_id != guarded_update.selected_belief.artifact_id:
        raise ValueError("selection does not bind the guarded selected belief")
    return GuardedBeliefSelectionReceiptV2(
        candidate_construction=construction,
        guard_kind=guard_kind,
        guard_certificate_id=decision.certificate_id,
        guard_decision_id=decision.decision_id,
        selection_id=selection.selection_id,
        selected_belief_id=selection.selected_belief_id,
        selected_candidate=selection.selected_candidate,
        exact_fallback=guarded_update.exact_fallback,
        metadata=metadata or {},
    )


__all__ = [
    "CANDIDATE_CONSTRUCTION_SCHEMA",
    "CANDIDATE_CONSTRUCTION_SCHEMA_VERSION",
    "DEFAULT_CONSTRUCTION_METHOD",
    "DEFAULT_GUARD_KIND",
    "GUARDED_SELECTION_SCHEMA",
    "GUARDED_SELECTION_SCHEMA_VERSION",
    "CandidateBeliefConstructionReceiptV1",
    "ClaimBearingCandidateInference",
    "GuardedBeliefSelectionReceiptV2",
    "bind_guarded_belief_selection_receipt",
    "build_candidate_belief_construction_receipt",
]
