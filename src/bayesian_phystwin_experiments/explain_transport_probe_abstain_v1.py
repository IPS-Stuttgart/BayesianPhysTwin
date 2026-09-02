"""Fail-closed orchestration for interventional physical-twin diagnosis.

The component certificates in this repository answer distinct questions:

1. can the registered cause family explain the residual at all;
2. is a held-intervention target invariant over the remaining cause ambiguity;
3. which minimum-cost registered intervention would identify that target.

This module composes those questions without silently promoting a point cause
representative.  The current result is one of:

- no detectable correction;
- none of the registered causes;
- explain and transport inside a unique registered explanation;
- transport without identifying a unique cause;
- acquire a target-identifying intervention and then reassess; or
- exact fallback.

A probe recommendation is not a transported correction.  Until its outcome is
observed and the complete certificate is recomputed, deployment returns the
caller-owned fallback by identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Final

import numpy as np

from .interventional_cause_adequacy_v1 import (
    CauseFamilyAdequacyStatus,
    InterventionalCauseFamilyAdequacyV1,
)
from .interventional_transport_quotient_v1 import (
    InterventionalTransportQuotientV1,
    TargetTransportQuotientV1,
    TransportQuotientStatus,
)
from .target_directed_intervention_design_v1 import (
    InterventionDesignStatus,
    TargetDirectedInterventionDesignV1,
)

EXPLAIN_TRANSPORT_PROBE_ABSTAIN_SCHEMA: Final = (
    "bayesian_phystwin.explain_transport_probe_abstain"
)
EXPLAIN_TRANSPORT_PROBE_ABSTAIN_VERSION: Final = 1
EXPLAIN_TRANSPORT_PROBE_ABSTAIN_SEMANTICS: Final = (
    "cause-family-adequacy-target-transport-and-target-directed-probing-v1"
)
EXPLAIN_TRANSPORT_PROBE_ABSTAIN_CLAIM_BOUNDARY: Final = (
    "The report is exact only for the supplied local linear cause family, "
    "whitened residual, deterministic adequacy radius, held-intervention target "
    "maps, finite intervention roster, additive intervention costs, coordinates, "
    "and numerical tolerances. A unique registered coefficient vector is not a "
    "proof of natural physical causation. A probe recommendation does not "
    "authorize transport before its outcome is observed and the certificate is "
    "recomputed. The report does not validate nonlinear closure, physical probe "
    "models, held-out transport, unseen-object generalization, deployment safety, "
    "or state of the art."
)


class DiagnosticDisposition(str, Enum):
    """Operational result for one registered held-intervention target."""

    NO_DETECTABLE_ERROR = "no_detectable_error"
    NONE_OF_THE_ABOVE = "none_of_the_above"
    EXPLAIN_AND_TRANSPORT = "explain_and_transport"
    TRANSPORT_WITHOUT_CAUSE = "transport_without_cause"
    PROBE_THEN_REASSESS = "probe_then_reassess"
    PARTIAL_ONLY_FALLBACK = "partial_only_fallback"
    ABSTAIN = "abstain"


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _canonical_id(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _frozen_json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    if type(result) is not dict:
        raise ValueError(f"{name} must be a mapping")
    return result


@dataclass(frozen=True, slots=True)
class TargetDiagnosticDecisionV1:
    """One target-level explain/transport/probe/fallback decision."""

    target_id: str
    disposition: DiagnosticDisposition
    adequacy_status: CauseFamilyAdequacyStatus
    transport_status: TransportQuotientStatus
    intervention_design_status: InterventionDesignStatus | None
    registered_explanation_unique: bool
    coefficient_ambiguity_dimension: int
    target_dimension: int
    target_identifiable_dimension: int
    target_ambiguity_dimension: int
    transport_permitted: bool
    partial_target_available: bool
    selected_interventions: tuple[str, ...]
    selected_intervention_cost: float | None
    minimum_full_cause_identification_cost: float | None
    target_cost_saving_vs_full_cause_identification: float | None
    none_of_the_above: bool
    fallback_required_now: bool
    reason: str
    representative_effect: np.ndarray
    identifiable_effect: np.ndarray

    def to_record(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "disposition": self.disposition.value,
            "adequacy_status": self.adequacy_status.value,
            "transport_status": self.transport_status.value,
            "intervention_design_status": (
                None
                if self.intervention_design_status is None
                else self.intervention_design_status.value
            ),
            "registered_explanation_unique": self.registered_explanation_unique,
            "coefficient_ambiguity_dimension": self.coefficient_ambiguity_dimension,
            "target_dimension": self.target_dimension,
            "target_identifiable_dimension": self.target_identifiable_dimension,
            "target_ambiguity_dimension": self.target_ambiguity_dimension,
            "transport_permitted": self.transport_permitted,
            "partial_target_available": self.partial_target_available,
            "selected_interventions": list(self.selected_interventions),
            "selected_intervention_cost": self.selected_intervention_cost,
            "minimum_full_cause_identification_cost": (
                self.minimum_full_cause_identification_cost
            ),
            "target_cost_saving_vs_full_cause_identification": (
                self.target_cost_saving_vs_full_cause_identification
            ),
            "none_of_the_above": self.none_of_the_above,
            "fallback_required_now": self.fallback_required_now,
            "reason": self.reason,
            "representative_effect": _array_record(self.representative_effect),
            "identifiable_effect": _array_record(self.identifiable_effect),
        }


@dataclass(frozen=True, slots=True)
class ExplainTransportProbeAbstainV1:
    """Compose adequacy, target transport, and target-directed intervention design."""

    adequacy_certificate: InterventionalCauseFamilyAdequacyV1
    target_intervention_roster_id: str
    target_transport_ids: Mapping[str, str]
    target_maps: Mapping[str, np.ndarray]
    candidate_roster_id: str
    candidate_intervention_ids: Mapping[str, str]
    candidate_designs: Mapping[str, np.ndarray]
    intervention_costs: Mapping[str, float]
    relative_rank_tolerance: float = 1e-10
    absolute_rank_tolerance: float = 1e-12
    cost_tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    transport_quotient: InterventionalTransportQuotientV1 = field(init=False)
    target_order: tuple[str, ...] = field(init=False)
    target_decisions: tuple[TargetDiagnosticDecisionV1, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.adequacy_certificate,
            InterventionalCauseFamilyAdequacyV1,
        ):
            raise TypeError(
                "adequacy_certificate must be an "
                "InterventionalCauseFamilyAdequacyV1"
            )
        _digest(
            self.adequacy_certificate.artifact_id,
            name="adequacy_certificate.artifact_id",
        )
        target_roster = _digest(
            self.target_intervention_roster_id,
            name="target_intervention_roster_id",
        )
        candidate_roster = _digest(
            self.candidate_roster_id,
            name="candidate_roster_id",
        )
        relative = _finite_nonnegative(
            self.relative_rank_tolerance,
            name="relative_rank_tolerance",
        )
        absolute = _finite_nonnegative(
            self.absolute_rank_tolerance,
            name="absolute_rank_tolerance",
        )
        cost_tolerance = _finite_nonnegative(
            self.cost_tolerance,
            name="cost_tolerance",
        )
        if relative == 0.0 and absolute == 0.0:
            raise ValueError("at least one rank tolerance must be positive")
        if not isinstance(self.candidate_intervention_ids, Mapping):
            raise TypeError("candidate_intervention_ids must be a mapping")
        if not isinstance(self.candidate_designs, Mapping):
            raise TypeError("candidate_designs must be a mapping")
        if not isinstance(self.intervention_costs, Mapping):
            raise TypeError("intervention_costs must be a mapping")
        if set(self.candidate_intervention_ids) != set(self.candidate_designs):
            raise ValueError(
                "candidate_intervention_ids must cover exactly candidate_designs"
            )
        if set(self.intervention_costs) != set(self.candidate_designs):
            raise ValueError("intervention_costs must cover exactly candidate_designs")

        candidate_ids = {
            candidate: _digest(
                self.candidate_intervention_ids[candidate],
                name=f"candidate_intervention_ids[{candidate!r}]",
            )
            for candidate in sorted(self.candidate_designs)
        }
        costs = {
            candidate: _finite_nonnegative(
                self.intervention_costs[candidate],
                name=f"intervention_costs[{candidate!r}]",
            )
            for candidate in sorted(self.candidate_designs)
        }
        metadata = _frozen_json_mapping(self.metadata, name="metadata")

        transport = InterventionalTransportQuotientV1(
            adequacy_certificate=self.adequacy_certificate,
            target_intervention_roster_id=target_roster,
            target_transport_ids=self.target_transport_ids,
            target_maps=self.target_maps,
            relative_rank_tolerance=relative,
            absolute_rank_tolerance=absolute,
            metadata={
                "composition": EXPLAIN_TRANSPORT_PROBE_ABSTAIN_SEMANTICS,
            },
        )

        decisions: list[TargetDiagnosticDecisionV1] = []
        for target_id in transport.target_order:
            target_record = transport.record_for(target_id)
            design: TargetDirectedInterventionDesignV1 | None = None
            if self.adequacy_certificate.family_adequate:
                design = TargetDirectedInterventionDesignV1(
                    source_design_id=self.adequacy_certificate.artifact_id,
                    target_query_id=self.target_transport_ids[target_id],
                    candidate_roster_id=candidate_roster,
                    source_design=self.adequacy_certificate.total_design,
                    target_map=self.target_maps[target_id],
                    candidate_intervention_ids=candidate_ids,
                    candidate_designs=self.candidate_designs,
                    intervention_costs=costs,
                    relative_rank_tolerance=relative,
                    absolute_rank_tolerance=absolute,
                    cost_tolerance=cost_tolerance,
                    metadata={
                        "adequacy_certificate_id": (
                            self.adequacy_certificate.artifact_id
                        ),
                        "target_transport_quotient_id": transport.artifact_id,
                    },
                )

            decision = self._classify_target(
                target_record=target_record,
                design=design,
            )
            decisions.append(decision)

        object.__setattr__(self, "target_intervention_roster_id", target_roster)
        object.__setattr__(self, "candidate_roster_id", candidate_roster)
        object.__setattr__(self, "candidate_intervention_ids", candidate_ids)
        object.__setattr__(self, "intervention_costs", costs)
        object.__setattr__(self, "relative_rank_tolerance", relative)
        object.__setattr__(self, "absolute_rank_tolerance", absolute)
        object.__setattr__(self, "cost_tolerance", cost_tolerance)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "transport_quotient", transport)
        object.__setattr__(self, "target_order", transport.target_order)
        object.__setattr__(self, "target_decisions", tuple(decisions))

        expected = _canonical_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match report content")
        object.__setattr__(self, "artifact_id", expected)

    def _classify_target(
        self,
        *,
        target_record: TargetTransportQuotientV1,
        design: TargetDirectedInterventionDesignV1 | None,
    ) -> TargetDiagnosticDecisionV1:
        adequacy_status = self.adequacy_certificate.status
        design_status = None if design is None else design.status

        if adequacy_status is CauseFamilyAdequacyStatus.NO_DETECTABLE_ERROR:
            disposition = DiagnosticDisposition.NO_DETECTABLE_ERROR
            reason = "residual-is-within-the-registered-noise-radius"
            transport_permitted = False
            none_of_the_above = False
            selected_interventions: tuple[str, ...] = ()
        elif adequacy_status is CauseFamilyAdequacyStatus.UNMODELED_CAUSE:
            disposition = DiagnosticDisposition.NONE_OF_THE_ABOVE
            reason = "residual-has-unexplained-energy-outside-the-cause-family"
            transport_permitted = False
            none_of_the_above = True
            selected_interventions = ()
        elif target_record.status is TransportQuotientStatus.FULLY_IDENTIFIABLE:
            transport_permitted = True
            none_of_the_above = False
            selected_interventions = ()
            if self.adequacy_certificate.unique_coefficients:
                disposition = DiagnosticDisposition.EXPLAIN_AND_TRANSPORT
                reason = "unique-registered-explanation-and-invariant-target"
            else:
                disposition = DiagnosticDisposition.TRANSPORT_WITHOUT_CAUSE
                reason = "target-is-invariant-over-the-set-valued-cause-explanation"
        elif design is not None and design.status is InterventionDesignStatus.TARGET_IDENTIFIED:
            disposition = DiagnosticDisposition.PROBE_THEN_REASSESS
            reason = "minimum-cost-registered-intervention-identifies-the-target"
            transport_permitted = False
            none_of_the_above = False
            selected_interventions = design.selected_interventions
        elif (
            target_record.status is TransportQuotientStatus.PARTIALLY_IDENTIFIABLE
            or (
                design is not None
                and design.status is InterventionDesignStatus.PARTIAL_IMPROVEMENT
            )
        ):
            disposition = DiagnosticDisposition.PARTIAL_ONLY_FALLBACK
            reason = "only-a-target-subspace-is-identifiable"
            transport_permitted = False
            none_of_the_above = False
            selected_interventions = ()
        else:
            disposition = DiagnosticDisposition.ABSTAIN
            reason = "no-registered-intervention-identifies-the-complete-target"
            transport_permitted = False
            none_of_the_above = False
            selected_interventions = ()

        if (
            design is not None
            and target_record.status is not TransportQuotientStatus.FULLY_IDENTIFIABLE
            and design.status is InterventionDesignStatus.ALREADY_IDENTIFIABLE
        ):
            disposition = DiagnosticDisposition.ABSTAIN
            reason = "transport-and-intervention-certificates-disagree"
            transport_permitted = False
            selected_interventions = ()

        selected_cost = None
        full_cost = None
        saving = None
        if design is not None:
            full_cost = design.minimum_full_cause_identification_cost
            if disposition is DiagnosticDisposition.PROBE_THEN_REASSESS:
                selected_cost = design.selected_total_cost
                saving = design.cost_saving_vs_full_cause_identification
            elif transport_permitted:
                selected_cost = 0.0
                if full_cost is not None:
                    saving = full_cost

        return TargetDiagnosticDecisionV1(
            target_id=target_record.target_id,
            disposition=disposition,
            adequacy_status=adequacy_status,
            transport_status=target_record.status,
            intervention_design_status=design_status,
            registered_explanation_unique=(
                self.adequacy_certificate.unique_coefficients
            ),
            coefficient_ambiguity_dimension=(
                self.adequacy_certificate.solution_nullity
            ),
            target_dimension=target_record.target_dimension,
            target_identifiable_dimension=target_record.identifiable_dimension,
            target_ambiguity_dimension=target_record.ambiguity_dimension,
            transport_permitted=transport_permitted,
            partial_target_available=target_record.partial_transport_available,
            selected_interventions=selected_interventions,
            selected_intervention_cost=selected_cost,
            minimum_full_cause_identification_cost=full_cost,
            target_cost_saving_vs_full_cause_identification=saving,
            none_of_the_above=none_of_the_above,
            fallback_required_now=not transport_permitted,
            reason=reason,
            representative_effect=target_record.representative_effect,
            identifiable_effect=target_record.identifiable_effect,
        )

    def decision_for(self, target_id: str) -> TargetDiagnosticDecisionV1:
        for decision in self.target_decisions:
            if decision.target_id == target_id:
                return decision
        raise KeyError(target_id)

    def deploy_or_exact_fallback(self, target_id: str, *, fallback: Any) -> Any:
        """Return a full identified effect or the exact caller-owned fallback.

        A `probe_then_reassess` disposition returns fallback now.  The caller may
        execute the selected registered intervention, append its response rows,
        and construct a new report.  This method never predicts a hypothetical
        post-probe outcome.
        """

        decision = self.decision_for(target_id)
        if decision.transport_permitted:
            return decision.identifiable_effect
        return fallback

    def descriptor(self) -> dict[str, object]:
        counts = {
            disposition.value: sum(
                decision.disposition is disposition
                for decision in self.target_decisions
            )
            for disposition in DiagnosticDisposition
        }
        return {
            "schema": EXPLAIN_TRANSPORT_PROBE_ABSTAIN_SCHEMA,
            "schema_version": EXPLAIN_TRANSPORT_PROBE_ABSTAIN_VERSION,
            "semantics": EXPLAIN_TRANSPORT_PROBE_ABSTAIN_SEMANTICS,
            "adequacy_certificate_id": self.adequacy_certificate.artifact_id,
            "transport_quotient_id": self.transport_quotient.artifact_id,
            "target_intervention_roster_id": self.target_intervention_roster_id,
            "candidate_roster_id": self.candidate_roster_id,
            "candidate_intervention_ids": dict(self.candidate_intervention_ids),
            "intervention_costs": dict(self.intervention_costs),
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "absolute_rank_tolerance": self.absolute_rank_tolerance,
            "cost_tolerance": self.cost_tolerance,
            "target_order": list(self.target_order),
            "target_decisions": [
                decision.to_record() for decision in self.target_decisions
            ],
            "disposition_counts": counts,
            "metadata": self.metadata,
            "claim_boundary": EXPLAIN_TRANSPORT_PROBE_ABSTAIN_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
