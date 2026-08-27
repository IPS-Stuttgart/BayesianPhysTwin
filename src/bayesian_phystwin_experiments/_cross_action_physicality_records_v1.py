"""Content-addressed placebo construction and score records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments._cross_action_physicality_common_v1 import (
    CROSS_ACTION_PHYSICALITY_SCHEMA,
    CROSS_ACTION_PHYSICALITY_VERSION,
    BrokenMechanismPolicy,
    _commit,
    _digest,
    _finite,
    _literal,
    _optional_literal,
    _optional_nonzero_integer,
)
from bayesian_phystwin_experiments.cross_action_transport_contracts_v1 import (
    PredictionDisposition,
)


@dataclass(frozen=True, slots=True)
class PlaceboConstructionV1:
    """One sealed, policy-specific source construction for one session."""

    protocol_id: str
    information_order_id: str
    object_session_id: str
    source_execution_id: str
    target_execution_id: str
    source_action_id: str
    target_action_id: str
    policy: BrokenMechanismPolicy
    policy_implementation_id: str
    parent_prediction_id: str
    parent_selected_belief_id: str
    source_evidence_id: str
    construction_artifact_id: str
    donor_object_session_id: str | None = None
    donor_source_execution_id: str | None = None
    donor_action_id: str | None = None
    phase_shift_steps: int | None = None
    phase_period_steps: int | None = None
    identity_permutation_id: str | None = None
    identity_permutation_size: int | None = None
    identity_permutation_moved_count: int | None = None
    source_prefix_only: bool = True
    construction_sealed_before_target: bool = True
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "information_order_id",
            "policy_implementation_id",
            "parent_prediction_id",
            "parent_selected_belief_id",
            "source_evidence_id",
            "construction_artifact_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        for name in (
            "object_session_id",
            "source_execution_id",
            "target_execution_id",
            "source_action_id",
            "target_action_id",
        ):
            object.__setattr__(self, name, _literal(getattr(self, name), name=name))
        if self.source_execution_id == self.target_execution_id:
            raise ValueError("source and target executions must be distinct")
        if self.source_action_id == self.target_action_id:
            raise ValueError("placebos must preserve a genuinely cross-action query")
        if not isinstance(self.policy, BrokenMechanismPolicy):
            raise TypeError("policy must be a BrokenMechanismPolicy")

        donor_session = _optional_literal(
            self.donor_object_session_id, name="donor_object_session_id"
        )
        donor_execution = _optional_literal(
            self.donor_source_execution_id, name="donor_source_execution_id"
        )
        donor_action = _optional_literal(self.donor_action_id, name="donor_action_id")
        phase_shift = _optional_nonzero_integer(
            self.phase_shift_steps, name="phase_shift_steps"
        )
        phase_period = self.phase_period_steps
        if phase_period is not None:
            phase_period = genuine_integer(
                phase_period, name="phase_period_steps", minimum=2
            )
        permutation = self.identity_permutation_id
        if permutation is not None:
            permutation = _digest(permutation, name="identity_permutation_id")
        permutation_size = self.identity_permutation_size
        if permutation_size is not None:
            permutation_size = genuine_integer(
                permutation_size,
                name="identity_permutation_size",
                minimum=2,
            )
        moved_count = self.identity_permutation_moved_count
        if moved_count is not None:
            moved_count = genuine_integer(
                moved_count,
                name="identity_permutation_moved_count",
                minimum=2,
            )

        if self.policy in {
            BrokenMechanismPolicy.WRONG_SOURCE_ACTION,
            BrokenMechanismPolicy.WRONG_OBJECT_SESSION,
        }:
            if donor_session is None or donor_execution is None or donor_action is None:
                raise ValueError("donor placebos require a complete donor identity")
            if donor_session == self.object_session_id:
                raise ValueError("donor placebo must use another physical session")
            if donor_execution == self.target_execution_id:
                raise ValueError(
                    "a placebo cannot consume the held-out target execution"
                )
            if any(
                value is not None
                for value in (
                    phase_shift,
                    phase_period,
                    permutation,
                    permutation_size,
                    moved_count,
                )
            ):
                raise ValueError("donor placebos cannot carry shift or permutation")
        elif self.policy is BrokenMechanismPolicy.PHASE_SHIFTED_SOURCE:
            if phase_shift is None or phase_period is None:
                raise ValueError(
                    "phase_shifted_source requires shift and period identities"
                )
            if phase_shift % phase_period == 0:
                raise ValueError("phase shift must be nontrivial modulo the period")
            if any(
                value is not None
                for value in (
                    donor_session,
                    donor_execution,
                    donor_action,
                    permutation,
                    permutation_size,
                    moved_count,
                )
            ):
                raise ValueError(
                    "phase_shifted_source cannot carry donor or permutation"
                )
        else:
            if (
                permutation is None
                or permutation_size is None
                or moved_count is None
            ):
                raise ValueError(
                    "identity_permuted requires permutation identity, size, "
                    "and moved count"
                )
            if moved_count > permutation_size:
                raise ValueError("permutation moved count cannot exceed its size")
            if any(
                value is not None
                for value in (
                    donor_session,
                    donor_execution,
                    donor_action,
                    phase_shift,
                    phase_period,
                )
            ):
                raise ValueError("identity_permuted cannot carry donor or phase shift")

        source_prefix_only = genuine_boolean(
            self.source_prefix_only, name="source_prefix_only"
        )
        sealed = genuine_boolean(
            self.construction_sealed_before_target,
            name="construction_sealed_before_target",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used, name="target_outcomes_used"
        )
        if not source_prefix_only or not sealed or target_used:
            raise ValueError(
                "placebo construction must be source-prefix-only and sealed "
                "before target"
            )
        object.__setattr__(self, "donor_object_session_id", donor_session)
        object.__setattr__(self, "donor_source_execution_id", donor_execution)
        object.__setattr__(self, "donor_action_id", donor_action)
        object.__setattr__(self, "phase_shift_steps", phase_shift)
        object.__setattr__(self, "phase_period_steps", phase_period)
        object.__setattr__(self, "identity_permutation_id", permutation)
        object.__setattr__(self, "identity_permutation_size", permutation_size)
        object.__setattr__(
            self, "identity_permutation_moved_count", moved_count
        )
        object.__setattr__(self, "source_prefix_only", source_prefix_only)
        object.__setattr__(self, "construction_sealed_before_target", sealed)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="construction metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PHYSICALITY_SCHEMA,
            "schema_version": CROSS_ACTION_PHYSICALITY_VERSION,
            "artifact_kind": "PlaceboConstructionV1",
            "protocol_id": self.protocol_id,
            "information_order_id": self.information_order_id,
            "object_session_id": self.object_session_id,
            "source_execution_id": self.source_execution_id,
            "target_execution_id": self.target_execution_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "policy": self.policy.value,
            "policy_implementation_id": self.policy_implementation_id,
            "parent_prediction_id": self.parent_prediction_id,
            "parent_selected_belief_id": self.parent_selected_belief_id,
            "source_evidence_id": self.source_evidence_id,
            "construction_artifact_id": self.construction_artifact_id,
            "donor_object_session_id": self.donor_object_session_id,
            "donor_source_execution_id": self.donor_source_execution_id,
            "donor_action_id": self.donor_action_id,
            "phase_shift_steps": self.phase_shift_steps,
            "phase_period_steps": self.phase_period_steps,
            "identity_permutation_id": self.identity_permutation_id,
            "identity_permutation_size": self.identity_permutation_size,
            "identity_permutation_moved_count": (
                self.identity_permutation_moved_count
            ),
            "source_prefix_only": self.source_prefix_only,
            "construction_sealed_before_target": (
                self.construction_sealed_before_target
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def construction_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class SealedPlaceboPredictionV1:
    """One target-blind placebo prediction bound to a construction record."""

    construction: PlaceboConstructionV1
    baseline_belief_id: str
    candidate_belief_id: str
    selected_belief_id: str
    disposition: PredictionDisposition
    prediction_artifact_id: str
    prediction_batch_id: str
    commit_id: str
    prediction_sealed_before_target: bool
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.construction, PlaceboConstructionV1):
            raise TypeError("construction must be a PlaceboConstructionV1")
        for name in (
            "baseline_belief_id",
            "candidate_belief_id",
            "selected_belief_id",
            "prediction_artifact_id",
            "prediction_batch_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        object.__setattr__(self, "commit_id", _commit(self.commit_id, name="commit_id"))
        if not isinstance(self.disposition, PredictionDisposition):
            raise TypeError("disposition must be a PredictionDisposition")
        if self.disposition is PredictionDisposition.BASELINE_REFERENCE:
            raise ValueError("a placebo prediction cannot be a baseline reference")
        if self.disposition is PredictionDisposition.EXACT_FALLBACK:
            if self.selected_belief_id != self.baseline_belief_id:
                raise ValueError("exact fallback must select the exact baseline belief")
        elif self.selected_belief_id != self.candidate_belief_id:
            raise ValueError("candidate_selected must select the placebo candidate")

        sealed = genuine_boolean(
            self.prediction_sealed_before_target,
            name="prediction_sealed_before_target",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used, name="target_outcomes_used"
        )
        if not sealed or target_used:
            raise ValueError("placebo predictions must be sealed before target access")
        object.__setattr__(self, "prediction_sealed_before_target", sealed)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="prediction metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PHYSICALITY_SCHEMA,
            "schema_version": CROSS_ACTION_PHYSICALITY_VERSION,
            "artifact_kind": "SealedPlaceboPredictionV1",
            "construction_id": self.construction.construction_id,
            "baseline_belief_id": self.baseline_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "selected_belief_id": self.selected_belief_id,
            "disposition": self.disposition.value,
            "prediction_artifact_id": self.prediction_artifact_id,
            "prediction_batch_id": self.prediction_batch_id,
            "commit_id": self.commit_id,
            "prediction_sealed_before_target": self.prediction_sealed_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def prediction_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class PlaceboScoreRowV1:
    """One post-access score for one sealed placebo prediction."""

    prediction: SealedPlaceboPredictionV1
    target_outcome_id: str
    target_access_attestation_id: str
    scorer_id: str
    proper_score: float
    target_side_selection_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, SealedPlaceboPredictionV1):
            raise TypeError("prediction must be a SealedPlaceboPredictionV1")
        for name in (
            "target_outcome_id",
            "target_access_attestation_id",
            "scorer_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        object.__setattr__(
            self, "proper_score", _finite(self.proper_score, name="proper_score")
        )
        selected = genuine_boolean(
            self.target_side_selection_used, name="target_side_selection_used"
        )
        if selected:
            raise ValueError("target-side placebo selection is forbidden")
        object.__setattr__(self, "target_side_selection_used", selected)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="score-row metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PHYSICALITY_SCHEMA,
            "schema_version": CROSS_ACTION_PHYSICALITY_VERSION,
            "artifact_kind": "PlaceboScoreRowV1",
            "prediction_id": self.prediction.prediction_id,
            "target_outcome_id": self.target_outcome_id,
            "target_access_attestation_id": self.target_access_attestation_id,
            "scorer_id": self.scorer_id,
            "proper_score": self.proper_score,
            "target_side_selection_used": self.target_side_selection_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def score_row_id(self) -> str:
        return cast(str, content_id(self.descriptor()))
