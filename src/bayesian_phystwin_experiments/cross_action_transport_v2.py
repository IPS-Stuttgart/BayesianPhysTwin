"""Chronological sparse-pair held-out action transport evidence.

Version 2 is intentionally specialized to the preregistered Causal4D
same-grasp acquisition: one chronological source execution and one held-out
target execution per independent physical session.  It does not weaken the
Cartesian-matrix semantics of cross-action transport v1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.cross_action_transport_contracts_v1 import (
    PredictionDisposition,
    TransportArm,
    TransportDecision,
    _arms,
    _commit,
    _digest,
    _finite,
    _optional_labels,
    _probability,
)
from bayesian_phystwin_experiments.cross_action_transport_evaluation_v1 import (
    _interval,
    _seed,
    _wilson_interval,
)

CROSS_ACTION_TRANSPORT_SCHEMA: Final = "bayesian_phystwin.cross_action_transport"
CROSS_ACTION_TRANSPORT_VERSION: Final = 2
CROSS_ACTION_TRANSPORT_SEMANTICS: Final = (
    "target-blind-chronological-sparse-pair-session-bootstrap-v2"
)
CAUSAL4D_SLOTH_MULTI_ACTION_V1_SHA256: Final = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY: Final = (
    "A positive result establishes bounded chronological held-out-action transport "
    "for the exact frozen Causal4D same-grasp session roster, source-to-target "
    "information order, physical query, score, candidate family, guard, software "
    "stack, and numerical environment. It does not establish a unique physical "
    "cause, arbitrary-action or unseen-object transfer, calibrated raw uncertainty, "
    "Prob4D provider competence, Causal4D intervention benefit, deployment safety, "
    "or deformable-object state of the art."
)

_REQUIRED_PRIMARY_ARMS: Final = frozenset(
    {
        TransportArm.PHYSICAL_FALLBACK,
        TransportArm.LAST_RESIDUAL,
        TransportArm.DISCREPANCY_ONLY,
        TransportArm.GUARDED_PHYSICAL,
    }
)


def _literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


@dataclass(frozen=True, slots=True)
class ChronologicalSessionPairV2:
    """One preregistered same-grasp source-to-target execution pair."""

    object_session_id: str
    source_execution_id: str
    target_execution_id: str
    source_action_id: str
    target_action_id: str
    contact_stratum_id: str
    information_order_id: str
    source_precedes_target: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "object_session_id",
            "source_execution_id",
            "target_execution_id",
            "source_action_id",
            "target_action_id",
        ):
            object.__setattr__(self, name, _literal(getattr(self, name), name=name))
        for name in ("contact_stratum_id", "information_order_id"):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.source_execution_id == self.target_execution_id:
            raise ValueError("source and target executions must be distinct")
        if self.source_action_id == self.target_action_id:
            raise ValueError("chronological transport requires a held-out action")
        precedes = genuine_boolean(
            self.source_precedes_target,
            name="source_precedes_target",
        )
        if not precedes:
            raise ValueError("v2 permits only the preregistered source-to-target order")
        object.__setattr__(self, "source_precedes_target", precedes)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="session-pair metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_VERSION,
            "artifact_kind": "ChronologicalSessionPairV2",
            "object_session_id": self.object_session_id,
            "source_execution_id": self.source_execution_id,
            "target_execution_id": self.target_execution_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "contact_stratum_id": self.contact_stratum_id,
            "information_order_id": self.information_order_id,
            "source_precedes_target": self.source_precedes_target,
            "metadata": plain_json(self.metadata),
        }

    @property
    def pair_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


def _session_pairs(
    values: Sequence[ChronologicalSessionPairV2],
) -> tuple[ChronologicalSessionPairV2, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("session_pairs must be a sequence")
    pairs = tuple(values)
    if not pairs or any(
        not isinstance(value, ChronologicalSessionPairV2) for value in pairs
    ):
        raise TypeError("session_pairs must contain ChronologicalSessionPairV2 values")
    ordered = tuple(sorted(pairs, key=lambda value: value.object_session_id))
    session_ids = [pair.object_session_id for pair in ordered]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("each physical session must appear exactly once")
    execution_ids = [
        execution_id
        for pair in ordered
        for execution_id in (pair.source_execution_id, pair.target_execution_id)
    ]
    if len(execution_ids) != len(set(execution_ids)):
        raise ValueError("each execution must appear exactly once in the frozen roster")
    if len({pair.pair_id for pair in ordered}) != len(ordered):
        raise ValueError("session-pair records must be unique")
    return ordered


@dataclass(frozen=True, slots=True)
class CrossActionProtocolV2:
    """Target-closed chronological sparse-pair protocol."""

    development_roster_id: str
    calibration_roster_id: str
    target_roster_id: str
    source_policy_id: str
    causal4d_design_sha256: str
    query_id: str
    query_jacobian_id: str
    score_definition_id: str
    grouping_rule_id: str
    interval_method_id: str
    target_access_policy_id: str
    model_stack_id: str
    numerical_environment_id: str
    technical_failure_policy_id: str
    candidate_family_id: str
    support_admission_id: str
    query_identifiability_id: str
    multi_action_identifiability_id: str
    nonlinear_closure_id: str
    guard_id: str
    session_pairs: tuple[ChronologicalSessionPairV2, ...]
    registered_arms: tuple[TransportArm, ...]
    minimum_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    minimum_gain: float
    minimum_discrepancy_contrast: float
    minimum_comparator_contrast: float
    maximum_harmful_session_fraction: float
    maximum_harmful_selected_fraction: float
    harmful_gain_margin: float = 0.0
    lower_is_better: bool = True
    method_frozen_before_target: bool = True
    roster_frozen_before_target: bool = True
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "development_roster_id",
            "calibration_roster_id",
            "target_roster_id",
            "source_policy_id",
            "causal4d_design_sha256",
            "query_id",
            "query_jacobian_id",
            "score_definition_id",
            "grouping_rule_id",
            "interval_method_id",
            "target_access_policy_id",
            "model_stack_id",
            "numerical_environment_id",
            "technical_failure_policy_id",
            "candidate_family_id",
            "support_admission_id",
            "query_identifiability_id",
            "multi_action_identifiability_id",
            "nonlinear_closure_id",
            "guard_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.causal4d_design_sha256 != CAUSAL4D_SLOTH_MULTI_ACTION_V1_SHA256:
            raise ValueError(
                "v2 must bind the frozen Causal4D sloth multi-action v1 design"
            )
        pairs = _session_pairs(self.session_pairs)
        object.__setattr__(self, "session_pairs", pairs)
        arms = _arms(self.registered_arms)
        if set(arms) != _REQUIRED_PRIMARY_ARMS:
            raise ValueError(
                "v2 claim-bearing execution requires exactly physical_fallback, "
                "last_residual, discrepancy_only, and guarded_physical"
            )
        object.__setattr__(self, "registered_arms", arms)
        object.__setattr__(
            self,
            "minimum_sessions",
            genuine_integer(self.minimum_sessions, name="minimum_sessions", minimum=2),
        )
        if self.minimum_sessions > len(pairs):
            raise ValueError("minimum_sessions cannot exceed the frozen session roster")
        object.__setattr__(
            self,
            "bootstrap_replicates",
            genuine_integer(
                self.bootstrap_replicates,
                name="bootstrap_replicates",
                minimum=100,
            ),
        )
        object.__setattr__(
            self,
            "bootstrap_seed",
            genuine_integer(self.bootstrap_seed, name="bootstrap_seed", minimum=0),
        )
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, name="confidence_level"),
        )
        for name in (
            "minimum_gain",
            "minimum_discrepancy_contrast",
            "minimum_comparator_contrast",
            "harmful_gain_margin",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name, minimum=0.0),
            )
        for name in (
            "maximum_harmful_session_fraction",
            "maximum_harmful_selected_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name, minimum=0.0, maximum=1.0),
            )
        lower_is_better = genuine_boolean(self.lower_is_better, name="lower_is_better")
        frozen = genuine_boolean(
            self.method_frozen_before_target,
            name="method_frozen_before_target",
        )
        roster_frozen = genuine_boolean(
            self.roster_frozen_before_target,
            name="roster_frozen_before_target",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not lower_is_better:
            raise ValueError("v2 requires a lower-is-better proper score")
        if not frozen or not roster_frozen or target_used:
            raise ValueError("protocol must be frozen and target-outcome free")
        object.__setattr__(self, "lower_is_better", lower_is_better)
        object.__setattr__(self, "method_frozen_before_target", frozen)
        object.__setattr__(self, "roster_frozen_before_target", roster_frozen)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="protocol metadata"),
        )

    @property
    def target_session_ids(self) -> tuple[str, ...]:
        return tuple(pair.object_session_id for pair in self.session_pairs)

    def pair_for_session(self, object_session_id: str) -> ChronologicalSessionPairV2:
        session_id = _literal(object_session_id, name="object_session_id")
        for pair in self.session_pairs:
            if pair.object_session_id == session_id:
                return pair
        raise KeyError(session_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_VERSION,
            "artifact_kind": "CrossActionProtocolV2",
            "semantics": CROSS_ACTION_TRANSPORT_SEMANTICS,
            "development_roster_id": self.development_roster_id,
            "calibration_roster_id": self.calibration_roster_id,
            "target_roster_id": self.target_roster_id,
            "source_policy_id": self.source_policy_id,
            "causal4d_design_sha256": self.causal4d_design_sha256,
            "query_id": self.query_id,
            "query_jacobian_id": self.query_jacobian_id,
            "score_definition_id": self.score_definition_id,
            "grouping_rule_id": self.grouping_rule_id,
            "interval_method_id": self.interval_method_id,
            "target_access_policy_id": self.target_access_policy_id,
            "model_stack_id": self.model_stack_id,
            "numerical_environment_id": self.numerical_environment_id,
            "technical_failure_policy_id": self.technical_failure_policy_id,
            "candidate_family_id": self.candidate_family_id,
            "support_admission_id": self.support_admission_id,
            "query_identifiability_id": self.query_identifiability_id,
            "multi_action_identifiability_id": self.multi_action_identifiability_id,
            "nonlinear_closure_id": self.nonlinear_closure_id,
            "guard_id": self.guard_id,
            "session_pair_ids": [pair.pair_id for pair in self.session_pairs],
            "registered_arms": [arm.value for arm in self.registered_arms],
            "minimum_sessions": self.minimum_sessions,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "minimum_gain": self.minimum_gain,
            "minimum_discrepancy_contrast": self.minimum_discrepancy_contrast,
            "minimum_comparator_contrast": self.minimum_comparator_contrast,
            "maximum_harmful_session_fraction": self.maximum_harmful_session_fraction,
            "maximum_harmful_selected_fraction": (
                self.maximum_harmful_selected_fraction
            ),
            "harmful_gain_margin": self.harmful_gain_margin,
            "lower_is_better": self.lower_is_better,
            "method_frozen_before_target": self.method_frozen_before_target,
            "roster_frozen_before_target": self.roster_frozen_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY,
        }

    @property
    def protocol_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class SealedTransportPredictionV2:
    """One target-blind prediction for one registered chronological session pair."""

    protocol_id: str
    registered_pair_id: str
    object_session_id: str
    source_execution_id: str
    target_execution_id: str
    source_action_id: str
    target_action_id: str
    arm: TransportArm
    baseline_belief_id: str
    candidate_belief_id: str | None
    selected_belief_id: str
    disposition: PredictionDisposition
    prediction_artifact_id: str
    source_evidence_id: str
    prediction_batch_id: str
    commit_id: str
    prediction_sealed_before_target: bool
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "registered_pair_id",
            "baseline_belief_id",
            "selected_belief_id",
            "prediction_artifact_id",
            "source_evidence_id",
            "prediction_batch_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.candidate_belief_id is not None:
            object.__setattr__(
                self,
                "candidate_belief_id",
                _digest(self.candidate_belief_id, name="candidate_belief_id"),
            )
        object.__setattr__(self, "commit_id", _commit(self.commit_id, name="commit_id"))
        for name in (
            "object_session_id",
            "source_execution_id",
            "target_execution_id",
            "source_action_id",
            "target_action_id",
        ):
            object.__setattr__(self, name, _literal(getattr(self, name), name=name))
        if not isinstance(self.arm, TransportArm):
            raise TypeError("arm must be a TransportArm")
        if not isinstance(self.disposition, PredictionDisposition):
            raise TypeError("disposition must be a PredictionDisposition")
        sealed = genuine_boolean(
            self.prediction_sealed_before_target,
            name="prediction_sealed_before_target",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not sealed or target_used:
            raise ValueError("predictions must be sealed before target access")
        if self.disposition is PredictionDisposition.BASELINE_REFERENCE:
            if self.arm is not TransportArm.PHYSICAL_FALLBACK:
                raise ValueError("baseline_reference is reserved for physical fallback")
            if self.candidate_belief_id is not None:
                raise ValueError("baseline_reference cannot bind a candidate belief")
            if self.selected_belief_id != self.baseline_belief_id:
                raise ValueError("baseline_reference must select the baseline")
        elif self.disposition is PredictionDisposition.EXACT_FALLBACK:
            if self.arm is TransportArm.PHYSICAL_FALLBACK:
                raise ValueError(
                    "physical fallback is a baseline reference, not a rejection"
                )
            if self.candidate_belief_id is None:
                raise ValueError("exact_fallback must retain the rejected candidate ID")
            if self.selected_belief_id != self.baseline_belief_id:
                raise ValueError("exact_fallback must select the exact baseline belief")
        else:
            if self.candidate_belief_id is None:
                raise ValueError("candidate_selected requires a candidate belief")
            if self.selected_belief_id != self.candidate_belief_id:
                raise ValueError("candidate_selected must select the candidate belief")
        object.__setattr__(self, "prediction_sealed_before_target", sealed)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="prediction metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_VERSION,
            "artifact_kind": "SealedTransportPredictionV2",
            "protocol_id": self.protocol_id,
            "registered_pair_id": self.registered_pair_id,
            "object_session_id": self.object_session_id,
            "source_execution_id": self.source_execution_id,
            "target_execution_id": self.target_execution_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "arm": self.arm.value,
            "baseline_belief_id": self.baseline_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "selected_belief_id": self.selected_belief_id,
            "disposition": self.disposition.value,
            "prediction_artifact_id": self.prediction_artifact_id,
            "source_evidence_id": self.source_evidence_id,
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
class TransportScoreRowV2:
    """One post-access score bound to a sealed chronological prediction."""

    prediction: SealedTransportPredictionV2
    target_outcome_id: str
    target_access_attestation_id: str
    scorer_id: str
    proper_score: float
    target_side_selection_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, SealedTransportPredictionV2):
            raise TypeError("prediction must be a SealedTransportPredictionV2")
        for name in (
            "target_outcome_id",
            "target_access_attestation_id",
            "scorer_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proper_score",
            _finite(self.proper_score, name="proper_score"),
        )
        selected = genuine_boolean(
            self.target_side_selection_used,
            name="target_side_selection_used",
        )
        if selected:
            raise ValueError("target-side model or threshold selection is forbidden")
        object.__setattr__(self, "target_side_selection_used", selected)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="score-row metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_VERSION,
            "artifact_kind": "TransportScoreRowV2",
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


@dataclass(frozen=True, slots=True)
class ArmTransportSummaryV2:
    """Independent-session summary for one non-fallback arm."""

    arm: TransportArm
    mean_gain: float
    gain_interval: tuple[float, float]
    win_sessions: int
    harmful_sessions: int
    harmful_fraction: float
    harmful_fraction_interval: tuple[float, float]
    selected_sessions: int
    fallback_sessions: int
    harmful_selected_sessions: int
    harmful_selected_fraction: float
    harmful_selected_fraction_interval: tuple[float, float]

    def descriptor(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "mean_gain": self.mean_gain,
            "gain_interval": list(self.gain_interval),
            "win_sessions": self.win_sessions,
            "harmful_sessions": self.harmful_sessions,
            "harmful_fraction": self.harmful_fraction,
            "harmful_fraction_interval": list(self.harmful_fraction_interval),
            "selected_sessions": self.selected_sessions,
            "fallback_sessions": self.fallback_sessions,
            "harmful_selected_sessions": self.harmful_selected_sessions,
            "harmful_selected_fraction": self.harmful_selected_fraction,
            "harmful_selected_fraction_interval": list(
                self.harmful_selected_fraction_interval
            ),
        }


@dataclass(frozen=True, slots=True)
class CrossActionTransportResultV2:
    """Evaluate the complete chronological sparse-pair score table."""

    protocol: CrossActionProtocolV2
    score_rows: tuple[TransportScoreRowV2, ...]
    target_accounting_id: str
    excluded_session_ids: tuple[str, ...] = ()
    technical_failure_session_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    independent_session_count: int = field(init=False)
    arm_summaries: tuple[ArmTransportSummaryV2, ...] = field(init=False)
    discrepancy_contrast: float = field(init=False)
    discrepancy_contrast_interval: tuple[float, float] = field(init=False)
    comparator_contrast: float = field(init=False)
    comparator_contrast_interval: tuple[float, float] = field(init=False)
    decision: TransportDecision = field(init=False)
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, CrossActionProtocolV2):
            raise TypeError("protocol must be a CrossActionProtocolV2")
        target_accounting_id = _digest(
            self.target_accounting_id,
            name="target_accounting_id",
        )
        excluded_sessions = _optional_labels(
            self.excluded_session_ids,
            name="excluded_session_ids",
        )
        technical_failures = _optional_labels(
            self.technical_failure_session_ids,
            name="technical_failure_session_ids",
        )
        if set(excluded_sessions) & set(technical_failures):
            raise ValueError("excluded and technical-failure sessions must be disjoint")

        rows = tuple(self.score_rows)
        if not rows or any(not isinstance(row, TransportScoreRowV2) for row in rows):
            raise TypeError("score_rows must contain TransportScoreRowV2 values")
        rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.prediction.object_session_id,
                    row.prediction.arm.value,
                ),
            )
        )
        if len({row.score_row_id for row in rows}) != len(rows):
            raise ValueError("score rows must be unique")
        if any(row.prediction.protocol_id != self.protocol.protocol_id for row in rows):
            raise ValueError("every prediction must bind the exact protocol")
        if len({row.prediction.prediction_batch_id for row in rows}) != 1:
            raise ValueError("one sealed prediction batch is required")
        if len({row.target_access_attestation_id for row in rows}) != 1:
            raise ValueError("one target-access attestation is required")
        if len({row.scorer_id for row in rows}) != 1:
            raise ValueError("one frozen scorer is required")
        if len({row.prediction.commit_id for row in rows}) != 1:
            raise ValueError("one exact BayesianPhysTwin revision is required")

        roster = {pair.object_session_id: pair for pair in self.protocol.session_pairs}
        by_session: dict[str, dict[TransportArm, TransportScoreRowV2]] = {}
        for row in rows:
            prediction = row.prediction
            pair = roster.get(prediction.object_session_id)
            if pair is None:
                raise ValueError(
                    "prediction references an unregistered physical session"
                )
            if (
                prediction.registered_pair_id != pair.pair_id
                or prediction.source_execution_id != pair.source_execution_id
                or prediction.target_execution_id != pair.target_execution_id
                or prediction.source_action_id != pair.source_action_id
                or prediction.target_action_id != pair.target_action_id
            ):
                raise ValueError(
                    "prediction must match the registered chronological "
                    "source-to-target pair"
                )
            arm_rows = by_session.setdefault(prediction.object_session_id, {})
            if prediction.arm in arm_rows:
                raise ValueError("duplicate arm within a physical session")
            arm_rows[prediction.arm] = row

        scored_sessions = tuple(sorted(by_session))
        if set(scored_sessions) & set(excluded_sessions):
            raise ValueError("scored and excluded target sessions must be disjoint")
        if set(scored_sessions) & set(technical_failures):
            raise ValueError("scored and technical-failure sessions must be disjoint")
        accounted = (
            set(scored_sessions) | set(excluded_sessions) | set(technical_failures)
        )
        if accounted != set(self.protocol.target_session_ids):
            raise ValueError("target accounting must cover the frozen session roster")

        expected_arms = set(self.protocol.registered_arms)
        for session in scored_sessions:
            arm_rows = by_session[session]
            if set(arm_rows) != expected_arms:
                raise ValueError(
                    "every scored session must contain every registered arm"
                )
            baseline_row = arm_rows[TransportArm.PHYSICAL_FALLBACK]
            if (
                baseline_row.prediction.disposition
                is not PredictionDisposition.BASELINE_REFERENCE
            ):
                raise ValueError("physical fallback must be the baseline reference")
            if len({row.target_outcome_id for row in arm_rows.values()}) != 1:
                raise ValueError("all arms for one session must score the same target")
            if (
                len({row.prediction.baseline_belief_id for row in arm_rows.values()})
                != 1
            ):
                raise ValueError("all arms for one session must share one baseline")
            for arm, row in arm_rows.items():
                if (
                    arm is not TransportArm.PHYSICAL_FALLBACK
                    and row.prediction.disposition
                    is PredictionDisposition.EXACT_FALLBACK
                    and row.proper_score != baseline_row.proper_score
                ):
                    raise ValueError(
                        "exact-fallback predictions must score identically to fallback"
                    )

        gains: dict[TransportArm, np.ndarray] = {}
        dispositions: dict[TransportArm, tuple[PredictionDisposition, ...]] = {}
        for arm in self.protocol.registered_arms:
            if arm is TransportArm.PHYSICAL_FALLBACK:
                continue
            arm_gains = []
            arm_dispositions = []
            for session in scored_sessions:
                arm_rows = by_session[session]
                baseline_score = arm_rows[TransportArm.PHYSICAL_FALLBACK].proper_score
                row = arm_rows[arm]
                arm_gains.append(baseline_score - row.proper_score)
                arm_dispositions.append(row.prediction.disposition)
            gains[arm] = np.asarray(arm_gains, dtype=np.float64)
            dispositions[arm] = tuple(arm_dispositions)

        summaries = []
        for stream, arm in enumerate(sorted(gains, key=lambda value: value.value)):
            values = gains[arm]
            harmful = values < -self.protocol.harmful_gain_margin
            selected_mask = np.asarray(
                [
                    disposition is PredictionDisposition.CANDIDATE_SELECTED
                    for disposition in dispositions[arm]
                ],
                dtype=bool,
            )
            fallback_mask = np.asarray(
                [
                    disposition is PredictionDisposition.EXACT_FALLBACK
                    for disposition in dispositions[arm]
                ],
                dtype=bool,
            )
            selected_count = int(np.count_nonzero(selected_mask))
            harmful_selected_count = int(np.count_nonzero(harmful & selected_mask))
            if selected_count:
                harmful_selected_fraction = harmful_selected_count / selected_count
                harmful_selected_interval = _wilson_interval(
                    harmful_selected_count,
                    selected_count,
                    confidence=self.protocol.confidence_level,
                )
            else:
                harmful_selected_fraction = 0.0
                harmful_selected_interval = (0.0, 1.0)
            summaries.append(
                ArmTransportSummaryV2(
                    arm=arm,
                    mean_gain=float(np.mean(values)),
                    gain_interval=_interval(
                        values,
                        replicates=self.protocol.bootstrap_replicates,
                        seed=_seed(self.protocol.bootstrap_seed, stream),
                        confidence=self.protocol.confidence_level,
                    ),
                    win_sessions=int(np.count_nonzero(values > 0.0)),
                    harmful_sessions=int(np.count_nonzero(harmful)),
                    harmful_fraction=float(np.mean(harmful)),
                    harmful_fraction_interval=_wilson_interval(
                        int(np.count_nonzero(harmful)),
                        values.size,
                        confidence=self.protocol.confidence_level,
                    ),
                    selected_sessions=selected_count,
                    fallback_sessions=int(np.count_nonzero(fallback_mask)),
                    harmful_selected_sessions=harmful_selected_count,
                    harmful_selected_fraction=float(harmful_selected_fraction),
                    harmful_selected_fraction_interval=harmful_selected_interval,
                )
            )

        physical = gains[TransportArm.GUARDED_PHYSICAL]
        discrepancy = gains[TransportArm.DISCREPANCY_ONLY]
        comparator = gains[TransportArm.LAST_RESIDUAL]
        discrepancy_contrast_values = physical - discrepancy
        comparator_contrast_values = physical - comparator
        discrepancy_interval = _interval(
            discrepancy_contrast_values,
            replicates=self.protocol.bootstrap_replicates,
            seed=_seed(self.protocol.bootstrap_seed, 2001),
            confidence=self.protocol.confidence_level,
        )
        comparator_interval = _interval(
            comparator_contrast_values,
            replicates=self.protocol.bootstrap_replicates,
            seed=_seed(self.protocol.bootstrap_seed, 2002),
            confidence=self.protocol.confidence_level,
        )
        physical_summary = next(
            summary
            for summary in summaries
            if summary.arm is TransportArm.GUARDED_PHYSICAL
        )
        if len(scored_sessions) < self.protocol.minimum_sessions:
            decision = TransportDecision.INSUFFICIENT_SESSIONS
        elif (
            physical_summary.gain_interval[0] > self.protocol.minimum_gain
            and discrepancy_interval[0] > self.protocol.minimum_discrepancy_contrast
            and comparator_interval[0] > self.protocol.minimum_comparator_contrast
            and physical_summary.harmful_fraction_interval[1]
            <= self.protocol.maximum_harmful_session_fraction
            and physical_summary.harmful_selected_fraction_interval[1]
            <= self.protocol.maximum_harmful_selected_fraction
            and physical_summary.selected_sessions > 0
        ):
            decision = TransportDecision.SUPPORTED
        else:
            decision = TransportDecision.NOT_SUPPORTED

        metadata = frozen_finite_json_mapping(self.metadata, name="result metadata")
        object.__setattr__(self, "score_rows", rows)
        object.__setattr__(self, "target_accounting_id", target_accounting_id)
        object.__setattr__(self, "excluded_session_ids", excluded_sessions)
        object.__setattr__(self, "technical_failure_session_ids", technical_failures)
        object.__setattr__(self, "independent_session_count", len(scored_sessions))
        object.__setattr__(self, "arm_summaries", tuple(summaries))
        object.__setattr__(
            self,
            "discrepancy_contrast",
            float(np.mean(discrepancy_contrast_values)),
        )
        object.__setattr__(self, "discrepancy_contrast_interval", discrepancy_interval)
        object.__setattr__(
            self,
            "comparator_contrast",
            float(np.mean(comparator_contrast_values)),
        )
        object.__setattr__(self, "comparator_contrast_interval", comparator_interval)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "result_id", cast(str, content_id(self.descriptor())))

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_VERSION,
            "artifact_kind": "CrossActionTransportResultV2",
            "protocol_id": self.protocol.protocol_id,
            "target_accounting_id": self.target_accounting_id,
            "excluded_session_ids": list(self.excluded_session_ids),
            "technical_failure_session_ids": list(
                self.technical_failure_session_ids
            ),
            "score_row_ids": [row.score_row_id for row in self.score_rows],
            "independent_session_count": self.independent_session_count,
            "arm_summaries": [summary.descriptor() for summary in self.arm_summaries],
            "discrepancy_contrast": self.discrepancy_contrast,
            "discrepancy_contrast_interval": list(self.discrepancy_contrast_interval),
            "comparator_contrast": self.comparator_contrast,
            "comparator_contrast_interval": list(self.comparator_contrast_interval),
            "decision": self.decision.value,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY,
        }

    @property
    def supports_physical_transport(self) -> bool:
        return self.decision is TransportDecision.SUPPORTED

    def summary(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "protocol_id": self.protocol.protocol_id,
            "decision": self.decision.value,
            "supports_physical_transport": self.supports_physical_transport,
            "independent_session_count": self.independent_session_count,
            "excluded_session_ids": list(self.excluded_session_ids),
            "technical_failure_session_ids": list(
                self.technical_failure_session_ids
            ),
            "discrepancy_contrast": self.discrepancy_contrast,
            "discrepancy_contrast_interval": list(self.discrepancy_contrast_interval),
            "comparator_contrast": self.comparator_contrast,
            "comparator_contrast_interval": list(self.comparator_contrast_interval),
            "arm_summaries": [summary.descriptor() for summary in self.arm_summaries],
            "claim_boundary": CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY,
        }


__all__ = [
    "ArmTransportSummaryV2",
    "CAUSAL4D_SLOTH_MULTI_ACTION_V1_SHA256",
    "CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY",
    "CROSS_ACTION_TRANSPORT_SCHEMA",
    "CROSS_ACTION_TRANSPORT_SEMANTICS",
    "CROSS_ACTION_TRANSPORT_VERSION",
    "ChronologicalSessionPairV2",
    "CrossActionProtocolV2",
    "CrossActionTransportResultV2",
    "PredictionDisposition",
    "SealedTransportPredictionV2",
    "TransportArm",
    "TransportDecision",
    "TransportScoreRowV2",
]
