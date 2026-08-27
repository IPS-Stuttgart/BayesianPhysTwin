"""Familywise evaluator for cross-action broken-mechanism controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments._cross_action_physicality_common_v1 import (
    CROSS_ACTION_PHYSICALITY_CLAIM_BOUNDARY,
    CROSS_ACTION_PHYSICALITY_SCHEMA,
    CROSS_ACTION_PHYSICALITY_VERSION,
    FAMILYWISE_METHOD,
    FAMILYWISE_METHOD_ID,
    REQUIRED_PLACEBO_POLICIES,
    BrokenMechanismPolicy,
    PhysicalityDecision,
)
from bayesian_phystwin_experiments._cross_action_physicality_protocol_v1 import (
    CrossActionPhysicalityProtocolV1,
)
from bayesian_phystwin_experiments._cross_action_physicality_records_v1 import (
    PlaceboScoreRowV1,
)
from bayesian_phystwin_experiments.cross_action_transport_contracts_v1 import (
    PredictionDisposition,
)
from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    CrossActionTransportResultV2,
    SparseTransportDecision,
)


@dataclass(frozen=True, slots=True)
class PlaceboContrastSummaryV1:
    """Session-level contrast against guarded physical transport."""

    policy: BrokenMechanismPolicy
    mean_contrast: float
    simultaneous_lower_bound: float
    win_sessions: int
    scored_sessions: int
    inherited_fallback_sessions: int

    def descriptor(self) -> dict[str, object]:
        return {
            "policy": self.policy.value,
            "mean_contrast": self.mean_contrast,
            "simultaneous_lower_bound": self.simultaneous_lower_bound,
            "win_sessions": self.win_sessions,
            "scored_sessions": self.scored_sessions,
            "inherited_fallback_sessions": self.inherited_fallback_sessions,
        }


def _simultaneous_lower_bounds(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> np.ndarray:
    """Paired Bonferroni percentile-bootstrap lower confidence bounds."""

    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("contrast values must be a nonempty session-by-policy matrix")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(replicates, values.shape[0]))
    means = np.mean(values[indices, :], axis=1)
    familywise_alpha = 1.0 - confidence
    marginal_alpha = familywise_alpha / values.shape[1]
    return np.asarray(
        np.quantile(means, marginal_alpha, axis=0, method="linear"),
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class CrossActionPhysicalityResultV1:
    """Evaluate a complete four-placebo table against one parent result."""

    protocol: CrossActionPhysicalityProtocolV1
    parent_result: CrossActionTransportResultV2
    score_rows: tuple[PlaceboScoreRowV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    independent_session_count: int = field(init=False)
    accepted_physical_session_count: int = field(init=False)
    inherited_fallback_session_count: int = field(init=False)
    placebo_summaries: tuple[PlaceboContrastSummaryV1, ...] = field(init=False)
    decision: PhysicalityDecision = field(init=False)
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, CrossActionPhysicalityProtocolV1):
            raise TypeError("protocol must be a CrossActionPhysicalityProtocolV1")
        if not isinstance(self.parent_result, CrossActionTransportResultV2):
            raise TypeError("parent_result must be a CrossActionTransportResultV2")
        if (
            self.parent_result.protocol.protocol_id
            != self.protocol.parent_protocol.protocol_id
        ):
            raise ValueError("parent result must bind the exact parent protocol")

        parent_rows = tuple(
            row
            for row in self.parent_result.score_rows
            if (
                row.prediction.arm
                is self.protocol.parent_protocol.physical_transport_arm
            )
        )
        parent_by_session = {
            row.prediction.object_session_id: row for row in parent_rows
        }
        if len(parent_by_session) != len(parent_rows):
            raise ValueError("parent result contains duplicate physical session rows")
        if len(parent_by_session) != self.parent_result.independent_session_count:
            raise ValueError("parent physical rows must cover every scored session")

        for row in parent_rows:
            if row.prediction.prediction_batch_id != self.protocol.prediction_batch_id:
                raise ValueError("parent prediction batch does not match the protocol")
            if row.prediction.commit_id != self.protocol.commit_id:
                raise ValueError(
                    "parent revision does not match the physicality protocol"
                )
            if row.scorer_id != self.protocol.scorer_id:
                raise ValueError(
                    "parent scorer does not match the physicality protocol"
                )

        parent_accepted_count = sum(
            row.prediction.disposition is PredictionDisposition.CANDIDATE_SELECTED
            for row in parent_rows
        )
        parent_fallback_count = sum(
            row.prediction.disposition is PredictionDisposition.EXACT_FALLBACK
            for row in parent_rows
        )

        rows = tuple(self.score_rows)
        if any(not isinstance(row, PlaceboScoreRowV1) for row in rows):
            raise TypeError("score_rows must contain PlaceboScoreRowV1 values")
        if (
            not rows
            and self.parent_result.decision is not SparseTransportDecision.SUPPORTED
        ):
            if self.parent_result.decision in {
                SparseTransportDecision.INSUFFICIENT_SESSIONS,
                SparseTransportDecision.INSUFFICIENT_ACCEPTED_UPDATES,
            }:
                decision = PhysicalityDecision.INSUFFICIENT
            else:
                decision = PhysicalityDecision.PARENT_NOT_SUPPORTED
            metadata = frozen_finite_json_mapping(
                self.metadata, name="result metadata"
            )
            object.__setattr__(self, "score_rows", ())
            object.__setattr__(
                self, "independent_session_count", len(parent_by_session)
            )
            object.__setattr__(
                self, "accepted_physical_session_count", parent_accepted_count
            )
            object.__setattr__(
                self,
                "inherited_fallback_session_count",
                parent_fallback_count,
            )
            object.__setattr__(self, "placebo_summaries", ())
            object.__setattr__(self, "decision", decision)
            object.__setattr__(self, "metadata", metadata)
            object.__setattr__(
                self, "result_id", cast(str, content_id(self.descriptor()))
            )
            return
        rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.prediction.construction.object_session_id,
                    row.prediction.construction.policy.value,
                ),
            )
        )
        if len({row.score_row_id for row in rows}) != len(rows):
            raise ValueError("placebo score rows must be unique")
        if len({row.prediction.construction.construction_id for row in rows}) != len(
            rows
        ):
            raise ValueError("placebo constructions must be unique")
        if len(
            {
                row.prediction.construction.construction_artifact_id
                for row in rows
            }
        ) != len(rows):
            raise ValueError("concrete construction artifacts must be unique")

        by_session: dict[
            str, dict[BrokenMechanismPolicy, PlaceboScoreRowV1]
        ] = {}
        pair_by_session = self.protocol.parent_protocol.pair_by_session
        for row in rows:
            prediction = row.prediction
            construction = prediction.construction
            if construction.protocol_id != self.protocol.protocol_id:
                raise ValueError("placebo construction must bind the exact protocol")
            pair = pair_by_session.get(construction.object_session_id)
            if pair is None:
                raise ValueError("placebo row uses an unregistered physical session")
            expected_identity = (
                pair.information_order_id,
                pair.source_execution_id,
                pair.target_execution_id,
                pair.source_action_id,
                pair.target_action_id,
            )
            observed_identity = (
                construction.information_order_id,
                construction.source_execution_id,
                construction.target_execution_id,
                construction.source_action_id,
                construction.target_action_id,
            )
            if observed_identity != expected_identity:
                raise ValueError("placebo must preserve the registered chronology")
            if construction.policy_implementation_id != (
                self.protocol.policy_implementation_id(construction.policy)
            ):
                raise ValueError("placebo policy implementation identity mismatch")
            parent_row = parent_by_session.get(construction.object_session_id)
            if parent_row is None:
                raise ValueError("placebos may score only parent-scored sessions")
            parent_prediction = parent_row.prediction

            expected_source_evidence_id = parent_prediction.source_evidence_id
            if construction.policy in {
                BrokenMechanismPolicy.WRONG_SOURCE_ACTION,
                BrokenMechanismPolicy.WRONG_OBJECT_SESSION,
            }:
                donor_session_id = cast(
                    str, construction.donor_object_session_id
                )
                donor_pair = pair_by_session.get(donor_session_id)
                if donor_pair is None:
                    raise ValueError("donor session must belong to the frozen roster")
                donor_parent_row = parent_by_session.get(donor_session_id)
                if donor_parent_row is None:
                    raise ValueError(
                        "donor session must have one sealed parent source prediction"
                    )
                expected_source_evidence_id = (
                    donor_parent_row.prediction.source_evidence_id
                )
                if (
                    construction.donor_source_execution_id
                    != donor_pair.source_execution_id
                    or construction.donor_action_id != donor_pair.source_action_id
                ):
                    raise ValueError(
                        "donor identity must match the registered donor source prefix"
                    )
                if (
                    construction.policy
                    is BrokenMechanismPolicy.WRONG_SOURCE_ACTION
                    and construction.donor_action_id == pair.source_action_id
                ):
                    raise ValueError(
                        "wrong_source_action must change the registered action profile"
                    )
                if (
                    construction.policy
                    is BrokenMechanismPolicy.WRONG_OBJECT_SESSION
                    and construction.donor_action_id != pair.source_action_id
                ):
                    raise ValueError(
                        "wrong_object_session must preserve the source action profile"
                    )
            if construction.source_evidence_id != expected_source_evidence_id:
                raise ValueError(
                    "placebo source evidence must match the registered source lineage"
                )
            if construction.parent_prediction_id != parent_prediction.prediction_id:
                raise ValueError(
                    "placebo must bind the exact guarded parent prediction"
                )
            if (
                construction.parent_selected_belief_id
                != parent_prediction.selected_belief_id
            ):
                raise ValueError("placebo must bind the parent selected belief")
            if prediction.baseline_belief_id != parent_prediction.baseline_belief_id:
                raise ValueError("parent and placebo must share the exact baseline")
            if prediction.prediction_batch_id != self.protocol.prediction_batch_id:
                raise ValueError("placebo prediction batch does not match the protocol")
            if prediction.commit_id != self.protocol.commit_id:
                raise ValueError("placebo revision does not match the protocol")
            if row.scorer_id != self.protocol.scorer_id:
                raise ValueError("placebo scorer does not match the protocol")
            if row.target_outcome_id != parent_row.target_outcome_id:
                raise ValueError(
                    "parent and placebo must score the same target outcome"
                )
            if (
                row.target_access_attestation_id
                != parent_row.target_access_attestation_id
            ):
                raise ValueError("parent and placebo must share one target attestation")

            if parent_prediction.disposition is PredictionDisposition.EXACT_FALLBACK:
                if prediction.disposition is not PredictionDisposition.EXACT_FALLBACK:
                    raise ValueError(
                        "parent fallback must be inherited by every placebo"
                    )
                if (
                    prediction.selected_belief_id
                    != parent_prediction.selected_belief_id
                ):
                    raise ValueError("inherited fallback must select the parent belief")
                if (
                    prediction.prediction_artifact_id
                    != parent_prediction.prediction_artifact_id
                ):
                    raise ValueError(
                        "inherited fallback must reuse the parent artifact"
                    )
                if row.proper_score != parent_row.proper_score:
                    raise ValueError(
                        "inherited fallback must receive the identical score"
                    )
            elif prediction.disposition is not PredictionDisposition.CANDIDATE_SELECTED:
                raise ValueError(
                    "an accepted parent requires a complete selected placebo candidate"
                )

            policy_rows = by_session.setdefault(construction.object_session_id, {})
            if construction.policy in policy_rows:
                raise ValueError("duplicate placebo policy within one session")
            policy_rows[construction.policy] = row

        if set(by_session) != set(parent_by_session):
            raise ValueError(
                "placebo sessions must exactly match every parent-scored session"
            )
        required = set(REQUIRED_PLACEBO_POLICIES)
        for policy_rows in by_session.values():
            if set(policy_rows) != required:
                raise ValueError(
                    "every parent-scored session requires all four placebos"
                )

        sessions = tuple(sorted(parent_by_session))
        accepted_count = parent_accepted_count
        fallback_count = parent_fallback_count

        summaries: list[PlaceboContrastSummaryV1] = []
        if sessions:
            contrast_values = np.asarray(
                [
                    [
                        by_session[session][policy].proper_score
                        - parent_by_session[session].proper_score
                        for policy in REQUIRED_PLACEBO_POLICIES
                    ]
                    for session in sessions
                ],
                dtype=np.float64,
            )
            lower_bounds = _simultaneous_lower_bounds(
                contrast_values,
                replicates=self.protocol.bootstrap_replicates,
                seed=self.protocol.bootstrap_seed,
                confidence=self.protocol.confidence_level,
            )
            for index, policy in enumerate(REQUIRED_PLACEBO_POLICIES):
                values = contrast_values[:, index]
                summaries.append(
                    PlaceboContrastSummaryV1(
                        policy=policy,
                        mean_contrast=float(np.mean(values)),
                        simultaneous_lower_bound=float(lower_bounds[index]),
                        win_sessions=int(np.count_nonzero(values > 0.0)),
                        scored_sessions=len(sessions),
                        inherited_fallback_sessions=fallback_count,
                    )
                )

        if self.parent_result.decision in {
            SparseTransportDecision.INSUFFICIENT_SESSIONS,
            SparseTransportDecision.INSUFFICIENT_ACCEPTED_UPDATES,
        }:
            decision = PhysicalityDecision.INSUFFICIENT
        elif self.parent_result.decision is not SparseTransportDecision.SUPPORTED:
            decision = PhysicalityDecision.PARENT_NOT_SUPPORTED
        elif len(sessions) < self.protocol.minimum_sessions or accepted_count == 0:
            decision = PhysicalityDecision.INSUFFICIENT
        elif all(
            summary.simultaneous_lower_bound
            > self.protocol.minimum_placebo_separation_margin
            for summary in summaries
        ):
            decision = PhysicalityDecision.SUPPORTED
        else:
            decision = PhysicalityDecision.NOT_SUPPORTED

        metadata = frozen_finite_json_mapping(self.metadata, name="result metadata")
        object.__setattr__(self, "score_rows", rows)
        object.__setattr__(self, "independent_session_count", len(sessions))
        object.__setattr__(self, "accepted_physical_session_count", accepted_count)
        object.__setattr__(self, "inherited_fallback_session_count", fallback_count)
        object.__setattr__(self, "placebo_summaries", tuple(summaries))
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "result_id", cast(str, content_id(self.descriptor())))

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PHYSICALITY_SCHEMA,
            "schema_version": CROSS_ACTION_PHYSICALITY_VERSION,
            "artifact_kind": "CrossActionPhysicalityResultV1",
            "protocol_id": self.protocol.protocol_id,
            "parent_result_id": self.parent_result.result_id,
            "score_row_ids": [row.score_row_id for row in self.score_rows],
            "independent_session_count": self.independent_session_count,
            "accepted_physical_session_count": self.accepted_physical_session_count,
            "inherited_fallback_session_count": (
                self.inherited_fallback_session_count
            ),
            "placebo_summaries": [
                summary.descriptor() for summary in self.placebo_summaries
            ],
            "familywise_method": FAMILYWISE_METHOD,
            "familywise_method_id": FAMILYWISE_METHOD_ID,
            "decision": self.decision.value,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_PHYSICALITY_CLAIM_BOUNDARY,
        }

    @property
    def supports_physicality(self) -> bool:
        return self.decision is PhysicalityDecision.SUPPORTED
