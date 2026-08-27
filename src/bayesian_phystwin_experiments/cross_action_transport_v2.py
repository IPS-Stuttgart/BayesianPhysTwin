"""Session-specific held-out action transport evidence.

Version 1 intentionally assumes that every independent session contains the same
complete action roster.  Version 2 retains the same target-blind prediction and
session-clustered inference semantics while allowing a frozen action subset per
session.  This supports balanced incomplete-block physical acquisitions without
promoting action pairs to independent replicates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist
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
    SealedTransportPredictionV1,
    TransportArm,
    TransportDecision,
    TransportScoreRowV1,
    _arms,
    _digest,
    _finite,
    _labels,
    _optional_labels,
    _probability,
)
from bayesian_phystwin_experiments.cross_action_transport_evaluation_v1 import (
    ArmTransportSummaryV1,
)

CROSS_ACTION_TRANSPORT_V2_SCHEMA: Final = (
    "bayesian_phystwin.cross_action_transport_session_specific"
)
CROSS_ACTION_TRANSPORT_V2_VERSION: Final = 2
CROSS_ACTION_TRANSPORT_V2_SEMANTICS: Final = (
    "target-blind-session-specific-cross-action-prediction-and-session-bootstrap-v2"
)
CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY: Final = (
    "A positive result establishes bounded held-out action transport only for "
    "the exact frozen session-specific action roster, physical query, score, "
    "candidates, guards, software stack, and numerical environment. It does not "
    "establish a unique physical cause, arbitrary-action or arbitrary-object "
    "generalization, calibrated raw uncertainty, deployment safety, real Prob4D "
    "provider competence, completion of the Causal4D primary experiment, or "
    "deformable-object state of the art."
)


def _label(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


@dataclass(frozen=True, slots=True)
class SessionActionSetV2:
    """Frozen action subset observed within one independent physical session."""

    object_session_id: str
    action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_session_id",
            _label(self.object_session_id, name="object_session_id"),
        )
        actions = _labels(self.action_ids, name="action_ids")
        if len(actions) < 2:
            raise ValueError("each transport session requires at least two actions")
        object.__setattr__(self, "action_ids", actions)

    @property
    def action_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (source, target) for source in self.action_ids for target in self.action_ids
        )

    @property
    def off_diagonal_action_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (source, target)
            for source, target in self.action_pairs
            if source != target
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "object_session_id": self.object_session_id,
            "action_ids": list(self.action_ids),
            "action_pairs": [list(pair) for pair in self.action_pairs],
        }


@dataclass(frozen=True, slots=True)
class CrossActionProtocolV2:
    """Target-closed protocol with one frozen action subset per session."""

    development_roster_id: str
    calibration_roster_id: str
    target_roster_id: str
    acquisition_binding_id: str
    query_id: str
    query_jacobian_id: str
    identifiability_certificate_id: str
    nonlinear_closure_certificate_id: str
    score_definition_id: str
    grouping_rule_id: str
    interval_method_id: str
    target_access_policy_id: str
    model_stack_id: str
    numerical_environment_id: str
    technical_failure_policy_id: str
    session_action_sets: tuple[SessionActionSetV2, ...]
    registered_arms: tuple[TransportArm, ...]
    physical_transport_arm: TransportArm
    discrepancy_reference_arm: TransportArm
    matched_comparator_arm: TransportArm
    minimum_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    minimum_off_diagonal_gain: float
    minimum_discrepancy_contrast: float
    minimum_comparator_contrast: float
    maximum_harmful_session_fraction: float
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
            "acquisition_binding_id",
            "query_id",
            "query_jacobian_id",
            "identifiability_certificate_id",
            "nonlinear_closure_certificate_id",
            "score_definition_id",
            "grouping_rule_id",
            "interval_method_id",
            "target_access_policy_id",
            "model_stack_id",
            "numerical_environment_id",
            "technical_failure_policy_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))

        session_sets = tuple(self.session_action_sets)
        if len(session_sets) < 2 or any(
            not isinstance(value, SessionActionSetV2) for value in session_sets
        ):
            raise TypeError(
                "session_action_sets must contain at least two "
                "SessionActionSetV2 values"
            )
        session_sets = tuple(
            sorted(session_sets, key=lambda value: value.object_session_id)
        )
        session_ids = tuple(value.object_session_id for value in session_sets)
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("session_action_sets must not repeat a session")
        object.__setattr__(self, "session_action_sets", session_sets)

        arms = _arms(self.registered_arms)
        object.__setattr__(self, "registered_arms", arms)
        for name in (
            "physical_transport_arm",
            "discrepancy_reference_arm",
            "matched_comparator_arm",
        ):
            arm = getattr(self, name)
            if not isinstance(arm, TransportArm) or arm not in arms:
                raise ValueError(f"{name} must be one registered arm")
        if TransportArm.PHYSICAL_FALLBACK not in arms:
            raise ValueError("the physical fallback arm is mandatory")
        if self.physical_transport_arm in {
            TransportArm.PHYSICAL_FALLBACK,
            self.discrepancy_reference_arm,
            self.matched_comparator_arm,
        }:
            raise ValueError("physical_transport_arm must be a distinct candidate")
        if self.discrepancy_reference_arm is TransportArm.PHYSICAL_FALLBACK:
            raise ValueError("discrepancy_reference_arm cannot be the fallback")
        if self.matched_comparator_arm is TransportArm.PHYSICAL_FALLBACK:
            raise ValueError("matched_comparator_arm cannot be the fallback")
        if self.discrepancy_reference_arm is self.matched_comparator_arm:
            raise ValueError("discrepancy and comparator arms must be distinct")

        minimum_sessions = genuine_integer(
            self.minimum_sessions,
            name="minimum_sessions",
            minimum=2,
        )
        if minimum_sessions > len(session_sets):
            raise ValueError("minimum_sessions exceeds the frozen target roster")
        object.__setattr__(self, "minimum_sessions", minimum_sessions)
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
            "minimum_off_diagonal_gain",
            "minimum_discrepancy_contrast",
            "minimum_comparator_contrast",
            "harmful_gain_margin",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name, minimum=0.0),
            )
        object.__setattr__(
            self,
            "maximum_harmful_session_fraction",
            _finite(
                self.maximum_harmful_session_fraction,
                name="maximum_harmful_session_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )

        lower_is_better = genuine_boolean(
            self.lower_is_better,
            name="lower_is_better",
        )
        if not lower_is_better:
            raise ValueError("v2 requires a lower-is-better proper score")
        object.__setattr__(self, "lower_is_better", lower_is_better)
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
        if not frozen or not roster_frozen or target_used:
            raise ValueError("protocol must be frozen and target-outcome free")
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
        return tuple(value.object_session_id for value in self.session_action_sets)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    action
                    for session in self.session_action_sets
                    for action in session.action_ids
                }
            )
        )

    def session_action_set(self, object_session_id: str) -> SessionActionSetV2:
        session_id = _label(object_session_id, name="object_session_id")
        for session in self.session_action_sets:
            if session.object_session_id == session_id:
                return session
        raise KeyError(f"unregistered object session: {session_id}")

    def action_pairs_for_session(
        self,
        object_session_id: str,
    ) -> tuple[tuple[str, str], ...]:
        return self.session_action_set(object_session_id).action_pairs

    def off_diagonal_action_pairs_for_session(
        self,
        object_session_id: str,
    ) -> tuple[tuple[str, str], ...]:
        return self.session_action_set(object_session_id).off_diagonal_action_pairs

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_V2_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_V2_VERSION,
            "artifact_kind": "CrossActionProtocolV2",
            "semantics": CROSS_ACTION_TRANSPORT_V2_SEMANTICS,
            "development_roster_id": self.development_roster_id,
            "calibration_roster_id": self.calibration_roster_id,
            "target_roster_id": self.target_roster_id,
            "acquisition_binding_id": self.acquisition_binding_id,
            "query_id": self.query_id,
            "query_jacobian_id": self.query_jacobian_id,
            "identifiability_certificate_id": self.identifiability_certificate_id,
            "nonlinear_closure_certificate_id": self.nonlinear_closure_certificate_id,
            "score_definition_id": self.score_definition_id,
            "grouping_rule_id": self.grouping_rule_id,
            "interval_method_id": self.interval_method_id,
            "target_access_policy_id": self.target_access_policy_id,
            "model_stack_id": self.model_stack_id,
            "numerical_environment_id": self.numerical_environment_id,
            "technical_failure_policy_id": self.technical_failure_policy_id,
            "action_ids": list(self.action_ids),
            "target_session_ids": list(self.target_session_ids),
            "session_action_sets": [
                value.descriptor() for value in self.session_action_sets
            ],
            "registered_arms": [arm.value for arm in self.registered_arms],
            "physical_transport_arm": self.physical_transport_arm.value,
            "discrepancy_reference_arm": self.discrepancy_reference_arm.value,
            "matched_comparator_arm": self.matched_comparator_arm.value,
            "minimum_sessions": self.minimum_sessions,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "minimum_off_diagonal_gain": self.minimum_off_diagonal_gain,
            "minimum_discrepancy_contrast": self.minimum_discrepancy_contrast,
            "minimum_comparator_contrast": self.minimum_comparator_contrast,
            "maximum_harmful_session_fraction": (
                self.maximum_harmful_session_fraction
            ),
            "harmful_gain_margin": self.harmful_gain_margin,
            "lower_is_better": self.lower_is_better,
            "method_frozen_before_target": self.method_frozen_before_target,
            "roster_frozen_before_target": self.roster_frozen_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY,
        }

    @property
    def protocol_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


def _interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = np.mean(values[indices], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha], method="linear")
    return float(lower), float(upper)


def _wilson_interval(
    events: int,
    total: int,
    *,
    confidence: float,
) -> tuple[float, float]:
    if total <= 0 or events < 0 or events > total:
        raise ValueError("Wilson interval requires 0 <= events <= total")
    z = NormalDist().inv_cdf(0.5 + 0.5 * confidence)
    fraction = events / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (fraction + z_squared / (2.0 * total)) / denominator
    radius = (
        z
        / denominator
        * np.sqrt(
            fraction * (1.0 - fraction) / total
            + z_squared / (4.0 * total * total)
        )
    )
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _seed(base: int, stream: int) -> int:
    return int(np.random.SeedSequence([base, stream]).generate_state(1)[0])


@dataclass(frozen=True, slots=True)
class CrossActionTransportResultV2:
    """Evaluate one complete session-specific crossed-action score table."""

    protocol: CrossActionProtocolV2
    score_rows: tuple[TransportScoreRowV1, ...]
    target_accounting_id: str
    excluded_session_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    independent_session_count: int = field(init=False)
    arm_summaries: tuple[ArmTransportSummaryV1, ...] = field(init=False)
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
        rows = tuple(self.score_rows)
        if not rows or any(not isinstance(row, TransportScoreRowV1) for row in rows):
            raise TypeError("score_rows must contain TransportScoreRowV1 values")
        rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.prediction.object_session_id,
                    row.prediction.source_action_id,
                    row.prediction.target_action_id,
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

        by_pair: dict[
            tuple[str, str, str], dict[TransportArm, TransportScoreRowV1]
        ] = {}
        for row in rows:
            prediction = row.prediction
            key = (
                prediction.object_session_id,
                prediction.source_action_id,
                prediction.target_action_id,
            )
            arm_rows = by_pair.setdefault(key, {})
            if prediction.arm in arm_rows:
                raise ValueError("duplicate arm within an action pair")
            arm_rows[prediction.arm] = row

        sessions = tuple(sorted({key[0] for key in by_pair}))
        if set(sessions) & set(excluded_sessions):
            raise ValueError("scored and excluded target sessions must be disjoint")
        if set(sessions) | set(excluded_sessions) != set(
            self.protocol.target_session_ids
        ):
            raise ValueError("target session accounting must cover the frozen roster")

        expected_arms = set(self.protocol.registered_arms)
        for session in sessions:
            expected_pairs = set(self.protocol.action_pairs_for_session(session))
            session_pairs = {
                (source, target)
                for object_session, source, target in by_pair
                if object_session == session
            }
            if session_pairs != expected_pairs:
                raise ValueError(
                    "every session must contain its complete registered action matrix"
                )
            for source, target in expected_pairs:
                arm_rows = by_pair[(session, source, target)]
                if set(arm_rows) != expected_arms:
                    raise ValueError(
                        "every action pair must contain every registered arm"
                    )
                baseline_row = arm_rows[TransportArm.PHYSICAL_FALLBACK]
                if (
                    baseline_row.prediction.disposition
                    is not PredictionDisposition.BASELINE_REFERENCE
                ):
                    raise ValueError("physical fallback must be the baseline reference")
                if len({row.target_outcome_id for row in arm_rows.values()}) != 1:
                    raise ValueError("all arms for one pair must score the same target")
                if (
                    len(
                        {row.prediction.baseline_belief_id for row in arm_rows.values()}
                    )
                    != 1
                ):
                    raise ValueError("all arms for one pair must share one baseline")

        gains: dict[TransportArm, dict[str, list[float]]] = {
            arm: {session: [] for session in sessions}
            for arm in self.protocol.registered_arms
            if arm is not TransportArm.PHYSICAL_FALLBACK
        }
        selected: dict[TransportArm, int] = {arm: 0 for arm in gains}
        fallback: dict[TransportArm, int] = {arm: 0 for arm in gains}
        for session in sessions:
            for source, target in self.protocol.off_diagonal_action_pairs_for_session(
                session
            ):
                arm_rows = by_pair[(session, source, target)]
                baseline_score = arm_rows[
                    TransportArm.PHYSICAL_FALLBACK
                ].proper_score
                for arm in gains:
                    row = arm_rows[arm]
                    gains[arm][session].append(baseline_score - row.proper_score)
                    if (
                        row.prediction.disposition
                        is PredictionDisposition.CANDIDATE_SELECTED
                    ):
                        selected[arm] += 1
                    elif (
                        row.prediction.disposition
                        is PredictionDisposition.EXACT_FALLBACK
                    ):
                        fallback[arm] += 1

        session_gain = {
            arm: np.asarray(
                [np.mean(gains[arm][session]) for session in sessions],
                dtype=np.float64,
            )
            for arm in gains
        }
        summaries = []
        for stream, arm in enumerate(sorted(gains, key=lambda value: value.value)):
            values = session_gain[arm]
            harmful = values < -self.protocol.harmful_gain_margin
            summaries.append(
                ArmTransportSummaryV1(
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
                    selected_pairs=selected[arm],
                    fallback_pairs=fallback[arm],
                )
            )

        physical = session_gain[self.protocol.physical_transport_arm]
        discrepancy = session_gain[self.protocol.discrepancy_reference_arm]
        comparator = session_gain[self.protocol.matched_comparator_arm]
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
            if summary.arm is self.protocol.physical_transport_arm
        )
        if len(sessions) < self.protocol.minimum_sessions:
            decision = TransportDecision.INSUFFICIENT_SESSIONS
        elif (
            physical_summary.gain_interval[0]
            > self.protocol.minimum_off_diagonal_gain
            and discrepancy_interval[0]
            > self.protocol.minimum_discrepancy_contrast
            and comparator_interval[0] > self.protocol.minimum_comparator_contrast
            and physical_summary.harmful_fraction_interval[1]
            <= self.protocol.maximum_harmful_session_fraction
            and physical_summary.selected_pairs > 0
        ):
            decision = TransportDecision.SUPPORTED
        else:
            decision = TransportDecision.NOT_SUPPORTED

        metadata = frozen_finite_json_mapping(self.metadata, name="result metadata")
        object.__setattr__(self, "score_rows", rows)
        object.__setattr__(self, "target_accounting_id", target_accounting_id)
        object.__setattr__(self, "excluded_session_ids", excluded_sessions)
        object.__setattr__(self, "independent_session_count", len(sessions))
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
            "schema": CROSS_ACTION_TRANSPORT_V2_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_V2_VERSION,
            "artifact_kind": "CrossActionTransportResultV2",
            "protocol_id": self.protocol.protocol_id,
            "target_accounting_id": self.target_accounting_id,
            "excluded_session_ids": list(self.excluded_session_ids),
            "score_row_ids": [row.score_row_id for row in self.score_rows],
            "independent_session_count": self.independent_session_count,
            "arm_summaries": [summary.descriptor() for summary in self.arm_summaries],
            "discrepancy_contrast": self.discrepancy_contrast,
            "discrepancy_contrast_interval": list(self.discrepancy_contrast_interval),
            "comparator_contrast": self.comparator_contrast,
            "comparator_contrast_interval": list(self.comparator_contrast_interval),
            "decision": self.decision.value,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY,
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
            "discrepancy_contrast": self.discrepancy_contrast,
            "discrepancy_contrast_interval": list(self.discrepancy_contrast_interval),
            "comparator_contrast": self.comparator_contrast,
            "comparator_contrast_interval": list(self.comparator_contrast_interval),
            "arm_summaries": [summary.descriptor() for summary in self.arm_summaries],
            "claim_boundary": CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY,
        }


__all__ = [
    "CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY",
    "CROSS_ACTION_TRANSPORT_V2_SCHEMA",
    "CROSS_ACTION_TRANSPORT_V2_SEMANTICS",
    "CROSS_ACTION_TRANSPORT_V2_VERSION",
    "CrossActionProtocolV2",
    "CrossActionTransportResultV2",
    "SealedTransportPredictionV1",
    "SessionActionSetV2",
    "TransportScoreRowV1",
]
