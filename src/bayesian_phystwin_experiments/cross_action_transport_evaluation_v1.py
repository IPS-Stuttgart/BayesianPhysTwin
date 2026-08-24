"""Session-level evaluation of sealed cross-action transport predictions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.cross_action_transport_contracts_v1 import (
    CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY,
    CROSS_ACTION_TRANSPORT_SCHEMA,
    CROSS_ACTION_TRANSPORT_VERSION,
    CrossActionProtocolV1,
    PredictionDisposition,
    TransportArm,
    TransportDecision,
    TransportScoreRowV1,
    _digest,
    _optional_labels,
)


@dataclass(frozen=True, slots=True)
class ArmTransportSummaryV1:
    """Session-level off-diagonal transport summary."""

    arm: TransportArm
    mean_gain: float
    gain_interval: tuple[float, float]
    win_sessions: int
    harmful_sessions: int
    harmful_fraction: float
    harmful_fraction_interval: tuple[float, float]
    selected_pairs: int
    fallback_pairs: int

    def descriptor(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "mean_gain": self.mean_gain,
            "gain_interval": list(self.gain_interval),
            "win_sessions": self.win_sessions,
            "harmful_sessions": self.harmful_sessions,
            "harmful_fraction": self.harmful_fraction,
            "harmful_fraction_interval": list(self.harmful_fraction_interval),
            "selected_pairs": self.selected_pairs,
            "fallback_pairs": self.fallback_pairs,
        }


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
    """Return a two-sided Wilson score interval for a Bernoulli fraction."""

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
class CrossActionTransportResultV1:
    """Evaluate one complete crossed-action score table."""

    protocol: CrossActionProtocolV1
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
        if not isinstance(self.protocol, CrossActionProtocolV1):
            raise TypeError("protocol must be a CrossActionProtocolV1")
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
            if prediction.source_action_id not in self.protocol.action_ids:
                raise ValueError("unregistered source action")
            if prediction.target_action_id not in self.protocol.action_ids:
                raise ValueError("unregistered target action")
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
        expected_pairs = set(self.protocol.action_pairs)
        expected_arms = set(self.protocol.registered_arms)
        for session in sessions:
            session_pairs = {
                (source, target)
                for object_session, source, target in by_pair
                if object_session == session
            }
            if session_pairs != expected_pairs:
                raise ValueError(
                    "every session must contain the complete action matrix"
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
                if len(
                    {
                        row.prediction.baseline_belief_id
                        for row in arm_rows.values()
                    }
                ) != 1:
                    raise ValueError("all arms for one pair must share one baseline")

        off_diagonal = [key for key in by_pair if key[1] != key[2]]
        gains: dict[TransportArm, dict[str, list[float]]] = {
            arm: {session: [] for session in sessions}
            for arm in self.protocol.registered_arms
            if arm is not TransportArm.PHYSICAL_FALLBACK
        }
        selected: dict[TransportArm, int] = {arm: 0 for arm in gains}
        fallback: dict[TransportArm, int] = {arm: 0 for arm in gains}
        for key in off_diagonal:
            arm_rows = by_pair[key]
            baseline_score = arm_rows[TransportArm.PHYSICAL_FALLBACK].proper_score
            for arm in gains:
                row = arm_rows[arm]
                gains[arm][key[0]].append(baseline_score - row.proper_score)
                if (
                    row.prediction.disposition
                    is PredictionDisposition.CANDIDATE_SELECTED
                ):
                    selected[arm] += 1
                elif row.prediction.disposition is PredictionDisposition.EXACT_FALLBACK:
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
            and comparator_interval[0]
            > self.protocol.minimum_comparator_contrast
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
            "schema": CROSS_ACTION_TRANSPORT_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_VERSION,
            "artifact_kind": "CrossActionTransportResultV1",
            "protocol_id": self.protocol.protocol_id,
            "target_accounting_id": self.target_accounting_id,
            "excluded_session_ids": list(self.excluded_session_ids),
            "score_row_ids": [row.score_row_id for row in self.score_rows],
            "independent_session_count": self.independent_session_count,
            "arm_summaries": [summary.descriptor() for summary in self.arm_summaries],
            "discrepancy_contrast": self.discrepancy_contrast,
            "discrepancy_contrast_interval": list(
                self.discrepancy_contrast_interval
            ),
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
            "discrepancy_contrast": self.discrepancy_contrast,
            "discrepancy_contrast_interval": list(
                self.discrepancy_contrast_interval
            ),
            "comparator_contrast": self.comparator_contrast,
            "comparator_contrast_interval": list(self.comparator_contrast_interval),
            "arm_summaries": [summary.descriptor() for summary in self.arm_summaries],
            "claim_boundary": CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY,
        }


__all__ = [
    "ArmTransportSummaryV1",
    "CrossActionTransportResultV1",
]
