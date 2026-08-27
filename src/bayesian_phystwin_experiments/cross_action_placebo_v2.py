"""Chronological broken-mechanism certificate for cross-action transport v2.

This target-closed supplement asks whether a supported guarded physical update
beats four preregistered controls that preserve superficial prediction structure
while breaking the registered physical relation. Complete physical sessions,
not frames, points, action labels, donors, or posterior samples, are the
independent statistical units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    CrossActionProtocolV2,
    CrossActionTransportResultV2,
    PredictionDisposition,
    SparseTransportDecision,
    TransportArm,
)

CROSS_ACTION_PLACEBO_V2_SCHEMA: Final = "bayesian_phystwin.cross_action_placebo"
CROSS_ACTION_PLACEBO_V2_VERSION: Final = 2
CROSS_ACTION_PLACEBO_V2_SEMANTICS: Final = (
    "target-blind-chronological-broken-mechanism-physicality-v2"
)
CROSS_ACTION_PLACEBO_V2_FAMILYWISE_METHOD: Final = (
    "paired-session-max-deviation-bootstrap-v1"
)
CROSS_ACTION_PLACEBO_V2_CLAIM_BOUNDARY: Final = (
    "A positive result establishes bounded separation of the registered guarded "
    "physical transport prediction from four preregistered broken-relation "
    "controls on the exact chronological physical-session roster, scorer, "
    "software revision, and target-access boundary. It does not establish a "
    "unique physical cause, reverse-direction reuse, arbitrary-action or "
    "unseen-object generalization, calibrated raw covariance, real Prob4D "
    "provider competence, Causal4D intervention benefit, deployment safety, or "
    "state of the art."
)


class PlaceboArmV2(str, Enum):
    """Guarded parent prediction and preregistered broken-relation controls."""

    GUARDED_PHYSICAL = "guarded_physical"
    WRONG_SOURCE_ACTION = "wrong_source_action"
    WRONG_OBJECT_SESSION = "wrong_object_session"
    PHASE_SHIFTED_SOURCE = "phase_shifted_source"
    IDENTITY_PERMUTED = "identity_permuted"

    @property
    def is_placebo(self) -> bool:
        return self is not PlaceboArmV2.GUARDED_PHYSICAL


PLACEBO_ARMS_V2: Final = (
    PlaceboArmV2.WRONG_SOURCE_ACTION,
    PlaceboArmV2.WRONG_OBJECT_SESSION,
    PlaceboArmV2.PHASE_SHIFTED_SOURCE,
    PlaceboArmV2.IDENTITY_PERMUTED,
)
ALL_PLACEBO_CERTIFICATE_ARMS_V2: Final = (
    PlaceboArmV2.GUARDED_PHYSICAL,
    *PLACEBO_ARMS_V2,
)


class PlaceboDecisionV2(str, Enum):
    """Conjunctive physicality decision for the chronological certificate."""

    SUPPORTED = "physicality_supported"
    NOT_SUPPORTED = "physicality_not_supported"
    PARENT_NOT_SUPPORTED = "parent_transport_not_supported"
    PARENT_INSUFFICIENT = "parent_transport_insufficient"
    INSUFFICIENT_SESSIONS = "insufficient_independent_sessions"


def _digest(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={64}))


def _commit(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={40, 64}))


def _literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _optional_literal(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _literal(value, name=name)


def _optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _digest(value, name=name)


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite(value, name=name, minimum=0.0, maximum=1.0)
    if result in {0.0, 1.0}:
        raise ValueError(f"{name} must be strictly between zero and one")
    return result


def _signed_nonzero_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value == 0:
        raise ValueError(f"{name} must be a nonzero literal integer")
    return value


def _optional_integer(
    value: object,
    *,
    name: str,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    return genuine_integer(value, name=name, minimum=minimum)


def _policy_ids(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("placebo_policy_ids must be a mapping")
    expected = {arm.value for arm in PLACEBO_ARMS_V2}
    if set(values) != expected:
        raise ValueError(
            "placebo_policy_ids must contain exactly the four registered controls"
        )
    return frozen_finite_json_mapping(
        {
            arm: _digest(values[arm], name=f"placebo_policy_ids[{arm!r}]")
            for arm in sorted(expected)
        },
        name="placebo policy IDs",
    )


def _marginal_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    vector = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, vector.size, size=(replicates, vector.size))
    means = np.mean(vector[indices], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha], method="linear")
    return float(lower), float(upper)


def _simultaneous_lower_bounds(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[np.ndarray, float]:
    """Joint one-sided lower bounds using a paired-session max deviation."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(
            "simultaneous bootstrap requires at least two sessions and one arm"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("simultaneous bootstrap values must be finite")
    observed = np.mean(matrix, axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[0], size=(replicates, matrix.shape[0]))
    bootstrap_means = np.mean(matrix[indices], axis=1)
    maximum_lower_error = np.max(observed[None, :] - bootstrap_means, axis=1)
    critical_value = max(
        0.0,
        float(np.quantile(maximum_lower_error, confidence, method="linear")),
    )
    return observed - critical_value, critical_value


@dataclass(frozen=True, slots=True)
class ChronologicalPlaceboConstructionV2:
    """One concrete source-prefix-only broken-relation construction."""

    object_session_id: str
    information_order_id: str
    source_execution_id: str
    target_execution_id: str
    source_action_id: str
    target_action_id: str
    arm: PlaceboArmV2
    policy_id: str
    source_prefix_artifact_id: str
    construction_artifact_id: str
    constructor_commit_id: str
    donor_object_session_id: str | None = None
    donor_source_execution_id: str | None = None
    donor_source_action_id: str | None = None
    phase_shift_steps: int | None = None
    permutation_artifact_id: str | None = None
    permutation_size: int | None = None
    permutation_fixed_point_count: int | None = None
    source_prefix_only: bool = True
    constructed_before_target: bool = True
    target_outcomes_used: bool = False
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
        if self.source_execution_id == self.target_execution_id:
            raise ValueError("source and target executions must be distinct")
        if self.source_action_id == self.target_action_id:
            raise ValueError("placebo v2 requires a genuinely cross-action pair")
        object.__setattr__(
            self,
            "information_order_id",
            _digest(self.information_order_id, name="information_order_id"),
        )
        if not isinstance(self.arm, PlaceboArmV2) or not self.arm.is_placebo:
            raise TypeError("construction arm must be one placebo control")
        object.__setattr__(self, "policy_id", _digest(self.policy_id, name="policy_id"))
        object.__setattr__(
            self,
            "source_prefix_artifact_id",
            _digest(self.source_prefix_artifact_id, name="source_prefix_artifact_id"),
        )
        object.__setattr__(
            self,
            "construction_artifact_id",
            _digest(self.construction_artifact_id, name="construction_artifact_id"),
        )
        object.__setattr__(
            self,
            "constructor_commit_id",
            _commit(self.constructor_commit_id, name="constructor_commit_id"),
        )
        for name in (
            "donor_object_session_id",
            "donor_source_execution_id",
            "donor_source_action_id",
        ):
            object.__setattr__(
                self,
                name,
                _optional_literal(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "permutation_artifact_id",
            _optional_digest(
                self.permutation_artifact_id,
                name="permutation_artifact_id",
            ),
        )
        object.__setattr__(
            self,
            "permutation_size",
            _optional_integer(
                self.permutation_size, name="permutation_size", minimum=2
            ),
        )
        object.__setattr__(
            self,
            "permutation_fixed_point_count",
            _optional_integer(
                self.permutation_fixed_point_count,
                name="permutation_fixed_point_count",
                minimum=0,
            ),
        )
        for name in (
            "source_prefix_only",
            "constructed_before_target",
            "target_outcomes_used",
        ):
            object.__setattr__(
                self, name, genuine_boolean(getattr(self, name), name=name)
            )
        if (
            not self.source_prefix_only
            or not self.constructed_before_target
            or self.target_outcomes_used
        ):
            raise ValueError(
                "placebo constructions must be source-prefix-only, sealed, and "
                "target-outcome free"
            )

        donor_values = (
            self.donor_object_session_id,
            self.donor_source_execution_id,
            self.donor_source_action_id,
        )
        if self.arm is PlaceboArmV2.WRONG_SOURCE_ACTION:
            if any(value is None for value in donor_values):
                raise ValueError(
                    "wrong_source_action requires a complete donor identity"
                )
            if self.donor_source_action_id == self.source_action_id:
                raise ValueError("wrong_source_action must change the source action")
            if any(
                value is not None
                for value in (
                    self.phase_shift_steps,
                    self.permutation_artifact_id,
                    self.permutation_size,
                    self.permutation_fixed_point_count,
                )
            ):
                raise ValueError("wrong_source_action cannot bind shift or permutation")
        elif self.arm is PlaceboArmV2.WRONG_OBJECT_SESSION:
            if any(value is None for value in donor_values):
                raise ValueError(
                    "wrong_object_session requires a complete donor identity"
                )
            if self.donor_object_session_id == self.object_session_id:
                raise ValueError(
                    "wrong_object_session must change the physical session"
                )
            if self.donor_source_action_id != self.source_action_id:
                raise ValueError(
                    "wrong_object_session must preserve the source action profile"
                )
            if any(
                value is not None
                for value in (
                    self.phase_shift_steps,
                    self.permutation_artifact_id,
                    self.permutation_size,
                    self.permutation_fixed_point_count,
                )
            ):
                raise ValueError(
                    "wrong_object_session cannot bind shift or permutation"
                )
        elif self.arm is PlaceboArmV2.PHASE_SHIFTED_SOURCE:
            if any(value is not None for value in donor_values):
                raise ValueError("phase_shifted_source cannot bind a donor")
            object.__setattr__(
                self,
                "phase_shift_steps",
                _signed_nonzero_integer(
                    self.phase_shift_steps,
                    name="phase_shift_steps",
                ),
            )
            if any(
                value is not None
                for value in (
                    self.permutation_artifact_id,
                    self.permutation_size,
                    self.permutation_fixed_point_count,
                )
            ):
                raise ValueError("phase_shifted_source cannot bind a permutation")
        else:
            if any(value is not None for value in donor_values):
                raise ValueError("identity_permuted cannot bind a donor")
            if self.phase_shift_steps is not None:
                raise ValueError("identity_permuted cannot bind a phase shift")
            if (
                self.permutation_artifact_id is None
                or self.permutation_size is None
                or self.permutation_fixed_point_count is None
            ):
                raise ValueError(
                    "identity_permuted requires a complete permutation identity"
                )
            if self.permutation_fixed_point_count >= self.permutation_size:
                raise ValueError(
                    "identity_permuted must encode a non-identity permutation"
                )

        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="construction metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "object_session_id": self.object_session_id,
            "information_order_id": self.information_order_id,
            "source_execution_id": self.source_execution_id,
            "target_execution_id": self.target_execution_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "arm": self.arm.value,
            "policy_id": self.policy_id,
            "source_prefix_artifact_id": self.source_prefix_artifact_id,
            "construction_artifact_id": self.construction_artifact_id,
            "constructor_commit_id": self.constructor_commit_id,
            "donor_object_session_id": self.donor_object_session_id,
            "donor_source_execution_id": self.donor_source_execution_id,
            "donor_source_action_id": self.donor_source_action_id,
            "phase_shift_steps": self.phase_shift_steps,
            "permutation_artifact_id": self.permutation_artifact_id,
            "permutation_size": self.permutation_size,
            "permutation_fixed_point_count": self.permutation_fixed_point_count,
            "source_prefix_only": self.source_prefix_only,
            "constructed_before_target": self.constructed_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def construction_id(self) -> str:
        return cast(str, content_id({"chronological_placebo_v2": self.descriptor()}))


@dataclass(frozen=True, slots=True)
class CrossActionPlaceboProtocolV2:
    """Target-closed physicality certificate bound to one transport v2 protocol."""

    parent_transport_protocol: CrossActionProtocolV2
    placebo_policy_ids: Mapping[str, str]
    constructions: Sequence[ChronologicalPlaceboConstructionV2]
    minimum_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    familywise_confidence_level: float
    minimum_placebo_separation: float
    familywise_method: str = CROSS_ACTION_PLACEBO_V2_FAMILYWISE_METHOD
    method_frozen_before_target: bool = True
    roster_frozen_before_target: bool = True
    constructions_sealed_before_target: bool = True
    target_outcomes_used_for_selection: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.parent_transport_protocol, CrossActionProtocolV2):
            raise TypeError("parent_transport_protocol must be CrossActionProtocolV2")
        parent = self.parent_transport_protocol
        if parent.physical_transport_arm is not TransportArm.GUARDED_PHYSICAL:
            raise ValueError("placebo v2 requires the guarded_physical parent arm")
        policies = _policy_ids(self.placebo_policy_ids)
        object.__setattr__(self, "placebo_policy_ids", policies)

        constructions = tuple(self.constructions)
        if any(
            not isinstance(value, ChronologicalPlaceboConstructionV2)
            for value in constructions
        ):
            raise TypeError(
                "constructions must contain ChronologicalPlaceboConstructionV2 values"
            )
        constructions = tuple(
            sorted(
                constructions,
                key=lambda value: (value.object_session_id, value.arm.value),
            )
        )
        if len({value.construction_id for value in constructions}) != len(
            constructions
        ):
            raise ValueError("placebo construction records must be unique")

        expected_keys = {
            (session_id, arm)
            for session_id in parent.target_session_ids
            for arm in PLACEBO_ARMS_V2
        }
        observed_keys = {
            (construction.object_session_id, construction.arm)
            for construction in constructions
        }
        if observed_keys != expected_keys or len(constructions) != len(expected_keys):
            raise ValueError(
                "constructions must cover every frozen session and placebo arm exactly once"
            )

        pair_by_session = parent.pair_by_session
        for construction in constructions:
            pair = pair_by_session[construction.object_session_id]
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
                raise ValueError(
                    "construction must preserve the exact registered chronology"
                )
            if construction.policy_id != policies[construction.arm.value]:
                raise ValueError("construction does not bind its registered policy")
            if construction.arm in {
                PlaceboArmV2.WRONG_SOURCE_ACTION,
                PlaceboArmV2.WRONG_OBJECT_SESSION,
            }:
                donor_session = construction.donor_object_session_id
                assert donor_session is not None
                donor = pair_by_session.get(donor_session)
                if donor is None:
                    raise ValueError("placebo donor must belong to the frozen roster")
                if (
                    construction.donor_source_execution_id != donor.source_execution_id
                    or construction.donor_source_action_id != donor.source_action_id
                ):
                    raise ValueError(
                        "placebo donor must bind the registered source execution"
                    )

        if len({value.constructor_commit_id for value in constructions}) != 1:
            raise ValueError("one exact constructor revision is required")
        object.__setattr__(self, "constructions", constructions)

        minimum_sessions = genuine_integer(
            self.minimum_sessions,
            name="minimum_sessions",
            minimum=2,
        )
        if minimum_sessions > len(parent.target_session_ids):
            raise ValueError("minimum_sessions exceeds the frozen parent roster")
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
            "familywise_confidence_level",
            _probability(
                self.familywise_confidence_level,
                name="familywise_confidence_level",
            ),
        )
        object.__setattr__(
            self,
            "minimum_placebo_separation",
            _finite(
                self.minimum_placebo_separation,
                name="minimum_placebo_separation",
                minimum=0.0,
            ),
        )
        method = _literal(self.familywise_method, name="familywise_method")
        if method != CROSS_ACTION_PLACEBO_V2_FAMILYWISE_METHOD:
            raise ValueError(
                "familywise_method must use the registered paired-session bootstrap"
            )
        object.__setattr__(self, "familywise_method", method)
        for name in (
            "method_frozen_before_target",
            "roster_frozen_before_target",
            "constructions_sealed_before_target",
            "target_outcomes_used_for_selection",
        ):
            object.__setattr__(
                self, name, genuine_boolean(getattr(self, name), name=name)
            )
        if (
            not self.method_frozen_before_target
            or not self.roster_frozen_before_target
            or not self.constructions_sealed_before_target
            or self.target_outcomes_used_for_selection
        ):
            raise ValueError(
                "placebo protocol must be frozen, sealed, and target-selection free"
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="protocol metadata"),
        )

    @property
    def construction_by_key(
        self,
    ) -> dict[tuple[str, PlaceboArmV2], ChronologicalPlaceboConstructionV2]:
        return {
            (construction.object_session_id, construction.arm): construction
            for construction in self.constructions
        }

    @property
    def constructor_commit_id(self) -> str:
        return self.constructions[0].constructor_commit_id

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PLACEBO_V2_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_V2_VERSION,
            "artifact_kind": "CrossActionPlaceboProtocolV2",
            "semantics": CROSS_ACTION_PLACEBO_V2_SEMANTICS,
            "parent_transport_protocol_id": (
                self.parent_transport_protocol.protocol_id
            ),
            "target_roster_id": self.parent_transport_protocol.target_roster_id,
            "physical_arm": PlaceboArmV2.GUARDED_PHYSICAL.value,
            "placebo_arms": [arm.value for arm in PLACEBO_ARMS_V2],
            "placebo_policy_ids": dict(self.placebo_policy_ids),
            "constructions": [
                {
                    **construction.descriptor(),
                    "construction_id": construction.construction_id,
                }
                for construction in self.constructions
            ],
            "minimum_sessions": self.minimum_sessions,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "familywise_confidence_level": self.familywise_confidence_level,
            "minimum_placebo_separation": self.minimum_placebo_separation,
            "familywise_method": self.familywise_method,
            "method_frozen_before_target": self.method_frozen_before_target,
            "roster_frozen_before_target": self.roster_frozen_before_target,
            "constructions_sealed_before_target": (
                self.constructions_sealed_before_target
            ),
            "target_outcomes_used_for_selection": (
                self.target_outcomes_used_for_selection
            ),
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_PLACEBO_V2_CLAIM_BOUNDARY,
        }

    @property
    def protocol_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class SealedCrossActionPlaceboPredictionV2:
    """One target-blind guarded-parent or broken-relation prediction."""

    protocol_id: str
    parent_transport_prediction_id: str
    information_order_id: str
    object_session_id: str
    source_execution_id: str
    target_execution_id: str
    source_action_id: str
    target_action_id: str
    arm: PlaceboArmV2
    construction_id: str | None
    prediction_artifact_id: str
    prediction_batch_id: str
    commit_id: str
    disposition: PredictionDisposition
    prediction_sealed_before_target: bool
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "parent_transport_prediction_id",
            "information_order_id",
            "prediction_artifact_id",
            "prediction_batch_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "construction_id",
            _optional_digest(self.construction_id, name="construction_id"),
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
        if self.source_execution_id == self.target_execution_id:
            raise ValueError("source and target executions must be distinct")
        if self.source_action_id == self.target_action_id:
            raise ValueError("placebo v2 requires a genuinely cross-action prediction")
        if not isinstance(self.arm, PlaceboArmV2):
            raise TypeError("arm must be PlaceboArmV2")
        if self.arm.is_placebo and self.construction_id is None:
            raise ValueError("placebo predictions must bind a construction record")
        if not self.arm.is_placebo and self.construction_id is not None:
            raise ValueError(
                "guarded parent prediction cannot bind a placebo construction"
            )
        if self.disposition not in {
            PredictionDisposition.CANDIDATE_SELECTED,
            PredictionDisposition.EXACT_FALLBACK,
        }:
            raise ValueError(
                "physicality predictions require selected-candidate or exact-fallback "
                "disposition"
            )
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
        object.__setattr__(self, "prediction_sealed_before_target", sealed)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="prediction metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PLACEBO_V2_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_V2_VERSION,
            "artifact_kind": "SealedCrossActionPlaceboPredictionV2",
            "protocol_id": self.protocol_id,
            "parent_transport_prediction_id": self.parent_transport_prediction_id,
            "information_order_id": self.information_order_id,
            "object_session_id": self.object_session_id,
            "source_execution_id": self.source_execution_id,
            "target_execution_id": self.target_execution_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "arm": self.arm.value,
            "construction_id": self.construction_id,
            "prediction_artifact_id": self.prediction_artifact_id,
            "prediction_batch_id": self.prediction_batch_id,
            "commit_id": self.commit_id,
            "disposition": self.disposition.value,
            "prediction_sealed_before_target": self.prediction_sealed_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def prediction_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class CrossActionPlaceboScoreRowV2:
    """One post-access score bound to one sealed physicality prediction."""

    prediction: SealedCrossActionPlaceboPredictionV2
    target_outcome_id: str
    target_access_attestation_id: str
    scorer_id: str
    proper_score: float
    target_side_selection_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, SealedCrossActionPlaceboPredictionV2):
            raise TypeError("prediction must be SealedCrossActionPlaceboPredictionV2")
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
            "schema": CROSS_ACTION_PLACEBO_V2_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_V2_VERSION,
            "artifact_kind": "CrossActionPlaceboScoreRowV2",
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
class PlaceboContrastSummaryV2:
    """One session-level guarded-physical versus placebo contrast."""

    arm: PlaceboArmV2
    mean_contrast: float
    marginal_interval: tuple[float, float] | None
    simultaneous_lower_bound: float | None
    win_sessions: int
    scored_sessions: int

    def descriptor(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "mean_contrast": self.mean_contrast,
            "marginal_interval": (
                None if self.marginal_interval is None else list(self.marginal_interval)
            ),
            "simultaneous_lower_bound": self.simultaneous_lower_bound,
            "win_sessions": self.win_sessions,
            "scored_sessions": self.scored_sessions,
        }


@dataclass(frozen=True, slots=True)
class CrossActionPlaceboResultV2:
    """Evaluate the complete chronological broken-mechanism certificate."""

    protocol: CrossActionPlaceboProtocolV2
    parent_transport_result: CrossActionTransportResultV2
    score_rows: Sequence[CrossActionPlaceboScoreRowV2]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    independent_session_count: int = field(init=False)
    selected_physical_session_count: int = field(init=False)
    session_placebo_contrasts: np.ndarray = field(init=False, repr=False)
    contrast_summaries: tuple[PlaceboContrastSummaryV2, ...] = field(init=False)
    familywise_critical_value: float | None = field(init=False)
    decision: PlaceboDecisionV2 = field(init=False)
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, CrossActionPlaceboProtocolV2):
            raise TypeError("protocol must be CrossActionPlaceboProtocolV2")
        if not isinstance(self.parent_transport_result, CrossActionTransportResultV2):
            raise TypeError(
                "parent_transport_result must be CrossActionTransportResultV2"
            )
        parent_result = self.parent_transport_result
        parent_protocol = self.protocol.parent_transport_protocol
        if parent_result.protocol.protocol_id != parent_protocol.protocol_id:
            raise ValueError("parent result belongs to another transport protocol")

        rows = tuple(self.score_rows)
        if any(not isinstance(row, CrossActionPlaceboScoreRowV2) for row in rows):
            raise TypeError(
                "score_rows must contain CrossActionPlaceboScoreRowV2 values"
            )
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
            raise ValueError("every prediction must bind the exact placebo protocol")

        parent_physical_rows = {
            row.prediction.object_session_id: row
            for row in parent_result.score_rows
            if row.prediction.arm is parent_protocol.physical_transport_arm
        }
        scored_sessions = tuple(sorted(parent_physical_rows))
        expected_keys = {
            (session_id, arm)
            for session_id in scored_sessions
            for arm in ALL_PLACEBO_CERTIFICATE_ARMS_V2
        }
        by_key: dict[tuple[str, PlaceboArmV2], CrossActionPlaceboScoreRowV2] = {}
        construction_by_key = self.protocol.construction_by_key
        for row in rows:
            prediction = row.prediction
            if prediction.object_session_id not in parent_physical_rows:
                raise ValueError("placebo score row uses an unscored parent session")
            pair = parent_protocol.pair_by_session[prediction.object_session_id]
            expected_identity = (
                pair.information_order_id,
                pair.source_execution_id,
                pair.target_execution_id,
                pair.source_action_id,
                pair.target_action_id,
            )
            observed_identity = (
                prediction.information_order_id,
                prediction.source_execution_id,
                prediction.target_execution_id,
                prediction.source_action_id,
                prediction.target_action_id,
            )
            if observed_identity != expected_identity:
                raise ValueError(
                    "prediction must preserve the exact registered chronology"
                )
            key = (prediction.object_session_id, prediction.arm)
            if key in by_key:
                raise ValueError("duplicate physicality arm within one session")
            if prediction.arm.is_placebo:
                construction = construction_by_key[key]
                if prediction.construction_id != construction.construction_id:
                    raise ValueError(
                        "placebo prediction does not bind its frozen construction"
                    )
                if prediction.commit_id != construction.constructor_commit_id:
                    raise ValueError(
                        "placebo prediction revision differs from its constructor"
                    )
            elif prediction.construction_id is not None:
                raise ValueError("guarded parent cannot bind a placebo construction")
            by_key[key] = row

        if set(by_key) != expected_keys or len(rows) != len(expected_keys):
            raise ValueError(
                "score rows must cover every parent-scored session and arm exactly once"
            )
        if rows:
            if len({row.prediction.prediction_batch_id for row in rows}) != 1:
                raise ValueError("one sealed physicality prediction batch is required")
            if len({row.prediction.commit_id for row in rows}) != 1:
                raise ValueError("one exact BayesianPhysTwin revision is required")
            if len({row.target_access_attestation_id for row in rows}) != 1:
                raise ValueError("one target-access attestation is required")
            if len({row.scorer_id for row in rows}) != 1:
                raise ValueError("one frozen scorer is required")

        selected_count = 0
        score_matrix = np.empty(
            (len(scored_sessions), len(ALL_PLACEBO_CERTIFICATE_ARMS_V2)),
            dtype=np.float64,
        )
        for session_index, session_id in enumerate(scored_sessions):
            parent_row = parent_physical_rows[session_id]
            pair_rows = [
                by_key[(session_id, arm)] for arm in ALL_PLACEBO_CERTIFICATE_ARMS_V2
            ]
            parent_prediction_id = parent_row.prediction.prediction_id
            if any(
                row.prediction.parent_transport_prediction_id != parent_prediction_id
                for row in pair_rows
            ):
                raise ValueError(
                    "all arms must bind the exact guarded parent prediction"
                )
            if any(
                row.prediction.disposition is not parent_row.prediction.disposition
                for row in pair_rows
            ):
                raise ValueError("all arms must preserve the parent disposition")
            if any(
                row.target_outcome_id != parent_row.target_outcome_id
                or row.target_access_attestation_id
                != parent_row.target_access_attestation_id
                or row.scorer_id != parent_row.scorer_id
                for row in pair_rows
            ):
                raise ValueError(
                    "physicality rows must reuse the parent target, attestation, and scorer"
                )
            if any(
                row.prediction.commit_id != parent_row.prediction.commit_id
                for row in pair_rows
            ):
                raise ValueError(
                    "physicality predictions must use the parent software revision"
                )
            physical_row = pair_rows[0]
            if (
                physical_row.prediction.prediction_artifact_id
                != parent_row.prediction.prediction_artifact_id
                or physical_row.proper_score != parent_row.proper_score
            ):
                raise ValueError(
                    "guarded physical row must reproduce the exact parent artifact and score"
                )
            if (
                parent_row.prediction.disposition
                is PredictionDisposition.EXACT_FALLBACK
            ):
                if (
                    len({row.prediction.prediction_artifact_id for row in pair_rows})
                    != 1
                    or len({row.proper_score for row in pair_rows}) != 1
                ):
                    raise ValueError(
                        "exact fallback must be artifact- and score-identical for every arm"
                    )
            elif (
                parent_row.prediction.disposition
                is PredictionDisposition.CANDIDATE_SELECTED
            ):
                selected_count += 1
            else:
                raise ValueError(
                    "parent physical row must be selected or exact fallback"
                )
            score_matrix[session_index] = [row.proper_score for row in pair_rows]

        contrasts = score_matrix[:, 1:] - score_matrix[:, [0]]
        summaries: list[PlaceboContrastSummaryV2] = []
        critical_value: float | None = None
        simultaneous_lower: np.ndarray | None = None
        if len(scored_sessions) >= 2:
            simultaneous_lower, critical_value = _simultaneous_lower_bounds(
                contrasts,
                replicates=self.protocol.bootstrap_replicates,
                seed=self.protocol.bootstrap_seed,
                confidence=self.protocol.familywise_confidence_level,
            )
        for index, arm in enumerate(PLACEBO_ARMS_V2):
            values = contrasts[:, index]
            mean_contrast = float(np.mean(values)) if values.size else 0.0
            marginal_interval = (
                _marginal_interval(
                    values,
                    replicates=self.protocol.bootstrap_replicates,
                    seed=self.protocol.bootstrap_seed + index + 1,
                    confidence=self.protocol.familywise_confidence_level,
                )
                if values.size >= 2
                else None
            )
            summaries.append(
                PlaceboContrastSummaryV2(
                    arm=arm,
                    mean_contrast=mean_contrast,
                    marginal_interval=marginal_interval,
                    simultaneous_lower_bound=(
                        None
                        if simultaneous_lower is None
                        else float(simultaneous_lower[index])
                    ),
                    win_sessions=int(np.count_nonzero(values > 0.0)),
                    scored_sessions=len(scored_sessions),
                )
            )

        if parent_result.decision in {
            SparseTransportDecision.INSUFFICIENT_SESSIONS,
            SparseTransportDecision.INSUFFICIENT_ACCEPTED_UPDATES,
        }:
            decision = PlaceboDecisionV2.PARENT_INSUFFICIENT
        elif parent_result.decision is not SparseTransportDecision.SUPPORTED:
            decision = PlaceboDecisionV2.PARENT_NOT_SUPPORTED
        elif len(scored_sessions) < self.protocol.minimum_sessions:
            decision = PlaceboDecisionV2.INSUFFICIENT_SESSIONS
        elif (
            simultaneous_lower is not None
            and selected_count > 0
            and bool(
                np.all(simultaneous_lower > self.protocol.minimum_placebo_separation)
            )
        ):
            decision = PlaceboDecisionV2.SUPPORTED
        else:
            decision = PlaceboDecisionV2.NOT_SUPPORTED

        score_matrix.setflags(write=False)
        contrasts.setflags(write=False)
        metadata = frozen_finite_json_mapping(self.metadata, name="result metadata")
        object.__setattr__(self, "score_rows", rows)
        object.__setattr__(self, "independent_session_count", len(scored_sessions))
        object.__setattr__(
            self,
            "selected_physical_session_count",
            selected_count,
        )
        object.__setattr__(self, "session_placebo_contrasts", contrasts)
        object.__setattr__(self, "contrast_summaries", tuple(summaries))
        object.__setattr__(self, "familywise_critical_value", critical_value)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "result_id", cast(str, content_id(self.descriptor())))

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PLACEBO_V2_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_V2_VERSION,
            "artifact_kind": "CrossActionPlaceboResultV2",
            "protocol_id": self.protocol.protocol_id,
            "parent_transport_result_id": self.parent_transport_result.result_id,
            "score_row_ids": [row.score_row_id for row in self.score_rows],
            "independent_session_count": self.independent_session_count,
            "selected_physical_session_count": (self.selected_physical_session_count),
            "contrast_summaries": [
                summary.descriptor() for summary in self.contrast_summaries
            ],
            "familywise_method": self.protocol.familywise_method,
            "familywise_critical_value": self.familywise_critical_value,
            "decision": self.decision.value,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_PLACEBO_V2_CLAIM_BOUNDARY,
        }

    @property
    def supports_physicality(self) -> bool:
        return self.decision is PlaceboDecisionV2.SUPPORTED

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "result_id": self.result_id,
            "session_ids": sorted(
                {row.prediction.object_session_id for row in self.score_rows}
            ),
            "arm_labels": [arm.value for arm in ALL_PLACEBO_CERTIFICATE_ARMS_V2],
            "session_placebo_contrasts": self.session_placebo_contrasts.tolist(),
        }


__all__ = [
    "ALL_PLACEBO_CERTIFICATE_ARMS_V2",
    "CROSS_ACTION_PLACEBO_V2_CLAIM_BOUNDARY",
    "CROSS_ACTION_PLACEBO_V2_FAMILYWISE_METHOD",
    "CROSS_ACTION_PLACEBO_V2_SCHEMA",
    "CROSS_ACTION_PLACEBO_V2_SEMANTICS",
    "CROSS_ACTION_PLACEBO_V2_VERSION",
    "ChronologicalPlaceboConstructionV2",
    "CrossActionPlaceboProtocolV2",
    "CrossActionPlaceboResultV2",
    "CrossActionPlaceboScoreRowV2",
    "PLACEBO_ARMS_V2",
    "PlaceboArmV2",
    "PlaceboContrastSummaryV2",
    "PlaceboDecisionV2",
    "PredictionDisposition",
    "SealedCrossActionPlaceboPredictionV2",
    "SparseTransportDecision",
    "TransportArm",
]
