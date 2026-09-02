"""Delayed-outcome admission controller for factor-enveloped e-processes.

This module combines the lower-envelope composition theorem from version 4
with the fail-closed lifecycle introduced by the earlier admission controllers:
content-addressed candidate and fallback identities, predictable trial issue,
delayed paired outcomes, geometric alpha spending across declared epochs, and
exact caller-owned fallback selection.

The controller is anytime-valid under the pointwise union null encoded by the
registered component e-factors. At every reveal, at least one component null
must hold conditionally on the pre-reveal information, but the active component
may change arbitrarily with the past. Candidate, fallback, score, harm
definition, information set, reveal policy, factor family, and factor grids are
frozen into the decision-contract digest.

This is a statistical admission mechanism. It does not certify physical safety,
causal identification, or validity after outcome-dependent contract redesign.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Final, Generic, TypeVar

from bayesian_phystwin.anytime_admission_v1 import GeometricAlphaSpending
from bayesian_phystwin.anytime_factor_envelope_v4 import (
    FactorEnvelopeUpdateV4,
    LowerEnvelopeMixtureEProcess,
)
from bayesian_phystwin.anytime_switching_admission_v3 import bounded_gain_score

SCHEMA: Final = "bayesian-phystwin.anytime-factor-envelope-controller-v4"
SCHEMA_VERSION: Final = 4

T = TypeVar("T")


def _identifier(value: str, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty literal string")
    return value.strip()


def _probability(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{label} must lie strictly between zero and one")
    return result


def _positive(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _nonnegative(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return result


def _literal_nonnegative_integer(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative literal integer")
    return value


def _literal_positive_integer(value: int, *, label: str) -> int:
    result = _literal_nonnegative_integer(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FactorEnvelopeAdmissionContractV4:
    candidate_id: str
    fallback_id: str
    gain_score_id: str
    harm_definition_id: str
    information_set_id: str
    reveal_policy_id: str
    factor_family_id: str = "lower-envelope-cartesian-v1"

    def __post_init__(self) -> None:
        for field in (
            "candidate_id",
            "fallback_id",
            "gain_score_id",
            "harm_definition_id",
            "information_set_id",
            "reveal_policy_id",
            "factor_family_id",
        ):
            _identifier(getattr(self, field), label=field)
        if self.candidate_id == self.fallback_id:
            raise ValueError("candidate_id and fallback_id must differ")

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            **asdict(self),
        }

    @property
    def digest(self) -> str:
        return _digest(self.descriptor())


@dataclass(frozen=True, slots=True)
class FactorEnvelopeAdmissionConfigV4:
    loss_cap: float
    minimum_mean_gain: float = 0.0
    harmful_margin: float = 0.0
    maximum_harm_rate: float = 0.10
    total_alpha: float = 0.05
    epoch_alpha_continuation: float = 0.5
    minimum_resolved_trials: int = 20
    gain_bet_fractions: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.40,
        0.60,
        0.80,
    )
    harm_alternative_fractions: tuple[float, ...] = (
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
    )

    def __post_init__(self) -> None:
        _positive(self.loss_cap, label="loss_cap")
        _nonnegative(self.minimum_mean_gain, label="minimum_mean_gain")
        _nonnegative(self.harmful_margin, label="harmful_margin")
        _probability(self.maximum_harm_rate, label="maximum_harm_rate")
        _probability(self.total_alpha, label="total_alpha")
        _probability(
            self.epoch_alpha_continuation,
            label="epoch_alpha_continuation",
        )
        _literal_positive_integer(
            self.minimum_resolved_trials,
            label="minimum_resolved_trials",
        )
        LowerEnvelopeMixtureEProcess(
            gain_bet_fractions=self.gain_bet_fractions,
            maximum_harm_rate=self.maximum_harm_rate,
            harm_alternative_fractions=self.harm_alternative_fractions,
        )

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FactorEnvelopePendingTrialV4:
    trial_id: str
    epoch_index: int
    issued_step: int
    maturity_step: int
    decision_contract_digest: str


@dataclass(frozen=True, slots=True)
class FactorEnvelopeResolvedTrialV4:
    trial_id: str
    epoch_index: int
    issued_step: int
    maturity_step: int
    resolved_step: int
    candidate_loss: float
    fallback_loss: float
    gain_score: float
    harmful: bool
    used_for_current_epoch: bool
    minimum_component_factor: float | None
    maximum_component_factor: float | None
    log_e_value: float | None
    maximum_log_e_value: float | None


@dataclass(frozen=True, slots=True)
class FactorEnvelopeAdmissionSnapshotV4:
    schema: str
    schema_version: int
    decision_contract_digest: str
    contract_digest: str
    candidate_id: str
    fallback_id: str
    factor_family_id: str
    selected_mode: str
    selected_artifact_id: str
    epoch_index: int
    epoch_reason: str
    issued_trial_count: int
    resolved_current_epoch_count: int
    pending_trial_count: int
    ignored_closed_epoch_outcome_count: int
    factor_component_count: int
    current_epoch_alpha: float
    cumulative_alpha: float
    log_threshold: float
    log_e_value: float
    maximum_log_e_value: float
    authorized: bool
    ever_authorized_in_epoch: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FactorEnvelopeAdmissionControllerV4(Generic[T]):
    """Fail-closed admission using independently tuned factor envelopes."""

    def __init__(
        self,
        config: FactorEnvelopeAdmissionConfigV4,
        contract: FactorEnvelopeAdmissionContractV4,
    ) -> None:
        if not isinstance(config, FactorEnvelopeAdmissionConfigV4):
            raise TypeError("config must be a FactorEnvelopeAdmissionConfigV4")
        if not isinstance(contract, FactorEnvelopeAdmissionContractV4):
            raise TypeError("contract must be a FactorEnvelopeAdmissionContractV4")
        self.config = config
        self.contract = contract
        self.decision_contract_digest = _digest(
            {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "contract": contract.descriptor(),
                "config": config.descriptor(),
            }
        )
        self._schedule = GeometricAlphaSpending(
            total_alpha=config.total_alpha,
            continuation=config.epoch_alpha_continuation,
        )
        self._pending: dict[str, FactorEnvelopePendingTrialV4] = {}
        self._resolved_ids: set[str] = set()
        self._resolved: list[FactorEnvelopeResolvedTrialV4] = []
        self._issued_count = 0
        self._ignored_closed_epoch_count = 0
        self._epoch_index = -1
        self._epoch_reason = ""
        self._authorized = False
        self._ever_authorized = False
        self._process = self._new_process()
        self.start_new_epoch(reason="initial")

    def _new_process(self) -> LowerEnvelopeMixtureEProcess:
        return LowerEnvelopeMixtureEProcess(
            gain_bet_fractions=self.config.gain_bet_fractions,
            maximum_harm_rate=self.config.maximum_harm_rate,
            harm_alternative_fractions=self.config.harm_alternative_fractions,
        )

    @property
    def authorized(self) -> bool:
        return self._authorized

    @property
    def epoch_index(self) -> int:
        return self._epoch_index

    @property
    def resolved_trials(self) -> tuple[FactorEnvelopeResolvedTrialV4, ...]:
        return tuple(self._resolved)

    def start_new_epoch(self, *, reason: str) -> FactorEnvelopeAdmissionSnapshotV4:
        reason_id = _identifier(reason, label="epoch reason")
        self._epoch_index += 1
        self._epoch_reason = reason_id
        self._authorized = False
        self._ever_authorized = False
        self._process = self._new_process()
        return self.snapshot()

    def issue_trial(
        self,
        *,
        trial_id: str,
        issued_step: int,
        maturity_step: int,
    ) -> FactorEnvelopePendingTrialV4:
        identifier = _identifier(trial_id, label="trial_id")
        if identifier in self._pending or identifier in self._resolved_ids:
            raise ValueError(f"trial_id was already registered: {identifier}")
        issued = _literal_nonnegative_integer(issued_step, label="issued_step")
        maturity = _literal_nonnegative_integer(
            maturity_step,
            label="maturity_step",
        )
        if maturity <= issued:
            raise ValueError("maturity_step must be strictly after issued_step")
        result = FactorEnvelopePendingTrialV4(
            trial_id=identifier,
            epoch_index=self._epoch_index,
            issued_step=issued,
            maturity_step=maturity,
            decision_contract_digest=self.decision_contract_digest,
        )
        self._pending[identifier] = result
        self._issued_count += 1
        return result

    def resolve_trial(
        self,
        *,
        trial_id: str,
        resolved_step: int,
        candidate_loss: float,
        fallback_loss: float,
    ) -> FactorEnvelopeResolvedTrialV4:
        identifier = _identifier(trial_id, label="trial_id")
        if identifier not in self._pending:
            if identifier in self._resolved_ids:
                raise ValueError(f"trial_id was already resolved: {identifier}")
            raise ValueError(f"unknown pending trial_id: {identifier}")
        trial = self._pending[identifier]
        resolved = _literal_nonnegative_integer(
            resolved_step,
            label="resolved_step",
        )
        if resolved < trial.maturity_step:
            raise ValueError("trial outcome cannot be resolved before maturity")

        # Validate and derive the complete observation before consuming the
        # pending record. A malformed reveal is therefore retryable rather than
        # silently destroying a registered trial.
        candidate = _nonnegative(candidate_loss, label="candidate_loss")
        fallback = _nonnegative(fallback_loss, label="fallback_loss")
        gain = bounded_gain_score(
            candidate_loss=candidate,
            fallback_loss=fallback,
            loss_cap=self.config.loss_cap,
            minimum_mean_gain=self.config.minimum_mean_gain,
        )
        harmful = candidate > fallback + self.config.harmful_margin
        used = (
            trial.epoch_index == self._epoch_index
            and trial.decision_contract_digest == self.decision_contract_digest
        )

        update: FactorEnvelopeUpdateV4 | None = None
        if used:
            update = self._process.update(gain_score=gain, harmful=harmful)
            alpha = self._schedule.alpha_for_epoch(self._epoch_index)
            minimum = self._process.count >= self.config.minimum_resolved_trials
            if minimum and self._process.maximum_log_e_value >= -math.log(alpha):
                self._authorized = True
                self._ever_authorized = True
        else:
            self._ignored_closed_epoch_count += 1

        del self._pending[identifier]
        result = FactorEnvelopeResolvedTrialV4(
            trial_id=trial.trial_id,
            epoch_index=trial.epoch_index,
            issued_step=trial.issued_step,
            maturity_step=trial.maturity_step,
            resolved_step=resolved,
            candidate_loss=candidate,
            fallback_loss=fallback,
            gain_score=gain,
            harmful=harmful,
            used_for_current_epoch=used,
            minimum_component_factor=(
                None if update is None else update.minimum_component_factor
            ),
            maximum_component_factor=(
                None if update is None else update.maximum_component_factor
            ),
            log_e_value=None if update is None else update.log_e_value,
            maximum_log_e_value=(
                None if update is None else update.maximum_log_e_value
            ),
        )
        self._resolved.append(result)
        self._resolved_ids.add(identifier)
        return result

    def select(
        self,
        *,
        fallback: T,
        candidate: T,
        fallback_id: str,
        candidate_id: str,
    ) -> T:
        if _identifier(fallback_id, label="fallback_id") != self.contract.fallback_id:
            raise ValueError("fallback_id does not match the frozen contract")
        if (
            _identifier(candidate_id, label="candidate_id")
            != self.contract.candidate_id
        ):
            raise ValueError("candidate_id does not match the frozen contract")
        return candidate if self._authorized else fallback

    def snapshot(self) -> FactorEnvelopeAdmissionSnapshotV4:
        alpha = self._schedule.alpha_for_epoch(self._epoch_index)
        selected_mode = "candidate" if self._authorized else "exact-fallback"
        return FactorEnvelopeAdmissionSnapshotV4(
            schema=SCHEMA,
            schema_version=SCHEMA_VERSION,
            decision_contract_digest=self.decision_contract_digest,
            contract_digest=self.contract.digest,
            candidate_id=self.contract.candidate_id,
            fallback_id=self.contract.fallback_id,
            factor_family_id=self.contract.factor_family_id,
            selected_mode=selected_mode,
            selected_artifact_id=(
                self.contract.candidate_id
                if self._authorized
                else self.contract.fallback_id
            ),
            epoch_index=self._epoch_index,
            epoch_reason=self._epoch_reason,
            issued_trial_count=self._issued_count,
            resolved_current_epoch_count=self._process.count,
            pending_trial_count=len(self._pending),
            ignored_closed_epoch_outcome_count=self._ignored_closed_epoch_count,
            factor_component_count=self._process.component_count,
            current_epoch_alpha=alpha,
            cumulative_alpha=self._schedule.cumulative_alpha_through(self._epoch_index),
            log_threshold=-math.log(alpha),
            log_e_value=self._process.log_e_value,
            maximum_log_e_value=self._process.maximum_log_e_value,
            authorized=self._authorized,
            ever_authorized_in_epoch=self._ever_authorized,
        )

    def theorem_boundary(self) -> dict[str, object]:
        return {
            "pointwise_switching_union_null": (
                "at every reveal at least one registered component e-factor has "
                "conditional expectation at most one; the active component may "
                "change predictably or adversarially with the past"
            ),
            "factor_composition": (
                "pointwise minimum across component factors, product over reveals, "
                "and outcome-independent mixture across fixed parameter tuples"
            ),
            "epoch_false_admission_bound": self._schedule.alpha_for_epoch(
                self._epoch_index
            ),
            "lifetime_false_admission_bound": self.config.total_alpha,
            "delayed_outcome_rule": (
                "only outcomes from trials issued under the current content-addressed "
                "decision contract may update the current epoch"
            ),
            "fallback_rule": (
                "selection returns the caller-owned registered fallback object unless "
                "the current epoch has crossed its factor-envelope boundary"
            ),
            "excluded_claims": (
                "physical safety, causal identification, validity after unregistered "
                "factor adaptation, and universal distribution shift"
            ),
        }
