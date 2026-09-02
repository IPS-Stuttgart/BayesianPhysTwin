"""Anytime-valid admission under switching invalidity modes.

The efficient shared-alpha intersection--union rule in version 2 assumes that
at least one fixed component null holds throughout an admission epoch. This
module supplies a conservative companion certificate for the stronger
pointwise union null: at every reveal, either conditional expected gain is
insufficient or conditional harmful-outcome probability is at least the
registered ceiling, but the active reason may change arbitrarily over time.

Let ``G_t in [-1, 1]`` be the registered gain score after its margin transform.
For binary harm ``H_t`` and ceiling ``rho``, define

``S_t = (rho - H_t) / max(rho, 1-rho)``.

Then ``S_t in [-1,1]`` and has nonpositive conditional mean whenever the
conditional harm probability is at least ``rho``. The robust score

``Z_t = min(G_t, S_t)``

is pointwise no larger than either component score. Consequently, if either
component null is true at each time, possibly a different one at every time,
``E[Z_t | F_{t-1}] <= 0``. A standard fixed-fraction mixture e-process on
``Z_t`` is therefore valid under optional stopping and adaptive switching of
the invalidity mode.

This robust certificate is sufficient rather than necessary and can be less
powerful than version 2. It does not assert physical safety or validity after
unregistered score, candidate, fallback, or information-set changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Final, Generic, TypeVar

from bayesian_phystwin.anytime_admission_v1 import (
    BoundedGainMixtureEProcess,
    GeometricAlphaSpending,
)

SCHEMA: Final = "bayesian-phystwin.anytime-switching-admission-v3"
SCHEMA_VERSION: Final = 3

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
class SwitchingAdmissionContractV3:
    candidate_id: str
    fallback_id: str
    gain_score_id: str
    harm_definition_id: str
    information_set_id: str
    reveal_policy_id: str

    def __post_init__(self) -> None:
        for field in (
            "candidate_id",
            "fallback_id",
            "gain_score_id",
            "harm_definition_id",
            "information_set_id",
            "reveal_policy_id",
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
class SwitchingAdmissionConfigV3:
    loss_cap: float
    minimum_mean_gain: float = 0.0
    harmful_margin: float = 0.0
    maximum_harm_rate: float = 0.10
    total_alpha: float = 0.05
    epoch_alpha_continuation: float = 0.5
    minimum_resolved_trials: int = 20
    bet_fractions: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.40,
        0.60,
        0.80,
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
        # Delegate exact mixture validation to the stable process.
        BoundedGainMixtureEProcess(self.bet_fractions)

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SwitchingPendingTrialV3:
    trial_id: str
    epoch_index: int
    issued_step: int
    maturity_step: int
    decision_contract_digest: str


@dataclass(frozen=True, slots=True)
class SwitchingResolvedTrialV3:
    trial_id: str
    epoch_index: int
    issued_step: int
    maturity_step: int
    resolved_step: int
    candidate_loss: float
    fallback_loss: float
    gain_score: float
    harm_score: float
    robust_score: float
    harmful: bool
    used_for_current_epoch: bool


@dataclass(frozen=True, slots=True)
class SwitchingAdmissionSnapshotV3:
    schema: str
    schema_version: int
    decision_contract_digest: str
    contract_digest: str
    candidate_id: str
    fallback_id: str
    selected_mode: str
    selected_artifact_id: str
    epoch_index: int
    epoch_reason: str
    issued_trial_count: int
    resolved_current_epoch_count: int
    pending_trial_count: int
    ignored_closed_epoch_outcome_count: int
    current_epoch_alpha: float
    cumulative_alpha: float
    log_threshold: float
    log_e_value: float
    maximum_log_e_value: float
    authorized: bool
    ever_authorized_in_epoch: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def bounded_gain_score(
    *,
    candidate_loss: float,
    fallback_loss: float,
    loss_cap: float,
    minimum_mean_gain: float,
) -> float:
    candidate = min(_nonnegative(candidate_loss, label="candidate_loss"), loss_cap)
    fallback = min(_nonnegative(fallback_loss, label="fallback_loss"), loss_cap)
    denominator = loss_cap + minimum_mean_gain
    value = (fallback - candidate - minimum_mean_gain) / denominator
    return max(-1.0, min(1.0, float(value)))


def bounded_harm_score(*, harmful: bool, maximum_harm_rate: float) -> float:
    """Return a bounded score with nonpositive mean when harm >= the ceiling."""

    if type(harmful) is not bool:
        raise ValueError("harmful must be a literal bool")
    ceiling = _probability(maximum_harm_rate, label="maximum_harm_rate")
    scale = max(ceiling, 1.0 - ceiling)
    value = (ceiling - float(harmful)) / scale
    if not -1.0 <= value <= 1.0:
        raise AssertionError("harm score escaped [-1, 1]")
    return value


def robust_switching_score(*, gain_score: float, harm_score: float) -> float:
    """Return the maximal pointwise score bounded above by both components."""

    values = (float(gain_score), float(harm_score))
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
        raise ValueError("component scores must be finite and lie in [-1, 1]")
    return min(values)


class SwitchingUnionAdmissionControllerV3(Generic[T]):
    """Fail-closed admission valid under pointwise switching union nulls."""

    def __init__(
        self,
        config: SwitchingAdmissionConfigV3,
        contract: SwitchingAdmissionContractV3,
    ) -> None:
        if not isinstance(config, SwitchingAdmissionConfigV3):
            raise TypeError("config must be a SwitchingAdmissionConfigV3")
        if not isinstance(contract, SwitchingAdmissionContractV3):
            raise TypeError("contract must be a SwitchingAdmissionContractV3")
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
        self._pending: dict[str, SwitchingPendingTrialV3] = {}
        self._resolved_ids: set[str] = set()
        self._resolved: list[SwitchingResolvedTrialV3] = []
        self._issued_count = 0
        self._ignored_closed_epoch_count = 0
        self._epoch_index = -1
        self._epoch_reason = ""
        self._authorized = False
        self._ever_authorized = False
        self._process = BoundedGainMixtureEProcess(config.bet_fractions)
        self.start_new_epoch(reason="initial")

    @property
    def authorized(self) -> bool:
        return self._authorized

    @property
    def epoch_index(self) -> int:
        return self._epoch_index

    @property
    def resolved_trials(self) -> tuple[SwitchingResolvedTrialV3, ...]:
        return tuple(self._resolved)

    def start_new_epoch(self, *, reason: str) -> SwitchingAdmissionSnapshotV3:
        self._epoch_index += 1
        self._epoch_reason = _identifier(reason, label="epoch reason")
        self._authorized = False
        self._ever_authorized = False
        self._process = BoundedGainMixtureEProcess(self.config.bet_fractions)
        return self.snapshot()

    def issue_trial(
        self,
        *,
        trial_id: str,
        issued_step: int,
        maturity_step: int,
    ) -> SwitchingPendingTrialV3:
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
        result = SwitchingPendingTrialV3(
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
    ) -> SwitchingResolvedTrialV3:
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
        del self._pending[identifier]
        candidate = _nonnegative(candidate_loss, label="candidate_loss")
        fallback = _nonnegative(fallback_loss, label="fallback_loss")
        gain = bounded_gain_score(
            candidate_loss=candidate,
            fallback_loss=fallback,
            loss_cap=self.config.loss_cap,
            minimum_mean_gain=self.config.minimum_mean_gain,
        )
        harmful = candidate > fallback + self.config.harmful_margin
        harm = bounded_harm_score(
            harmful=harmful,
            maximum_harm_rate=self.config.maximum_harm_rate,
        )
        robust = robust_switching_score(gain_score=gain, harm_score=harm)
        used = (
            trial.epoch_index == self._epoch_index
            and trial.decision_contract_digest == self.decision_contract_digest
        )
        if used:
            self._process.update(robust)
            alpha = self._schedule.alpha_for_epoch(self._epoch_index)
            minimum = self._process.count >= self.config.minimum_resolved_trials
            if minimum and self._process.maximum_log_e_value >= -math.log(alpha):
                self._authorized = True
                self._ever_authorized = True
        else:
            self._ignored_closed_epoch_count += 1
        result = SwitchingResolvedTrialV3(
            trial_id=trial.trial_id,
            epoch_index=trial.epoch_index,
            issued_step=trial.issued_step,
            maturity_step=trial.maturity_step,
            resolved_step=resolved,
            candidate_loss=candidate,
            fallback_loss=fallback,
            gain_score=gain,
            harm_score=harm,
            robust_score=robust,
            harmful=harmful,
            used_for_current_epoch=used,
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
        if _identifier(candidate_id, label="candidate_id") != self.contract.candidate_id:
            raise ValueError("candidate_id does not match the frozen contract")
        return candidate if self._authorized else fallback

    def snapshot(self) -> SwitchingAdmissionSnapshotV3:
        alpha = self._schedule.alpha_for_epoch(self._epoch_index)
        selected_mode = "candidate" if self._authorized else "exact-fallback"
        return SwitchingAdmissionSnapshotV3(
            schema=SCHEMA,
            schema_version=SCHEMA_VERSION,
            decision_contract_digest=self.decision_contract_digest,
            contract_digest=self.contract.digest,
            candidate_id=self.contract.candidate_id,
            fallback_id=self.contract.fallback_id,
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
            current_epoch_alpha=alpha,
            cumulative_alpha=self._schedule.cumulative_alpha_through(
                self._epoch_index
            ),
            log_threshold=-math.log(alpha),
            log_e_value=self._process.log_e_value,
            maximum_log_e_value=self._process.maximum_log_e_value,
            authorized=self._authorized,
            ever_authorized_in_epoch=self._ever_authorized,
        )

    def theorem_boundary(self) -> dict[str, object]:
        return {
            "pointwise_switching_union_null": (
                "at each reveal, either conditional expected registered gain is "
                "nonpositive or conditional harm probability is at least the "
                "registered ceiling; the active component may switch over time"
            ),
            "robust_score": "minimum of bounded gain and bounded harm scores",
            "epoch_false_admission_bound": self._schedule.alpha_for_epoch(
                self._epoch_index
            ),
            "lifetime_false_admission_bound": self.config.total_alpha,
            "price_of_robustness": (
                "the pointwise minimum is sufficient but can be substantially less "
                "powerful than stable-component intersection--union admission"
            ),
            "excluded_claims": (
                "physical safety, arbitrary score adaptation, universal transport, "
                "unclipped-loss validity, and causal identification"
            ),
        }
