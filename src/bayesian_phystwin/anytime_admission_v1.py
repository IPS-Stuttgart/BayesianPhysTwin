"""Anytime-valid admission of learned corrections with delayed outcomes.

The module implements two nonnegative mixture e-processes:

* a bounded-gain process for testing whether a candidate improves a registered
  capped loss over an exact fallback by more than a fixed margin; and
* a Bernoulli process for testing whether the candidate's materially harmful
  update rate lies below a registered ceiling.

A correction is authorized only when both processes cross their current
thresholds.  Geometric alpha spending supports an unbounded sequence of
externally declared epochs without reusing type-I error.  Forecast trials are
registered before their outcomes mature; outcomes from closed epochs are
retained for audit but cannot update a later epoch.

The guarantee is statistical and conditional on the registered score, loss
cap, harm definition, candidate roster, reveal filtration, and null
assumptions.  It is not a deployment-safety guarantee.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

SCHEMA: Final = "bayesian-phystwin.anytime-admission-v1"
SCHEMA_VERSION: Final = 1


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


def _weights(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("a mixture must contain at least one component")
    result = np.full(count, 1.0 / count, dtype=np.float64)
    result.setflags(write=False)
    return result


def _log_mixture(log_wealth: np.ndarray, weights: np.ndarray) -> float:
    if log_wealth.ndim != 1 or weights.shape != log_wealth.shape:
        raise ValueError("mixture arrays must be aligned one-dimensional vectors")
    terms = log_wealth + np.log(weights)
    maximum = float(np.max(terms))
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(float(np.sum(np.exp(terms - maximum))))


@dataclass(frozen=True, slots=True)
class GeometricAlphaSpending:
    """Allocate a finite error budget over an unbounded epoch sequence."""

    total_alpha: float = 0.05
    continuation: float = 0.5

    def __post_init__(self) -> None:
        _probability(self.total_alpha, label="total_alpha")
        _probability(self.continuation, label="continuation")

    def alpha_for_epoch(self, epoch_index: int) -> float:
        """Return alpha_k with sum_k alpha_k equal to ``total_alpha``."""

        index = _literal_nonnegative_integer(epoch_index, label="epoch_index")
        return self.total_alpha * (1.0 - self.continuation) * (
            self.continuation**index
        )

    def cumulative_alpha_through(self, epoch_index: int) -> float:
        """Return the error budget allocated through the requested epoch."""

        index = _literal_nonnegative_integer(epoch_index, label="epoch_index")
        return self.total_alpha * (1.0 - self.continuation ** (index + 1))


@dataclass(frozen=True, slots=True)
class AnytimeAdmissionConfig:
    """Frozen statistical contract for one correction candidate."""

    loss_cap: float
    minimum_mean_gain: float = 0.0
    harmful_margin: float = 0.0
    maximum_harm_rate: float = 0.25
    total_alpha_gain: float = 0.05
    total_alpha_harm: float = 0.05
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
        _probability(self.total_alpha_gain, label="total_alpha_gain")
        _probability(self.total_alpha_harm, label="total_alpha_harm")
        _probability(
            self.epoch_alpha_continuation,
            label="epoch_alpha_continuation",
        )
        _literal_positive_integer(
            self.minimum_resolved_trials,
            label="minimum_resolved_trials",
        )
        if not self.gain_bet_fractions:
            raise ValueError("gain_bet_fractions must not be empty")
        if not self.harm_alternative_fractions:
            raise ValueError("harm_alternative_fractions must not be empty")
        for value in self.gain_bet_fractions:
            if not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
                raise ValueError("gain bet fractions must lie in (0, 1)")
        for value in self.harm_alternative_fractions:
            if not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
                raise ValueError("harm alternative fractions must lie in (0, 1)")
        if len(set(self.gain_bet_fractions)) != len(self.gain_bet_fractions):
            raise ValueError("gain bet fractions must be unique")
        if len(set(self.harm_alternative_fractions)) != len(
            self.harm_alternative_fractions
        ):
            raise ValueError("harm alternative fractions must be unique")


class BoundedGainMixtureEProcess:
    """Mixture betting e-process for bounded conditional mean improvement."""

    def __init__(self, bet_fractions: tuple[float, ...]) -> None:
        if not bet_fractions:
            raise ValueError("bet_fractions must not be empty")
        bets = np.asarray(bet_fractions, dtype=np.float64)
        if (
            bets.ndim != 1
            or not np.isfinite(bets).all()
            or np.any(bets <= 0.0)
            or np.any(bets >= 1.0)
        ):
            raise ValueError("bet fractions must be a finite vector in (0, 1)")
        self._bets = bets
        self._weights = _weights(len(bets))
        self._log_wealth = np.zeros(len(bets), dtype=np.float64)
        self._count = 0
        self._maximum_log_e_value = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def log_e_value(self) -> float:
        return _log_mixture(self._log_wealth, self._weights)

    @property
    def maximum_log_e_value(self) -> float:
        return self._maximum_log_e_value

    def update(self, score: float) -> float:
        """Update with a score in [-1, 1], positive when candidate is better."""

        value = float(score)
        if not math.isfinite(value) or value < -1.0 or value > 1.0:
            raise ValueError("bounded gain score must lie in [-1, 1]")
        factors = 1.0 + self._bets * value
        if np.any(factors <= 0.0) or not np.isfinite(factors).all():
            raise ValueError("gain betting factor is not positive and finite")
        self._log_wealth += np.log(factors)
        self._count += 1
        current = self.log_e_value
        self._maximum_log_e_value = max(self._maximum_log_e_value, current)
        return current


class BernoulliHarmMixtureEProcess:
    """Mixture likelihood-ratio e-process for a harmful-update-rate ceiling."""

    def __init__(
        self,
        *,
        maximum_harm_rate: float,
        alternative_fractions: tuple[float, ...],
    ) -> None:
        ceiling = _probability(maximum_harm_rate, label="maximum_harm_rate")
        if not alternative_fractions:
            raise ValueError("alternative_fractions must not be empty")
        fractions = np.asarray(alternative_fractions, dtype=np.float64)
        if (
            fractions.ndim != 1
            or not np.isfinite(fractions).all()
            or np.any(fractions <= 0.0)
            or np.any(fractions >= 1.0)
        ):
            raise ValueError("harm alternative fractions must lie in (0, 1)")
        self._ceiling = ceiling
        self._alternatives = ceiling * fractions
        self._weights = _weights(len(fractions))
        self._log_wealth = np.zeros(len(fractions), dtype=np.float64)
        self._count = 0
        self._harm_count = 0
        self._maximum_log_e_value = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def harm_count(self) -> int:
        return self._harm_count

    @property
    def log_e_value(self) -> float:
        return _log_mixture(self._log_wealth, self._weights)

    @property
    def maximum_log_e_value(self) -> float:
        return self._maximum_log_e_value

    def update(self, harmful: bool) -> float:
        """Update after revealing one registered binary harm outcome."""

        if type(harmful) is not bool:
            raise ValueError("harmful must be a literal bool")
        if harmful:
            factors = self._alternatives / self._ceiling
            self._harm_count += 1
        else:
            factors = (1.0 - self._alternatives) / (1.0 - self._ceiling)
        if np.any(factors <= 0.0) or not np.isfinite(factors).all():
            raise ValueError("harm likelihood-ratio factor is invalid")
        self._log_wealth += np.log(factors)
        self._count += 1
        current = self.log_e_value
        self._maximum_log_e_value = max(self._maximum_log_e_value, current)
        return current


@dataclass(frozen=True, slots=True)
class PendingTrial:
    """A shadow candidate/fallback comparison registered before outcome reveal."""

    trial_id: str
    epoch_index: int
    issued_step: int
    maturity_step: int


@dataclass(frozen=True, slots=True)
class ResolvedTrial:
    """Auditable delayed outcome and whether it informed the active epoch."""

    trial_id: str
    epoch_index: int
    issued_step: int
    maturity_step: int
    resolved_step: int
    candidate_loss: float
    fallback_loss: float
    bounded_gain_score: float
    harmful: bool
    used_for_current_epoch: bool


@dataclass(frozen=True, slots=True)
class AnytimeAdmissionSnapshot:
    """Serializable state of the current anytime-valid admission epoch."""

    schema: str
    schema_version: int
    epoch_index: int
    epoch_reason: str
    issued_trial_count: int
    resolved_current_epoch_count: int
    pending_trial_count: int
    ignored_closed_epoch_outcome_count: int
    gain_log_e_value: float
    harm_log_e_value: float
    gain_maximum_log_e_value: float
    harm_maximum_log_e_value: float
    gain_log_threshold: float
    harm_log_threshold: float
    current_epoch_alpha_gain: float
    current_epoch_alpha_harm: float
    cumulative_alpha_gain: float
    cumulative_alpha_harm: float
    harm_count: int
    empirical_harm_fraction: float | None
    utility_evidence_passed: bool
    harm_evidence_passed: bool
    minimum_evidence_passed: bool
    authorized: bool
    ever_authorized_in_epoch: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class AnytimeAdmissionController:
    """Delayed-outcome, epochal admission controller for one fixed candidate."""

    def __init__(self, config: AnytimeAdmissionConfig) -> None:
        if not isinstance(config, AnytimeAdmissionConfig):
            raise TypeError("config must be an AnytimeAdmissionConfig")
        self.config = config
        self._gain_schedule = GeometricAlphaSpending(
            total_alpha=config.total_alpha_gain,
            continuation=config.epoch_alpha_continuation,
        )
        self._harm_schedule = GeometricAlphaSpending(
            total_alpha=config.total_alpha_harm,
            continuation=config.epoch_alpha_continuation,
        )
        self._pending: dict[str, PendingTrial] = {}
        self._resolved_ids: set[str] = set()
        self._resolved: list[ResolvedTrial] = []
        self._issued_count = 0
        self._ignored_closed_epoch_count = 0
        self._epoch_index = -1
        self._epoch_reason = ""
        self._ever_authorized = False
        self._gain_process = BoundedGainMixtureEProcess(
            config.gain_bet_fractions
        )
        self._harm_process = BernoulliHarmMixtureEProcess(
            maximum_harm_rate=config.maximum_harm_rate,
            alternative_fractions=config.harm_alternative_fractions,
        )
        self.start_new_epoch(reason="initial")

    @property
    def epoch_index(self) -> int:
        return self._epoch_index

    @property
    def resolved_trials(self) -> tuple[ResolvedTrial, ...]:
        return tuple(self._resolved)

    @property
    def authorized(self) -> bool:
        return self.snapshot().authorized

    def start_new_epoch(self, *, reason: str) -> AnytimeAdmissionSnapshot:
        """Start a fresh alpha-spent epoch; pending old trials remain auditable."""

        if type(reason) is not str or not reason.strip():
            raise ValueError("epoch reason must be a nonempty literal string")
        self._epoch_index += 1
        self._epoch_reason = reason.strip()
        self._gain_process = BoundedGainMixtureEProcess(
            self.config.gain_bet_fractions
        )
        self._harm_process = BernoulliHarmMixtureEProcess(
            maximum_harm_rate=self.config.maximum_harm_rate,
            alternative_fractions=self.config.harm_alternative_fractions,
        )
        self._ever_authorized = False
        return self.snapshot()

    def issue_trial(
        self,
        *,
        trial_id: str,
        issued_step: int,
        maturity_step: int,
    ) -> PendingTrial:
        """Register a comparison before its target-dependent loss is available."""

        if type(trial_id) is not str or not trial_id:
            raise ValueError("trial_id must be a nonempty literal string")
        if trial_id in self._pending or trial_id in self._resolved_ids:
            raise ValueError(f"trial_id was already registered: {trial_id}")
        issued = _literal_nonnegative_integer(issued_step, label="issued_step")
        maturity = _literal_nonnegative_integer(
            maturity_step,
            label="maturity_step",
        )
        if maturity <= issued:
            raise ValueError("maturity_step must be strictly after issued_step")
        trial = PendingTrial(
            trial_id=trial_id,
            epoch_index=self._epoch_index,
            issued_step=issued,
            maturity_step=maturity,
        )
        self._pending[trial_id] = trial
        self._issued_count += 1
        return trial

    def resolve_trial(
        self,
        *,
        trial_id: str,
        resolved_step: int,
        candidate_loss: float,
        fallback_loss: float,
    ) -> ResolvedTrial:
        """Reveal a matured loss pair and update only its issuing epoch."""

        if trial_id not in self._pending:
            if trial_id in self._resolved_ids:
                raise ValueError(f"trial_id was already resolved: {trial_id}")
            raise ValueError(f"unknown pending trial_id: {trial_id}")
        trial = self._pending.pop(trial_id)
        resolved = _literal_nonnegative_integer(
            resolved_step,
            label="resolved_step",
        )
        if resolved < trial.maturity_step:
            self._pending[trial_id] = trial
            raise ValueError("trial outcome cannot be resolved before maturity")
        candidate = _nonnegative(candidate_loss, label="candidate_loss")
        fallback = _nonnegative(fallback_loss, label="fallback_loss")
        capped_candidate = min(candidate, self.config.loss_cap)
        capped_fallback = min(fallback, self.config.loss_cap)
        denominator = self.config.loss_cap + self.config.minimum_mean_gain
        score = (
            capped_fallback
            - capped_candidate
            - self.config.minimum_mean_gain
        ) / denominator
        score = float(np.clip(score, -1.0, 1.0))
        harmful = candidate > fallback + self.config.harmful_margin
        used = trial.epoch_index == self._epoch_index
        if used:
            self._gain_process.update(score)
            self._harm_process.update(harmful)
        else:
            self._ignored_closed_epoch_count += 1
        result = ResolvedTrial(
            trial_id=trial.trial_id,
            epoch_index=trial.epoch_index,
            issued_step=trial.issued_step,
            maturity_step=trial.maturity_step,
            resolved_step=resolved,
            candidate_loss=candidate,
            fallback_loss=fallback,
            bounded_gain_score=score,
            harmful=harmful,
            used_for_current_epoch=used,
        )
        self._resolved.append(result)
        self._resolved_ids.add(trial_id)
        if self._authorization_components()[-1]:
            self._ever_authorized = True
        return result

    def _authorization_components(self) -> tuple[bool, bool, bool, bool]:
        alpha_gain = self._gain_schedule.alpha_for_epoch(self._epoch_index)
        alpha_harm = self._harm_schedule.alpha_for_epoch(self._epoch_index)
        utility = self._gain_process.log_e_value >= -math.log(alpha_gain)
        harm = self._harm_process.log_e_value >= -math.log(alpha_harm)
        minimum = (
            self._gain_process.count >= self.config.minimum_resolved_trials
            and self._harm_process.count >= self.config.minimum_resolved_trials
        )
        return utility, harm, minimum, utility and harm and minimum

    def snapshot(self) -> AnytimeAdmissionSnapshot:
        alpha_gain = self._gain_schedule.alpha_for_epoch(self._epoch_index)
        alpha_harm = self._harm_schedule.alpha_for_epoch(self._epoch_index)
        utility, harm, minimum, authorized = self._authorization_components()
        harm_fraction: float | None = None
        if self._harm_process.count:
            harm_fraction = self._harm_process.harm_count / self._harm_process.count
        return AnytimeAdmissionSnapshot(
            schema=SCHEMA,
            schema_version=SCHEMA_VERSION,
            epoch_index=self._epoch_index,
            epoch_reason=self._epoch_reason,
            issued_trial_count=self._issued_count,
            resolved_current_epoch_count=self._gain_process.count,
            pending_trial_count=len(self._pending),
            ignored_closed_epoch_outcome_count=self._ignored_closed_epoch_count,
            gain_log_e_value=self._gain_process.log_e_value,
            harm_log_e_value=self._harm_process.log_e_value,
            gain_maximum_log_e_value=self._gain_process.maximum_log_e_value,
            harm_maximum_log_e_value=self._harm_process.maximum_log_e_value,
            gain_log_threshold=-math.log(alpha_gain),
            harm_log_threshold=-math.log(alpha_harm),
            current_epoch_alpha_gain=alpha_gain,
            current_epoch_alpha_harm=alpha_harm,
            cumulative_alpha_gain=self._gain_schedule.cumulative_alpha_through(
                self._epoch_index
            ),
            cumulative_alpha_harm=self._harm_schedule.cumulative_alpha_through(
                self._epoch_index
            ),
            harm_count=self._harm_process.harm_count,
            empirical_harm_fraction=harm_fraction,
            utility_evidence_passed=utility,
            harm_evidence_passed=harm,
            minimum_evidence_passed=minimum,
            authorized=authorized,
            ever_authorized_in_epoch=self._ever_authorized or authorized,
        )
