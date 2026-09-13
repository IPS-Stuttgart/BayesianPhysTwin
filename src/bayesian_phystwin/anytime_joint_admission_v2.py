"""Joint anytime-valid admission for learned simulator corrections.

Version 2 replaces two independently alpha-split admission gates with one
intersection--union decision. A candidate is admissible only after both:

* a bounded-gain e-process rejects insufficient mean improvement; and
* a Bernoulli e-process rejects a harmful-update rate at or above a ceiling.

The invalid-candidate null is the union of those component nulls. Therefore,
testing each component at the same epoch-wise alpha and requiring both to cross
controls false admission at that alpha; no Bonferroni split is needed. Component
crossings are latched within an epoch, which preserves the same bound while
avoiding the requirement that both e-values be simultaneously large.

A separate heavy-tailed change-point mixture monitors reverse gain after
promotion. All trial identities and delayed outcomes are bound to a canonical,
content-addressed decision contract. Selection returns either the exact
candidate object or the exact caller-owned fallback object.

The guarantees are conditional on one fixed component null holding throughout
an admission epoch, predictable trial registration, the frozen score and harm
definitions, and paired candidate/fallback outcomes. They are not physical
safety, arbitrary-shift, or unclipped-loss guarantees.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Final, Generic, TypeVar

import numpy as np

from bayesian_phystwin.anytime_admission_v1 import (
    BernoulliHarmMixtureEProcess,
    BoundedGainMixtureEProcess,
    GeometricAlphaSpending,
)

SCHEMA: Final = "bayesian-phystwin.anytime-joint-admission-v2"
SCHEMA_VERSION: Final = 2

T = TypeVar("T")


def _nonempty_identifier(value: str, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty literal string")
    return value.strip()


def _probability(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{label} must lie strictly between zero and one")
    return result


def _nonnegative(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return result


def _positive(value: float, *, label: str) -> float:
    result = _nonnegative(value, label=label)
    if result == 0.0:
        raise ValueError(f"{label} must be positive")
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


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _logsumexp(values: np.ndarray) -> float:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("log-sum-exp input must be a nonempty vector")
    maximum = float(np.max(values))
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


@dataclass(frozen=True, slots=True)
class AdmissionContractV2:
    """Immutable identities defining one paired admission experiment."""

    candidate_id: str
    fallback_id: str
    score_id: str
    harm_definition_id: str
    information_set_id: str
    reveal_policy_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_id",
            "fallback_id",
            "score_id",
            "harm_definition_id",
            "information_set_id",
            "reveal_policy_id",
        ):
            _nonempty_identifier(getattr(self, field_name), label=field_name)
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
        return _canonical_digest(self.descriptor())


@dataclass(frozen=True, slots=True)
class JointAdmissionConfigV2:
    """Frozen statistical thresholds for joint admission and revocation."""

    loss_cap: float
    minimum_mean_gain: float = 0.0
    harmful_margin: float = 0.0
    maximum_harm_rate: float = 0.10
    total_alpha: float = 0.05
    total_beta: float = 0.05
    epoch_alpha_continuation: float = 0.5
    minimum_resolved_trials: int = 20
    allow_reentry: bool = True
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
    revocation_bet_fractions: tuple[float, ...] = (
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
        _probability(self.total_beta, label="total_beta")
        _probability(
            self.epoch_alpha_continuation,
            label="epoch_alpha_continuation",
        )
        _literal_positive_integer(
            self.minimum_resolved_trials,
            label="minimum_resolved_trials",
        )
        for label, values in (
            ("gain_bet_fractions", self.gain_bet_fractions),
            ("harm_alternative_fractions", self.harm_alternative_fractions),
            ("revocation_bet_fractions", self.revocation_bet_fractions),
        ):
            if not values:
                raise ValueError(f"{label} must not be empty")
            numeric = tuple(float(value) for value in values)
            if any(
                not math.isfinite(value) or not 0.0 < value < 1.0 for value in numeric
            ) or len(set(numeric)) != len(numeric):
                raise ValueError(f"{label} must contain unique finite values in (0, 1)")

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PendingTrialV2:
    trial_id: str
    epoch_index: int
    issued_step: int
    maturity_step: int
    evidence_role: str
    decision_contract_digest: str


@dataclass(frozen=True, slots=True)
class ResolvedTrialV2:
    trial_id: str
    epoch_index: int
    issued_step: int
    maturity_step: int
    resolved_step: int
    evidence_role: str
    candidate_loss: float
    fallback_loss: float
    bounded_gain_score: float
    harmful: bool
    used_for_current_epoch: bool
    event: str


@dataclass(frozen=True, slots=True)
class JointAdmissionSnapshotV2:
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
    closed: bool
    issued_trial_count: int
    resolved_current_epoch_admission_count: int
    resolved_current_epoch_revocation_count: int
    pending_trial_count: int
    ignored_closed_epoch_outcome_count: int
    current_epoch_alpha: float | None
    current_epoch_beta: float | None
    cumulative_alpha: float
    cumulative_beta: float
    shared_log_threshold: float | None
    revocation_log_threshold: float | None
    gain_log_e_value: float
    harm_log_e_value: float
    gain_maximum_log_e_value: float
    harm_maximum_log_e_value: float
    revocation_log_e_value: float | None
    revocation_maximum_log_e_value: float | None
    gain_evidence_ever_passed: bool
    harm_evidence_ever_passed: bool
    minimum_evidence_passed: bool
    authorized: bool
    ever_authorized_in_epoch: bool
    harm_count: int
    empirical_harm_fraction: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ChangePointBoundedGainEProcess:
    """Mixture over all deterministic start times for an unknown mean reversal."""

    def __init__(self, bet_fractions: tuple[float, ...]) -> None:
        if not bet_fractions:
            raise ValueError("bet_fractions must not be empty")
        self._bet_fractions = tuple(float(value) for value in bet_fractions)
        # Constructor validation is delegated to the stable component process.
        BoundedGainMixtureEProcess(self._bet_fractions)
        self._processes: list[BoundedGainMixtureEProcess] = []
        self._count = 0
        self._maximum_log_e_value = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def log_e_value(self) -> float:
        terms = [math.log(1.0 / (self._count + 1.0))]
        terms.extend(
            math.log(1.0 / (start * (start + 1.0))) + process.log_e_value
            for start, process in enumerate(self._processes, start=1)
        )
        return _logsumexp(np.asarray(terms, dtype=np.float64))

    @property
    def maximum_log_e_value(self) -> float:
        return self._maximum_log_e_value

    def update(self, score: float) -> float:
        value = float(score)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("score must be finite and lie in [-1, 1]")
        for process in self._processes:
            process.update(value)
        process = BoundedGainMixtureEProcess(self._bet_fractions)
        process.update(value)
        self._processes.append(process)
        self._count += 1
        current = self.log_e_value
        self._maximum_log_e_value = max(self._maximum_log_e_value, current)
        return current

    def snapshot(self) -> dict[str, object]:
        return {
            "start_prior": "1/(s*(s+1))",
            "future_start_mass": 1.0 / (self._count + 1.0),
            "observation_count": self._count,
            "start_count": len(self._processes),
            "log_e_value": self.log_e_value,
            "maximum_log_e_value": self.maximum_log_e_value,
        }


class JointAnytimeAdmissionControllerV2(Generic[T]):
    """Content-addressed joint admission with delayed outcomes and revocation."""

    def __init__(
        self,
        config: JointAdmissionConfigV2,
        contract: AdmissionContractV2,
    ) -> None:
        if not isinstance(config, JointAdmissionConfigV2):
            raise TypeError("config must be a JointAdmissionConfigV2")
        if not isinstance(contract, AdmissionContractV2):
            raise TypeError("contract must be an AdmissionContractV2")
        self.config = config
        self.contract = contract
        self.decision_contract_digest = _canonical_digest(
            {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "contract": contract.descriptor(),
                "config": config.descriptor(),
            }
        )
        self._alpha_schedule = GeometricAlphaSpending(
            total_alpha=config.total_alpha,
            continuation=config.epoch_alpha_continuation,
        )
        self._beta_schedule = GeometricAlphaSpending(
            total_alpha=config.total_beta,
            continuation=config.epoch_alpha_continuation,
        )
        self._pending: dict[str, PendingTrialV2] = {}
        self._resolved_ids: set[str] = set()
        self._resolved: list[ResolvedTrialV2] = []
        self._issued_count = 0
        self._ignored_closed_epoch_count = 0
        self._epoch_index = -1
        self._epoch_reason = ""
        self._closed = False
        self._authorized = False
        self._ever_authorized = False
        self._gain_crossed = False
        self._harm_crossed = False
        self._gain_process = BoundedGainMixtureEProcess(config.gain_bet_fractions)
        self._harm_process = BernoulliHarmMixtureEProcess(
            maximum_harm_rate=config.maximum_harm_rate,
            alternative_fractions=config.harm_alternative_fractions,
        )
        self._revocation_process: ChangePointBoundedGainEProcess | None = None
        self._begin_epoch(reason="initial")

    @property
    def authorized(self) -> bool:
        return self._authorized

    @property
    def epoch_index(self) -> int:
        return self._epoch_index

    @property
    def resolved_trials(self) -> tuple[ResolvedTrialV2, ...]:
        return tuple(self._resolved)

    def _begin_epoch(self, *, reason: str) -> None:
        if self._closed:
            raise RuntimeError("controller is closed")
        self._epoch_index += 1
        self._epoch_reason = _nonempty_identifier(reason, label="epoch reason")
        self._authorized = False
        self._ever_authorized = False
        self._gain_crossed = False
        self._harm_crossed = False
        self._gain_process = BoundedGainMixtureEProcess(self.config.gain_bet_fractions)
        self._harm_process = BernoulliHarmMixtureEProcess(
            maximum_harm_rate=self.config.maximum_harm_rate,
            alternative_fractions=self.config.harm_alternative_fractions,
        )
        self._revocation_process = None

    def start_new_epoch(self, *, reason: str) -> JointAdmissionSnapshotV2:
        """Revert to exact fallback and spend a fresh, summable epoch budget."""

        self._begin_epoch(reason=reason)
        return self.snapshot()

    def issue_trial(
        self,
        *,
        trial_id: str,
        issued_step: int,
        maturity_step: int,
    ) -> PendingTrialV2:
        """Register a paired shadow comparison before its outcome is available."""

        if self._closed:
            raise RuntimeError("controller is closed")
        identifier = _nonempty_identifier(trial_id, label="trial_id")
        if identifier in self._pending or identifier in self._resolved_ids:
            raise ValueError(f"trial_id was already registered: {identifier}")
        issued = _literal_nonnegative_integer(issued_step, label="issued_step")
        maturity = _literal_nonnegative_integer(
            maturity_step,
            label="maturity_step",
        )
        if maturity <= issued:
            raise ValueError("maturity_step must be strictly after issued_step")
        trial = PendingTrialV2(
            trial_id=identifier,
            epoch_index=self._epoch_index,
            issued_step=issued,
            maturity_step=maturity,
            evidence_role="revocation" if self._authorized else "admission",
            decision_contract_digest=self.decision_contract_digest,
        )
        self._pending[identifier] = trial
        self._issued_count += 1
        return trial

    def _bounded_gain_score(
        self,
        *,
        candidate_loss: float,
        fallback_loss: float,
    ) -> float:
        candidate = min(candidate_loss, self.config.loss_cap)
        fallback = min(fallback_loss, self.config.loss_cap)
        denominator = self.config.loss_cap + self.config.minimum_mean_gain
        return float(
            np.clip(
                (fallback - candidate - self.config.minimum_mean_gain) / denominator,
                -1.0,
                1.0,
            )
        )

    def resolve_trial(
        self,
        *,
        trial_id: str,
        resolved_step: int,
        candidate_loss: float,
        fallback_loss: float,
    ) -> ResolvedTrialV2:
        """Reveal one matured pair and update only its issuing epoch and role."""

        identifier = _nonempty_identifier(trial_id, label="trial_id")
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
        candidate = _nonnegative(candidate_loss, label="candidate_loss")
        fallback = _nonnegative(fallback_loss, label="fallback_loss")
        del self._pending[identifier]
        score = self._bounded_gain_score(
            candidate_loss=candidate,
            fallback_loss=fallback,
        )
        harmful = candidate > fallback + self.config.harmful_margin
        used = (
            trial.epoch_index == self._epoch_index
            and trial.decision_contract_digest == self.decision_contract_digest
        )
        event = "closed-epoch-outcome"

        if used and trial.evidence_role == "admission":
            self._gain_process.update(score)
            self._harm_process.update(harmful)
            log_threshold = -math.log(
                self._alpha_schedule.alpha_for_epoch(self._epoch_index)
            )
            self._gain_crossed = (
                self._gain_crossed
                or self._gain_process.maximum_log_e_value >= log_threshold
            )
            self._harm_crossed = (
                self._harm_crossed
                or self._harm_process.maximum_log_e_value >= log_threshold
            )
            minimum = self._gain_process.count >= self.config.minimum_resolved_trials
            if (
                not self._authorized
                and self._gain_crossed
                and self._harm_crossed
                and minimum
            ):
                self._authorized = True
                self._ever_authorized = True
                self._revocation_process = ChangePointBoundedGainEProcess(
                    self.config.revocation_bet_fractions
                )
                event = "admit"
            elif self._authorized:
                event = "admission-outcome-after-promotion"
            else:
                event = "remain-fallback"
        elif used and trial.evidence_role == "revocation":
            if not self._authorized or self._revocation_process is None:
                used = False
                self._ignored_closed_epoch_count += 1
                event = "inactive-revocation-outcome"
            else:
                reverse_score = -score
                self._revocation_process.update(reverse_score)
                beta = self._beta_schedule.alpha_for_epoch(self._epoch_index)
                if self._revocation_process.maximum_log_e_value >= -math.log(beta):
                    self._authorized = False
                    event = "revoke"
                    if self.config.allow_reentry:
                        old_epoch = self._epoch_index
                        self._begin_epoch(reason=f"revocation-after-epoch-{old_epoch}")
                    else:
                        self._closed = True
                else:
                    event = "remain-candidate"
        else:
            self._ignored_closed_epoch_count += 1

        result = ResolvedTrialV2(
            trial_id=trial.trial_id,
            epoch_index=trial.epoch_index,
            issued_step=trial.issued_step,
            maturity_step=trial.maturity_step,
            resolved_step=resolved,
            evidence_role=trial.evidence_role,
            candidate_loss=candidate,
            fallback_loss=fallback,
            bounded_gain_score=score,
            harmful=harmful,
            used_for_current_epoch=used,
            event=event,
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
        """Return the exact registered object selected for the next decision."""

        if (
            _nonempty_identifier(fallback_id, label="fallback_id")
            != self.contract.fallback_id
        ):
            raise ValueError("fallback_id does not match the frozen contract")
        if (
            _nonempty_identifier(candidate_id, label="candidate_id")
            != self.contract.candidate_id
        ):
            raise ValueError("candidate_id does not match the frozen contract")
        return candidate if self._authorized else fallback

    def snapshot(self) -> JointAdmissionSnapshotV2:
        alpha = (
            None
            if self._closed
            else self._alpha_schedule.alpha_for_epoch(self._epoch_index)
        )
        beta = (
            None
            if self._closed
            else self._beta_schedule.alpha_for_epoch(self._epoch_index)
        )
        minimum = self._gain_process.count >= self.config.minimum_resolved_trials
        harm_fraction: float | None = None
        if self._harm_process.count:
            harm_fraction = self._harm_process.harm_count / self._harm_process.count
        selected_mode = "candidate" if self._authorized else "exact-fallback"
        selected_id = (
            self.contract.candidate_id
            if self._authorized
            else self.contract.fallback_id
        )
        return JointAdmissionSnapshotV2(
            schema=SCHEMA,
            schema_version=SCHEMA_VERSION,
            decision_contract_digest=self.decision_contract_digest,
            contract_digest=self.contract.digest,
            candidate_id=self.contract.candidate_id,
            fallback_id=self.contract.fallback_id,
            selected_mode=selected_mode,
            selected_artifact_id=selected_id,
            epoch_index=self._epoch_index,
            epoch_reason=self._epoch_reason,
            closed=self._closed,
            issued_trial_count=self._issued_count,
            resolved_current_epoch_admission_count=self._gain_process.count,
            resolved_current_epoch_revocation_count=(
                0
                if self._revocation_process is None
                else self._revocation_process.count
            ),
            pending_trial_count=len(self._pending),
            ignored_closed_epoch_outcome_count=self._ignored_closed_epoch_count,
            current_epoch_alpha=alpha,
            current_epoch_beta=beta,
            cumulative_alpha=self._alpha_schedule.cumulative_alpha_through(
                self._epoch_index
            ),
            cumulative_beta=self._beta_schedule.cumulative_alpha_through(
                self._epoch_index
            ),
            shared_log_threshold=None if alpha is None else -math.log(alpha),
            revocation_log_threshold=None if beta is None else -math.log(beta),
            gain_log_e_value=self._gain_process.log_e_value,
            harm_log_e_value=self._harm_process.log_e_value,
            gain_maximum_log_e_value=self._gain_process.maximum_log_e_value,
            harm_maximum_log_e_value=self._harm_process.maximum_log_e_value,
            revocation_log_e_value=(
                None
                if self._revocation_process is None
                else self._revocation_process.log_e_value
            ),
            revocation_maximum_log_e_value=(
                None
                if self._revocation_process is None
                else self._revocation_process.maximum_log_e_value
            ),
            gain_evidence_ever_passed=self._gain_crossed,
            harm_evidence_ever_passed=self._harm_crossed,
            minimum_evidence_passed=minimum,
            authorized=self._authorized,
            ever_authorized_in_epoch=self._ever_authorized,
            harm_count=self._harm_process.harm_count,
            empirical_harm_fraction=harm_fraction,
        )

    def theorem_boundary(self) -> dict[str, object]:
        return {
            "admission_null": (
                "insufficient conditional mean capped gain OR conditional harm "
                "probability at or above the registered ceiling"
            ),
            "intersection_union_rule": (
                "both latched component e-processes must cross the same epoch-wise "
                "alpha boundary"
            ),
            "epoch_false_admission_bound": (
                "at most current_epoch_alpha when one fixed component null holds "
                "throughout the epoch"
            ),
            "lifetime_false_admission_bound": self.config.total_alpha,
            "revocation_rule": (
                "heavy-tailed mixture over deterministic post-admission change "
                "starts, tested with the epoch-wise beta budget"
            ),
            "required_information_order": (
                "contract, candidate, fallback, score, harm definition, trial, "
                "and betting grids fixed before each paired outcome reveal"
            ),
            "excluded_claims": (
                "physical safety, arbitrary nonstationarity, universal transport, "
                "unclipped-loss validity, and causal identification"
            ),
        }
