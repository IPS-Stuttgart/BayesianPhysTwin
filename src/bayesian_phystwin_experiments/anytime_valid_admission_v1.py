"""Anytime-valid admission and revocation for shadow-evaluated simulator corrections.

The controller treats a candidate correction and the caller-owned physical
fallback as a paired sequential experiment.  It promotes the candidate only
when a nonnegative e-process crosses an alpha-spending boundary.  A second
one-sided e-process can revoke an admitted candidate after evidence of harm.

The guarantee concerns the registered clipped paired gain, not universal
physical safety.  Candidate and fallback losses must both be computed when the
delayed outcome becomes available, even if the candidate was not deployed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Generic, TypeVar

import numpy as np


T = TypeVar("T")


class DeploymentState(str, Enum):
    """Deployment choice for the next decision."""

    FALLBACK = "fallback"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class AnytimeAdmissionConfig:
    """Frozen configuration for an anytime-valid admission stream.

    ``gain_margin`` is the minimum normalized expected gain that admission is
    required to establish.  ``harm_margin`` is the tolerated normalized loss
    disadvantage after admission.  Both are expressed relative to
    ``loss_cap``.  The registered loss is clipped before testing, so all
    guarantees are explicitly about that clipped loss.
    """

    alpha: float = 0.05
    beta: float = 0.05
    loss_cap: float = 1.0
    gain_margin: float = 0.0
    harm_margin: float = 0.0
    allow_reentry: bool = True
    lambdas: tuple[float, ...] = (
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.70,
        0.90,
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        if not math.isfinite(self.loss_cap) or self.loss_cap <= 0.0:
            raise ValueError("loss_cap must be finite and positive")
        if not 0.0 <= self.gain_margin < 1.0:
            raise ValueError("gain_margin must lie in [0, 1)")
        if not 0.0 <= self.harm_margin < 1.0:
            raise ValueError("harm_margin must lie in [0, 1)")
        if not self.lambdas:
            raise ValueError("at least one betting fraction is required")
        if any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.lambdas
        ):
            raise ValueError("every betting fraction must lie in (0, 1)")
        if len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("betting fractions must be unique")


@dataclass(frozen=True)
class EProcessRecord:
    """One update of a mixture betting e-process."""

    observation_count: int
    evidence: float
    log_e_value: float
    e_value: float


@dataclass(frozen=True)
class AdmissionRecord:
    """Auditable result of revealing one paired delayed outcome."""

    reveal_index: int
    epoch: int
    state_before: str
    state_after: str
    event: str
    candidate_loss: float
    fallback_loss: float
    raw_gain: float
    clipped_normalized_gain: float
    test_evidence: float
    e_value: float
    boundary: float
    budget: float
    clipped: bool


class MixtureBettingEProcess:
    """Convex mixture of fixed-fraction betting e-processes.

    For adapted observations ``X_t in [-1, 1]`` satisfying
    ``E[X_t | F_{t-1}] <= 0``, each component

    ``prod_i (1 + lambda * X_i)``

    is a nonnegative supermartingale for ``lambda in (0, 1)``.  A fixed convex
    mixture is therefore an e-process as well.  Ville's inequality permits
    arbitrary stopping and continuous monitoring.
    """

    def __init__(self, lambdas: tuple[float, ...]) -> None:
        values = np.asarray(lambdas, dtype=np.float64)
        if (
            values.ndim != 1
            or len(values) == 0
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
            or np.any(values >= 1.0)
        ):
            raise ValueError("lambdas must be a finite nonempty vector in (0, 1)")
        self._lambdas = values
        self._log_components = np.zeros_like(values)
        self._log_weights = np.full_like(values, -math.log(len(values)))
        self._count = 0

    @property
    def observation_count(self) -> int:
        return self._count

    @property
    def log_e_value(self) -> float:
        terms = self._log_weights + self._log_components
        maximum = float(np.max(terms))
        return maximum + math.log(float(np.sum(np.exp(terms - maximum))))

    @property
    def e_value(self) -> float:
        value = self.log_e_value
        return math.inf if value >= math.log(np.finfo(float).max) else math.exp(value)

    def update(self, evidence: float) -> EProcessRecord:
        if not math.isfinite(evidence) or not -1.0 <= evidence <= 1.0:
            raise ValueError("evidence must be finite and lie in [-1, 1]")
        increments = 1.0 + self._lambdas * evidence
        if np.any(increments <= 0.0):
            raise AssertionError("betting increment lost nonnegativity")
        self._log_components += np.log(increments)
        self._count += 1
        return EProcessRecord(
            observation_count=self._count,
            evidence=float(evidence),
            log_e_value=self.log_e_value,
            e_value=self.e_value,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "lambdas": self._lambdas.tolist(),
            "log_components": self._log_components.tolist(),
            "log_weights": self._log_weights.tolist(),
            "observation_count": self._count,
            "log_e_value": self.log_e_value,
            "e_value": self.e_value,
        }


def epoch_budget(total: float, epoch: int, *, allow_reentry: bool) -> float:
    """Allocate a summable error budget to a one-indexed admission epoch.

    With re-entry, ``total / (epoch * (epoch + 1))`` telescopes and sums to
    ``total`` over infinitely many epochs.  Without re-entry, the only epoch
    receives the full budget.
    """

    if not 0.0 < total < 1.0:
        raise ValueError("total budget must lie in (0, 1)")
    if epoch < 1:
        raise ValueError("epoch must be positive")
    if not allow_reentry and epoch != 1:
        raise ValueError("non-reentrant controllers have only one epoch")
    return total if not allow_reentry else total / (epoch * (epoch + 1))


def clipped_gain(
    *,
    candidate_loss: float,
    fallback_loss: float,
    loss_cap: float,
) -> tuple[float, float, bool]:
    """Return raw and clipped normalized fallback-minus-candidate gain."""

    values = (candidate_loss, fallback_loss, loss_cap)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("losses and loss_cap must be finite")
    if candidate_loss < 0.0 or fallback_loss < 0.0:
        raise ValueError("losses must be nonnegative")
    if loss_cap <= 0.0:
        raise ValueError("loss_cap must be positive")
    raw = fallback_loss - candidate_loss
    normalized = raw / loss_cap
    clipped = float(np.clip(normalized, -1.0, 1.0))
    return raw, clipped, not math.isclose(normalized, clipped, abs_tol=0.0)


def margin_evidence(value: float, margin: float) -> float:
    """Map a bounded gain to a bounded zero-mean-null test variable.

    If ``value in [-1, 1]`` and the null is ``E[value | past] <= margin``, then
    ``(value - margin) / (1 + margin)`` lies in ``[-1, 1]`` and has conditional
    mean at most zero.
    """

    if not -1.0 <= value <= 1.0:
        raise ValueError("value must lie in [-1, 1]")
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must lie in [0, 1)")
    transformed = (value - margin) / (1.0 + margin)
    if not -1.0 <= transformed <= 1.0:
        raise AssertionError("margin transform escaped [-1, 1]")
    return transformed


class AnytimeAdmissionController(Generic[T]):
    """Fail-closed sequential controller for one fixed candidate correction.

    Outcomes may be delayed.  The controller's deployment choice changes only
    after ``observe`` receives the paired candidate and fallback losses, so the
    newly crossed boundary affects the next decision rather than the decision
    whose outcome supplied the evidence.
    """

    def __init__(
        self,
        config: AnytimeAdmissionConfig,
        *,
        candidate_id: str,
    ) -> None:
        if not candidate_id:
            raise ValueError("candidate_id must be nonempty")
        self.config = config
        self.candidate_id = candidate_id
        self.state = DeploymentState.FALLBACK
        self.epoch = 1
        self.reveal_index = 0
        self._promotion = MixtureBettingEProcess(config.lambdas)
        self._harm: MixtureBettingEProcess | None = None
        self._history: list[AdmissionRecord] = []
        self._closed = False

    @property
    def history(self) -> tuple[AdmissionRecord, ...]:
        return tuple(self._history)

    @property
    def current_alpha(self) -> float:
        return epoch_budget(
            self.config.alpha,
            self.epoch,
            allow_reentry=self.config.allow_reentry,
        )

    @property
    def current_beta(self) -> float:
        return epoch_budget(
            self.config.beta,
            self.epoch,
            allow_reentry=self.config.allow_reentry,
        )

    def select(self, *, fallback: T, candidate: T) -> T:
        """Return the selected object without copying or reconstructing it."""

        return candidate if self.state is DeploymentState.CANDIDATE else fallback

    def observe(
        self,
        *,
        candidate_loss: float,
        fallback_loss: float,
    ) -> AdmissionRecord:
        """Reveal one paired outcome and update the appropriate e-process."""

        if self._closed:
            raise RuntimeError("controller is closed after a terminal revocation")
        raw, gain, was_clipped = clipped_gain(
            candidate_loss=candidate_loss,
            fallback_loss=fallback_loss,
            loss_cap=self.config.loss_cap,
        )
        self.reveal_index += 1
        state_before = self.state

        if state_before is DeploymentState.FALLBACK:
            evidence = margin_evidence(gain, self.config.gain_margin)
            update = self._promotion.update(evidence)
            budget = self.current_alpha
            boundary = 1.0 / budget
            if update.e_value >= boundary:
                self.state = DeploymentState.CANDIDATE
                self._harm = MixtureBettingEProcess(self.config.lambdas)
                event = "admit"
            else:
                event = "remain-fallback"
        else:
            if self._harm is None:
                raise AssertionError("candidate state has no harm monitor")
            evidence = margin_evidence(-gain, self.config.harm_margin)
            update = self._harm.update(evidence)
            budget = self.current_beta
            boundary = 1.0 / budget
            if update.e_value >= boundary:
                self.state = DeploymentState.FALLBACK
                event = "revoke"
                if self.config.allow_reentry:
                    self.epoch += 1
                    self._promotion = MixtureBettingEProcess(self.config.lambdas)
                    self._harm = None
                else:
                    self._closed = True
            else:
                event = "remain-candidate"

        record = AdmissionRecord(
            reveal_index=self.reveal_index,
            epoch=self.epoch if event != "revoke" else self.epoch - int(self.config.allow_reentry),
            state_before=state_before.value,
            state_after=self.state.value,
            event=event,
            candidate_loss=float(candidate_loss),
            fallback_loss=float(fallback_loss),
            raw_gain=float(raw),
            clipped_normalized_gain=float(gain),
            test_evidence=float(evidence),
            e_value=float(update.e_value),
            boundary=float(boundary),
            budget=float(budget),
            clipped=was_clipped,
        )
        self._history.append(record)
        return record

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "contract": "anytime-valid-simulator-admission-state-v1",
            "candidate_id": self.candidate_id,
            "state": self.state.value,
            "epoch": self.epoch,
            "reveal_index": self.reveal_index,
            "closed": self._closed,
            "config": asdict(self.config),
            "current_alpha": self.current_alpha if not self._closed else None,
            "current_beta": self.current_beta if not self._closed else None,
            "promotion_e_process": self._promotion.snapshot(),
            "harm_e_process": None if self._harm is None else self._harm.snapshot(),
            "history": [asdict(record) for record in self._history],
            "guarantee_boundary": (
                "Under the registered clipped-gain null and predictable delayed "
                "feedback, Ville plus the summable epoch budget bounds the "
                "probability of any false admission by alpha. This is not a "
                "deployment-safety or unclipped-loss guarantee."
            ),
        }
