"""Lower-envelope e-factor composition for switching-union admission.

Suppose a conjunctive deployment decision has component nulls H_1,...,H_K.
For every component k, let F_{t,k,theta_k} be a nonnegative e-factor under
H_k, conditional on the information available before outcome t is revealed.
Under the pointwise union null, at least one component null holds at every
reveal, but the active component may change over time.

For a fixed tuple theta=(theta_1,...,theta_K), define

    L_t(theta) = min_k F_{t,k,theta_k}.

Whichever component null is active, L_t is bounded above by a valid factor.
Therefore E[L_t(theta) | F_{t-1}] <= 1. Products of these factors and any
outcome-independent mixture over fixed tuples are e-processes. This theorem
does not require a common scalar score or a stable invalidity mode.

This module instantiates the construction for two physical-twin admission
conditions:

* bounded mean utility, using F_gain = 1 + lambda * G for G in [-1,1]; and
* harmful-update probability below rho, using a Bernoulli likelihood-ratio
  factor toward a fixed alternative q < rho.

The Cartesian mixture allows the gain and harm factors to be tuned
independently. Version 3 is recovered by restricting both components to a
shared betting fraction after linearizing the Bernoulli factor.

The guarantee is statistical and conditional on frozen factor families,
candidate, fallback, score, harm definition, and reveal filtration. It is not
a physical-safety or causal-identification guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np

SCHEMA: Final = "bayesian-phystwin.anytime-factor-envelope-v4"
SCHEMA_VERSION: Final = 4


def _probability(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{label} must lie strictly between zero and one")
    return result


def _bounded_score(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and lie in [-1, 1]")
    return result


def _fraction_vector(values: tuple[float, ...], *, label: str) -> np.ndarray:
    if not values:
        raise ValueError(f"{label} must not be empty")
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 1
        or not np.isfinite(result).all()
        or np.any(result <= 0.0)
        or np.any(result >= 1.0)
    ):
        raise ValueError(f"{label} must be a finite vector in (0, 1)")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{label} must contain unique values")
    result.setflags(write=False)
    return result


def _log_mean_exp(log_values: np.ndarray) -> float:
    flattened = np.asarray(log_values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        raise ValueError("log_values must not be empty")
    if np.isnan(flattened).any() or np.isposinf(flattened).any():
        raise ValueError("log_values must not contain NaN or positive infinity")
    maximum = float(np.max(flattened))
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(float(np.mean(np.exp(flattened - maximum))))


def bounded_gain_factor(*, gain_score: float, bet_fraction: float) -> float:
    """Return ``1 + lambda G``, a gain-null e-factor for ``G in [-1,1]``."""

    score = _bounded_score(gain_score, label="gain_score")
    bet = _probability(bet_fraction, label="bet_fraction")
    factor = 1.0 + bet * score
    if not math.isfinite(factor) or factor <= 0.0:
        raise AssertionError("bounded gain factor must be positive and finite")
    return factor


def bernoulli_harm_factor(
    *,
    harmful: bool,
    maximum_harm_rate: float,
    alternative_fraction: float,
) -> float:
    """Return the Bernoulli LR factor for an alternative below the ceiling."""

    if type(harmful) is not bool:
        raise ValueError("harmful must be a literal bool")
    ceiling = _probability(maximum_harm_rate, label="maximum_harm_rate")
    fraction = _probability(alternative_fraction, label="alternative_fraction")
    alternative = ceiling * fraction
    factor = alternative / ceiling if harmful else (1.0 - alternative) / (1.0 - ceiling)
    if not math.isfinite(factor) or factor <= 0.0:
        raise AssertionError("harm factor must be positive and finite")
    return factor


def lower_envelope_factor(factors: Iterable[float]) -> float:
    """Return the minimum of positive finite component factors."""

    values = tuple(float(value) for value in factors)
    if not values:
        raise ValueError("at least one component factor is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("component factors must be positive and finite")
    return min(values)


@dataclass(frozen=True, slots=True)
class FactorEnvelopeUpdateV4:
    count: int
    gain_score: float
    harmful: bool
    minimum_component_factor: float
    maximum_component_factor: float
    log_e_value: float
    maximum_log_e_value: float


class LowerEnvelopeMixtureEProcess:
    """Cartesian mixture of independently tuned lower-envelope e-factors."""

    def __init__(
        self,
        *,
        gain_bet_fractions: tuple[float, ...],
        maximum_harm_rate: float,
        harm_alternative_fractions: tuple[float, ...],
    ) -> None:
        self._gain_bets = _fraction_vector(
            gain_bet_fractions,
            label="gain_bet_fractions",
        )
        self._ceiling = _probability(
            maximum_harm_rate,
            label="maximum_harm_rate",
        )
        self._harm_fractions = _fraction_vector(
            harm_alternative_fractions,
            label="harm_alternative_fractions",
        )
        self._harm_alternatives = self._ceiling * self._harm_fractions
        self._log_wealth = np.zeros(
            (len(self._gain_bets), len(self._harm_fractions)),
            dtype=np.float64,
        )
        self._count = 0
        self._maximum_log_e_value = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def component_count(self) -> int:
        return int(self._log_wealth.size)

    @property
    def log_e_value(self) -> float:
        return _log_mean_exp(self._log_wealth)

    @property
    def maximum_log_e_value(self) -> float:
        return self._maximum_log_e_value

    def update(
        self,
        *,
        gain_score: float,
        harmful: bool,
    ) -> FactorEnvelopeUpdateV4:
        score = _bounded_score(gain_score, label="gain_score")
        if type(harmful) is not bool:
            raise ValueError("harmful must be a literal bool")
        gain_factors = 1.0 + self._gain_bets * score
        harm_factors = (
            self._harm_alternatives / self._ceiling
            if harmful
            else (1.0 - self._harm_alternatives) / (1.0 - self._ceiling)
        )
        envelope = np.minimum(gain_factors[:, None], harm_factors[None, :])
        if np.any(envelope <= 0.0) or not np.isfinite(envelope).all():
            raise AssertionError("factor envelope must be positive and finite")
        self._log_wealth += np.log(envelope)
        self._count += 1
        current = self.log_e_value
        self._maximum_log_e_value = max(self._maximum_log_e_value, current)
        return FactorEnvelopeUpdateV4(
            count=self._count,
            gain_score=score,
            harmful=harmful,
            minimum_component_factor=float(np.min(envelope)),
            maximum_component_factor=float(np.max(envelope)),
            log_e_value=current,
            maximum_log_e_value=self._maximum_log_e_value,
        )

    def theorem_boundary(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "pointwise_union_null": (
                "at each reveal at least one registered component factor has "
                "conditional expectation at most one; the active component may "
                "change arbitrarily with the past"
            ),
            "composition": (
                "for every fixed parameter tuple, take the pointwise minimum "
                "across component e-factors, multiply over time, then mix across "
                "tuples using outcome-independent weights"
            ),
            "specialization": (
                "bounded-gain linear factors crossed with Bernoulli harm-rate "
                "likelihood-ratio factors"
            ),
            "component_count": self.component_count,
            "excluded_claims": (
                "physical safety, validity after unregistered factor adaptation, "
                "causal identification, and universal distribution shift"
            ),
        }
