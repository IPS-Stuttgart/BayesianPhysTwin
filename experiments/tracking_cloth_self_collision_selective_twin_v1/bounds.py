"""Finite-sample bounds for selective simulator competence.

The module intentionally separates three claims:

1. exact fallback is a deterministic software property;
2. source-side context admission can be certified simultaneously over a finite
   family using bounded-loss concentration; and
3. a frozen confirmation policy can receive an exact one-sided binomial bound
   on harmful accepted uses under the registered exchangeability assumption.

None of the routines turns a small public benchmark into a universal safety
claim. The caller must retain the statistical unit and deployment population.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

SCHEMA: Final = "bayesian-phystwin.selective-competence-bound"
SCHEMA_VERSION: Final = 1


def _probability(value: float, *, name: str, closed_zero: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real probability")
    result = float(value)
    lower_ok = result >= 0.0 if closed_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok or result >= 1.0:
        relation = "[0, 1)" if closed_zero else "(0, 1)"
        raise ValueError(f"{name} must lie in {relation}")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """Return P[X <= k] for X ~ Binomial(n, probability)."""

    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * log_p
        + (n - i) * log_q
        for i in range(k + 1)
    ]
    largest = max(terms)
    return math.exp(largest) * sum(math.exp(term - largest) for term in terms)


def clopper_pearson_upper(
    harmful_count: int,
    accepted_count: int,
    *,
    alpha: float = 0.05,
    iterations: int = 100,
) -> float:
    """Return the exact one-sided Clopper--Pearson upper endpoint.

    An empty accepted set receives the vacuous bound 1.0.
    """

    if isinstance(accepted_count, bool) or not isinstance(accepted_count, int):
        raise ValueError("accepted_count must be an integer")
    if isinstance(harmful_count, bool) or not isinstance(harmful_count, int):
        raise ValueError("harmful_count must be an integer")
    if accepted_count < 0 or harmful_count < 0 or harmful_count > accepted_count:
        raise ValueError("harmful_count must lie between zero and accepted_count")
    alpha_value = _probability(alpha, name="alpha", closed_zero=False)
    _positive_integer(iterations, name="iterations")
    if accepted_count == 0 or harmful_count == accepted_count:
        return 1.0

    low, high = 0.0, 1.0
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        if _binomial_cdf(harmful_count, accepted_count, midpoint) > alpha_value:
            low = midpoint
        else:
            high = midpoint
    return high


def hoeffding_upper(
    empirical_mean: float,
    sample_count: int,
    *,
    lower: float,
    upper: float,
    alpha: float = 0.05,
    family_size: int = 1,
) -> float:
    """Return a simultaneous one-sided Hoeffding upper endpoint."""

    mean = float(empirical_mean)
    lo, hi = float(lower), float(upper)
    if not all(math.isfinite(value) for value in (mean, lo, hi)) or hi <= lo:
        raise ValueError("mean and finite ordered bounds are required")
    if mean < lo - 1e-12 or mean > hi + 1e-12:
        raise ValueError("empirical_mean must lie inside [lower, upper]")
    n = _positive_integer(sample_count, name="sample_count")
    m = _positive_integer(family_size, name="family_size")
    alpha_value = _probability(alpha, name="alpha", closed_zero=False)
    radius = (hi - lo) * math.sqrt(math.log(m / alpha_value) / (2.0 * n))
    return min(hi, mean + radius)


@dataclass(frozen=True, slots=True)
class SelectiveCompetenceCertificateV1:
    """Compact certificate for one frozen selective policy."""

    accepted_count: int
    harmful_count: int
    empirical_excess_loss: float
    harm_alpha: float
    regret_alpha: float
    loss_lower: float
    loss_upper: float
    group_count: int
    family_size: int
    harm_upper_bound: float | None = None
    regret_upper_bound: float | None = None

    def __post_init__(self) -> None:
        if self.accepted_count < 0 or self.harmful_count < 0:
            raise ValueError("counts must be nonnegative")
        if self.harmful_count > self.accepted_count:
            raise ValueError("harmful_count exceeds accepted_count")
        harm = clopper_pearson_upper(
            self.harmful_count,
            self.accepted_count,
            alpha=self.harm_alpha,
        )
        regret = hoeffding_upper(
            self.empirical_excess_loss,
            self.group_count,
            lower=self.loss_lower,
            upper=self.loss_upper,
            alpha=self.regret_alpha,
            family_size=self.family_size,
        )
        if self.harm_upper_bound is not None and not math.isclose(
            float(self.harm_upper_bound), harm, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("harm_upper_bound does not match the counts")
        if self.regret_upper_bound is not None and not math.isclose(
            float(self.regret_upper_bound), regret, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("regret_upper_bound does not match the evidence")
        object.__setattr__(self, "harm_upper_bound", harm)
        object.__setattr__(self, "regret_upper_bound", regret)

    def authorizes(self, *, maximum_harm: float, maximum_regret: float = 0.0) -> bool:
        maximum_harm_value = _probability(maximum_harm, name="maximum_harm")
        maximum_regret_value = float(maximum_regret)
        if not math.isfinite(maximum_regret_value):
            raise ValueError("maximum_regret must be finite")
        harm_upper_bound = self.harm_upper_bound
        regret_upper_bound = self.regret_upper_bound
        if harm_upper_bound is None or regret_upper_bound is None:
            raise RuntimeError("certificate endpoints were not initialized")
        return bool(
            self.accepted_count > 0
            and harm_upper_bound <= maximum_harm_value
            and regret_upper_bound <= maximum_regret_value
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "accepted_count": self.accepted_count,
            "harmful_count": self.harmful_count,
            "empirical_excess_loss": self.empirical_excess_loss,
            "harm_alpha": self.harm_alpha,
            "regret_alpha": self.regret_alpha,
            "loss_lower": self.loss_lower,
            "loss_upper": self.loss_upper,
            "group_count": self.group_count,
            "family_size": self.family_size,
            "harm_upper_bound": self.harm_upper_bound,
            "regret_upper_bound": self.regret_upper_bound,
        }
