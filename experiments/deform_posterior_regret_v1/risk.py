"""Future-blind regret risk for fixed forecasts; all arrays use metres.

Squared-loss regret is affine in the future observation, not quadratic:
R = (||C||^2 - ||B||^2 - 2(C-B)'Y) / d.
Its expectation depends only on E[Y]. Covariance affects tails, not this mean.
These are established squared-loss/Gaussian identities, not a new theorem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from scipy.special import ndtr


class CovarianceOperator(Protocol):
    def quadratic_form(self, direction: np.ndarray) -> float: ...


@dataclass(frozen=True)
class TemporalBlocks:
    """[node, coordinate, time, time], independent node/coordinate blocks."""

    blocks: np.ndarray

    def quadratic_form(self, direction: np.ndarray) -> float:
        a = np.asarray(self.blocks, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        if d.ndim != 3 or a.shape != (d.shape[1], d.shape[2], len(d), len(d)):
            raise ValueError("covariance and forecast dimensions differ")
        if not np.isfinite(a).all():
            raise ValueError("nonfinite covariance")
        if not np.allclose(a, a.swapaxes(-1, -2), rtol=1e-10, atol=1e-15):
            raise ValueError("asymmetric covariance")
        # No clipping/jitter/pseudoinverse: an invalid operator must abstain.
        for block in a.reshape(-1, len(d), len(d)):
            np.linalg.cholesky(block)
        result = np.einsum("tnc,ncts,snc->", d, a, d)
        if not np.isfinite(result) or result < 0:
            raise ValueError("invalid covariance quadratic form")
        return float(result)


@dataclass(frozen=True)
class RiskResult:
    regret_mean_mm2: float | None
    regret_variance_mm4: float | None
    harm_probability: float | None
    valid: bool
    failure: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def forecast_risk(
    baseline: np.ndarray,
    candidate: np.ndarray,
    predictive_mean: np.ndarray,
    covariance: CovarianceOperator,
    *,
    harmful_regret_mm2: float = 1.0,
) -> RiskResult:
    """No future observations accepted; invalid uncertainty implies fallback."""
    try:
        b, c, mu = [
            np.asarray(x, dtype=np.float64)
            for x in (baseline, candidate, predictive_mean)
        ]
        if b.shape != c.shape or b.shape != mu.shape or not b.size:
            raise ValueError("forecast shapes differ or are empty")
        if not all(np.isfinite(x).all() for x in (b, c, mu)):
            raise ValueError("nonfinite forecast")
        if not np.isfinite(harmful_regret_mm2) or harmful_regret_mm2 < 0:
            raise ValueError("invalid harm threshold")
        difference = c - b
        mean = float(np.mean(c * c - b * b - 2 * difference * mu)) * 1e6
        variance = 4 * covariance.quadratic_form(difference) / b.size**2 * 1e12
        probability = (
            float(ndtr((mean - harmful_regret_mm2) / np.sqrt(variance)))
            if variance > 0
            else float(mean > harmful_regret_mm2)
        )
        return RiskResult(mean, variance, probability, True)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return RiskResult(None, None, None, False, str(error))


def ranked_acceptance(scores, keys, valid, fraction=0.5):
    """Fixed denominator, deterministic lexical ties; invalid rows never update."""
    scores = np.asarray(scores, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if len(scores) != len(keys) or len(valid) != len(keys):
        raise ValueError("selection dimensions differ")
    if len(set(keys)) != len(keys) or not 0 <= fraction <= 1:
        raise ValueError("duplicate context identifiers or invalid coverage")
    order = sorted(
        np.flatnonzero(valid & np.isfinite(scores)),
        key=lambda i: (float(scores[i]), keys[i]),
    )
    selected = np.zeros(len(keys), dtype=bool)
    selected[order[: int(np.floor(fraction * len(keys)))]] = True
    return selected


def deploy(baseline, candidate, accept):
    """Do not blend, cast, or modify fallback bytes."""
    return candidate.copy() if accept else baseline.copy()
