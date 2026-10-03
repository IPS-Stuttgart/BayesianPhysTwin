"""Query and conditioning interfaces deliberately exclude future observations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .gp import ResidualGP

FREE = np.arange(2, 10)
PANELS = (tuple(FREE[::2]), tuple(FREE[1::2]))
OFFSETS = np.arange(-46, 0, 5)


def pairs(panel: int, budget: int = 2) -> tuple[tuple[int, ...], ...]:
    if panel not in (0, 1) or budget not in (1, 2, 4):
        raise ValueError("unregistered panel or budget")
    return tuple(combinations(PANELS[panel], budget))


def prefix_times(endpoint: int) -> np.ndarray:
    if not isinstance(endpoint, (int, np.integer)) or endpoint < 46:
        raise ValueError("prefix endpoint has insufficient history")
    return endpoint + OFFSETS


class PrefixProvider:
    """Owns only permitted past samples, not a complete trajectory."""

    def __init__(
        self,
        times: np.ndarray,
        nodes: tuple[int, ...],
        values: np.ndarray,
        *,
        endpoint: int,
    ):
        self.times = np.asarray(times, dtype=np.int64).copy()
        self.nodes = tuple(nodes)
        self._values = np.asarray(values, dtype=np.float64).copy()
        if (
            self._values.shape != (10, len(nodes), 3)
            or len(set(nodes)) != len(nodes)
            or not np.isfinite(self._values).all()
            or self.times.shape != (10,)
            or not np.array_equal(self.times, prefix_times(endpoint))
        ):
            raise ValueError("invalid restricted prefix")
        self.revealed: list[tuple[int, ...]] = []

    def reveal(self, selected: tuple[int, ...]) -> np.ndarray:
        selected = tuple(sorted(selected))
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("empty or duplicate query")
        if not set(selected).issubset(self.nodes):
            raise PermissionError("attempt to reveal a held-out identity")
        self.revealed.append(selected)
        return self._values[:, [self.nodes.index(n) for n in selected]].copy()


@dataclass(frozen=True)
class QueryScore:
    nodes: tuple[int, ...]
    expected_mse_reduction_m2: float
    valid: bool
    failure: str | None = None


@dataclass(frozen=True)
class Forecast:
    positions: np.ndarray
    valid: bool
    failure: str | None = None


def choose(scores: list[QueryScore]) -> QueryScore:
    valid = [s for s in scores if s.valid and np.isfinite(s.expected_mse_reduction_m2)]
    if not valid:
        return QueryScore((), 0.0, False, "no_valid_query")
    return min(valid, key=lambda s: (-s.expected_mse_reduction_m2, s.nodes))


@dataclass
class Context:
    open_loop: np.ndarray
    design: np.ndarray
    coefficient_covariance: np.ndarray
    noise_m2: np.ndarray
    rotation: np.ndarray
    endpoint: int
    horizon: int
    panel: int

    def __post_init__(self) -> None:
        h = len(self.open_loop)
        p = self.design.shape[-1]
        if (
            self.open_loop.shape != (h, 12, 3)
            or self.design.shape[:2] != (h, 8)
            or self.coefficient_covariance.shape != (3, p, p)
            or self.noise_m2.shape != (3,)
            or self.rotation.shape != (3, 3)
            or self.panel not in (0, 1)
            or self.horizon < 1
            or self.endpoint + self.horizon > h
        ):
            raise ValueError("invalid forecast context")
        prefix_times(self.endpoint)
        if not all(
            np.isfinite(a).all()
            for a in (
                self.open_loop,
                self.design,
                self.coefficient_covariance,
                self.noise_m2,
                self.rotation,
            )
        ) or np.any(self.noise_m2 <= 0):
            raise ValueError("nonfinite context or nonpositive observation noise")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-10):
            raise ValueError("nonorthogonal coordinate frame")

    @property
    def forecast_times(self) -> np.ndarray:
        return np.arange(self.endpoint, self.endpoint + self.horizon)

    @property
    def scoring_nodes(self) -> tuple[int, ...]:
        return PANELS[1 - self.panel]

    def _rows(self, times: np.ndarray, nodes: tuple[int, ...]) -> np.ndarray:
        return self.design[np.ix_(times, np.asarray(nodes) - 2)].reshape(
            -1, self.design.shape[-1]
        )

    def _operator(self, selected: tuple[int, ...]):
        selected = tuple(sorted(selected))
        if selected not in pairs(self.panel, len(selected)):
            raise ValueError("query outside registered panel")
        q = self._rows(prefix_times(self.endpoint), selected)
        f = self._rows(self.forecast_times, self.scoring_nodes)
        covariance = self.coefficient_covariance
        if not np.allclose(covariance, covariance.transpose(0, 2, 1), atol=1e-12):
            raise ValueError("asymmetric covariance")
        # Reject genuinely indefinite operators; roundoff near zero is tolerated.
        eig = np.linalg.eigvalsh(covariance)
        scale = max(float(np.max(np.abs(eig))), 1e-12)
        if float(eig.min()) < -1e-10 * scale:
            raise ValueError("indefinite covariance")
        cross = np.einsum("fp,cpq,oq->cfo", f, covariance, q, optimize=True)
        qq = np.einsum("op,cpq,rq->cor", q, covariance, q, optimize=True)
        factors = [
            cho_factor((a + a.T) / 2 + noise * np.eye(len(q)), lower=True)
            for a, noise in zip(qq, self.noise_m2, strict=True)
        ]
        return selected, q, f, cross, factors

    def score(self, selected: tuple[int, ...]) -> QueryScore:
        try:
            selected, _, f, cross, factors = self._operator(selected)
            reduction = sum(
                float(np.sum(c * cho_solve(fac, c.T).T))
                for c, fac in zip(cross, factors, strict=True)
            )
            return QueryScore(selected, reduction / (3 * len(f)), True)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return QueryScore(tuple(sorted(selected)), 0.0, False, str(exc))

    def forecast(self, selected: tuple[int, ...], observed: np.ndarray) -> Forecast:
        baseline = self.open_loop[self.forecast_times].copy()
        if not selected:
            return Forecast(baseline, True)
        try:
            original = tuple(selected)
            selected, q, _, _, factors = self._operator(selected)
            obs = np.asarray(observed, dtype=np.float64)
            if obs.shape != (10, len(selected), 3) or not np.isfinite(obs).all():
                raise ValueError("invalid observation values")
            obs = obs[:, [original.index(n) for n in selected]]
            mean = self.open_loop[np.ix_(prefix_times(self.endpoint), selected)]
            innovation = ((obs - mean) @ self.rotation).reshape(-1, 3)
            f = self._rows(self.forecast_times, tuple(FREE))
            cross = np.einsum(
                "fp,cpq,oq->cfo", f, self.coefficient_covariance, q, optimize=True
            )
            delta = np.stack(
                [
                    c @ cho_solve(fac, innovation[:, k])
                    for k, (c, fac) in enumerate(zip(cross, factors, strict=True))
                ],
                axis=-1,
            )
            world_delta = delta.reshape(self.horizon, 8, 3) @ self.rotation.T
            result = baseline.copy()
            result[:, 2:-2] += world_delta
            if not np.isfinite(result).all():
                raise ValueError("nonfinite conditional forecast")
            return Forecast(result, True)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return Forecast(baseline, False, str(exc))


def context_from_gp(
    model: ResidualGP,
    features: np.ndarray,
    baseline: np.ndarray,
    rotation: np.ndarray,
    *,
    endpoint: int,
    horizon: int,
    panel: int,
    inflation: float,
    shrinkage: float = 0.25,
) -> Context:
    design = model.design(features.reshape(-1, features.shape[-1])).reshape(
        *features.shape[:2], -1
    )
    open_loop = baseline.copy()
    open_loop[:, 2:-2] += shrinkage * (design @ model.weights) @ rotation.T
    covariance = (
        shrinkage**2
        * model.residual_variance[:, None, None]
        * model.precision_inverse[None]
    )
    noise = np.maximum(inflation * model.residual_variance, 1e-12)
    return Context(
        open_loop, design, covariance, noise, rotation, endpoint, horizon, panel
    )


def headroom_summary(
    mse: np.ndarray, names: list[str], *, minimum: float = 0.10
) -> dict:
    """Scoring-only function: [recording, endpoint, panel, candidate pair]."""
    mse = np.asarray(mse, dtype=np.float64)
    if (
        mse.shape != (len(names), 3, 2, 6)
        or not np.isfinite(mse).all()
        or np.any(mse < 0)
        or len(set(names)) != len(names)
    ):
        raise ValueError("invalid headroom score tensor")
    fixed_indices = np.argmin(mse.mean(axis=(0, 1)), axis=-1)
    fixed = np.stack([mse[:, :, p, i] for p, i in enumerate(fixed_indices)], axis=-1)
    oracle = mse.min(axis=-1)
    fixed_rmse, oracle_rmse = (
        float(np.sqrt(fixed.mean())),
        float(np.sqrt(oracle.mean())),
    )
    improvement = 0.0 if fixed_rmse == 0 else 1 - oracle_rmse / fixed_rmse
    return {
        "fixed_pair_indices": fixed_indices.tolist(),
        "best_fixed_rmse_mm": fixed_rmse * 1000,
        "oracle_rmse_mm": oracle_rmse * 1000,
        "relative_rmse_headroom": improvement,
        "minimum_required": minimum,
        "passed": bool(len(names) == 9 and improvement >= minimum),
        "decision": "advance_to_selector_development"
        if len(names) == 9 and improvement >= minimum
        else "stop_insufficient_validation_headroom",
        "statistical_unit": "complete_recording",
        "per_recording": [
            {
                "name": name,
                "fixed_rmse_mm": float(np.sqrt(fixed[i].mean()) * 1000),
                "oracle_rmse_mm": float(np.sqrt(oracle[i].mean()) * 1000),
            }
            for i, name in enumerate(names)
        ],
        "oracle_is_scoring_only": True,
    }
