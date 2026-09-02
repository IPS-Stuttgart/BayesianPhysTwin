"""Decision-directed probe selection on the public RCT force release."""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .rct_real_decision_protocol import (
    HELD_INTERVENTION,
    MANDATORY_ANCHOR,
    REGISTERED_INDENTATIONS_MM,
    SELECTABLE_PROBES,
)

Trajectory = tuple[int, int]
SelectorName = Literal["decision_directed", "system_identification", "fixed"]
TRAJECTORY_ORDER: tuple[Trajectory, ...] = (
    MANDATORY_ANCHOR,
    *SELECTABLE_PROBES,
    HELD_INTERVENTION,
)
COORDINATE_COUNT = len(TRAJECTORY_ORDER) * len(REGISTERED_INDENTATIONS_MM)
_MAXIMUM_INDENTATION_MISMATCH_MM = 0.051


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_material_id(value: object) -> str:
    material_id = str(value).strip()
    if material_id.startswith("material_"):
        material_id = material_id.removeprefix("material_")
    _require(bool(material_id), "material ID is empty")
    return material_id


def _coordinate_indices(trajectory: Trajectory) -> tuple[int, ...]:
    _require(trajectory in TRAJECTORY_ORDER, f"unregistered trajectory: {trajectory}")
    start = TRAJECTORY_ORDER.index(trajectory) * len(REGISTERED_INDENTATIONS_MM)
    return tuple(range(start, start + len(REGISTERED_INDENTATIONS_MM)))


HELD_INDICES = _coordinate_indices(HELD_INTERVENTION)


@dataclass(frozen=True)
class RCTMaterialResponse:
    """Registered force coordinates for one physical material sample."""

    material_id: str
    force_n: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.force_n, dtype=np.float64)
        _require(values.shape == (COORDINATE_COUNT,), "response shape changed")
        _require(np.all(np.isfinite(values)), "response contains non-finite values")
        _require(np.all(values >= 0.0), "normal-force increment is negative")
        object.__setattr__(self, "force_n", values.copy())

    def values_for(self, trajectory: Trajectory) -> np.ndarray:
        """Return the three registered force values for one press."""

        return self.force_n[np.asarray(_coordinate_indices(trajectory), dtype=int)].copy()


def _registered_force_values(rows: Sequence[tuple[float, float]]) -> np.ndarray:
    _require(len(rows) >= len(REGISTERED_INDENTATIONS_MM) + 1, "force trace is too short")
    z = np.asarray([row[0] for row in rows], dtype=np.float64)
    raw_force = np.asarray([row[1] for row in rows], dtype=np.float64)
    _require(np.all(np.isfinite(z)), "z_frame contains non-finite values")
    _require(np.all(np.isfinite(raw_force)), "raw_fz contains non-finite values")
    shallow_index = int(np.argmax(z))
    shallow_z = float(z[shallow_index])
    baseline_force = float(raw_force[shallow_index])
    indentation = shallow_z - z
    _require(np.min(indentation) >= -1e-9, "indentation alignment is invalid")
    force_increment = np.abs(raw_force - baseline_force)
    selected: list[float] = []
    for requested in REGISTERED_INDENTATIONS_MM:
        mismatch = np.abs(indentation - requested)
        index = int(np.argmin(mismatch))
        _require(
            float(mismatch[index]) <= _MAXIMUM_INDENTATION_MISMATCH_MM,
            f"registered indentation {requested:g} mm is unavailable",
        )
        selected.append(float(force_increment[index]))
    return np.asarray(selected, dtype=np.float64)


def load_rct_force_responses(
    force_metadata_csv: str | Path,
    *,
    allowed_material_ids: Iterable[str],
    forbidden_material_ids: Iterable[str] = (),
) -> tuple[RCTMaterialResponse, ...]:
    """Load only explicitly allowed rows from RCT ``force_metadata.csv``.

    Rows outside ``allowed_material_ids`` are rejected before any numerical force
    field is parsed. This lets source executors stream past confirmation rows
    without deriving a confirmation outcome.
    """

    allowed = frozenset(_canonical_material_id(value) for value in allowed_material_ids)
    forbidden = frozenset(
        _canonical_material_id(value) for value in forbidden_material_ids
    )
    _require(bool(allowed), "allowed material roster is empty")
    _require(not (allowed & forbidden), "allowed and forbidden material rosters overlap")
    grouped: dict[tuple[str, int, int], list[tuple[float, float]]] = {}
    seen_allowed: set[str] = set()
    with Path(force_metadata_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"material_id", "position", "sensor", "z_frame", "raw_fz"}
        _require(
            reader.fieldnames is not None
            and required_columns <= set(reader.fieldnames),
            "RCT force metadata schema changed",
        )
        for row in reader:
            material_id = _canonical_material_id(row["material_id"])
            if material_id not in allowed:
                continue
            _require(material_id not in forbidden, "forbidden material was admitted")
            seen_allowed.add(material_id)
            position = int(row["position"])
            sensor = int(row["sensor"])
            trajectory = (position, sensor)
            if trajectory not in TRAJECTORY_ORDER:
                continue
            grouped.setdefault((material_id, position, sensor), []).append(
                (float(row["z_frame"]), float(row["raw_fz"]))
            )
    _require(seen_allowed == allowed, "one or more allowed materials are absent")
    responses: list[RCTMaterialResponse] = []
    for material_id in sorted(allowed):
        vectors: list[np.ndarray] = []
        for position, sensor in TRAJECTORY_ORDER:
            rows = grouped.get((material_id, position, sensor))
            _require(
                rows is not None,
                f"registered trajectory p{position}/s{sensor} is missing: {material_id}",
            )
            vectors.append(_registered_force_values(rows))
        responses.append(RCTMaterialResponse(material_id, np.concatenate(vectors)))
    return tuple(responses)


@dataclass(frozen=True)
class GaussianState:
    """Conditional material-response belief."""

    mean: np.ndarray
    covariance: np.ndarray
    observed_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        _require(mean.shape == (COORDINATE_COUNT,), "Gaussian mean shape changed")
        _require(
            covariance.shape == (COORDINATE_COUNT, COORDINATE_COUNT),
            "Gaussian covariance shape changed",
        )
        _require(np.all(np.isfinite(mean)), "Gaussian mean is non-finite")
        _require(np.all(np.isfinite(covariance)), "Gaussian covariance is non-finite")
        _require(
            np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10),
            "Gaussian covariance is not symmetric",
        )
        _require(
            len(self.observed_indices) == len(set(self.observed_indices)),
            "observed coordinate repeated",
        )
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "covariance", covariance.copy())


@dataclass(frozen=True)
class RCTGaussianTwin:
    """Frozen empirical Gaussian twin fitted on material-disjoint source data."""

    mean: np.ndarray
    covariance: np.ndarray
    covariance_diagonal_shrinkage: float = 0.25
    jitter_fraction_of_median_variance: float = 1e-8

    def __post_init__(self) -> None:
        state = GaussianState(self.mean, self.covariance)
        _require(
            self.covariance_diagonal_shrinkage == 0.25,
            "covariance shrinkage changed",
        )
        _require(
            self.jitter_fraction_of_median_variance == 1e-8,
            "covariance jitter changed",
        )
        object.__setattr__(self, "mean", state.mean)
        object.__setattr__(self, "covariance", state.covariance)

    @classmethod
    def fit(cls, responses: Sequence[RCTMaterialResponse]) -> RCTGaussianTwin:
        """Fit the registered mean and diagonal-shrinkage covariance."""

        _require(len(responses) >= 3, "at least three fit materials are required")
        material_ids = [response.material_id for response in responses]
        _require(len(material_ids) == len(set(material_ids)), "fit material repeated")
        matrix = np.stack([response.force_n for response in responses])
        mean = np.mean(matrix, axis=0)
        sample_covariance = np.cov(matrix, rowvar=False, ddof=1)
        diagonal = np.diag(np.diag(sample_covariance))
        covariance = 0.75 * sample_covariance + 0.25 * diagonal
        positive_variances = np.diag(covariance)[np.diag(covariance) > 0.0]
        _require(len(positive_variances) > 0, "fit covariance has no positive variance")
        jitter = 1e-8 * float(np.median(positive_variances))
        covariance = covariance + max(jitter, np.finfo(np.float64).eps) * np.eye(
            COORDINATE_COUNT
        )
        return cls(mean=mean, covariance=covariance)

    def prior(self) -> GaussianState:
        return GaussianState(self.mean, self.covariance)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "covariance_diagonal_shrinkage": self.covariance_diagonal_shrinkage,
            "jitter_fraction_of_median_variance": (
                self.jitter_fraction_of_median_variance
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RCTGaussianTwin:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            covariance=np.asarray(payload["covariance"], dtype=np.float64),
            covariance_diagonal_shrinkage=float(
                payload["covariance_diagonal_shrinkage"]
            ),
            jitter_fraction_of_median_variance=float(
                payload["jitter_fraction_of_median_variance"]
            ),
        )


def condition_gaussian(
    state: GaussianState,
    indices: Sequence[int],
    values: Sequence[float],
) -> GaussianState:
    """Condition a Gaussian state on exact registered force coordinates."""

    new_indices = tuple(int(index) for index in indices)
    observations = np.asarray(values, dtype=np.float64)
    _require(len(new_indices) == len(observations), "observation length changed")
    _require(len(new_indices) == len(set(new_indices)), "coordinate repeated in update")
    _require(
        not (set(new_indices) & set(state.observed_indices)),
        "coordinate was observed twice",
    )
    _require(
        all(0 <= index < COORDINATE_COUNT for index in new_indices),
        "observation coordinate is out of range",
    )
    _require(np.all(np.isfinite(observations)), "observation is non-finite")
    index_array = np.asarray(new_indices, dtype=int)
    covariance_yy = state.covariance[np.ix_(index_array, index_array)]
    cross = state.covariance[:, index_array]
    gain = np.linalg.solve(covariance_yy, cross.T).T
    residual = observations - state.mean[index_array]
    mean = state.mean + gain @ residual
    covariance = state.covariance - gain @ cross.T
    covariance = 0.5 * (covariance + covariance.T)
    covariance[index_array, :] = 0.0
    covariance[:, index_array] = 0.0
    return GaussianState(
        mean=mean,
        covariance=covariance,
        observed_indices=(*state.observed_indices, *new_indices),
    )


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return 0.5 * (
        1.0
        + np.asarray(
            [math.erf(float(item) / math.sqrt(2.0)) for item in value.ravel()],
            dtype=np.float64,
        ).reshape(value.shape)
    )


def _expected_action_utility(
    mean_force: np.ndarray,
    variance: float,
    *,
    indentation_mm: float,
    force_limit_n: float,
) -> np.ndarray:
    _require(force_limit_n > 0.0, "force limit must be positive")
    mean = np.asarray(mean_force, dtype=np.float64)
    standard_deviation = math.sqrt(max(float(variance), 0.0))
    safe_utility = indentation_mm / max(REGISTERED_INDENTATIONS_MM)
    if standard_deviation <= np.finfo(np.float64).eps:
        safe = mean <= force_limit_n
        excess = np.maximum(0.0, mean - force_limit_n)
        return np.where(safe, safe_utility, -1.0 - excess / force_limit_n)
    standardized_limit = (force_limit_n - mean) / standard_deviation
    probability_safe = _normal_cdf(standardized_limit)
    probability_unsafe = 1.0 - probability_safe
    density = np.exp(-0.5 * standardized_limit**2) / math.sqrt(2.0 * math.pi)
    expected_excess = (
        (mean - force_limit_n) * probability_unsafe
        + standard_deviation * density
    )
    return (
        probability_safe * safe_utility
        - probability_unsafe
        - expected_excess / force_limit_n
    )


def _expected_best_utility(
    held_mean: np.ndarray, held_covariance: np.ndarray, force_limit_n: float
) -> float:
    expected = [0.0]
    for index, indentation in enumerate(REGISTERED_INDENTATIONS_MM):
        expected.append(
            float(
                _expected_action_utility(
                    np.asarray([held_mean[index]]),
                    float(held_covariance[index, index]),
                    indentation_mm=indentation,
                    force_limit_n=force_limit_n,
                )[0]
            )
        )
    return max(expected)


def _stable_cholesky(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    positive = np.diag(covariance)[np.diag(covariance) > 0.0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    for multiplier in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        try:
            return np.linalg.cholesky(
                covariance + multiplier * max(scale, 1.0) * np.eye(len(covariance))
            )
        except np.linalg.LinAlgError:
            continue
    raise ValueError("predictive covariance is not positive definite")


def _common_standard_draws(draw_count: int = 4096, seed: int = 20260902) -> np.ndarray:
    _require(draw_count == 4096, "predictive draw count changed")
    _require(seed == 20260902, "predictive seed changed")
    generator = np.random.default_rng(seed)
    half = generator.standard_normal((draw_count // 2, len(REGISTERED_INDENTATIONS_MM)))
    return np.concatenate((half, -half), axis=0)


def decision_value_of_probe(
    state: GaussianState,
    trajectory: Trajectory,
    *,
    force_limit_n: float,
    standard_draws: np.ndarray | None = None,
) -> float:
    """Expected reduction in Bayes decision regret from one logged press."""

    indices = np.asarray(_coordinate_indices(trajectory), dtype=int)
    _require(
        not (set(indices.tolist()) & set(state.observed_indices)),
        "candidate probe was already observed",
    )
    held = np.asarray(HELD_INDICES, dtype=int)
    candidate_mean = state.mean[indices]
    candidate_covariance = state.covariance[np.ix_(indices, indices)]
    cross = state.covariance[:, indices]
    gain = np.linalg.solve(candidate_covariance, cross.T).T
    posterior_covariance = state.covariance - gain @ cross.T
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    current_best = _expected_best_utility(
        state.mean[held], state.covariance[np.ix_(held, held)], force_limit_n
    )
    draws = _common_standard_draws() if standard_draws is None else standard_draws
    _require(
        draws.shape == (4096, len(REGISTERED_INDENTATIONS_MM)),
        "common-random-number shape changed",
    )
    candidate_samples = candidate_mean + draws @ _stable_cholesky(
        candidate_covariance
    ).T
    posterior_held_mean = state.mean[held] + (
        candidate_samples - candidate_mean
    ) @ gain[held, :].T
    held_covariance = posterior_covariance[np.ix_(held, held)]
    expected_action_values = np.zeros((len(draws), 1 + len(held)), dtype=np.float64)
    for action_index, indentation in enumerate(REGISTERED_INDENTATIONS_MM):
        expected_action_values[:, action_index + 1] = _expected_action_utility(
            posterior_held_mean[:, action_index],
            float(held_covariance[action_index, action_index]),
            indentation_mm=indentation,
            force_limit_n=force_limit_n,
        )
    expected_post_probe_best = float(np.mean(np.max(expected_action_values, axis=1)))
    return max(0.0, expected_post_probe_best - current_best)


def _logdet(covariance: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(covariance)
    _require(sign > 0 and np.isfinite(value), "covariance log determinant failed")
    return float(value)


def system_identification_value_of_probe(
    state: GaussianState, trajectory: Trajectory
) -> float:
    """Gaussian information gain about all still-unobserved response coordinates."""

    candidate = tuple(_coordinate_indices(trajectory))
    observed = set(state.observed_indices)
    _require(not (set(candidate) & observed), "candidate probe was already observed")
    remaining = tuple(
        index
        for index in range(COORDINATE_COUNT)
        if index not in observed and index not in candidate
    )
    _require(bool(remaining), "candidate leaves no response coordinates")
    candidate_array = np.asarray(candidate, dtype=int)
    remaining_array = np.asarray(remaining, dtype=int)
    covariance_cc = state.covariance[np.ix_(candidate_array, candidate_array)]
    covariance_rr = state.covariance[np.ix_(remaining_array, remaining_array)]
    cross = state.covariance[np.ix_(remaining_array, candidate_array)]
    conditional = covariance_rr - cross @ np.linalg.solve(covariance_cc, cross.T)
    conditional = 0.5 * (conditional + conditional.T)
    return max(0.0, 0.5 * (_logdet(covariance_rr) - _logdet(conditional)))


@dataclass(frozen=True)
class RCTPolicyTrace:
    """Posterior states after the mandatory anchor and each selected probe."""

    selector: str
    probe_order: tuple[Trajectory, ...]
    states: tuple[GaussianState, ...]

    def __post_init__(self) -> None:
        _require(len(self.probe_order) == 3, "probe order length changed")
        _require(set(self.probe_order) == set(SELECTABLE_PROBES), "probe roster changed")
        _require(len(self.states) == 4, "policy budget roster changed")


def _select_next_probe(
    state: GaussianState,
    remaining: Sequence[Trajectory],
    *,
    selector: SelectorName,
    force_limit_n: float,
    standard_draws: np.ndarray,
) -> Trajectory:
    scores: list[tuple[float, Trajectory]] = []
    for trajectory in remaining:
        if selector == "decision_directed":
            value = decision_value_of_probe(
                state,
                trajectory,
                force_limit_n=force_limit_n,
                standard_draws=standard_draws,
            )
        elif selector == "system_identification":
            value = system_identification_value_of_probe(state, trajectory)
        elif selector == "fixed":
            value = 0.0
        else:  # pragma: no cover
            raise ValueError(f"unknown selector: {selector}")
        scores.append((float(value), trajectory))
    return min(scores, key=lambda item: (-item[0], item[1]))[1]


def trace_policy(
    twin: RCTGaussianTwin,
    response: RCTMaterialResponse,
    *,
    selector: SelectorName,
    force_limit_n: float,
) -> RCTPolicyTrace:
    """Run one registered adaptive selector without revealing held outcomes."""

    anchor_indices = _coordinate_indices(MANDATORY_ANCHOR)
    state = condition_gaussian(
        twin.prior(), anchor_indices, response.values_for(MANDATORY_ANCHOR)
    )
    states = [state]
    remaining = list(SELECTABLE_PROBES)
    order: list[Trajectory] = []
    standard_draws = _common_standard_draws()
    while remaining:
        selected = _select_next_probe(
            state,
            remaining,
            selector=selector,
            force_limit_n=force_limit_n,
            standard_draws=standard_draws,
        )
        order.append(selected)
        state = condition_gaussian(
            state, _coordinate_indices(selected), response.values_for(selected)
        )
        states.append(state)
        remaining.remove(selected)
    return RCTPolicyTrace(selector=selector, probe_order=tuple(order), states=tuple(states))


def trace_probe_order(
    twin: RCTGaussianTwin,
    response: RCTMaterialResponse,
    probe_order: Sequence[Trajectory],
    *,
    selector: str,
) -> RCTPolicyTrace:
    """Trace an explicit fixed or random-control probe order."""

    order = tuple(probe_order)
    _require(len(order) == 3 and set(order) == set(SELECTABLE_PROBES), "order changed")
    state = condition_gaussian(
        twin.prior(),
        _coordinate_indices(MANDATORY_ANCHOR),
        response.values_for(MANDATORY_ANCHOR),
    )
    states = [state]
    for trajectory in order:
        state = condition_gaussian(
            state, _coordinate_indices(trajectory), response.values_for(trajectory)
        )
        states.append(state)
    return RCTPolicyTrace(selector=selector, probe_order=order, states=tuple(states))


def _all_registered_traces(
    twin: RCTGaussianTwin,
    response: RCTMaterialResponse,
    *,
    force_limit_n: float,
) -> tuple[RCTPolicyTrace, ...]:
    traces = [
        trace_policy(
            twin,
            response,
            selector="decision_directed",
            force_limit_n=force_limit_n,
        ),
        trace_policy(
            twin,
            response,
            selector="system_identification",
            force_limit_n=force_limit_n,
        ),
        trace_policy(
            twin, response, selector="fixed", force_limit_n=force_limit_n
        ),
    ]
    traces.extend(
        trace_probe_order(twin, response, order, selector=f"permutation_{index}")
        for index, order in enumerate(itertools.permutations(SELECTABLE_PROBES))
    )
    return tuple(traces)


def calibrate_simultaneous_force_multiplier(
    twin: RCTGaussianTwin,
    calibration_responses: Sequence[RCTMaterialResponse],
    *,
    force_limit_n: float,
    coverage: float = 0.9,
) -> tuple[float, tuple[float, ...], int]:
    """Calibrate one upper-force multiplier across all methods and budgets."""

    _require(coverage == 0.9, "conformal coverage changed")
    _require(len(calibration_responses) == 20, "calibration material count changed")
    material_ids = [response.material_id for response in calibration_responses]
    _require(len(material_ids) == len(set(material_ids)), "calibration material repeated")
    held = np.asarray(HELD_INDICES, dtype=int)
    scores: list[float] = []
    for response in calibration_responses:
        actual = response.force_n[held]
        material_score = -math.inf
        for trace in _all_registered_traces(
            twin, response, force_limit_n=force_limit_n
        ):
            for state in trace.states:
                mean = state.mean[held]
                variance = np.diag(state.covariance[np.ix_(held, held)])
                standard_deviation = np.sqrt(np.maximum(variance, 0.0))
                _require(
                    np.all(standard_deviation > 0.0),
                    "held predictive standard deviation vanished",
                )
                material_score = max(
                    material_score,
                    float(np.max((actual - mean) / standard_deviation)),
                )
        _require(np.isfinite(material_score), "calibration score is non-finite")
        scores.append(material_score)
    rank = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    multiplier = max(0.0, float(np.sort(np.asarray(scores))[rank - 1]))
    return multiplier, tuple(scores), rank


@dataclass(frozen=True)
class RCTDecisionMethod:
    """Frozen fit and calibration artifacts used by source-test and confirmation."""

    twin: RCTGaussianTwin
    force_limit_n: float
    conformal_multiplier: float
    calibration_scores: tuple[float, ...]
    conformal_rank: int

    def __post_init__(self) -> None:
        _require(self.force_limit_n > 0.0, "force limit must be positive")
        _require(self.conformal_multiplier >= 0.0, "conformal multiplier is negative")
        _require(len(self.calibration_scores) == 20, "calibration score count changed")
        _require(self.conformal_rank == 19, "conformal finite-sample rank changed")

    @classmethod
    def fit(
        cls,
        fit_responses: Sequence[RCTMaterialResponse],
        calibration_responses: Sequence[RCTMaterialResponse],
    ) -> RCTDecisionMethod:
        twin = RCTGaussianTwin.fit(fit_responses)
        held_max_index = HELD_INDICES[-1]
        force_limit = float(
            np.quantile(
                np.asarray(
                    [response.force_n[held_max_index] for response in fit_responses]
                ),
                0.6,
                method="linear",
            )
        )
        _require(force_limit > 0.0, "fit-only force limit is not positive")
        multiplier, scores, rank = calibrate_simultaneous_force_multiplier(
            twin,
            calibration_responses,
            force_limit_n=force_limit,
        )
        return cls(
            twin=twin,
            force_limit_n=force_limit,
            conformal_multiplier=multiplier,
            calibration_scores=scores,
            conformal_rank=rank,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "RCTRealDecisionMethodV1",
            "twin": self.twin.as_dict(),
            "force_limit_n": self.force_limit_n,
            "conformal_multiplier": self.conformal_multiplier,
            "calibration_scores": list(self.calibration_scores),
            "conformal_rank": self.conformal_rank,
            "confirmation_opened": False,
            "held_v8_accessed": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RCTDecisionMethod:
        _require(
            payload.get("artifact_kind") == "RCTRealDecisionMethodV1",
            "method artifact kind changed",
        )
        _require(payload.get("confirmation_opened") is False, "confirmation opened")
        _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
        twin = payload.get("twin")
        _require(isinstance(twin, Mapping), "twin artifact is missing")
        return cls(
            twin=RCTGaussianTwin.from_dict(twin),
            force_limit_n=float(payload["force_limit_n"]),
            conformal_multiplier=float(payload["conformal_multiplier"]),
            calibration_scores=tuple(float(value) for value in payload["calibration_scores"]),
            conformal_rank=int(payload["conformal_rank"]),
        )


def _realized_utility(force_n: float, indentation_mm: float, force_limit_n: float) -> float:
    if indentation_mm == 0.0:
        return 0.0
    if force_n <= force_limit_n:
        return indentation_mm / max(REGISTERED_INDENTATIONS_MM)
    return -1.0 - (force_n - force_limit_n) / force_limit_n


def _evaluate_state(
    state: GaussianState,
    actual_force: np.ndarray,
    method: RCTDecisionMethod,
) -> dict[str, Any]:
    held = np.asarray(HELD_INDICES, dtype=int)
    mean = state.mean[held]
    standard_deviation = np.sqrt(
        np.maximum(np.diag(state.covariance[np.ix_(held, held)]), 0.0)
    )
    upper = mean + method.conformal_multiplier * standard_deviation
    certified = [
        indentation
        for indentation, bound in zip(REGISTERED_INDENTATIONS_MM, upper, strict=True)
        if float(bound) <= method.force_limit_n
    ]
    selected_depth = max(certified, default=0.0)
    action_index = (
        None
        if selected_depth == 0.0
        else REGISTERED_INDENTATIONS_MM.index(selected_depth)
    )
    selected_force = 0.0 if action_index is None else float(actual_force[action_index])
    realized_utility = _realized_utility(
        selected_force, selected_depth, method.force_limit_n
    )
    safe_actions = [
        indentation
        for indentation, force in zip(
            REGISTERED_INDENTATIONS_MM, actual_force, strict=True
        )
        if float(force) <= method.force_limit_n
    ]
    oracle_depth = max(safe_actions, default=0.0)
    oracle_index = (
        None if oracle_depth == 0.0 else REGISTERED_INDENTATIONS_MM.index(oracle_depth)
    )
    oracle_force = 0.0 if oracle_index is None else float(actual_force[oracle_index])
    oracle_utility = _realized_utility(oracle_force, oracle_depth, method.force_limit_n)
    simultaneous_covered = bool(np.all(actual_force <= upper + 1e-12))
    unsafe = bool(action_index is not None and selected_force > method.force_limit_n)
    return {
        "selected_indentation_mm": selected_depth,
        "selected_force_n": selected_force,
        "oracle_indentation_mm": oracle_depth,
        "regret": oracle_utility - realized_utility,
        "abstained": action_index is None,
        "unsafe": unsafe,
        "false_safe": unsafe,
        "simultaneous_force_covered": simultaneous_covered,
        "predicted_force_mean_n": mean.tolist(),
        "predicted_force_upper_n": upper.tolist(),
    }


def _trace_result(
    trace: RCTPolicyTrace,
    response: RCTMaterialResponse,
    method: RCTDecisionMethod,
) -> dict[str, Any]:
    actual = response.force_n[np.asarray(HELD_INDICES, dtype=int)]
    budgets = [
        _evaluate_state(state, actual, method)
        for state in trace.states
    ]
    regrets = np.asarray([record["regret"] for record in budgets], dtype=np.float64)
    return {
        "probe_order": [
            {"position": position, "sensor": sensor}
            for position, sensor in trace.probe_order
        ],
        "budgets": budgets,
        "regret_auc": float(np.trapezoid(regrets, dx=1.0)),
        "simultaneous_force_covered_all_budgets": bool(
            all(record["simultaneous_force_covered"] for record in budgets)
        ),
    }


def evaluate_material(
    method: RCTDecisionMethod, response: RCTMaterialResponse
) -> dict[str, Any]:
    """Evaluate all registered policies on one already-authorized material."""

    decision = trace_policy(
        method.twin,
        response,
        selector="decision_directed",
        force_limit_n=method.force_limit_n,
    )
    identification = trace_policy(
        method.twin,
        response,
        selector="system_identification",
        force_limit_n=method.force_limit_n,
    )
    fixed = trace_policy(
        method.twin,
        response,
        selector="fixed",
        force_limit_n=method.force_limit_n,
    )
    permutation_results = [
        _trace_result(
            trace_probe_order(
                method.twin,
                response,
                order,
                selector=f"permutation_{index}",
            ),
            response,
            method,
        )
        for index, order in enumerate(itertools.permutations(SELECTABLE_PROBES))
    ]
    return {
        "material_id": response.material_id,
        "decision_directed": _trace_result(decision, response, method),
        "system_identification": _trace_result(identification, response, method),
        "fixed": _trace_result(fixed, response, method),
        "random_permutation_mean_regret_auc": float(
            np.mean([record["regret_auc"] for record in permutation_results])
        ),
        "random_permutations": permutation_results,
    }


def _exact_one_sided_sign_flip_paired_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    _require(differences.shape == (20,), "confirmation contrast must have 20 materials")
    observed = float(np.mean(differences))
    count = 0
    total = 1 << len(differences)
    chunk_size = 1 << 15
    bit_positions = np.arange(len(differences), dtype=np.uint64)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        masks = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((masks >> bit_positions) & 1).astype(np.float64)
        permuted = signs @ differences / len(differences)
        count += int(np.count_nonzero(permuted <= observed + 1e-15))
    return count / total


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repetitions: int = 10000,
    seed: int = 20260902,
) -> tuple[float, float]:
    _require(repetitions == 10000, "bootstrap repetitions changed")
    _require(seed == 20260902, "bootstrap seed changed")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repetitions, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_evaluation(
    material_results: Sequence[Mapping[str, Any]],
    *,
    require_confirmation_count: bool,
) -> dict[str, Any]:
    """Aggregate object-level regret, coverage, safety, and paired inference."""

    expected_count = 20 if require_confirmation_count else len(material_results)
    _require(len(material_results) == expected_count, "evaluation material count changed")
    material_ids = [str(record["material_id"]) for record in material_results]
    _require(len(material_ids) == len(set(material_ids)), "evaluation material repeated")
    decision_auc = np.asarray(
        [record["decision_directed"]["regret_auc"] for record in material_results],
        dtype=np.float64,
    )
    identification_auc = np.asarray(
        [record["system_identification"]["regret_auc"] for record in material_results],
        dtype=np.float64,
    )
    differences = decision_auc - identification_auc
    decision_budgets = [
        budget
        for record in material_results
        for budget in record["decision_directed"]["budgets"]
    ]
    identification_budgets = [
        budget
        for record in material_results
        for budget in record["system_identification"]["budgets"]
    ]
    mean_identification = float(np.mean(identification_auc))
    relative_improvement = (
        0.0
        if mean_identification <= 0.0
        else (mean_identification - float(np.mean(decision_auc)))
        / mean_identification
    )
    interval = _bootstrap_mean_interval(differences)
    summary: dict[str, Any] = {
        "material_count": len(material_results),
        "decision_directed_mean_regret_auc": float(np.mean(decision_auc)),
        "system_identification_mean_regret_auc": mean_identification,
        "paired_mean_auc_difference": float(np.mean(differences)),
        "paired_mean_auc_difference_bootstrap_95": list(interval),
        "relative_auc_improvement": relative_improvement,
        "material_improvement_count": int(np.count_nonzero(differences < -1e-12)),
        "material_tie_count": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
        "material_regression_count": int(np.count_nonzero(differences > 1e-12)),
        "decision_directed_false_safe_rate": float(
            np.mean([record["false_safe"] for record in decision_budgets])
        ),
        "decision_directed_unsafe_action_rate": float(
            np.mean([record["unsafe"] for record in decision_budgets])
        ),
        "system_identification_unsafe_action_rate": float(
            np.mean([record["unsafe"] for record in identification_budgets])
        ),
        "decision_directed_simultaneous_force_coverage": float(
            np.mean(
                [
                    record["decision_directed"][
                        "simultaneous_force_covered_all_budgets"
                    ]
                    for record in material_results
                ]
            )
        ),
        "decision_directed_abstention_rate": float(
            np.mean([record["abstained"] for record in decision_budgets])
        ),
        "selected_probe_count": len(
            {
                (
                    record["decision_directed"]["probe_order"][0]["position"],
                    record["decision_directed"]["probe_order"][0]["sensor"],
                )
                for record in material_results
            }
        ),
    }
    if require_confirmation_count:
        summary["one_sided_exact_paired_sign_flip_p"] = (
            _exact_one_sided_sign_flip_paired_p(differences)
        )
    return summary


def source_promotion_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen source-side gate without accessing confirmation outcomes."""

    checks = {
        "relative_auc_improvement": float(summary["relative_auc_improvement"]) >= 0.05,
        "material_improvement_count": int(summary["material_improvement_count"]) >= 12,
        "simultaneous_force_coverage": (
            float(summary["decision_directed_simultaneous_force_coverage"]) >= 0.9
        ),
        "false_safe_rate": float(summary["decision_directed_false_safe_rate"]) <= 0.1,
        "unsafe_rate_noninferiority": (
            float(summary["decision_directed_unsafe_action_rate"])
            <= float(summary["system_identification_unsafe_action_rate"]) + 0.05
        ),
        "selected_probe_diversity": int(summary["selected_probe_count"]) >= 2,
    }
    return {
        "artifact_kind": "RCTRealDecisionSourceGateV1",
        "checks": checks,
        "passed": all(checks.values()),
        "target_authorized": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }


__all__ = [
    "COORDINATE_COUNT",
    "GaussianState",
    "RCTDecisionMethod",
    "RCTGaussianTwin",
    "RCTMaterialResponse",
    "RCTPolicyTrace",
    "TRAJECTORY_ORDER",
    "calibrate_simultaneous_force_multiplier",
    "condition_gaussian",
    "decision_value_of_probe",
    "evaluate_material",
    "load_rct_force_responses",
    "source_promotion_gate",
    "summarize_evaluation",
    "system_identification_value_of_probe",
    "trace_policy",
    "trace_probe_order",
]
