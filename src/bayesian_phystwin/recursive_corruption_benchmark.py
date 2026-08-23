"""Controlled recursive benchmark for corrupted and missing observations.

The benchmark isolates a question that average point-error comparisons cannot
answer: when does a recursive uncertainty-bearing belief add value over residual
persistence? It uses a deterministic physical rollout plus one latent
autoregressive discrepancy and evaluates one-step forecasts under missing data,
outlier bursts, coherent drift, identity substitution, stale observations, and
reduced observation density.

This module provides controlled mechanism evidence only. It does not exercise a
real PhysTwin, Prob4D provider, sealed object/session cohort, or Causal4D
intervention.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Final

import numpy as np

RECURSIVE_CORRUPTION_SCHEMA: Final = "bayesian-phystwin.recursive-corruption-benchmark"
RECURSIVE_CORRUPTION_SCHEMA_VERSION: Final = 1

METHODS: Final[tuple[str, ...]] = (
    "physical_baseline",
    "last_residual",
    "exponential_residual",
    "recursive_gaussian",
    "guarded_recursive",
)
CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "missing_burst",
    "outlier_burst",
    "coherent_drift",
    "identity_switch",
    "delayed_observation",
    "density_drop",
)
FALLBACK_REASONS: Final[tuple[str, ...]] = (
    "none",
    "missing-observation",
    "low-reliability",
    "stale-observation",
    "innovation-gate",
    "trust-region",
)
_Z90: Final = NormalDist().inv_cdf(0.95)


@dataclass(frozen=True, slots=True)
class RecursiveCorruptionBenchmarkConfig:
    """Configuration for the controlled recursive discrepancy benchmark."""

    step_count: int = 180
    time_step: float = 0.04
    residual_persistence: float = 0.94
    residual_process_std_m: float = 0.0015
    observation_std_m: float = 0.004
    initial_residual_std_m: float = 0.010
    corruption_start: int = 60
    corruption_length: int = 30
    recovery_window: int = 45
    exponential_decay: float = 0.85
    minimum_reliability: float = 0.50
    maximum_nis: float = 9.0
    maximum_update_m: float = 0.025
    maximum_residual_m: float = 0.080
    materially_harmful_margin_m: float = 0.002
    outlier_std_m: float = 0.050
    drift_magnitude_m: float = 0.070
    identity_offset_m: float = 0.060
    delay_steps: int = 6
    density_keep_probability: float = 0.20

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True, slots=True)
class CorruptedSequence:
    """One generated sequence with target-free reliability and lineage cues."""

    physical_position_m: np.ndarray
    true_position_m: np.ndarray
    observation_m: np.ndarray
    observation_available: np.ndarray
    prior_reliability: np.ndarray
    source_step: np.ndarray
    corruption_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class _MethodTrace:
    forecast_mean_m: np.ndarray
    forecast_variance_m2: np.ndarray
    accepted_update: np.ndarray
    exact_fallback: np.ndarray
    exact_fallback_valid: np.ndarray
    materially_harmful_update: np.ndarray
    fallback_reason: np.ndarray


def _genuine_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _validate_config(config: RecursiveCorruptionBenchmarkConfig) -> None:
    step_count = _genuine_integer(
        config.step_count,
        name="step_count",
        minimum=20,
    )
    corruption_start = _genuine_integer(
        config.corruption_start,
        name="corruption_start",
        minimum=2,
    )
    corruption_length = _genuine_integer(
        config.corruption_length,
        name="corruption_length",
        minimum=1,
    )
    recovery_window = _genuine_integer(
        config.recovery_window,
        name="recovery_window",
        minimum=3,
    )
    delay_steps = _genuine_integer(
        config.delay_steps,
        name="delay_steps",
        minimum=1,
    )
    if corruption_start + corruption_length + recovery_window >= step_count:
        raise ValueError(
            "corruption and recovery windows must end before the final step"
        )
    if delay_steps >= corruption_start:
        raise ValueError("delay_steps must be smaller than corruption_start")

    positive = {
        "time_step": config.time_step,
        "residual_process_std_m": config.residual_process_std_m,
        "observation_std_m": config.observation_std_m,
        "initial_residual_std_m": config.initial_residual_std_m,
        "maximum_nis": config.maximum_nis,
        "maximum_update_m": config.maximum_update_m,
        "maximum_residual_m": config.maximum_residual_m,
        "outlier_std_m": config.outlier_std_m,
        "drift_magnitude_m": config.drift_magnitude_m,
        "identity_offset_m": config.identity_offset_m,
    }
    for name, raw in positive.items():
        if _finite(raw, name=name) <= 0.0:
            raise ValueError(f"{name} must be positive")

    harmful_margin = _finite(
        config.materially_harmful_margin_m,
        name="materially_harmful_margin_m",
    )
    if harmful_margin < 0.0:
        raise ValueError("materially_harmful_margin_m must be nonnegative")

    unit_interval = {
        "residual_persistence": config.residual_persistence,
        "exponential_decay": config.exponential_decay,
        "minimum_reliability": config.minimum_reliability,
        "density_keep_probability": config.density_keep_probability,
    }
    for name, raw in unit_interval.items():
        value = _finite(raw, name=name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0, 1)")


def _canonical_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    canonical = tuple(_genuine_integer(seed, name="seed", minimum=0) for seed in seeds)
    if not canonical:
        raise ValueError("at least one seed is required")
    if len(canonical) != len(set(canonical)):
        raise ValueError("seeds must be unique")
    return canonical


def _canonical_conditions(conditions: Sequence[str]) -> tuple[str, ...]:
    canonical = tuple(conditions)
    if not canonical:
        raise ValueError("at least one condition is required")
    if any(type(condition) is not str for condition in canonical):
        raise ValueError("conditions must be literal strings")
    unknown = sorted(set(canonical) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    if len(canonical) != len(set(canonical)):
        raise ValueError("conditions must be unique")
    return canonical


def _physical_rollout(
    config: RecursiveCorruptionBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(config.step_count, dtype=np.float64) * config.time_step
    action = 0.75 * np.sin(2.0 * np.pi * 0.45 * time)
    action += 0.30 * np.sign(np.sin(2.0 * np.pi * 0.11 * time + 0.2))
    position: np.ndarray = np.zeros(config.step_count, dtype=np.float64)
    velocity = 0.0
    for step in range(config.step_count - 1):
        acceleration = -2.2 * position[step] - 0.55 * velocity + action[step]
        velocity += config.time_step * acceleration
        position[step + 1] = position[step] + config.time_step * velocity
    return position, action


def generate_corrupted_sequence(
    condition: str,
    *,
    seed: int,
    config: RecursiveCorruptionBenchmarkConfig | None = None,
) -> CorruptedSequence:
    """Generate one deterministic sequence and declared target-free cues."""

    cfg = RecursiveCorruptionBenchmarkConfig() if config is None else config
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    canonical_seed = _genuine_integer(seed, name="seed", minimum=0)
    rng = np.random.default_rng(canonical_seed)
    physical, action = _physical_rollout(cfg)

    discrepancy: np.ndarray = np.zeros(cfg.step_count, dtype=np.float64)
    discrepancy[0] = rng.normal(scale=cfg.initial_residual_std_m)
    for step in range(cfg.step_count - 1):
        action_coupling = 0.0004 * np.tanh(action[step])
        discrepancy[step + 1] = (
            cfg.residual_persistence * discrepancy[step]
            + action_coupling
            + rng.normal(scale=cfg.residual_process_std_m)
        )

    truth = physical + discrepancy
    observation = truth + rng.normal(
        scale=cfg.observation_std_m,
        size=cfg.step_count,
    )
    available: np.ndarray = np.ones(cfg.step_count, dtype=bool)
    reliability: np.ndarray = np.full(cfg.step_count, 0.95, dtype=np.float64)
    source_step: np.ndarray = np.arange(cfg.step_count, dtype=np.int64)
    corruption: np.ndarray = np.zeros(cfg.step_count, dtype=bool)
    start = cfg.corruption_start
    stop = start + cfg.corruption_length
    corruption[start:stop] = True

    if condition == "clean":
        corruption[:] = False
    elif condition == "missing_burst":
        available[start:stop] = False
        observation[start:stop] = np.nan
        reliability[start:stop] = 0.0
    elif condition == "outlier_burst":
        observation[start:stop] += rng.normal(
            scale=cfg.outlier_std_m,
            size=cfg.corruption_length,
        )
        reliability[start:stop] = 0.08
    elif condition == "coherent_drift":
        observation[start:stop] += np.linspace(
            0.0,
            cfg.drift_magnitude_m,
            cfg.corruption_length,
        )
        reliability[start:stop] = np.linspace(
            0.75,
            0.45,
            cfg.corruption_length,
        )
    elif condition == "identity_switch":
        phase = np.linspace(
            0.0,
            2.0 * np.pi,
            cfg.corruption_length,
            endpoint=False,
        )
        observation[start:stop] = (
            physical[start:stop]
            - discrepancy[start:stop]
            + cfg.identity_offset_m
            + 0.012 * np.sin(phase)
            + rng.normal(
                scale=cfg.observation_std_m,
                size=cfg.corruption_length,
            )
        )
        reliability[start:stop] = 0.12
    elif condition == "delayed_observation":
        for step in range(start, stop):
            source = step - cfg.delay_steps
            observation[step] = truth[source] + rng.normal(scale=cfg.observation_std_m)
            source_step[step] = source
        reliability[start:stop] = 0.65
    elif condition == "density_drop":
        retained = rng.random(cfg.corruption_length) < cfg.density_keep_probability
        selected: np.ndarray = np.arange(start, stop, dtype=np.int64)
        missing = selected[~retained]
        available[missing] = False
        observation[missing] = np.nan
        reliability[missing] = 0.0

    arrays = (
        physical,
        truth,
        observation,
        available,
        reliability,
        source_step,
        corruption,
    )
    for array in arrays:
        array.setflags(write=False)
    return CorruptedSequence(
        physical_position_m=physical,
        true_position_m=truth,
        observation_m=observation,
        observation_available=available,
        prior_reliability=reliability,
        source_step=source_step,
        corruption_mask=corruption,
    )


def _empty_trace(step_count: int) -> _MethodTrace:
    forecast_count = step_count - 1
    return _MethodTrace(
        forecast_mean_m=np.zeros(forecast_count, dtype=np.float64),
        forecast_variance_m2=np.full(
            forecast_count,
            np.nan,
            dtype=np.float64,
        ),
        accepted_update=np.zeros(forecast_count, dtype=bool),
        exact_fallback=np.zeros(forecast_count, dtype=bool),
        exact_fallback_valid=np.ones(forecast_count, dtype=bool),
        materially_harmful_update=np.zeros(forecast_count, dtype=bool),
        fallback_reason=np.full(forecast_count, "none", dtype="<U24"),
    )


def _materially_harmful(
    candidate_forecast: float,
    fallback_forecast: float,
    target: float,
    *,
    margin_m: float,
) -> bool:
    return abs(candidate_forecast - target) > (
        abs(fallback_forecast - target) + margin_m
    )


def _run_methods(
    sequence: CorruptedSequence,
    config: RecursiveCorruptionBenchmarkConfig,
) -> dict[str, _MethodTrace]:
    traces = {method: _empty_trace(config.step_count) for method in METHODS}
    physical = sequence.physical_position_m
    truth = sequence.true_position_m
    observation = sequence.observation_m
    available = sequence.observation_available
    reliability = sequence.prior_reliability
    source_step = sequence.source_step
    persistence = config.residual_persistence
    process_variance = config.residual_process_std_m**2
    observation_variance = config.observation_std_m**2

    last_residual = 0.0
    exponential_residual = 0.0
    recursive_mean = 0.0
    recursive_variance = config.initial_residual_std_m**2
    guarded_mean = 0.0
    guarded_variance = config.initial_residual_std_m**2

    for step in range(config.step_count - 1):
        target = float(truth[step + 1])
        traces["physical_baseline"].forecast_mean_m[step] = physical[step + 1]

        last_prior = persistence * last_residual
        if available[step]:
            last_residual = float(observation[step] - physical[step])
            traces["last_residual"].accepted_update[step] = True
        else:
            last_residual = last_prior
            traces["last_residual"].exact_fallback[step] = True
            traces["last_residual"].fallback_reason[step] = "missing-observation"
        traces["last_residual"].forecast_mean_m[step] = (
            physical[step + 1] + persistence * last_residual
        )

        exponential_prior = persistence * exponential_residual
        if available[step]:
            measured = float(observation[step] - physical[step])
            exponential_residual = (
                config.exponential_decay * exponential_prior
                + (1.0 - config.exponential_decay) * measured
            )
            traces["exponential_residual"].accepted_update[step] = True
        else:
            exponential_residual = exponential_prior
            traces["exponential_residual"].exact_fallback[step] = True
            traces["exponential_residual"].fallback_reason[step] = "missing-observation"
        traces["exponential_residual"].forecast_mean_m[step] = (
            physical[step + 1] + persistence * exponential_residual
        )

        recursive_prior_mean = persistence * recursive_mean
        recursive_prior_variance = (
            persistence**2 * recursive_variance + process_variance
        )
        recursive_fallback_forecast = (
            physical[step + 1] + persistence * recursive_prior_mean
        )
        if available[step]:
            innovation = float(
                observation[step] - physical[step] - recursive_prior_mean
            )
            innovation_variance = recursive_prior_variance + observation_variance
            gain = recursive_prior_variance / innovation_variance
            recursive_mean = recursive_prior_mean + gain * innovation
            recursive_variance = (1.0 - gain) * recursive_prior_variance
            traces["recursive_gaussian"].accepted_update[step] = True
        else:
            recursive_mean = recursive_prior_mean
            recursive_variance = recursive_prior_variance
            traces["recursive_gaussian"].exact_fallback[step] = True
            traces["recursive_gaussian"].fallback_reason[step] = "missing-observation"
        recursive_forecast = physical[step + 1] + persistence * recursive_mean
        traces["recursive_gaussian"].forecast_mean_m[step] = recursive_forecast
        traces["recursive_gaussian"].forecast_variance_m2[step] = (
            persistence**2 * recursive_variance + process_variance
        )
        if traces["recursive_gaussian"].accepted_update[step]:
            traces["recursive_gaussian"].materially_harmful_update[step] = (
                _materially_harmful(
                    recursive_forecast,
                    recursive_fallback_forecast,
                    target,
                    margin_m=config.materially_harmful_margin_m,
                )
            )

        guarded_prior_mean = persistence * guarded_mean
        guarded_prior_variance = persistence**2 * guarded_variance + process_variance
        guarded_fallback_forecast = (
            physical[step + 1] + persistence * guarded_prior_mean
        )
        accepted = False
        reason = "none"
        candidate_mean = guarded_prior_mean
        candidate_variance = guarded_prior_variance
        if not available[step]:
            reason = "missing-observation"
        elif reliability[step] < config.minimum_reliability:
            reason = "low-reliability"
        elif source_step[step] != step:
            reason = "stale-observation"
        else:
            effective_observation_variance = observation_variance / float(
                reliability[step]
            )
            innovation = float(observation[step] - physical[step] - guarded_prior_mean)
            innovation_variance = (
                guarded_prior_variance + effective_observation_variance
            )
            gain = guarded_prior_variance / innovation_variance
            candidate_mean = guarded_prior_mean + gain * innovation
            candidate_variance = (1.0 - gain) * guarded_prior_variance
            nis = innovation**2 / innovation_variance
            if nis > config.maximum_nis:
                reason = "innovation-gate"
            elif (
                abs(candidate_mean - guarded_prior_mean) > config.maximum_update_m
                or abs(candidate_mean) > config.maximum_residual_m
            ):
                reason = "trust-region"
            else:
                accepted = True

        if accepted:
            guarded_mean = candidate_mean
            guarded_variance = candidate_variance
            traces["guarded_recursive"].accepted_update[step] = True
        else:
            guarded_mean = guarded_prior_mean
            guarded_variance = guarded_prior_variance
            traces["guarded_recursive"].exact_fallback[step] = True
            traces["guarded_recursive"].fallback_reason[step] = reason
            traces["guarded_recursive"].exact_fallback_valid[step] = bool(
                guarded_mean == guarded_prior_mean
                and guarded_variance == guarded_prior_variance
            )
        guarded_forecast = physical[step + 1] + persistence * guarded_mean
        traces["guarded_recursive"].forecast_mean_m[step] = guarded_forecast
        traces["guarded_recursive"].forecast_variance_m2[step] = (
            persistence**2 * guarded_variance + process_variance
        )
        if accepted:
            traces["guarded_recursive"].materially_harmful_update[step] = (
                _materially_harmful(
                    guarded_forecast,
                    guarded_fallback_forecast,
                    target,
                    margin_m=config.materially_harmful_margin_m,
                )
            )

    for trace in traces.values():
        for array in (
            trace.forecast_mean_m,
            trace.forecast_variance_m2,
            trace.accepted_update,
            trace.exact_fallback,
            trace.exact_fallback_valid,
            trace.materially_harmful_update,
            trace.fallback_reason,
        ):
            array.setflags(write=False)
    return traces


def _recovery_half_life_steps(
    absolute_error: np.ndarray,
    *,
    corruption_stop: int,
    recovery_window: int,
) -> int:
    segment = absolute_error[corruption_stop : corruption_stop + recovery_window]
    if not len(segment):
        return recovery_window + 1
    head = float(np.mean(segment[: min(3, len(segment))]))
    floor = float(np.mean(segment[-min(10, len(segment)) :]))
    if head <= floor:
        return 0
    threshold = floor + 0.5 * (head - floor)
    for offset in range(len(segment)):
        local = segment[offset : offset + 3]
        if len(local) and float(np.mean(local)) <= threshold:
            return offset
    return len(segment) + 1


def _probabilistic_metrics(
    error: np.ndarray,
    variance: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if not np.all(np.isfinite(variance)):
        return None, None, None
    safe_variance = np.maximum(variance, np.finfo(np.float64).tiny)
    nll = 0.5 * (np.log(2.0 * np.pi * safe_variance) + np.square(error) / safe_variance)
    half_width = _Z90 * np.sqrt(safe_variance)
    coverage = np.mean(np.abs(error) <= half_width)
    return (
        float(np.mean(nll)),
        float(coverage),
        float(np.mean(2.0 * half_width)),
    )


def _record(
    *,
    condition: str,
    seed: int,
    method: str,
    sequence: CorruptedSequence,
    trace: _MethodTrace,
    config: RecursiveCorruptionBenchmarkConfig,
) -> dict[str, Any]:
    target = sequence.true_position_m[1:]
    error = trace.forecast_mean_m - target
    start = config.corruption_start
    stop = start + config.corruption_length
    recovery_stop = min(stop + config.recovery_window, len(error))
    gaussian_nll, coverage_90, width_90 = _probabilistic_metrics(
        error,
        trace.forecast_variance_m2,
    )
    reason_counts = Counter(map(str, trace.fallback_reason))
    return {
        "condition": condition,
        "seed": seed,
        "method": method,
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "pre_corruption_rmse_m": float(np.sqrt(np.mean(np.square(error[:start])))),
        "corruption_rmse_m": float(np.sqrt(np.mean(np.square(error[start:stop])))),
        "recovery_rmse_m": float(
            np.sqrt(np.mean(np.square(error[stop:recovery_stop])))
        ),
        "maximum_absolute_error_m": float(np.max(np.abs(error))),
        "recovery_half_life_steps": _recovery_half_life_steps(
            np.abs(error),
            corruption_stop=stop,
            recovery_window=config.recovery_window,
        ),
        "accepted_update_count": int(np.sum(trace.accepted_update)),
        "fallback_count": int(np.sum(trace.exact_fallback)),
        "materially_harmful_accepted_update_count": int(
            np.sum(trace.materially_harmful_update)
        ),
        "exact_fallback_violation_count": int(
            np.sum(trace.exact_fallback & ~trace.exact_fallback_valid)
        ),
        "gaussian_nll": gaussian_nll,
        "coverage_90": coverage_90,
        "mean_full_interval_width_90_m": width_90,
        "fallback_reasons": {
            reason: int(reason_counts.get(reason, 0))
            for reason in FALLBACK_REASONS
            if reason != "none"
        },
    }


def _numeric_mean_sem(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> tuple[float | None, float | None]:
    values = [record[key] for record in records if record[key] is not None]
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    sem = 0.0
    if len(array) > 1:
        sem = float(np.std(array, ddof=1) / np.sqrt(len(array)))
    return mean, sem


def _aggregate_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = (
        "rmse_m",
        "pre_corruption_rmse_m",
        "corruption_rmse_m",
        "recovery_rmse_m",
        "maximum_absolute_error_m",
        "recovery_half_life_steps",
        "accepted_update_count",
        "fallback_count",
        "materially_harmful_accepted_update_count",
        "exact_fallback_violation_count",
        "gaussian_nll",
        "coverage_90",
        "mean_full_interval_width_90_m",
    )
    result: dict[str, Any] = {"sequence_count": len(records)}
    for metric in metrics:
        mean, sem = _numeric_mean_sem(records, metric)
        result[f"{metric}_mean"] = mean
        result[f"{metric}_sem"] = sem
    reasons: Counter[str] = Counter()
    for record in records:
        reasons.update(record["fallback_reasons"])
    result["fallback_reason_totals"] = {
        reason: int(reasons.get(reason, 0))
        for reason in FALLBACK_REASONS
        if reason != "none"
    }
    return result


def _relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return 100.0 * (candidate / baseline - 1.0)


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    corrupted = [record for record in records if record["condition"] != "clean"]
    claim_boundary = (
        "Controlled mechanism evidence only; no real-provider competence, "
        "physical-object transfer, covariance calibration, intervention "
        "benefit, deployment safety, or state-of-the-art claim is authorized."
    )
    if not corrupted:
        return {
            "corrupted_sequence_count_per_method": 0,
            "guarded_vs_last_residual_rmse_change_percent": None,
            "guarded_vs_recursive_gaussian_rmse_change_percent": None,
            "guarded_vs_recursive_gaussian_harmful_update_change_percent": None,
            "guarded_exact_fallback_violation_count": 0,
            "claim_boundary": claim_boundary,
        }

    by_method = {
        method: [record for record in corrupted if record["method"] == method]
        for method in METHODS
    }
    aggregates = {
        method: _aggregate_records(selected) for method, selected in by_method.items()
    }
    guarded = aggregates["guarded_recursive"]
    unguarded = aggregates["recursive_gaussian"]
    last = aggregates["last_residual"]
    return {
        "corrupted_sequence_count_per_method": len(by_method["guarded_recursive"]),
        "guarded_vs_last_residual_rmse_change_percent": _relative_change(
            float(guarded["rmse_m_mean"]),
            float(last["rmse_m_mean"]),
        ),
        "guarded_vs_recursive_gaussian_rmse_change_percent": _relative_change(
            float(guarded["rmse_m_mean"]),
            float(unguarded["rmse_m_mean"]),
        ),
        "guarded_vs_recursive_gaussian_harmful_update_change_percent": (
            _relative_change(
                float(guarded["materially_harmful_accepted_update_count_mean"]),
                float(unguarded["materially_harmful_accepted_update_count_mean"]),
            )
        ),
        "guarded_exact_fallback_violation_count": int(
            sum(
                int(record["exact_fallback_violation_count"])
                for record in by_method["guarded_recursive"]
            )
        ),
        "claim_boundary": claim_boundary,
    }


def run_recursive_corruption_benchmark(
    *,
    seeds: Sequence[int] = tuple(range(10)),
    conditions: Sequence[str] = CONDITIONS,
    config: RecursiveCorruptionBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run the complete deterministic benchmark and return JSON-ready results."""

    cfg = RecursiveCorruptionBenchmarkConfig() if config is None else config
    canonical_seeds = _canonical_seeds(seeds)
    canonical_conditions = _canonical_conditions(conditions)
    records: list[dict[str, Any]] = []
    for condition in canonical_conditions:
        for seed in canonical_seeds:
            sequence = generate_corrupted_sequence(
                condition,
                seed=seed,
                config=cfg,
            )
            traces = _run_methods(sequence, cfg)
            for method in METHODS:
                records.append(
                    _record(
                        condition=condition,
                        seed=seed,
                        method=method,
                        sequence=sequence,
                        trace=traces[method],
                        config=cfg,
                    )
                )

    aggregate: dict[str, Any] = {}
    for condition in canonical_conditions:
        aggregate[condition] = {}
        for method in METHODS:
            selected = [
                record
                for record in records
                if record["condition"] == condition and record["method"] == method
            ]
            aggregate[condition][method] = _aggregate_records(selected)
    aggregate["all_corruptions"] = {
        method: _aggregate_records(
            [
                record
                for record in records
                if record["condition"] != "clean" and record["method"] == method
            ]
        )
        for method in METHODS
    }

    result = {
        "schema": RECURSIVE_CORRUPTION_SCHEMA,
        "schema_version": RECURSIVE_CORRUPTION_SCHEMA_VERSION,
        "config": asdict(cfg),
        "seeds": list(canonical_seeds),
        "conditions": list(canonical_conditions),
        "methods": list(METHODS),
        "records": records,
        "aggregate": aggregate,
        "summary": _summary(records),
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def write_recursive_corruption_json(
    result: Mapping[str, Any],
    path: str | Path,
) -> None:
    """Write one deterministic, finite JSON benchmark record."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_recursive_corruption_csv(
    result: Mapping[str, Any],
    path: str | Path,
) -> None:
    """Write the long-form per-sequence metric table."""

    records = result.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("result records must be a nonempty list")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in records[0] if key != "fallback_reasons"]
    reason_fields = [
        f"fallback_reason__{reason}" for reason in FALLBACK_REASONS if reason != "none"
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[*fieldnames, *reason_fields],
        )
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames}
            reasons = record["fallback_reasons"]
            for reason in FALLBACK_REASONS:
                if reason == "none":
                    continue
                row[f"fallback_reason__{reason}"] = reasons.get(reason, 0)
            writer.writerow(row)


__all__ = [
    "CONDITIONS",
    "METHODS",
    "RECURSIVE_CORRUPTION_SCHEMA",
    "RECURSIVE_CORRUPTION_SCHEMA_VERSION",
    "CorruptedSequence",
    "RecursiveCorruptionBenchmarkConfig",
    "generate_corrupted_sequence",
    "run_recursive_corruption_benchmark",
    "write_recursive_corruption_csv",
    "write_recursive_corruption_json",
]
