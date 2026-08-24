"""Registered recursive-corruption benchmark with matched guards and fresh domains.

Version 2 isolates the value of Gaussian recursive belief propagation from the
value of a corruption guard by comparing it with a deterministic last-residual
method that receives the identical reliability, lineage, innovation, trust-
region, and exact-fallback logic.  The generator varies physical dynamics,
action trajectories, discrepancy dynamics, observation noise, corruption timing,
and corruption severity across independent seeds.

The benchmark is controlled synthetic mechanism evidence only.  It does not use
a real observation provider, physical target cohort, Deform360 confirmation
outcome, Causal4D execution, DLO4/DLO5 reserve, or held-v8 payload.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Final

import numpy as np

SCHEMA: Final = "bayesian-phystwin.recursive-corruption-benchmark-v2"
SCHEMA_VERSION: Final = 2
TRACE_SCHEMA: Final = "bayesian-phystwin.recursive-corruption-traces-v2"
TRACE_SCHEMA_VERSION: Final = 2

METHODS: Final[tuple[str, ...]] = (
    "physical_baseline",
    "last_residual",
    "guarded_last_residual",
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
    "reliability_false_negative",
    "reliability_false_positive",
    "timestamp_jitter",
    "partial_stale",
    "mixed_identity",
)
STRESS_CONDITIONS: Final[tuple[str, ...]] = tuple(
    condition for condition in CONDITIONS if condition != "clean"
)
IMPERFECT_CUE_CONDITIONS: Final[tuple[str, ...]] = (
    "reliability_false_negative",
    "reliability_false_positive",
    "timestamp_jitter",
    "partial_stale",
    "mixed_identity",
)
PRIMARY_ENDPOINTS: Final[tuple[str, ...]] = (
    "stress_full_rmse_guarded_recursive_minus_guarded_last_residual",
    "stress_harmful_updates_guarded_recursive_minus_recursive_gaussian",
)

FALLBACK_REASONS: Final[tuple[str, ...]] = (
    "none",
    "missing-observation",
    "low-reliability",
    "stale-observation",
    "innovation-gate",
    "trust-region",
)
FALLBACK_REASON_TO_CODE: Final = {
    reason: index for index, reason in enumerate(FALLBACK_REASONS)
}
_Z90: Final = NormalDist().inv_cdf(0.95)


@dataclass(frozen=True, slots=True)
class RecursiveCorruptionV2Config:
    """Frozen benchmark-level settings; seed-specific physics are drawn below."""

    step_count: int = 260
    recovery_window: int = 60
    minimum_reliability: float = 0.50
    maximum_reported_age_steps: int = 1
    maximum_nis: float = 9.0
    maximum_update_m: float = 0.025
    maximum_residual_m: float = 0.080
    materially_harmful_margin_m: float = 0.002
    clean_noninferiority_margin_m: float = 0.0001
    harmful_seed_margin_m: float = 0.0005
    maximum_harmful_seed_fraction: float = 0.25
    maximum_worst_seed_regret_m: float = 0.001
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 20260825

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True, slots=True)
class SeedDomain:
    """One independent physical/dynamical domain and corruption schedule."""

    seed: int
    time_step: float
    stiffness: float
    damping: float
    cubic_stiffness: float
    action_amplitude_1: float
    action_frequency_1: float
    action_phase_1: float
    action_amplitude_2: float
    action_frequency_2: float
    action_phase_2: float
    action_amplitude_3: float
    action_frequency_3: float
    action_phase_3: float
    discrepancy_persistence: float
    discrepancy_process_std_m: float
    observation_std_m: float
    initial_residual_std_m: float
    action_coupling_m: float
    corruption_start: int
    corruption_length: int
    outlier_std_m: float
    drift_magnitude_m: float
    identity_offset_m: float
    delay_steps: int
    density_keep_probability: float

    @property
    def corruption_stop(self) -> int:
        return self.corruption_start + self.corruption_length


@dataclass(frozen=True, slots=True)
class CorruptedSequenceV2:
    """One condition-specific observation stream over a shared seed domain."""

    physical_position_m: np.ndarray
    true_position_m: np.ndarray
    observation_m: np.ndarray
    observation_available: np.ndarray
    reliability: np.ndarray
    actual_source_step: np.ndarray
    reported_source_step: np.ndarray
    corruption_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class MethodTraceV2:
    """One method's one-step forecast trace."""

    forecast_mean_m: np.ndarray
    forecast_variance_m2: np.ndarray
    accepted_update: np.ndarray
    exact_fallback: np.ndarray
    exact_fallback_valid: np.ndarray
    materially_harmful_update: np.ndarray
    fallback_reason_code: np.ndarray


@dataclass(frozen=True, slots=True)
class GuardDecision:
    accepted: bool
    reason: str
    candidate_variance_m2: float


def _validate_config(config: RecursiveCorruptionV2Config) -> None:
    integer_values = {
        "step_count": config.step_count,
        "recovery_window": config.recovery_window,
        "maximum_reported_age_steps": config.maximum_reported_age_steps,
        "bootstrap_replicates": config.bootstrap_replicates,
        "bootstrap_seed": config.bootstrap_seed,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a literal integer")
    if config.step_count < 180:
        raise ValueError("step_count must be at least 180")
    if config.recovery_window < 20:
        raise ValueError("recovery_window must be at least 20")
    if config.maximum_reported_age_steps < 0:
        raise ValueError("maximum_reported_age_steps must be nonnegative")
    if config.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    if config.bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be nonnegative")

    finite_values = {
        "minimum_reliability": config.minimum_reliability,
        "maximum_nis": config.maximum_nis,
        "maximum_update_m": config.maximum_update_m,
        "maximum_residual_m": config.maximum_residual_m,
        "materially_harmful_margin_m": config.materially_harmful_margin_m,
        "clean_noninferiority_margin_m": config.clean_noninferiority_margin_m,
        "harmful_seed_margin_m": config.harmful_seed_margin_m,
        "maximum_harmful_seed_fraction": config.maximum_harmful_seed_fraction,
        "maximum_worst_seed_regret_m": config.maximum_worst_seed_regret_m,
    }
    for name, raw in finite_values.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.number)):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(float(raw)):
            raise ValueError(f"{name} must be a finite number")
    if not 0.0 < config.minimum_reliability < 1.0:
        raise ValueError("minimum_reliability must lie in (0, 1)")
    if config.maximum_nis <= 0.0:
        raise ValueError("maximum_nis must be positive")
    if config.maximum_update_m <= 0.0 or config.maximum_residual_m <= 0.0:
        raise ValueError("trust-region limits must be positive")
    if config.materially_harmful_margin_m < 0.0:
        raise ValueError("materially_harmful_margin_m must be nonnegative")
    if config.clean_noninferiority_margin_m < 0.0:
        raise ValueError("clean_noninferiority_margin_m must be nonnegative")
    if config.harmful_seed_margin_m < 0.0:
        raise ValueError("harmful_seed_margin_m must be nonnegative")
    if not 0.0 <= config.maximum_harmful_seed_fraction <= 1.0:
        raise ValueError("maximum_harmful_seed_fraction must lie in [0, 1]")
    if config.maximum_worst_seed_regret_m < 0.0:
        raise ValueError("maximum_worst_seed_regret_m must be nonnegative")


def _canonical_seeds(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("seeds must be an integer sequence")
    seeds = tuple(values)
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must be nonnegative literal integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    return seeds


def _canonical_conditions(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("conditions must be a string sequence")
    conditions = tuple(values)
    if not conditions:
        raise ValueError("at least one condition is required")
    if any(type(condition) is not str for condition in conditions):
        raise ValueError("conditions must be literal strings")
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    if len(conditions) != len(set(conditions)):
        raise ValueError("conditions must be unique")
    return conditions


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def draw_seed_domain(seed: int, config: RecursiveCorruptionV2Config) -> SeedDomain:
    """Draw one registered physical domain independently from the evidence roster."""

    canonical_seed = _canonical_seeds((seed,))[0]
    rng = _rng(canonical_seed, 10)
    corruption_start = int(rng.integers(75, 111))
    corruption_length = int(rng.integers(28, 45))
    if (
        corruption_start + corruption_length + config.recovery_window
        >= config.step_count
    ):
        raise ValueError(
            "drawn corruption schedule leaves no registered recovery window"
        )
    return SeedDomain(
        seed=canonical_seed,
        time_step=float(rng.uniform(0.035, 0.055)),
        stiffness=float(rng.uniform(1.4, 3.0)),
        damping=float(rng.uniform(0.30, 0.90)),
        cubic_stiffness=float(rng.uniform(0.02, 0.18)),
        action_amplitude_1=float(rng.uniform(0.50, 1.00)),
        action_frequency_1=float(rng.uniform(0.22, 0.70)),
        action_phase_1=float(rng.uniform(-math.pi, math.pi)),
        action_amplitude_2=float(rng.uniform(0.10, 0.40)),
        action_frequency_2=float(rng.uniform(0.05, 0.18)),
        action_phase_2=float(rng.uniform(-math.pi, math.pi)),
        action_amplitude_3=float(rng.uniform(0.05, 0.25)),
        action_frequency_3=float(rng.uniform(0.75, 1.35)),
        action_phase_3=float(rng.uniform(-math.pi, math.pi)),
        discrepancy_persistence=float(rng.uniform(0.88, 0.98)),
        discrepancy_process_std_m=float(rng.uniform(0.0008, 0.0025)),
        observation_std_m=float(rng.uniform(0.0030, 0.0060)),
        initial_residual_std_m=float(rng.uniform(0.006, 0.015)),
        action_coupling_m=float(rng.uniform(0.00015, 0.00070)),
        corruption_start=corruption_start,
        corruption_length=corruption_length,
        outlier_std_m=float(rng.uniform(0.025, 0.070)),
        drift_magnitude_m=float(rng.uniform(0.030, 0.090)),
        identity_offset_m=float(rng.uniform(0.030, 0.080)),
        delay_steps=int(rng.integers(3, 11)),
        density_keep_probability=float(rng.uniform(0.10, 0.40)),
    )


def _base_sequence(
    domain: SeedDomain,
    config: RecursiveCorruptionV2Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = _rng(domain.seed, 20)
    time = np.arange(config.step_count, dtype=np.float64) * domain.time_step
    action = domain.action_amplitude_1 * np.sin(
        2.0 * np.pi * domain.action_frequency_1 * time + domain.action_phase_1
    )
    action += domain.action_amplitude_2 * np.sign(
        np.sin(2.0 * np.pi * domain.action_frequency_2 * time + domain.action_phase_2)
    )
    action += domain.action_amplitude_3 * np.sin(
        2.0 * np.pi * domain.action_frequency_3 * time + domain.action_phase_3
    )

    physical: np.ndarray = np.zeros(config.step_count, dtype=np.float64)
    velocity = float(rng.normal(scale=0.02))
    physical[0] = float(rng.normal(scale=0.01))
    for step in range(config.step_count - 1):
        position = physical[step]
        acceleration = (
            -domain.stiffness * position
            - domain.damping * velocity
            - domain.cubic_stiffness * position**3
            + action[step]
        )
        velocity += domain.time_step * acceleration
        physical[step + 1] = position + domain.time_step * velocity

    discrepancy: np.ndarray = np.zeros(config.step_count, dtype=np.float64)
    discrepancy[0] = float(rng.normal(scale=domain.initial_residual_std_m))
    for step in range(config.step_count - 1):
        discrepancy[step + 1] = (
            domain.discrepancy_persistence * discrepancy[step]
            + domain.action_coupling_m * math.tanh(float(action[step]))
            + float(rng.normal(scale=domain.discrepancy_process_std_m))
        )
    truth = physical + discrepancy
    base_observation = truth + rng.normal(
        scale=domain.observation_std_m,
        size=config.step_count,
    )
    reliability = np.clip(
        0.90 + rng.normal(scale=0.035, size=config.step_count),
        0.65,
        0.995,
    )
    return physical, truth, base_observation, reliability


def generate_corrupted_sequence_v2(
    condition: str,
    *,
    domain: SeedDomain,
    config: RecursiveCorruptionV2Config,
) -> CorruptedSequenceV2:
    """Generate one condition with potentially imperfect provider cues."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    physical, truth, observation, reliability = _base_sequence(domain, config)
    observation = observation.copy()
    reliability = reliability.copy()
    available: np.ndarray = np.ones(config.step_count, dtype=bool)
    actual_source: np.ndarray = np.arange(config.step_count, dtype=np.int64)
    reported_source = actual_source.copy()
    corruption: np.ndarray = np.zeros(config.step_count, dtype=bool)
    start = domain.corruption_start
    stop = domain.corruption_stop
    selected: np.ndarray = np.arange(start, stop, dtype=np.int64)
    corruption[start:stop] = True
    rng = _rng(domain.seed, 100 + CONDITIONS.index(condition))

    phase = np.linspace(0.0, 2.0 * np.pi, domain.corruption_length, endpoint=False)
    distractor = (
        physical[start:stop]
        - 0.6 * (truth[start:stop] - physical[start:stop])
        + domain.identity_offset_m
        + 0.012 * np.sin(phase + float(rng.uniform(-math.pi, math.pi)))
        + rng.normal(scale=domain.observation_std_m, size=domain.corruption_length)
    )

    if condition == "clean":
        corruption[:] = False
    elif condition == "missing_burst":
        available[start:stop] = False
        observation[start:stop] = np.nan
        reliability[start:stop] = 0.0
    elif condition == "outlier_burst":
        observation[start:stop] += rng.normal(
            scale=domain.outlier_std_m,
            size=domain.corruption_length,
        )
        reliability[start:stop] = rng.uniform(0.05, 0.30, domain.corruption_length)
    elif condition == "coherent_drift":
        observation[start:stop] += np.linspace(
            0.0,
            domain.drift_magnitude_m,
            domain.corruption_length,
        )
        reliability[start:stop] = np.linspace(0.78, 0.35, domain.corruption_length)
    elif condition == "identity_switch":
        observation[start:stop] = distractor
        reliability[start:stop] = rng.uniform(0.08, 0.32, domain.corruption_length)
    elif condition == "delayed_observation":
        for step in selected:
            source = max(0, int(step) - domain.delay_steps)
            observation[step] = truth[source] + float(
                rng.normal(scale=domain.observation_std_m)
            )
            actual_source[step] = source
            reported_source[step] = source
        reliability[start:stop] = rng.uniform(0.65, 0.90, domain.corruption_length)
    elif condition == "density_drop":
        retained = (
            rng.random(domain.corruption_length) < domain.density_keep_probability
        )
        missing = selected[~retained]
        available[missing] = False
        observation[missing] = np.nan
        reliability[missing] = 0.0
    elif condition == "reliability_false_negative":
        observation[start:stop] += rng.normal(
            scale=0.75 * domain.outlier_std_m,
            size=domain.corruption_length,
        )
        reliability[start:stop] = rng.uniform(0.72, 0.98, domain.corruption_length)
    elif condition == "reliability_false_positive":
        low = rng.random(domain.corruption_length) < 0.55
        reliability[selected[low]] = rng.uniform(0.05, 0.42, int(np.sum(low)))
    elif condition == "timestamp_jitter":
        jitter = rng.integers(-2, 3, size=domain.corruption_length)
        reported_source[start:stop] = np.clip(
            selected + jitter,
            0,
            config.step_count - 1,
        )
        reliability[start:stop] = rng.uniform(0.72, 0.97, domain.corruption_length)
    elif condition == "partial_stale":
        stale = rng.random(domain.corruption_length) < 0.55
        delays = rng.integers(2, domain.delay_steps + 2, size=domain.corruption_length)
        report_jitter = rng.integers(-2, 3, size=domain.corruption_length)
        for offset, step in enumerate(selected):
            if stale[offset]:
                source = max(0, int(step) - int(delays[offset]))
                observation[step] = truth[source] + float(
                    rng.normal(scale=domain.observation_std_m)
                )
                actual_source[step] = source
            reported_source[step] = int(
                np.clip(
                    actual_source[step] + report_jitter[offset],
                    0,
                    config.step_count - 1,
                )
            )
        reliability[start:stop] = rng.uniform(0.68, 0.96, domain.corruption_length)
    elif condition == "mixed_identity":
        mixing = rng.uniform(0.20, 0.80, domain.corruption_length)
        observation[start:stop] = (
            (1.0 - mixing) * truth[start:stop]
            + mixing * distractor
            + rng.normal(
                scale=0.5 * domain.observation_std_m,
                size=domain.corruption_length,
            )
        )
        reliability[start:stop] = rng.uniform(0.55, 0.90, domain.corruption_length)

    arrays = (
        physical,
        truth,
        observation,
        available,
        reliability,
        actual_source,
        reported_source,
        corruption,
    )
    for array in arrays:
        array.setflags(write=False)
    return CorruptedSequenceV2(
        physical_position_m=physical,
        true_position_m=truth,
        observation_m=observation,
        observation_available=available,
        reliability=reliability,
        actual_source_step=actual_source,
        reported_source_step=reported_source,
        corruption_mask=corruption,
    )


def _empty_trace(step_count: int) -> MethodTraceV2:
    forecast_count = step_count - 1
    return MethodTraceV2(
        forecast_mean_m=np.zeros(forecast_count, dtype=np.float64),
        forecast_variance_m2=np.full(forecast_count, np.nan, dtype=np.float64),
        accepted_update=np.zeros(forecast_count, dtype=bool),
        exact_fallback=np.zeros(forecast_count, dtype=bool),
        exact_fallback_valid=np.ones(forecast_count, dtype=bool),
        materially_harmful_update=np.zeros(forecast_count, dtype=bool),
        fallback_reason_code=np.zeros(forecast_count, dtype=np.uint8),
    )


def _materially_harmful(
    candidate_forecast: float,
    fallback_forecast: float,
    target: float,
    *,
    margin_m: float,
) -> bool:
    return abs(candidate_forecast - target) > abs(fallback_forecast - target) + margin_m


def _guard_decision(
    *,
    available: bool,
    reliability: float,
    reported_age_steps: int,
    prior_mean: float,
    prior_variance_m2: float,
    candidate_mean: float,
    innovation: float,
    observation_variance_m2: float,
    config: RecursiveCorruptionV2Config,
) -> GuardDecision:
    """Apply the identical cue, innovation, and trust-region rule to both arms."""

    if not available:
        return GuardDecision(False, "missing-observation", prior_variance_m2)
    if reliability < config.minimum_reliability:
        return GuardDecision(False, "low-reliability", prior_variance_m2)
    if abs(reported_age_steps) > config.maximum_reported_age_steps:
        return GuardDecision(False, "stale-observation", prior_variance_m2)

    effective_observation_variance = observation_variance_m2 / max(
        reliability,
        float(np.finfo(np.float64).eps),
    )
    innovation_variance = prior_variance_m2 + effective_observation_variance
    gain = prior_variance_m2 / innovation_variance
    candidate_variance = (1.0 - gain) * prior_variance_m2
    nis = innovation**2 / innovation_variance
    if nis > config.maximum_nis:
        return GuardDecision(False, "innovation-gate", prior_variance_m2)
    if (
        abs(candidate_mean - prior_mean) > config.maximum_update_m
        or abs(candidate_mean) > config.maximum_residual_m
    ):
        return GuardDecision(False, "trust-region", prior_variance_m2)
    return GuardDecision(True, "none", candidate_variance)


def _fallback_identity(
    *,
    prior_mean: float,
    prior_variance: float,
    forecast: float,
) -> bytes:
    return np.asarray(
        (prior_mean, prior_variance, forecast),
        dtype="<f8",
    ).tobytes(order="C")


def run_methods_v2(
    sequence: CorruptedSequenceV2,
    *,
    domain: SeedDomain,
    config: RecursiveCorruptionV2Config,
) -> dict[str, MethodTraceV2]:
    """Run matched deterministic and Gaussian recursion with shared guards."""

    traces = {method: _empty_trace(config.step_count) for method in METHODS}
    physical = sequence.physical_position_m
    truth = sequence.true_position_m
    observation = sequence.observation_m
    available = sequence.observation_available
    reliability = sequence.reliability
    reported_source = sequence.reported_source_step
    phi = domain.discrepancy_persistence
    process_variance = domain.discrepancy_process_std_m**2
    observation_variance = domain.observation_std_m**2

    last_mean = 0.0
    guarded_last_mean = 0.0
    guarded_last_variance = domain.initial_residual_std_m**2
    gaussian_mean = 0.0
    gaussian_variance = domain.initial_residual_std_m**2
    guarded_gaussian_mean = 0.0
    guarded_gaussian_variance = domain.initial_residual_std_m**2

    for step in range(config.step_count - 1):
        target = float(truth[step + 1])
        physical_forecast = float(physical[step + 1])
        traces["physical_baseline"].forecast_mean_m[step] = physical_forecast

        # Unguarded deterministic residual persistence.
        last_prior = phi * last_mean
        last_fallback_forecast = physical_forecast + phi * last_prior
        if available[step]:
            last_candidate = float(observation[step] - physical[step])
            last_mean = last_candidate
            traces["last_residual"].accepted_update[step] = True
        else:
            last_mean = last_prior
            traces["last_residual"].exact_fallback[step] = True
            traces["last_residual"].fallback_reason_code[step] = (
                FALLBACK_REASON_TO_CODE["missing-observation"]
            )
        last_forecast = physical_forecast + phi * last_mean
        traces["last_residual"].forecast_mean_m[step] = last_forecast
        if traces["last_residual"].accepted_update[step]:
            traces["last_residual"].materially_harmful_update[step] = (
                _materially_harmful(
                    last_forecast,
                    last_fallback_forecast,
                    target,
                    margin_m=config.materially_harmful_margin_m,
                )
            )

        # Deterministic residual persistence with the same guard as Gaussian recursion.
        guarded_last_prior = phi * guarded_last_mean
        guarded_last_prior_variance = phi**2 * guarded_last_variance + process_variance
        guarded_last_fallback = physical_forecast + phi * guarded_last_prior
        guarded_last_candidate = guarded_last_prior
        guarded_last_innovation = 0.0
        if available[step]:
            guarded_last_candidate = float(observation[step] - physical[step])
            guarded_last_innovation = guarded_last_candidate - guarded_last_prior
        last_decision = _guard_decision(
            available=bool(available[step]),
            reliability=float(reliability[step]),
            reported_age_steps=step - int(reported_source[step]),
            prior_mean=guarded_last_prior,
            prior_variance_m2=guarded_last_prior_variance,
            candidate_mean=guarded_last_candidate,
            innovation=guarded_last_innovation,
            observation_variance_m2=observation_variance,
            config=config,
        )
        if last_decision.accepted:
            guarded_last_mean = guarded_last_candidate
            guarded_last_variance = last_decision.candidate_variance_m2
            traces["guarded_last_residual"].accepted_update[step] = True
        else:
            reference = _fallback_identity(
                prior_mean=guarded_last_prior,
                prior_variance=guarded_last_prior_variance,
                forecast=guarded_last_fallback,
            )
            guarded_last_mean = guarded_last_prior
            guarded_last_variance = guarded_last_prior_variance
            traces["guarded_last_residual"].exact_fallback[step] = True
            traces["guarded_last_residual"].fallback_reason_code[step] = (
                FALLBACK_REASON_TO_CODE[last_decision.reason]
            )
            selected = _fallback_identity(
                prior_mean=guarded_last_mean,
                prior_variance=guarded_last_variance,
                forecast=physical_forecast + phi * guarded_last_mean,
            )
            traces["guarded_last_residual"].exact_fallback_valid[step] = (
                selected == reference
            )
        guarded_last_forecast = physical_forecast + phi * guarded_last_mean
        traces["guarded_last_residual"].forecast_mean_m[step] = guarded_last_forecast
        if last_decision.accepted:
            traces["guarded_last_residual"].materially_harmful_update[step] = (
                _materially_harmful(
                    guarded_last_forecast,
                    guarded_last_fallback,
                    target,
                    margin_m=config.materially_harmful_margin_m,
                )
            )

        # Unguarded Gaussian recursive discrepancy belief.
        gaussian_prior = phi * gaussian_mean
        gaussian_prior_variance = phi**2 * gaussian_variance + process_variance
        gaussian_fallback = physical_forecast + phi * gaussian_prior
        if available[step]:
            innovation = float(observation[step] - physical[step] - gaussian_prior)
            innovation_variance = gaussian_prior_variance + observation_variance
            gain = gaussian_prior_variance / innovation_variance
            gaussian_mean = gaussian_prior + gain * innovation
            gaussian_variance = (1.0 - gain) * gaussian_prior_variance
            traces["recursive_gaussian"].accepted_update[step] = True
        else:
            gaussian_mean = gaussian_prior
            gaussian_variance = gaussian_prior_variance
            traces["recursive_gaussian"].exact_fallback[step] = True
            traces["recursive_gaussian"].fallback_reason_code[step] = (
                FALLBACK_REASON_TO_CODE["missing-observation"]
            )
        gaussian_forecast = physical_forecast + phi * gaussian_mean
        traces["recursive_gaussian"].forecast_mean_m[step] = gaussian_forecast
        traces["recursive_gaussian"].forecast_variance_m2[step] = (
            phi**2 * gaussian_variance + process_variance
        )
        if traces["recursive_gaussian"].accepted_update[step]:
            traces["recursive_gaussian"].materially_harmful_update[step] = (
                _materially_harmful(
                    gaussian_forecast,
                    gaussian_fallback,
                    target,
                    margin_m=config.materially_harmful_margin_m,
                )
            )

        # Guarded Gaussian recursion, using the same helper as guarded last residual.
        guarded_prior = phi * guarded_gaussian_mean
        guarded_prior_variance = phi**2 * guarded_gaussian_variance + process_variance
        guarded_fallback = physical_forecast + phi * guarded_prior
        guarded_innovation = 0.0
        guarded_candidate = guarded_prior
        if available[step]:
            guarded_innovation = float(
                observation[step] - physical[step] - guarded_prior
            )
            effective_variance = observation_variance / max(
                float(reliability[step]),
                float(np.finfo(np.float64).eps),
            )
            gain = guarded_prior_variance / (
                guarded_prior_variance + effective_variance
            )
            guarded_candidate = guarded_prior + gain * guarded_innovation
        guarded_decision = _guard_decision(
            available=bool(available[step]),
            reliability=float(reliability[step]),
            reported_age_steps=step - int(reported_source[step]),
            prior_mean=guarded_prior,
            prior_variance_m2=guarded_prior_variance,
            candidate_mean=guarded_candidate,
            innovation=guarded_innovation,
            observation_variance_m2=observation_variance,
            config=config,
        )
        if guarded_decision.accepted:
            guarded_gaussian_mean = guarded_candidate
            guarded_gaussian_variance = guarded_decision.candidate_variance_m2
            traces["guarded_recursive"].accepted_update[step] = True
        else:
            reference = _fallback_identity(
                prior_mean=guarded_prior,
                prior_variance=guarded_prior_variance,
                forecast=guarded_fallback,
            )
            guarded_gaussian_mean = guarded_prior
            guarded_gaussian_variance = guarded_prior_variance
            traces["guarded_recursive"].exact_fallback[step] = True
            traces["guarded_recursive"].fallback_reason_code[step] = (
                FALLBACK_REASON_TO_CODE[guarded_decision.reason]
            )
            selected = _fallback_identity(
                prior_mean=guarded_gaussian_mean,
                prior_variance=guarded_gaussian_variance,
                forecast=physical_forecast + phi * guarded_gaussian_mean,
            )
            traces["guarded_recursive"].exact_fallback_valid[step] = (
                selected == reference
            )
        guarded_forecast = physical_forecast + phi * guarded_gaussian_mean
        traces["guarded_recursive"].forecast_mean_m[step] = guarded_forecast
        traces["guarded_recursive"].forecast_variance_m2[step] = (
            phi**2 * guarded_gaussian_variance + process_variance
        )
        if guarded_decision.accepted:
            traces["guarded_recursive"].materially_harmful_update[step] = (
                _materially_harmful(
                    guarded_forecast,
                    guarded_fallback,
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
            trace.fallback_reason_code,
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
    if segment.size == 0:
        return recovery_window + 1
    head = float(np.mean(segment[: min(3, segment.size)]))
    floor = float(np.mean(segment[-min(10, segment.size) :]))
    if head <= floor:
        return 0
    threshold = floor + 0.5 * (head - floor)
    for offset in range(segment.size):
        local = segment[offset : offset + 3]
        if local.size and float(np.mean(local)) <= threshold:
            return offset
    return segment.size + 1


def _probabilistic_metrics(
    error: np.ndarray,
    variance: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if not np.all(np.isfinite(variance)):
        return None, None, None
    safe_variance = np.maximum(variance, np.finfo(np.float64).tiny)
    nll = 0.5 * (np.log(2.0 * np.pi * safe_variance) + np.square(error) / safe_variance)
    half_width = _Z90 * np.sqrt(safe_variance)
    return (
        float(np.mean(nll)),
        float(np.mean(np.abs(error) <= half_width)),
        float(np.mean(2.0 * half_width)),
    )


def _record(
    *,
    seed: int,
    condition: str,
    method: str,
    domain: SeedDomain,
    sequence: CorruptedSequenceV2,
    trace: MethodTraceV2,
    config: RecursiveCorruptionV2Config,
) -> dict[str, Any]:
    target = sequence.true_position_m[1:]
    error = trace.forecast_mean_m - target
    start = domain.corruption_start
    stop = domain.corruption_stop
    recovery_stop = min(stop + config.recovery_window, error.size)
    nll, coverage, width = _probabilistic_metrics(
        error,
        trace.forecast_variance_m2,
    )
    reason_counts = Counter(
        FALLBACK_REASONS[int(code)] for code in trace.fallback_reason_code
    )
    harmful_count: int | None = None
    if method != "physical_baseline":
        harmful_count = int(np.sum(trace.materially_harmful_update))
    return {
        "seed": seed,
        "condition": condition,
        "method": method,
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
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
        "materially_harmful_accepted_update_count": harmful_count,
        "exact_fallback_violation_count": int(
            np.sum(trace.exact_fallback & ~trace.exact_fallback_valid)
        ),
        "gaussian_nll": nll,
        "coverage_90": coverage,
        "mean_full_interval_width_90_m": width,
        "fallback_reasons": {
            reason: int(reason_counts.get(reason, 0))
            for reason in FALLBACK_REASONS
            if reason != "none"
        },
    }


def _canonical_payload_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_recursive_corruption_benchmark_v2(
    *,
    seeds: Sequence[int],
    conditions: Sequence[str] = CONDITIONS,
    config: RecursiveCorruptionV2Config | None = None,
    retain_traces: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Run v2 and return finite summary records plus optional per-step traces."""

    cfg = RecursiveCorruptionV2Config() if config is None else config
    canonical_seeds = _canonical_seeds(seeds)
    canonical_conditions = _canonical_conditions(conditions)
    seed_domains = tuple(draw_seed_domain(seed, cfg) for seed in canonical_seeds)
    forecast_count = cfg.step_count - 1

    records: list[dict[str, Any]] = []
    trace_arrays: dict[str, np.ndarray] | None = None
    if retain_traces:
        shape = (
            len(canonical_seeds),
            len(canonical_conditions),
            len(METHODS),
            forecast_count,
        )
        trace_arrays = {
            "absolute_error_m": np.zeros(shape, dtype=np.float32),
            "accepted_update": np.zeros(shape, dtype=np.uint8),
            "exact_fallback": np.zeros(shape, dtype=np.uint8),
            "materially_harmful_update": np.zeros(shape, dtype=np.uint8),
            "fallback_reason_code": np.zeros(shape, dtype=np.uint8),
            "corruption_mask": np.zeros(
                (len(canonical_seeds), len(canonical_conditions), forecast_count),
                dtype=np.uint8,
            ),
            "reliability": np.zeros(
                (len(canonical_seeds), len(canonical_conditions), forecast_count),
                dtype=np.float32,
            ),
            "reported_age_steps": np.zeros(
                (len(canonical_seeds), len(canonical_conditions), forecast_count),
                dtype=np.int16,
            ),
        }

    for seed_index, domain in enumerate(seed_domains):
        for condition_index, condition in enumerate(canonical_conditions):
            sequence = generate_corrupted_sequence_v2(
                condition,
                domain=domain,
                config=cfg,
            )
            traces = run_methods_v2(sequence, domain=domain, config=cfg)
            target = sequence.true_position_m[1:]
            for method_index, method in enumerate(METHODS):
                trace = traces[method]
                records.append(
                    _record(
                        seed=domain.seed,
                        condition=condition,
                        method=method,
                        domain=domain,
                        sequence=sequence,
                        trace=trace,
                        config=cfg,
                    )
                )
                if trace_arrays is not None:
                    trace_arrays["absolute_error_m"][
                        seed_index, condition_index, method_index
                    ] = np.abs(trace.forecast_mean_m - target).astype(np.float32)
                    trace_arrays["accepted_update"][
                        seed_index, condition_index, method_index
                    ] = trace.accepted_update.astype(np.uint8)
                    trace_arrays["exact_fallback"][
                        seed_index, condition_index, method_index
                    ] = trace.exact_fallback.astype(np.uint8)
                    trace_arrays["materially_harmful_update"][
                        seed_index, condition_index, method_index
                    ] = trace.materially_harmful_update.astype(np.uint8)
                    trace_arrays["fallback_reason_code"][
                        seed_index, condition_index, method_index
                    ] = trace.fallback_reason_code
            if trace_arrays is not None:
                trace_arrays["corruption_mask"][seed_index, condition_index] = (
                    sequence.corruption_mask[1:].astype(np.uint8)
                )
                trace_arrays["reliability"][seed_index, condition_index] = (
                    sequence.reliability[:-1].astype(np.float32)
                )
                trace_arrays["reported_age_steps"][seed_index, condition_index] = (
                    np.arange(forecast_count, dtype=np.int64)
                    - sequence.reported_source_step[:-1]
                ).astype(np.int16)

    expected_count = len(canonical_seeds) * len(canonical_conditions) * len(METHODS)
    if len(records) != expected_count:
        raise RuntimeError("record matrix is incomplete")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evidence_role": "controlled-synthetic-mechanism",
        "config": asdict(cfg),
        "seeds": list(canonical_seeds),
        "conditions": list(canonical_conditions),
        "stress_conditions": [
            condition
            for condition in STRESS_CONDITIONS
            if condition in canonical_conditions
        ],
        "imperfect_cue_conditions": [
            condition
            for condition in IMPERFECT_CUE_CONDITIONS
            if condition in canonical_conditions
        ],
        "methods": list(METHODS),
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "seed_domains": [asdict(domain) for domain in seed_domains],
        "records": records,
        "trace_schema": TRACE_SCHEMA if retain_traces else None,
        "trace_schema_version": TRACE_SCHEMA_VERSION if retain_traces else None,
        "scientific_boundary": (
            "Controlled synthetic mechanism evidence only. No real-provider "
            "competence, independent physical transfer, physical-data calibration, "
            "physical-state validity, Causal4D benefit, deployment safety, or "
            "state-of-the-art claim is authorized."
        ),
        "access_boundary": {
            "real_provider_used": False,
            "physical_target_used": False,
            "causal4d_outcome_used": False,
            "deform360_confirmation_opened": False,
            "dlo4_dlo5_opened": False,
            "held_v8_opened": False,
        },
    }
    result["result_id"] = _canonical_payload_id(result)
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result, trace_arrays


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_records_csv(result: Mapping[str, Any], path: str | Path) -> None:
    raw_records = result.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("result records must be a nonempty list")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in raw_records[0] if key != "fallback_reasons"]
    reason_fields = [
        f"fallback_reason__{reason}" for reason in FALLBACK_REASONS if reason != "none"
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fieldnames, *reason_fields])
        writer.writeheader()
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise ValueError("each record must be a mapping")
            row = {key: raw_record.get(key) for key in fieldnames}
            raw_reasons = raw_record.get("fallback_reasons")
            if not isinstance(raw_reasons, Mapping):
                raise ValueError("fallback_reasons must be a mapping")
            for reason in FALLBACK_REASONS:
                if reason != "none":
                    row[f"fallback_reason__{reason}"] = raw_reasons.get(reason, 0)
            writer.writerow(row)


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def write_deterministic_trace_npz(
    *,
    arrays: Mapping[str, np.ndarray],
    result: Mapping[str, Any],
    path: str | Path,
) -> None:
    """Write deterministic compressed per-step traces with fixed ZIP metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        "result_id": result["result_id"],
        "seeds": result["seeds"],
        "conditions": result["conditions"],
        "methods": result["methods"],
        "fallback_reasons": list(FALLBACK_REASONS),
    }
    entries: dict[str, bytes] = {
        f"{name}.npy": _npy_bytes(np.asarray(array))
        for name, array in sorted(arrays.items())
    }
    entries["metadata.json"] = (
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, content in sorted(entries.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info,
                    content,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CONDITIONS",
    "FALLBACK_REASONS",
    "IMPERFECT_CUE_CONDITIONS",
    "METHODS",
    "PRIMARY_ENDPOINTS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STRESS_CONDITIONS",
    "TRACE_SCHEMA",
    "TRACE_SCHEMA_VERSION",
    "CorruptedSequenceV2",
    "MethodTraceV2",
    "RecursiveCorruptionV2Config",
    "SeedDomain",
    "draw_seed_domain",
    "generate_corrupted_sequence_v2",
    "run_methods_v2",
    "run_recursive_corruption_benchmark_v2",
    "sha256_file",
    "write_deterministic_trace_npz",
    "write_json",
    "write_records_csv",
]
