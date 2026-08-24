"""End-to-end SBC execution for the controlled BayesianPhysTwin benchmark.

The existing simulation-based-calibration module validates and summarizes PIT
values. This module closes the remaining execution gap: it samples physical
parameters from the benchmark prior, generates trajectories and observations,
runs the exact finite-grid posterior, and emits matched, underdispersed, and
overdispersed calibration summaries from one target-free simulation roster.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np

from bayesian_phystwin.simulation_based_calibration import (
    SimulationBasedCalibrationSummaryV1,
    posterior_pit_matrix,
)
from bayesian_phystwin.synthetic_benchmark import (
    PARAMETER_NAMES,
    SyntheticBenchmarkConfig,
    make_action,
    parameter_grid,
    simulate_parameter_particles,
)

SYNTHETIC_BENCHMARK_SBC_SCHEMA: Final = (
    "bayesian_phystwin.synthetic_benchmark_sbc"
)
SYNTHETIC_BENCHMARK_SBC_VERSION: Final = 1
SYNTHETIC_BENCHMARK_SBC_CLAIM_BOUNDARY: Final = (
    "This is controlled self-consistency evidence for the finite synthetic "
    "benchmark posterior. It does not establish simulator adequacy, real-data "
    "calibration, physical identifiability, unseen-object transfer, deployment "
    "safety, Prob4D competence, Causal4D benefit, or state of the art."
)


@dataclass(frozen=True, slots=True)
class SyntheticBenchmarkSBCConfigV1:
    """Frozen execution settings for one controlled SBC run."""

    replicate_count: int = 512
    seed: int = 20260824
    bin_count: int = 10
    action_modes: tuple[str, ...] = ("dynamic", "quasi_static")
    likelihood_scale_multipliers: tuple[float, ...] = (1.0, 0.5, 2.0)

    def __post_init__(self) -> None:
        if type(self.replicate_count) is not int or self.replicate_count < 20:
            raise ValueError("replicate_count must be an integer of at least 20")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.bin_count) is not int or self.bin_count < 2:
            raise ValueError("bin_count must be an integer of at least two")
        modes = tuple(self.action_modes)
        if not modes or any(
            type(mode) is not str or mode not in {"dynamic", "quasi_static"}
            for mode in modes
        ):
            raise ValueError("action_modes must contain supported literal modes")
        if len(modes) != len(set(modes)):
            raise ValueError("action_modes must not contain duplicates")
        object.__setattr__(self, "action_modes", tuple(sorted(modes)))
        multipliers = np.asarray(self.likelihood_scale_multipliers, dtype=float)
        if (
            multipliers.ndim != 1
            or len(multipliers) < 2
            or not np.all(np.isfinite(multipliers))
            or np.any(multipliers <= 0.0)
        ):
            raise ValueError("likelihood_scale_multipliers must be finite and positive")
        if len(set(map(float, multipliers))) != len(multipliers):
            raise ValueError("likelihood_scale_multipliers must not contain duplicates")
        normalized_multipliers = tuple(sorted(map(float, multipliers)))
        if 1.0 not in set(normalized_multipliers):
            raise ValueError(
                "likelihood_scale_multipliers must include the matched arm 1.0"
            )
        object.__setattr__(
            self,
            "likelihood_scale_multipliers",
            normalized_multipliers,
        )


def _strict_json(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("JSON floats must be finite")
        return value
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise ValueError("JSON mapping keys must be strings")
            result[key] = _strict_json(nested)
        return result
    if isinstance(value, (tuple, list)):
        return [_strict_json(nested) for nested in value]
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    raise ValueError(f"unsupported JSON value {type(value).__name__}")


def _record_id(record: dict[str, object]) -> str:
    payload = json.dumps(
        _strict_json(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("log_weights must be a finite vector")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("posterior normalization failed")
    return weights / total


def _arm_name(multiplier: float) -> str:
    if multiplier == 1.0:
        return "matched_likelihood"
    if multiplier < 1.0:
        return f"underdispersed_{multiplier:g}x"
    return f"overdispersed_{multiplier:g}x"


def _array_record(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _summary_diagnostics(
    summary: SimulationBasedCalibrationSummaryV1,
) -> dict[str, object]:
    return {
        "mean_ks_distance": float(np.mean(summary.ks_distance)),
        "maximum_ks_distance": float(np.max(summary.ks_distance)),
        "mean_absolute_pit_bias": float(np.mean(np.abs(summary.mean_pit - 0.5))),
        "mean_absolute_90_coverage_error": float(
            np.mean(np.abs(summary.central_90_coverage - 0.9))
        ),
        "mean_lower_5_tail_rate": float(np.mean(summary.tail_5_rates[:, 0])),
        "mean_upper_5_tail_rate": float(np.mean(summary.tail_5_rates[:, 1])),
    }


def _summary_record(
    summary: SimulationBasedCalibrationSummaryV1,
    *,
    metadata: dict[str, object],
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "bayesian_phystwin.synthetic_benchmark_sbc_arm",
        "schema_version": 1,
        "group_count": len(summary.group_ids),
        "parameter_names": list(summary.parameter_names),
        "bin_count": summary.bin_count,
        "pit_values": _array_record(summary.pit_values),
        "histogram_counts": _array_record(summary.histogram_counts),
        "mean_pit": summary.mean_pit.tolist(),
        "ks_distance": summary.ks_distance.tolist(),
        "cramer_von_mises": summary.cramer_von_mises.tolist(),
        "central_50_coverage": summary.central_50_coverage.tolist(),
        "central_90_coverage": summary.central_90_coverage.tolist(),
        "central_95_coverage": summary.central_95_coverage.tolist(),
        "tail_5_rates": summary.tail_5_rates.tolist(),
        "metadata": metadata,
    }
    record["summary_id"] = _record_id(record)
    return record


def run_synthetic_benchmark_sbc_v1(
    *,
    config: SyntheticBenchmarkSBCConfigV1 | None = None,
    benchmark_config: SyntheticBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Run exact finite-grid SBC and two deliberate dispersion controls."""

    run_config = config or SyntheticBenchmarkSBCConfigV1()
    model_config = benchmark_config or SyntheticBenchmarkConfig()
    if model_config.train_step_count >= model_config.step_count:
        raise ValueError("train_step_count must be below step_count")
    if model_config.observation_std <= 0.0 or not np.isfinite(
        model_config.observation_std
    ):
        raise ValueError("observation_std must be finite and positive")

    particles = parameter_grid(model_config)
    if particles.ndim != 2 or particles.shape[1] != len(PARAMETER_NAMES):
        raise RuntimeError("synthetic benchmark parameter grid changed unexpectedly")
    trajectory_vectors: dict[str, np.ndarray] = {}
    trajectory_norms: dict[str, np.ndarray] = {}
    for mode in run_config.action_modes:
        trajectories = simulate_parameter_particles(
            particles,
            make_action(model_config, mode),
            model_config,
        )
        vectors = np.ascontiguousarray(
            trajectories[:, : model_config.train_step_count].reshape(len(particles), -1)
        )
        trajectory_vectors[mode] = vectors
        trajectory_norms[mode] = np.einsum("ij,ij->i", vectors, vectors)

    rng = np.random.default_rng(run_config.seed)
    truth_indices = rng.integers(0, len(particles), size=run_config.replicate_count)
    balanced_modes = np.resize(
        np.asarray(run_config.action_modes, dtype=object),
        run_config.replicate_count,
    )
    rng.shuffle(balanced_modes)
    tie_breakers = rng.random(
        (run_config.replicate_count, len(PARAMETER_NAMES)),
        dtype=np.float64,
    )
    truths = particles[truth_indices]
    weights_by_multiplier = {
        float(multiplier): np.empty(
            (run_config.replicate_count, len(particles)),
            dtype=np.float64,
        )
        for multiplier in run_config.likelihood_scale_multipliers
    }

    for replicate in range(run_config.replicate_count):
        mode = str(balanced_modes[replicate])
        vectors = trajectory_vectors[mode]
        truth = vectors[truth_indices[replicate]]
        observation = truth + rng.normal(
            scale=model_config.observation_std,
            size=truth.shape,
        )
        squared_error = (
            trajectory_norms[mode]
            - 2.0 * (vectors @ observation)
            + float(observation @ observation)
        )
        squared_error = np.maximum(squared_error, 0.0)
        for multiplier, posterior_rows in weights_by_multiplier.items():
            likelihood_std = model_config.observation_std * multiplier
            posterior_rows[replicate] = _normalize_log_weights(
                -0.5 * squared_error / (likelihood_std * likelihood_std)
            )

    posterior_support = np.broadcast_to(
        particles[None, :, :],
        (run_config.replicate_count, len(particles), len(PARAMETER_NAMES)),
    )
    group_ids = tuple(
        f"synthetic-sbc-{index:05d}" for index in range(run_config.replicate_count)
    )
    action_counts = {
        mode: int(np.sum(balanced_modes == mode)) for mode in run_config.action_modes
    }
    summaries: dict[str, SimulationBasedCalibrationSummaryV1] = {}
    summary_records: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    for multiplier in run_config.likelihood_scale_multipliers:
        arm = _arm_name(float(multiplier))
        pit = posterior_pit_matrix(
            posterior_support,
            truths,
            weights=weights_by_multiplier[float(multiplier)],
            tie_breakers=tie_breakers,
        )
        arm_metadata: dict[str, object] = {
            "generator": "bayesian_phystwin.synthetic_benchmark",
            "inference": "exact_finite_grid_gaussian_posterior",
            "action_mode_counts": action_counts,
            "likelihood_scale_multiplier": float(multiplier),
            "seed": run_config.seed,
            "target_outcomes_used": False,
            "claim_boundary": SYNTHETIC_BENCHMARK_SBC_CLAIM_BOUNDARY,
        }
        summary = SimulationBasedCalibrationSummaryV1(
            group_ids=group_ids,
            parameter_names=PARAMETER_NAMES,
            pit_values=pit,
            bin_count=run_config.bin_count,
            metadata=arm_metadata,
        )
        summaries[arm] = summary
        summary_records[arm] = _summary_record(summary, metadata=arm_metadata)
        diagnostics[arm] = _summary_diagnostics(summary)

    matched = diagnostics["matched_likelihood"]
    deliberate_controls = {
        arm: values
        for arm, values in diagnostics.items()
        if arm != "matched_likelihood"
    }
    control_separation = {
        "matched_has_smallest_mean_ks": all(
            float(matched["mean_ks_distance"]) < float(values["mean_ks_distance"])
            for values in deliberate_controls.values()
        ),
        "matched_has_smallest_90_coverage_error": all(
            float(matched["mean_absolute_90_coverage_error"])
            < float(values["mean_absolute_90_coverage_error"])
            for values in deliberate_controls.values()
        ),
    }

    record: dict[str, object] = {
        "schema": SYNTHETIC_BENCHMARK_SBC_SCHEMA,
        "schema_version": SYNTHETIC_BENCHMARK_SBC_VERSION,
        "interpretation": (
            "controlled finite-grid self-consistency and dispersion controls"
        ),
        "claim_boundary": SYNTHETIC_BENCHMARK_SBC_CLAIM_BOUNDARY,
        "run_config": asdict(run_config),
        "benchmark_config": asdict(model_config),
        "parameter_grid_size": int(len(particles)),
        "action_mode_counts": action_counts,
        "summary_ids": {
            arm: summary_record["summary_id"]
            for arm, summary_record in summary_records.items()
        },
        "summaries": summary_records,
        "diagnostics": diagnostics,
        "normative_control_separation": control_separation,
    }
    record["result_id"] = _record_id(record)
    return record


def write_synthetic_benchmark_sbc_v1(
    result: dict[str, object],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one finite JSON result without silent replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to replace existing result: {target}"
        )
    payload = json.dumps(
        _strict_json(result),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to replace existing result: {target}"
                ) from error
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "SYNTHETIC_BENCHMARK_SBC_CLAIM_BOUNDARY",
    "SYNTHETIC_BENCHMARK_SBC_SCHEMA",
    "SYNTHETIC_BENCHMARK_SBC_VERSION",
    "SyntheticBenchmarkSBCConfigV1",
    "run_synthetic_benchmark_sbc_v1",
    "write_synthetic_benchmark_sbc_v1",
]
