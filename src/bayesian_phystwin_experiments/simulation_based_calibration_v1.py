"""Controlled simulation-based calibration for the fixed-graph posterior.

The experiment samples truth uniformly from the complete discrete parameter
grid, simulates observations, and evaluates randomized discrete posterior PITs.
The exact clean model is separated from one fixed correlated-misspecification
stress. No external, source, confirmation, or target outcome is consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Final

import numpy as np

from bayesian_phystwin.synthetic_benchmark import (
    SyntheticBenchmarkConfig,
    generate_observations,
    make_action,
    parameter_grid,
    simulate_parameter_particles,
)

PROTOCOL_SCHEMA: Final = "bayesian-phystwin.simulation-based-calibration-protocol"
RESULT_SCHEMA: Final = "bayesian-phystwin.simulation-based-calibration-result"
SUMMARY_SCHEMA: Final = "bayesian-phystwin.simulation-based-calibration-summary"
SCHEMA_VERSION: Final = 1
CONDITIONS: Final = ("clean", "correlated")
ACTION_MODES: Final = ("dynamic", "quasi_static")
QUANTITIES: Final = (
    "stiffness",
    "damping",
    "control_scale",
    "terminal_last_node_displacement",
)
CREDIBLE_LEVELS: Final = (0.5, 0.8, 0.9, 0.95)
TRUTH_PRIOR: Final = "uniform-over-complete-discrete-parameter-grid-v1"
INFERENCE_MODEL: Final = "uniform-prior-clean-gaussian-grid-posterior-v1"
RANDOMIZED_PIT: Final = "strict-less-mass-plus-uniform-tie-mass-v1"
SCALAR_QUERY: Final = "final-frame-last-node-displacement-v1"
REPLICATE_RNG: Final = (
    "numpy-default-rng-seed-truth-index-observation-seed-four-pit-uniforms-v1"
)


def _plain_json(value: Any) -> Any:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return json.loads(encoded)


def _content_id(payload: Mapping[str, Any], identity_field: str) -> str:
    document = dict(_plain_json(payload))
    document.pop(identity_field, None)
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    return value


def _literal_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be Boolean")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _literal_string(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return result


def seal_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Content-address and validate a simulation-calibration protocol."""

    document = dict(_plain_json(payload))
    document.pop("protocol_id", None)
    document["protocol_id"] = _content_id(document, "protocol_id")
    validate_protocol(document)
    return document


def validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete target-free v1 protocol."""

    document = dict(_plain_json(_mapping(payload, name="protocol")))
    if (
        document.get("schema") != PROTOCOL_SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("simulation-calibration protocol schema changed")
    declared = _sha256(document.get("protocol_id"), name="protocol_id")
    if declared != _content_id(document, "protocol_id"):
        raise ValueError("protocol_id does not match protocol content")

    if tuple(_sequence(document.get("conditions"), name="conditions")) != CONDITIONS:
        raise ValueError("conditions changed")
    if (
        tuple(_sequence(document.get("action_modes"), name="action_modes"))
        != ACTION_MODES
    ):
        raise ValueError("action modes changed")
    if tuple(_sequence(document.get("quantities"), name="quantities")) != QUANTITIES:
        raise ValueError("quantities changed")
    levels = tuple(
        _finite(value, name="credible_levels")
        for value in _sequence(document.get("credible_levels"), name="credible_levels")
    )
    if levels != CREDIBLE_LEVELS:
        raise ValueError("credible levels changed")

    seed_start = _integer(document.get("seed_start"), name="seed_start")
    replicate_count = _integer(
        document.get("replicate_count"),
        name="replicate_count",
    )
    if seed_start < 0 or replicate_count < 32:
        raise ValueError("seed_start must be nonnegative and replicate_count at least 32")
    if seed_start + replicate_count > 2**63 - 1:
        raise ValueError("registered seed range overflows")

    if _integer(
        document.get("histogram_bin_count"),
        name="histogram_bin_count",
    ) != 10:
        raise ValueError("histogram bin count changed")
    familywise_alpha = _finite(
        document.get("familywise_alpha"),
        name="familywise_alpha",
    )
    if familywise_alpha != 0.05:
        raise ValueError("familywise alpha changed")
    required_failed_fraction = _finite(
        document.get("misspecification_required_failed_fraction"),
        name="misspecification_required_failed_fraction",
    )
    if required_failed_fraction != 1.0:
        raise ValueError("misspecification decision rule changed")

    fixed_strings = {
        "truth_prior": TRUTH_PRIOR,
        "inference_model": INFERENCE_MODEL,
        "randomized_pit": RANDOMIZED_PIT,
        "scalar_query": SCALAR_QUERY,
        "replicate_rng": REPLICATE_RNG,
    }
    for field, expected in fixed_strings.items():
        if document.get(field) != expected:
            raise ValueError(f"{field} changed")
    _literal_string(document.get("claim_boundary"), name="claim_boundary")

    boundary = _mapping(
        document.get("information_boundary"),
        name="information_boundary",
    )
    expected_boundary = {
        "external_data_used",
        "source_outcomes_used",
        "confirmation_outcomes_used",
        "target_outcomes_used",
        "model_selection_used",
    }
    if set(boundary) != expected_boundary:
        raise ValueError("information-boundary fields changed")
    for field in sorted(expected_boundary):
        if _boolean(boundary[field], name=f"information_boundary.{field}"):
            raise ValueError(f"forbidden information use: {field}")

    config_payload = dict(
        _plain_json(
            _mapping(document.get("benchmark_config"), name="benchmark_config")
        )
    )
    expected_config_fields = set(asdict(SyntheticBenchmarkConfig()))
    if set(config_payload) != expected_config_fields:
        raise ValueError("benchmark configuration fields changed")
    config = SyntheticBenchmarkConfig(**config_payload)
    if asdict(config) != config_payload:
        raise ValueError("benchmark configuration is not canonical")
    if not 3 <= config.train_step_count < config.step_count:
        raise ValueError("benchmark train/future split is invalid")
    if config.node_count < 3 or config.observation_std <= 0.0:
        raise ValueError("benchmark geometry or observation model is invalid")
    for field in ("stiffness_count", "damping_count", "control_scale_count"):
        if getattr(config, field) < 2:
            raise ValueError(f"{field} must be at least two")
    return document


def randomized_discrete_pit(
    values: np.ndarray,
    weights: np.ndarray,
    truth: float,
    tie_uniform: float,
) -> float:
    """Return P(X<truth|y) + U P(X=truth|y) for a discrete posterior."""

    support = np.asarray(values, dtype=float)
    posterior = np.asarray(weights, dtype=float)
    if support.ndim != 1 or posterior.shape != support.shape:
        raise ValueError("values and weights must have the same vector shape")
    if (
        not np.all(np.isfinite(support))
        or not np.all(np.isfinite(posterior))
        or np.any(posterior < 0.0)
    ):
        raise ValueError("posterior support and weights must be finite and nonnegative")
    total = float(np.sum(posterior))
    if total <= 0.0:
        raise ValueError("posterior weights must have positive mass")
    if not math.isfinite(tie_uniform) or not 0.0 <= tie_uniform <= 1.0:
        raise ValueError("tie_uniform must be in [0, 1]")
    normalized = posterior / total
    less = float(np.sum(normalized[support < truth]))
    equal = float(np.sum(normalized[support == truth]))
    if equal <= 0.0:
        raise ValueError("truth is absent from the registered discrete support")
    result = less + tie_uniform * equal
    return min(1.0, max(0.0, result))


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("log weights must be a finite vector")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("posterior normalization failed")
    return weights / total


def _posterior_weights(
    observed: np.ndarray,
    trajectories: np.ndarray,
    config: SyntheticBenchmarkConfig,
) -> np.ndarray:
    train = config.train_step_count
    residual = observed[:train][None, :, :] - trajectories[:, :train]
    standardized = residual / config.observation_std
    log_weights = -0.5 * np.sum(np.square(standardized), axis=(1, 2))
    return _normalize_log_weights(log_weights)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, probability, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _ks_distance(pits: Sequence[float]) -> float:
    values = np.sort(np.asarray(pits, dtype=float))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("at least one PIT value is required")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("PIT values must be in [0, 1]")
    count = values.size
    indices = np.arange(1, count + 1, dtype=float)
    upper = np.max(indices / count - values)
    lower = np.max(values - (indices - 1.0) / count)
    return float(max(upper, lower))


def _wilson_interval(successes: int, count: int) -> list[float]:
    if successes < 0 or count <= 0 or successes > count:
        raise ValueError("invalid Wilson interval counts")
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def _histogram(pits: Sequence[float], bin_count: int) -> list[int]:
    counts, _ = np.histogram(
        np.asarray(pits, dtype=float),
        bins=np.linspace(0.0, 1.0, bin_count + 1),
    )
    return [int(value) for value in counts]


def _quantity_values(
    particles: np.ndarray,
    trajectories: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "stiffness": particles[:, 0],
        "damping": particles[:, 1],
        "control_scale": particles[:, 2],
        "terminal_last_node_displacement": trajectories[:, -1, -1],
    }


def _replicate_row(
    *,
    seed: int,
    condition: str,
    action_mode: str,
    particles: np.ndarray,
    trajectories: np.ndarray,
    quantity_values: Mapping[str, np.ndarray],
    credible_levels: Sequence[float],
    config: SyntheticBenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    truth_index = int(rng.integers(particles.shape[0]))
    observation_seed = int(rng.integers(0, 2**32 - 1))
    truth_trajectory = trajectories[truth_index]
    observed = generate_observations(
        truth_trajectory,
        condition=condition,
        seed=observation_seed,
        config=config,
    ).observed
    weights = _posterior_weights(observed, trajectories, config)
    effective_sample_size = float(1.0 / np.sum(np.square(weights)))

    quantities: dict[str, Any] = {}
    for quantity in QUANTITIES:
        values = np.asarray(quantity_values[quantity], dtype=float)
        truth = float(values[truth_index])
        posterior_mean = float(np.dot(weights, values))
        posterior_variance = float(
            np.dot(weights, np.square(values - posterior_mean))
        )
        covered: dict[str, bool] = {}
        for level in credible_levels:
            tail = (1.0 - level) / 2.0
            lower = _weighted_quantile(values, weights, tail)
            upper = _weighted_quantile(values, weights, 1.0 - tail)
            covered[f"{level:g}"] = bool(lower <= truth <= upper)
        signed_error = posterior_mean - truth
        quantities[quantity] = {
            "truth": truth,
            "posterior_mean": posterior_mean,
            "posterior_std": math.sqrt(max(0.0, posterior_variance)),
            "signed_error": signed_error,
            "absolute_error": abs(signed_error),
            "randomized_pit": randomized_discrete_pit(
                values,
                weights,
                truth,
                float(rng.random()),
            ),
            "covered": covered,
        }
    return {
        "seed": seed,
        "condition": condition,
        "action_mode": action_mode,
        "truth_index": truth_index,
        "observation_seed": observation_seed,
        "posterior_effective_sample_size": effective_sample_size,
        "quantities": quantities,
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    replicate_count = int(protocol["replicate_count"])
    test_count = len(ACTION_MODES) * len(QUANTITIES)
    per_test_alpha = float(protocol["familywise_alpha"]) / test_count
    dkw_threshold = math.sqrt(
        math.log(2.0 / per_test_alpha) / (2.0 * replicate_count)
    )
    aggregate: list[dict[str, Any]] = []
    for action_mode in ACTION_MODES:
        for condition in CONDITIONS:
            group = [
                row
                for row in rows
                if row["action_mode"] == action_mode
                and row["condition"] == condition
            ]
            if len(group) != replicate_count:
                raise ValueError("replicate table is incomplete")
            for quantity in QUANTITIES:
                quantity_rows = [
                    _mapping(row["quantities"], name="quantities")[quantity]
                    for row in group
                ]
                pits = [
                    _finite(item["randomized_pit"], name="randomized_pit")
                    for item in quantity_rows
                ]
                ks_distance = _ks_distance(pits)
                coverage: dict[str, Any] = {}
                for level in CREDIBLE_LEVELS:
                    key = f"{level:g}"
                    successes = sum(
                        _boolean(
                            _mapping(item["covered"], name="covered")[key],
                            name=f"covered.{key}",
                        )
                        for item in quantity_rows
                    )
                    coverage[key] = {
                        "count": successes,
                        "rate": successes / replicate_count,
                        "wilson_95": _wilson_interval(
                            successes,
                            replicate_count,
                        ),
                    }
                aggregate.append(
                    {
                        "action_mode": action_mode,
                        "condition": condition,
                        "quantity": quantity,
                        "replicate_count": replicate_count,
                        "pit_mean": float(np.mean(pits)),
                        "pit_variance": float(np.var(pits)),
                        "pit_ks_distance": ks_distance,
                        "pit_histogram": _histogram(
                            pits,
                            int(protocol["histogram_bin_count"]),
                        ),
                        "bonferroni_dkw_95_threshold": dkw_threshold,
                        "uniformity_not_rejected": bool(
                            ks_distance <= dkw_threshold
                        ),
                        "mean_signed_error": float(
                            np.mean(
                                [
                                    _finite(
                                        item["signed_error"],
                                        name="signed_error",
                                    )
                                    for item in quantity_rows
                                ]
                            )
                        ),
                        "mean_absolute_error": float(
                            np.mean(
                                [
                                    _finite(
                                        item["absolute_error"],
                                        name="absolute_error",
                                    )
                                    for item in quantity_rows
                                ]
                            )
                        ),
                        "mean_posterior_std": float(
                            np.mean(
                                [
                                    _finite(
                                        item["posterior_std"],
                                        name="posterior_std",
                                    )
                                    for item in quantity_rows
                                ]
                            )
                        ),
                        "coverage": coverage,
                        "mean_effective_sample_size": float(
                            np.mean(
                                [
                                    _finite(
                                        row["posterior_effective_sample_size"],
                                        name="posterior_effective_sample_size",
                                    )
                                    for row in group
                                ]
                            )
                        ),
                    }
                )
    return aggregate, per_test_alpha


def run_simulation_based_calibration(
    protocol_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the complete registered target-free calibration study."""

    protocol = validate_protocol(protocol_payload)
    config = SyntheticBenchmarkConfig(**protocol["benchmark_config"])
    particles = parameter_grid(config)
    rows: list[dict[str, Any]] = []
    seed_start = int(protocol["seed_start"])
    replicate_count = int(protocol["replicate_count"])
    seeds = range(seed_start, seed_start + replicate_count)

    for action_mode in ACTION_MODES:
        action = make_action(config, action_mode)
        trajectories = simulate_parameter_particles(particles, action, config)
        quantity_values = _quantity_values(particles, trajectories)
        for condition in CONDITIONS:
            rows.extend(
                _replicate_row(
                    seed=seed,
                    condition=condition,
                    action_mode=action_mode,
                    particles=particles,
                    trajectories=trajectories,
                    quantity_values=quantity_values,
                    credible_levels=CREDIBLE_LEVELS,
                    config=config,
                )
                for seed in seeds
            )

    aggregate, per_test_alpha = _aggregate_rows(rows, protocol)
    clean_tests = [row for row in aggregate if row["condition"] == "clean"]
    correlated_tests = [
        row for row in aggregate if row["condition"] == "correlated"
    ]
    exact_model_not_rejected = all(
        bool(row["uniformity_not_rejected"]) for row in clean_tests
    )
    correlated_failed_fraction = sum(
        not bool(row["uniformity_not_rejected"]) for row in correlated_tests
    ) / len(correlated_tests)
    misspecification_detected = correlated_failed_fraction >= float(
        protocol["misspecification_required_failed_fraction"]
    )

    if exact_model_not_rejected and misspecification_detected:
        decision = "exact-model-calibration-not-rejected-and-misspecification-detected"
    elif not exact_model_not_rejected:
        decision = "exact-model-calibration-rejected"
    else:
        decision = "exact-model-calibration-not-rejected-but-stress-insensitive"

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "exact_model_calibration_not_rejected": exact_model_not_rejected,
        "correlated_misspecification_detected": misspecification_detected,
        "correlated_failed_test_fraction": correlated_failed_fraction,
        "familywise_test_count": len(clean_tests),
        "bonferroni_per_test_alpha": per_test_alpha,
        "aggregate": aggregate,
        "replicate_rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = _content_id(result, "result_id")
    return result


def compact_summary(result_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, content-addressed summary retaining row-table identity."""

    result = dict(_plain_json(_mapping(result_payload, name="result")))
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("simulation-calibration result schema changed")
    declared_result_id = _sha256(result.get("result_id"), name="result_id")
    if declared_result_id != _content_id(result, "result_id"):
        raise ValueError("result_id does not match result content")
    rows = list(_sequence(result.get("replicate_rows"), name="replicate_rows"))
    rows_digest = hashlib.sha256(
        json.dumps(
            rows,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": result["protocol_id"],
        "result_id": declared_result_id,
        "decision": result["decision"],
        "exact_model_calibration_not_rejected": result[
            "exact_model_calibration_not_rejected"
        ],
        "correlated_misspecification_detected": result[
            "correlated_misspecification_detected"
        ],
        "correlated_failed_test_fraction": result[
            "correlated_failed_test_fraction"
        ],
        "familywise_test_count": result["familywise_test_count"],
        "bonferroni_per_test_alpha": result["bonferroni_per_test_alpha"],
        "replicate_row_count": len(rows),
        "replicate_rows_sha256": rows_digest,
        "aggregate": result["aggregate"],
        "claim_boundary": result["claim_boundary"],
    }
    summary["summary_id"] = _content_id(summary, "summary_id")
    return summary


__all__ = [
    "ACTION_MODES",
    "CONDITIONS",
    "CREDIBLE_LEVELS",
    "INFERENCE_MODEL",
    "PROTOCOL_SCHEMA",
    "QUANTITIES",
    "RANDOMIZED_PIT",
    "REPLICATE_RNG",
    "RESULT_SCHEMA",
    "SCALAR_QUERY",
    "SCHEMA_VERSION",
    "SUMMARY_SCHEMA",
    "TRUTH_PRIOR",
    "compact_summary",
    "randomized_discrete_pit",
    "run_simulation_based_calibration",
    "seal_protocol",
    "validate_protocol",
]
