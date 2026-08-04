#!/usr/bin/env python3
"""Model and contract helpers for the controlled Prob4D-to-BPT study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from prob4d.gauge import GaugeEstimate
from prob4d.observation_factors import ObservationFactor, ObservationFactorBundle
from prob4d.sim3 import Sim3
from prob4d.sparse_observation_factors import (
    SparseStackedObservationFactors,
    stack_sparse_observation_factors,
)

from bayesian_phystwin._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,
)

PROTOCOL_SCHEMA = "bayesian-phystwin-prob4d-controlled-decisive-study"
REPORT_SCHEMA = "bayesian-phystwin-prob4d-controlled-decisive-report"
TRIAL_SCHEMA = "bayesian-phystwin-prob4d-controlled-decisive-trial"
CHI_SQUARE_3_90 = 6.251388631170325
REJECT_ALL_THRESHOLD = -1.0e300
FINITE_INFINITY = 1.0e300
PRIMARY_METHOD = "P3_explicit_gauge_persistent"
BASELINE_METHOD = "B0_physical_fallback"
MARGINAL_METHOD = "P1_marginal_gauge_persistent"
METHODS = (
    BASELINE_METHOD,
    "B1_naive_last_frame_state",
    MARGINAL_METHOD,
    "P2_explicit_gauge_framewise",
    PRIMARY_METHOD,
    "P4_explicit_gauge_persistent_metric_anchor",
)


@dataclass(frozen=True)
class StudyConfig:
    point_count: int
    frame_count: int
    state_count: int
    bias_mode_count: int
    window_count: int
    calibration_groups_per_scenario: int
    target_groups_per_scenario: int
    calibration_seed: int
    target_seed: int
    bootstrap_resamples: int
    bootstrap_seed: int
    harmful_margin_m: float
    guard_harmful_rate_at_most: float
    guard_minimum_accepted_groups: int
    conditional_noise_std_m: float
    state_mode_maximum_m: float
    query_progress: float
    state_prior_std: float
    source_revision: str
    scenarios: tuple[str, ...]


@dataclass(frozen=True)
class GroupData:
    group_id: str
    scenario: str
    stack: SparseStackedObservationFactors
    physical_prediction_m: np.ndarray
    state_jacobian: np.ndarray
    shared_bias_jacobian: np.ndarray
    query_state_jacobian: np.ndarray
    true_state: np.ndarray
    true_query_correction_m: np.ndarray
    true_gauge_delta: np.ndarray
    metric_anchor_observation: np.ndarray
    metric_anchor_covariance: np.ndarray
    physical_response_scale_m: float


@dataclass(frozen=True)
class Candidate:
    method_id: str
    inference_admissible: bool
    reason: str
    correction_m: np.ndarray
    covariance_m2: np.ndarray
    risk_score: float
    nominal_probability: float
    identifiable_fraction: float
    query_sensitivity_fraction: float
    fixed_point_converged: bool


@dataclass(frozen=True)
class CandidateScore:
    group_id: str
    scenario: str
    method_id: str
    candidate: Candidate
    baseline_rmse_m: float
    raw_rmse_m: float
    harmful_raw: bool
    coverage_90: float | None
    predictive_width_rms_m: float | None


@dataclass(frozen=True)
class GuardCalibration:
    method_id: str
    risk_threshold: float
    accepted_group_count: int
    harmful_accepted_count: int
    harmful_accepted_rate: float
    deployed_mean_rmse_m: float
    baseline_mean_rmse_m: float
    calibration_group_count: int
    fallback_only: bool


@dataclass(frozen=True)
class TrialResult:
    schema: str
    group_id: str
    scenario: str
    method_id: str
    solver_admissible: bool
    solver_reason: str
    risk_score: float
    guard_threshold: float
    guard_accepted: bool
    exact_fallback: bool
    baseline_rmse_m: float
    raw_rmse_m: float
    deployed_rmse_m: float
    raw_harmful: bool
    harmful_accepted: bool
    raw_improvement_fraction: float
    deployed_improvement_fraction: float
    coverage_90: float | None
    predictive_width_rms_m: float | None
    nominal_probability: float
    identifiable_fraction: float
    query_sensitivity_fraction: float
    fixed_point_converged: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--prob4d-revision", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def load_protocol(path: Path) -> tuple[dict[str, Any], StudyConfig]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    _require(protocol["schema"] == PROTOCOL_SCHEMA, "unexpected protocol schema")
    _require(protocol["schema_version"] == 1, "unexpected protocol version")
    geometry = protocol["geometry"]
    calibration = protocol["calibration"]
    target = protocol["target"]
    guard = protocol["guard_calibration"]
    generator = protocol["generator"]
    config = StudyConfig(
        point_count=int(geometry["point_count"]),
        frame_count=int(geometry["frame_count"]),
        state_count=int(geometry["state_count"]),
        bias_mode_count=int(geometry["bias_mode_count"]),
        window_count=int(geometry["window_count"]),
        calibration_groups_per_scenario=int(calibration["groups_per_scenario"]),
        target_groups_per_scenario=int(target["groups_per_scenario"]),
        calibration_seed=int(calibration["seed_start"]),
        target_seed=int(target["seed_start"]),
        bootstrap_resamples=int(protocol["bootstrap"]["resamples"]),
        bootstrap_seed=int(protocol["bootstrap"]["seed"]),
        harmful_margin_m=float(protocol["endpoints"]["harmful_margin_m"]),
        guard_harmful_rate_at_most=float(guard["harmful_accepted_rate_at_most"]),
        guard_minimum_accepted_groups=int(guard["minimum_accepted_groups"]),
        conditional_noise_std_m=float(generator["conditional_noise_std_m"]),
        state_mode_maximum_m=float(generator["state_mode_maximum_m"]),
        query_progress=float(generator["query_progress"]),
        state_prior_std=float(generator["state_prior_std"]),
        source_revision=str(protocol["repository_pins"]["prob4d"]),
        scenarios=tuple(map(str, protocol["scenarios"])),
    )
    _require(config.frame_count >= 3, "frame count must be at least three")
    _require(config.window_count == 2, "v1 requires exactly two gauges")
    _require(config.state_count >= 1, "state count must be positive")
    _require(config.point_count >= 8, "point count is too small")
    _require(len(config.scenarios) >= 3, "at least three scenarios are required")
    _require(tuple(protocol["methods"]) == METHODS, "method registry changed")
    return protocol, config


def _vector_bias_design(basis: np.ndarray) -> np.ndarray:
    point_count, mode_count = basis.shape
    design = np.zeros((point_count, 3, 3 * mode_count), dtype=np.float64)
    for coordinate in range(3):
        start = coordinate * mode_count
        design[:, coordinate, start : start + mode_count] = basis
    return design


def _orthogonal_state_modes(
    rng: np.random.Generator,
    point_count: int,
    state_count: int,
    bias_design: np.ndarray,
    maximum_m: float,
    *,
    weak_identifiability: bool,
) -> np.ndarray:
    raw = rng.normal(size=(point_count * 3, state_count))
    bias_flat = bias_design.reshape(point_count * 3, -1)
    bias_space = np.linalg.qr(bias_flat)[0]
    raw -= bias_space @ (bias_space.T @ raw)
    modes = np.linalg.qr(raw)[0][:, :state_count]
    if weak_identifiability:
        ambiguous = bias_flat[:, :state_count]
        ambiguous = np.linalg.qr(ambiguous)[0][:, :state_count]
        modes = 0.18 * modes + 0.982 * ambiguous
    modes = modes.reshape(point_count, 3, state_count)
    for state in range(state_count):
        maximum = float(np.max(np.linalg.norm(modes[:, :, state], axis=1)))
        modes[:, :, state] *= maximum_m / maximum
    return modes


def _scenario_parameters(scenario: str) -> dict[str, float]:
    table = {
        "nominal_correlated": {
            "gauge_correlation": 0.80,
            "gauge_scale": 1.0,
            "bias_scale_m": 0.0025,
            "outlier_group_probability": 0.0,
            "outlier_shift_m": 0.0,
            "nominal_probability": 0.96,
            "weak_identifiability": 0.0,
        },
        "common_mode_bias": {
            "gauge_correlation": 0.80,
            "gauge_scale": 1.0,
            "bias_scale_m": 0.009,
            "outlier_group_probability": 0.0,
            "outlier_shift_m": 0.0,
            "nominal_probability": 0.94,
            "weak_identifiability": 0.0,
        },
        "outlier_groups": {
            "gauge_correlation": 0.80,
            "gauge_scale": 1.0,
            "bias_scale_m": 0.003,
            "outlier_group_probability": 0.25,
            "outlier_shift_m": 0.020,
            "nominal_probability": 0.78,
            "weak_identifiability": 0.0,
        },
        "weak_identifiability": {
            "gauge_correlation": 0.80,
            "gauge_scale": 1.0,
            "bias_scale_m": 0.006,
            "outlier_group_probability": 0.0,
            "outlier_shift_m": 0.0,
            "nominal_probability": 0.94,
            "weak_identifiability": 1.0,
        },
        "large_gauge_uncertainty": {
            "gauge_correlation": 0.95,
            "gauge_scale": 2.2,
            "bias_scale_m": 0.003,
            "outlier_group_probability": 0.0,
            "outlier_shift_m": 0.0,
            "nominal_probability": 0.94,
            "weak_identifiability": 0.0,
        },
        "mixed_stress": {
            "gauge_correlation": 0.90,
            "gauge_scale": 1.6,
            "bias_scale_m": 0.007,
            "outlier_group_probability": 0.20,
            "outlier_shift_m": 0.015,
            "nominal_probability": 0.82,
            "weak_identifiability": 0.0,
        },
    }
    if scenario not in table:
        raise ValueError(f"unknown scenario: {scenario}")
    return table[scenario]


def _joint_gauge_covariance(
    correlation: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    standard_deviation = (
        np.asarray(
            [0.004, 0.007, 0.007, 0.007, 0.0035, 0.0035, 0.0035],
            dtype=np.float64,
        )
        * scale
    )
    block = np.diag(np.square(standard_deviation))
    cross = correlation * block
    joint = np.block([[block, cross], [cross, block]])
    return block, 0.5 * (joint + joint.T)


def _make_bundle(
    group_id: str,
    points_by_frame: np.ndarray,
    local_covariance: np.ndarray,
    reliability: np.ndarray,
    association: np.ndarray,
    nominal_probability: float,
    joint_gauge_covariance: np.ndarray,
    source_revision: str,
) -> ObservationFactorBundle:
    frame_count, point_count, _ = points_by_frame.shape
    gauge_block = joint_gauge_covariance[:7, :7]
    gauges = (
        GaugeEstimate("window-0", Sim3.identity(), gauge_block),
        GaugeEstimate("window-1", Sim3.identity(), gauge_block),
    )
    factors: list[ObservationFactor] = []
    for frame in range(frame_count):
        window = frame // max(1, frame_count // 2)
        window = min(window, 1)
        factors.append(
            ObservationFactor(
                factor_id=f"{group_id}:factor-{frame}",
                frame_index=frame,
                view_id="camera-0",
                window_id=f"window-{window}",
                gauge_id=f"window-{window}",
                point_ids=np.arange(point_count, dtype=np.int64),
                points_local_m=points_by_frame[frame],
                valid_mask=np.ones(point_count, dtype=bool),
                local_covariance_m2=local_covariance[frame],
                association_probability=association[frame],
                prior_reliability=reliability[frame],
                prior_nominal_probability=nominal_probability,
                composite_weight=min(1.0, 12.0 / point_count),
                correlation_group_id=f"{group_id}:frame-{frame}",
                causal_frame_stop=frame_count,
            )
        )
    return ObservationFactorBundle(
        sequence_id=f"{group_id}:sequence",
        case_id=group_id,
        stream_id="prob4d:controlled-explicit-gauge",
        factors=tuple(factors),
        gauges=gauges,
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision=source_revision,
        causal_frame_stop=frame_count,
        joint_gauge_covariance=joint_gauge_covariance,
        gauge_covariance_semantics="joint-cross-window",
        metadata={
            "study": "prob4d-bpt-controlled-decisive-v1",
            "persistent_point_ids": True,
        },
    )


def generate_group(
    seed: int,
    scenario: str,
    config: StudyConfig,
    *,
    group_prefix: str,
) -> GroupData:
    rng = np.random.default_rng(seed)
    parameters = _scenario_parameters(scenario)
    point_count = config.point_count
    frame_count = config.frame_count
    group_id = f"{group_prefix}-{scenario}-{seed}"

    coordinate = rng.uniform(-1.0, 1.0, size=(point_count, 3))
    base_points = np.column_stack(
        (
            0.18 * coordinate[:, 0],
            0.18 * coordinate[:, 1],
            0.85 + 0.12 * coordinate[:, 2],
        )
    )
    bias_basis = np.column_stack(
        (
            np.ones(point_count, dtype=np.float64),
            coordinate[:, 0] - np.mean(coordinate[:, 0]),
        )
    )[:, : config.bias_mode_count]
    bias_basis /= np.sqrt(np.mean(np.square(bias_basis), axis=0, keepdims=True))
    bias_design = _vector_bias_design(bias_basis)
    modes = _orthogonal_state_modes(
        rng,
        point_count,
        config.state_count,
        bias_design,
        config.state_mode_maximum_m,
        weak_identifiability=bool(parameters["weak_identifiability"]),
    )
    progress = np.linspace(0.30, 1.0, frame_count, dtype=np.float64)
    state_jacobian = progress[:, None, None, None] * modes[None]
    query_state_jacobian = config.query_progress * modes
    true_state = np.clip(
        rng.normal(scale=0.75, size=config.state_count),
        -1.5,
        1.5,
    )
    bias_coefficients = rng.normal(
        scale=parameters["bias_scale_m"],
        size=3 * config.bias_mode_count,
    )
    shared_bias = np.einsum("nck,k->nc", bias_design, bias_coefficients, optimize=True)
    state_signal = np.einsum("tncs,s->tnc", state_jacobian, true_state, optimize=True)

    conditional_variance = config.conditional_noise_std_m**2
    local_covariance = np.repeat(
        np.eye(3, dtype=np.float64)[None, None] * conditional_variance,
        frame_count * point_count,
        axis=0,
    ).reshape(frame_count, point_count, 3, 3)
    noise = rng.normal(
        scale=config.conditional_noise_std_m,
        size=(frame_count, point_count, 3),
    )
    outlier = np.zeros_like(noise)
    for frame in range(frame_count):
        if rng.random() < parameters["outlier_group_probability"]:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            outlier[frame] = parameters["outlier_shift_m"] * direction

    reliability = np.clip(
        0.96 - 0.16 * np.abs(coordinate[None, :, 0]),
        0.58,
        0.98,
    )
    reliability = np.repeat(reliability, frame_count, axis=0)
    association = np.clip(
        reliability * rng.uniform(0.92, 1.0, size=reliability.shape),
        0.50,
        0.99,
    )
    physical_prediction = np.repeat(base_points[None], frame_count, axis=0)
    nominal_points = (
        physical_prediction + state_signal + shared_bias[None] + noise + outlier
    )
    _, joint_covariance = _joint_gauge_covariance(
        parameters["gauge_correlation"],
        parameters["gauge_scale"],
    )
    initial_bundle = _make_bundle(
        group_id,
        nominal_points,
        local_covariance,
        reliability,
        association,
        parameters["nominal_probability"],
        joint_covariance,
        config.source_revision,
    )
    initial_stack = stack_sparse_observation_factors(initial_bundle)
    true_gauge_delta = rng.multivariate_normal(
        np.zeros(7 * config.window_count),
        joint_covariance,
    )
    gauge_effect = initial_stack.apply_gauge_delta(true_gauge_delta)
    gauge_effect = gauge_effect.reshape(frame_count, point_count, 3)
    final_bundle = _make_bundle(
        group_id,
        nominal_points + gauge_effect,
        local_covariance,
        reliability,
        association,
        parameters["nominal_probability"],
        joint_covariance,
        config.source_revision,
    )
    stack = stack_sparse_observation_factors(final_bundle)
    expected_frames = np.repeat(np.arange(frame_count), point_count)
    _require(
        np.array_equal(stack.frame_indices, expected_frames),
        "unexpected Prob4D factor stacking order",
    )
    _require(
        np.array_equal(
            stack.point_ids,
            np.tile(np.arange(point_count), frame_count),
        ),
        "persistent Prob4D point identities changed",
    )

    anchor_std = np.asarray(
        [0.001, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001],
        dtype=np.float64,
    )
    anchor_covariance = np.diag(np.square(anchor_std))
    anchor_observation = true_gauge_delta[:7] + rng.multivariate_normal(
        np.zeros(7), anchor_covariance
    )
    true_query = np.einsum("ncs,s->nc", query_state_jacobian, true_state, optimize=True)
    response_scale = float(
        np.max(
            np.linalg.norm(
                np.sum(np.abs(query_state_jacobian), axis=2),
                axis=1,
            )
        )
    )
    return GroupData(
        group_id=group_id,
        scenario=scenario,
        stack=stack,
        physical_prediction_m=physical_prediction.reshape(-1, 3),
        state_jacobian=state_jacobian.reshape(-1, 3, config.state_count),
        shared_bias_jacobian=np.repeat(bias_design[None], frame_count, axis=0).reshape(
            -1, 3, 3 * config.bias_mode_count
        ),
        query_state_jacobian=query_state_jacobian,
        true_state=true_state,
        true_query_correction_m=true_query,
        true_gauge_delta=true_gauge_delta,
        metric_anchor_observation=anchor_observation,
        metric_anchor_covariance=anchor_covariance,
        physical_response_scale_m=response_scale,
    )


def _condition_gauge_prior(
    prior_covariance: np.ndarray,
    anchor_observation: np.ndarray,
    anchor_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = len(prior_covariance)
    observation = np.zeros((7, dimension), dtype=np.float64)
    observation[:, :7] = np.eye(7, dtype=np.float64)
    innovation_covariance = (
        observation @ prior_covariance @ observation.T + anchor_covariance
    )
    gain = np.linalg.solve(
        innovation_covariance,
        observation @ prior_covariance,
    ).T
    posterior_mean = gain @ anchor_observation
    posterior_covariance = prior_covariance - gain @ observation @ prior_covariance
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    eigenvalues = np.linalg.eigvalsh(posterior_covariance)
    _require(np.min(eigenvalues) >= -1e-12, "conditioned gauge prior is not PSD")
    return posterior_mean, posterior_covariance


def _batch_for_method(
    group: GroupData,
    method_id: str,
    config: StudyConfig,
) -> GaugeAwareObservationBatch:
    stack = group.stack
    framewise = method_id == "P2_explicit_gauge_framewise"
    selected = (
        stack.frame_indices == np.max(stack.frame_indices)
        if framewise
        else np.ones(stack.observation_count, dtype=bool)
    )
    innovation = (stack.world_mean_m - group.physical_prediction_m)[selected]
    state = group.state_jacobian[selected]
    shared = group.shared_bias_jacobian[selected]
    query = group.query_state_jacobian
    correlation_groups = tuple(
        value
        for value, keep in zip(stack.correlation_group_ids, selected, strict=True)
        if keep
    )
    reliability = stack.prior_reliability[selected]
    nominal = stack.prior_nominal_probability[selected]
    composite = stack.composite_weight[selected]

    if method_id == MARGINAL_METHOD:
        observation_covariance = stack.marginal_world_covariance_m2[selected]
        gauge_jacobian = np.zeros((len(innovation), 3, 0), dtype=np.float64)
        gauge_prior = np.zeros((0, 0), dtype=np.float64)
    else:
        observation_covariance = stack.conditional_world_covariance_m2[selected]
        gauge_jacobian = stack.dense_gauge_jacobian()[selected]
        gauge_prior = stack.gauge_prior_covariance
        if method_id == "P4_explicit_gauge_persistent_metric_anchor":
            gauge_mean, gauge_prior = _condition_gauge_prior(
                gauge_prior,
                group.metric_anchor_observation,
                group.metric_anchor_covariance,
            )
            innovation = innovation - np.einsum(
                "mci,i->mc", gauge_jacobian, gauge_mean, optimize=True
            )

    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=observation_covariance,
        state_jacobian=state,
        gauge_jacobian=gauge_jacobian,
        shared_bias_jacobian=shared,
        view_bias_jacobian=np.zeros((len(innovation), 3, 0), dtype=np.float64),
        query_state_jacobian=query,
        gauge_prior_covariance=gauge_prior,
        correlation_group_ids=correlation_groups,
        prior_reliability=reliability,
        prior_nominal_probability=nominal,
        composite_weight=composite,
        state_prior_covariance_m2=(
            np.eye(config.state_count, dtype=np.float64) * config.state_prior_std**2
        ),
        physical_response_scale_m=group.physical_response_scale_m,
        composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
        metadata={
            "group_id": group.group_id,
            "scenario": group.scenario,
            "method_id": method_id,
            "prob4d_source_revision": config.source_revision,
            "gauge_covariance_semantics": (
                "marginalized-per-row"
                if method_id == MARGINAL_METHOD
                else "joint-cross-window-explicit"
            ),
        },
    )


def _query_covariance(
    query_jacobian: np.ndarray,
    state_covariance: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "nci,ij,ndj->ncd",
        query_jacobian,
        state_covariance,
        query_jacobian,
        optimize=True,
    )


def _risk_from_result(
    group: GroupData,
    result: Any,
    query_covariance: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    width = float(np.sqrt(np.mean(np.trace(query_covariance, axis1=1, axis2=2))))
    diagnostics = result.diagnostics
    nominal_values = diagnostics.get(
        "observation_group_posterior_nominal_probability",
        [],
    )
    nominal = float(np.mean(nominal_values)) if nominal_values else 0.0
    identifiable = (
        float(np.min(result.identifiable_fractions))
        if len(result.identifiable_fractions)
        else 0.0
    )
    sensitivity = (
        float(np.min(result.query_sensitivity_fractions))
        if len(result.query_sensitivity_fractions)
        else 0.0
    )
    converged = bool(diagnostics.get("mixture_fixed_point_converged", False))
    risk = (
        width / max(group.physical_response_scale_m, 1e-12)
        + (1.0 - nominal)
        + 0.50 * (1.0 - identifiable)
        + 0.25 * (1.0 - sensitivity)
        + (0.0 if converged else 0.35)
    )
    return risk, nominal, identifiable, sensitivity, converged
