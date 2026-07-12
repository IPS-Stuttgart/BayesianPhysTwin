"""Synthetic operating-point controls for the v3 mechanism-promotion gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from causal4d.benchmark import make_objects
from causal4d.simulator import (
    Action,
    SimulatorConfig,
    WorldCondition,
    graph_laplacian,
    simulate,
)
from causal4d.source_panel_signatures import heldout_mechanism_eligibility


ArmName = Literal["placebo_null", "positive_control", "placebo_on_positive"]


@dataclass(frozen=True)
class MechanismGateControlConfig:
    """Audit v3's 10% aggregate shrinkage and 8/12 positive-session gate."""

    simulation_count: int = 512
    random_seed: int = 20260712
    frame_count: int = 48
    dt_s: float = 0.03
    prefix_frame_count: int = 6
    graph_rank: int = 4
    force_magnitude_n: float = 1.00
    positive_actuation_gain: float = 0.90
    persistent_offset_std_m: float = 0.00002
    temporal_noise_std_m: float = 0.00003
    temporal_noise_correlation: float = 0.70
    actuation_gain_grid: tuple[float, ...] = (
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
    )
    placebo_response_rotation_deg: float = 90.0
    minimum_shrinkage_fraction: float = 0.10
    minimum_positive_sessions: int = 8
    maximum_placebo_false_positive_rate: float = 0.05
    minimum_positive_control_power: float = 0.80
    interval_z: float = 1.96

    def __post_init__(self) -> None:
        if self.simulation_count < 1:
            raise ValueError("simulation_count must be positive")
        if self.frame_count < self.prefix_frame_count + 8:
            raise ValueError("frame_count leaves too little untouched future")
        if self.prefix_frame_count < 2:
            raise ValueError("prefix_frame_count must be at least two")
        if not 1 <= self.graph_rank <= 4:
            raise ValueError("graph_rank must be in [1, 4]")
        if self.force_magnitude_n <= 0.0:
            raise ValueError("force_magnitude_n must be positive")
        if self.positive_actuation_gain <= 0.0:
            raise ValueError("positive_actuation_gain must be positive")
        if self.persistent_offset_std_m <= 0.0:
            raise ValueError("persistent_offset_std_m must be positive")
        if self.temporal_noise_std_m <= 0.0:
            raise ValueError("temporal_noise_std_m must be positive")
        if not 0.0 <= self.temporal_noise_correlation < 1.0:
            raise ValueError("temporal_noise_correlation must be in [0, 1)")
        if tuple(sorted(self.actuation_gain_grid)) != self.actuation_gain_grid:
            raise ValueError("actuation_gain_grid must be sorted")
        if self.positive_actuation_gain not in self.actuation_gain_grid:
            raise ValueError("positive_actuation_gain must lie on its fit grid")
        if not np.isclose(abs(self.placebo_response_rotation_deg), 90.0):
            raise ValueError("placebo response must be rotated by 90 degrees")
        if not 0.0 < self.minimum_shrinkage_fraction < 1.0:
            raise ValueError("minimum_shrinkage_fraction must be in (0, 1)")
        if not 1 <= self.minimum_positive_sessions <= 12:
            raise ValueError("minimum_positive_sessions must be in [1, 12]")
        if not 0.0 < self.maximum_placebo_false_positive_rate < 1.0:
            raise ValueError("maximum placebo rate must be in (0, 1)")
        if not 0.0 < self.minimum_positive_control_power < 1.0:
            raise ValueError("minimum positive-control power must be in (0, 1)")
        if self.interval_z <= 0.0:
            raise ValueError("interval_z must be positive")

    @property
    def simulator(self) -> SimulatorConfig:
        return SimulatorConfig(frame_count=self.frame_count, dt=self.dt_s)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SyntheticPanel:
    truth_by_world: dict[str, np.ndarray]
    action_indices: np.ndarray
    replicate_indices: np.ndarray


def _minimum_jerk(size: int) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, size)
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5


def _envelope(frame_count: int, profile: str) -> np.ndarray:
    transition_count = frame_count - 1
    if profile == "high":
        rise, hold, fall = 3, 14, 3
    elif profile == "slow":
        rise, hold, fall = 12, 8, 12
    elif profile == "long_hold":
        rise, hold, fall = 3, 30, 3
    else:
        raise ValueError(f"unknown synthetic source profile: {profile}")
    if rise + hold + fall > transition_count:
        raise ValueError("profile does not fit the configured frame count")
    result = np.zeros(transition_count, dtype=float)
    cursor = 0
    result[cursor : cursor + rise] = _minimum_jerk(rise)
    cursor += rise
    result[cursor : cursor + hold] = 1.0
    cursor += hold
    result[cursor : cursor + fall] = _minimum_jerk(fall)[::-1]
    return result


def _source_actions(config: MechanismGateControlConfig) -> tuple[Action, ...]:
    definitions = (
        ("lift_high", "high", 1.0),
        ("lower_high", "high", -1.0),
        ("lift_high_slow", "slow", 1.0),
        ("lift_high_long_hold", "long_hold", 1.0),
    )
    actions = []
    for action_id, profile, direction in definitions:
        envelope = _envelope(config.frame_count, profile)
        force = np.zeros((config.frame_count - 1, 1, 2), dtype=float)
        force[:, 0, 1] = direction * config.force_magnitude_n * envelope
        actions.append(Action(action_id, "train", (4,), force))
    return tuple(actions)


def _graph_basis(config: MechanismGateControlConfig) -> tuple[np.ndarray, np.ndarray]:
    graph_object = make_objects()[1]
    positions = graph_object.rest_positions
    x = positions[:, 0] - np.mean(positions[:, 0])
    y = positions[:, 1] - np.mean(positions[:, 1])
    features = np.column_stack((np.ones(len(x)), x, y, x * y))
    basis = np.zeros((len(x), config.graph_rank), dtype=float)
    for mode in range(config.graph_rank):
        vector = features[:, mode].copy()
        for previous in range(mode):
            vector -= float(basis[:, previous] @ vector) * basis[:, previous]
        norm = float(np.sqrt(vector @ vector))
        if norm <= 1.0e-12:
            raise ValueError("deterministic graph feature basis is rank deficient")
        basis[:, mode] = vector / norm
    laplacian = graph_laplacian(graph_object)
    frequencies = np.asarray(
        [float(vector @ laplacian @ vector) for vector in basis.T],
        dtype=float,
    )
    return basis, frequencies


def _trajectory_banks(
    config: MechanismGateControlConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph_object = make_objects()[1]
    actions = _source_actions(config)
    baseline = np.stack(
        [
            simulate(
                graph_object,
                action,
                graph_object.true_parameters,
                WorldCondition("nominal"),
                config.simulator,
            )
            for action in actions
        ]
    )
    positive = np.stack(
        [
            np.stack(
                [
                    simulate(
                        graph_object,
                        action,
                        graph_object.true_parameters,
                        WorldCondition(
                            "actuation_gain",
                            contact_gain_multiplier=value,
                        ),
                        config.simulator,
                    )
                    for action in actions
                ]
            )
            for value in config.actuation_gain_grid
        ]
    )
    angle = np.deg2rad(config.placebo_response_rotation_deg)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    response = positive - baseline[None, :, :, :, :]
    placebo = baseline[None, :, :, :, :] + response @ rotation.T
    return baseline, positive, placebo


def _panel_noise(
    rng: np.random.Generator,
    config: MechanismGateControlConfig,
    basis: np.ndarray,
    node_count: int,
) -> np.ndarray:
    offset_coefficients = rng.normal(
        scale=config.persistent_offset_std_m,
        size=(config.graph_rank, 2),
    )
    persistent = basis @ offset_coefficients
    rho = config.temporal_noise_correlation
    temporal = np.empty((config.frame_count, node_count, 2), dtype=float)
    temporal[0] = rng.normal(
        scale=config.temporal_noise_std_m,
        size=(node_count, 2),
    )
    innovation_scale = config.temporal_noise_std_m * np.sqrt(1.0 - rho**2)
    for frame in range(1, config.frame_count):
        temporal[frame] = rho * temporal[frame - 1] + rng.normal(
            scale=innovation_scale,
            size=(node_count, 2),
        )
    return persistent[None, :, :] + temporal


def _make_panel(
    rng: np.random.Generator,
    config: MechanismGateControlConfig,
    baseline: np.ndarray,
    positive_bank: np.ndarray,
    basis: np.ndarray,
) -> _SyntheticPanel:
    action_indices = np.repeat(np.arange(4, dtype=int), 3)
    replicate_indices = np.tile(np.arange(3, dtype=int), 4)
    positive_index = config.actuation_gain_grid.index(config.positive_actuation_gain)
    null_truth = []
    positive_truth = []
    for action_index in action_indices:
        noise = _panel_noise(
            rng,
            config,
            basis,
            node_count=baseline.shape[2],
        )
        null_truth.append(baseline[action_index] + noise)
        positive_truth.append(positive_bank[positive_index, action_index] + noise)
    return _SyntheticPanel(
        truth_by_world={
            "null": np.stack(null_truth),
            "positive": np.stack(positive_truth),
        },
        action_indices=action_indices,
        replicate_indices=replicate_indices,
    )


def _fit_grid_value(
    truth: np.ndarray,
    action_indices: np.ndarray,
    candidate_bank: np.ndarray,
) -> int:
    candidate = candidate_bank[:, action_indices]
    loss = np.mean(
        np.square(candidate - truth[None, :, :, :, :]),
        axis=(1, 2, 3, 4),
    )
    return int(np.argmin(loss))


def _fit_readout_correction(
    truth: np.ndarray,
    model: np.ndarray,
    basis: np.ndarray,
    prefix_frame_count: int,
) -> np.ndarray:
    prefix = slice(1, 1 + prefix_frame_count)
    mean_residual = np.mean(truth[prefix] - model[prefix], axis=0)
    return basis @ (basis.T @ mean_residual)


def _weighted_correction_rms(
    correction: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
) -> float:
    coefficients = basis.T @ correction
    weighted_energy = np.sum((1.0 + eigenvalues[:, None]) * coefficients**2)
    return max(float(np.sqrt(weighted_energy / correction.size)), 1.0e-15)


def _track_error(truth: np.ndarray, prediction: np.ndarray) -> float:
    sensor_nodes = np.asarray((0, 4, 8), dtype=int)
    difference = truth[:, sensor_nodes] - prediction[:, sensor_nodes]
    return float(np.sqrt(np.mean(np.sum(np.square(difference), axis=-1))))


def _chamfer_error(truth: np.ndarray, prediction: np.ndarray) -> float:
    pairwise = np.linalg.norm(
        truth[:, :, None, :] - prediction[:, None, :, :],
        axis=-1,
    )
    symmetric = 0.5 * (
        np.mean(np.min(pairwise, axis=2), axis=1)
        + np.mean(np.min(pairwise, axis=1), axis=1)
    )
    return float(np.mean(symmetric))


def _repeatability_scales(
    truth: np.ndarray,
    action_indices: np.ndarray,
    future_start: int,
) -> tuple[float, float, float]:
    track_values = []
    late_values = []
    cd_values = []
    sensor_nodes = np.asarray((0, 4, 8), dtype=int)
    future_count = truth.shape[1] - future_start
    late_start = future_start + 2 * future_count // 3
    for action_index in range(4):
        repeated = truth[action_indices == action_index]
        centre = np.mean(repeated, axis=0)
        difference = repeated - centre[None, :, :, :]
        track_values.extend(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(difference[:, future_start:, sensor_nodes]),
                        axis=-1,
                    ),
                    axis=(1, 2),
                )
            )
        )
        late_values.extend(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(difference[:, late_start:, sensor_nodes]),
                        axis=-1,
                    ),
                    axis=(1, 2),
                )
            )
        )
        cd_values.extend(
            np.sqrt(
                np.mean(
                    np.square(difference[:, future_start:]),
                    axis=(1, 2, 3),
                )
            )
        )
    return tuple(
        max(float(np.sqrt(np.mean(np.square(values)))), 1.0e-15)
        for values in (track_values, late_values, cd_values)
    )


def _evaluate_arm(
    panel: _SyntheticPanel,
    world: Literal["null", "positive"],
    candidate_bank: np.ndarray,
    candidate_values: tuple[float, ...],
    baseline: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    config: MechanismGateControlConfig,
) -> dict[str, Any]:
    truth = panel.truth_by_world[world]
    future_start = 1 + config.prefix_frame_count
    future_count = config.frame_count - future_start
    late_start = future_start + 2 * future_count // 3
    baseline_correction = []
    mechanism_correction = []
    track_gain = []
    late_track_gain = []
    cd_degradation = []
    fitted_values = []
    for held_replicate in range(3):
        fit_mask = panel.replicate_indices != held_replicate
        held_mask = ~fit_mask
        fitted_index = _fit_grid_value(
            truth[fit_mask],
            panel.action_indices[fit_mask],
            candidate_bank,
        )
        fitted_values.append(float(candidate_values[fitted_index]))
        for session_index in np.flatnonzero(held_mask):
            action_index = int(panel.action_indices[session_index])
            session_truth = truth[session_index]
            baseline_model = baseline[action_index]
            mechanism_model = candidate_bank[fitted_index, action_index]
            base_correction = _fit_readout_correction(
                session_truth,
                baseline_model,
                basis,
                config.prefix_frame_count,
            )
            mechanism_c = _fit_readout_correction(
                session_truth,
                mechanism_model,
                basis,
                config.prefix_frame_count,
            )
            baseline_correction.append(
                _weighted_correction_rms(base_correction, basis, eigenvalues)
            )
            mechanism_correction.append(
                _weighted_correction_rms(mechanism_c, basis, eigenvalues)
            )
            base_prediction = baseline_model + base_correction[None, :, :]
            mechanism_prediction = mechanism_model + mechanism_c[None, :, :]
            base_future = base_prediction[future_start:]
            mechanism_future = mechanism_prediction[future_start:]
            truth_future = session_truth[future_start:]
            track_gain.append(
                _track_error(truth_future, base_future)
                - _track_error(truth_future, mechanism_future)
            )
            late_track_gain.append(
                _track_error(
                    session_truth[late_start:],
                    base_prediction[late_start:],
                )
                - _track_error(
                    session_truth[late_start:],
                    mechanism_prediction[late_start:],
                )
            )
            cd_degradation.append(
                _chamfer_error(truth_future, mechanism_future)
                - _chamfer_error(truth_future, base_future)
            )
    repeatability = _repeatability_scales(
        truth,
        panel.action_indices,
        future_start,
    )
    eligibility = heldout_mechanism_eligibility(
        baseline_correction,
        mechanism_correction,
        track_gain_m=track_gain,
        late_track_gain_m=late_track_gain,
        cd_degradation_m=cd_degradation,
        track_repeatability_sd_m=repeatability[0],
        late_track_repeatability_sd_m=repeatability[1],
        cd_repeatability_sd_m=repeatability[2],
        minimum_shrinkage_fraction=config.minimum_shrinkage_fraction,
        minimum_positive_sessions=config.minimum_positive_sessions,
    )
    eligibility["fitted_parameter_by_fold"] = fitted_values
    eligibility["repeatability_scales_m"] = {
        "track": repeatability[0],
        "late_track": repeatability[1],
        "cd_proxy": repeatability[2],
    }
    return eligibility


def _wilson_interval(successes: int, trials: int, z: float) -> list[float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial count")
    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (probability + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(probability * (1.0 - probability) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


def _rate_summary(
    records: list[dict[str, Any]],
    config: MechanismGateControlConfig,
) -> dict[str, Any]:
    full_passes = sum(
        record["eligible_for_confirmatory_evaluation"] for record in records
    )
    primary_passes = sum(
        record["gates"]["shrinkage"] and record["gates"]["session_direction"]
        for record in records
    )
    fitted = np.asarray(
        [value for record in records for value in record["fitted_parameter_by_fold"]],
        dtype=float,
    )
    shrinkage = np.asarray(
        [record["shrinkage_fraction"] for record in records], dtype=float
    )
    positive_sessions = np.asarray(
        [record["positive_shrinkage_session_count"] for record in records],
        dtype=float,
    )
    return {
        "simulation_count": len(records),
        "primary_shrinkage_gate_pass_count": int(primary_passes),
        "primary_shrinkage_gate_pass_rate": float(primary_passes / len(records)),
        "primary_shrinkage_gate_wilson_95": _wilson_interval(
            primary_passes, len(records), config.interval_z
        ),
        "full_eligibility_pass_count": int(full_passes),
        "full_eligibility_pass_rate": float(full_passes / len(records)),
        "full_eligibility_wilson_95": _wilson_interval(
            full_passes, len(records), config.interval_z
        ),
        "shrinkage_fraction_quantiles": {
            str(quantile): float(np.quantile(shrinkage, quantile))
            for quantile in (0.05, 0.50, 0.95)
        },
        "positive_session_count_quantiles": {
            str(quantile): float(np.quantile(positive_sessions, quantile))
            for quantile in (0.05, 0.50, 0.95)
        },
        "fitted_parameter_quantiles": {
            str(quantile): float(np.quantile(fitted, quantile))
            for quantile in (0.05, 0.50, 0.95)
        },
    }


def _threshold_sensitivity(
    records_by_arm: dict[ArmName, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    result = []
    for shrinkage_threshold in (0.05, 0.10, 0.15):
        for session_threshold in (7, 8, 9):
            row: dict[str, Any] = {
                "minimum_shrinkage_fraction": shrinkage_threshold,
                "minimum_positive_sessions": session_threshold,
            }
            for arm, records in records_by_arm.items():
                primary = [
                    record["shrinkage_fraction"] >= shrinkage_threshold
                    and record["positive_shrinkage_session_count"] >= session_threshold
                    for record in records
                ]
                full = [
                    primary[index]
                    and record["gates"]["track"]
                    and record["gates"]["late_track"]
                    and record["gates"]["cd_non_degradation"]
                    for index, record in enumerate(records)
                ]
                row[f"{arm}_primary_rate"] = float(np.mean(primary))
                row[f"{arm}_full_rate"] = float(np.mean(full))
            result.append(row)
    return result


def mechanism_gate_control_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    serialized = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _canonicalize_report_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [_canonicalize_report_floats(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize_report_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonicalize_report_floats(item) for key, item in value.items()}
    return value


def run_mechanism_gate_controls(
    config: MechanismGateControlConfig | None = None,
) -> dict[str, Any]:
    """Estimate false-positive and detection rates for the frozen v3 gate."""

    cfg = config or MechanismGateControlConfig()
    baseline, positive_bank, placebo_bank = _trajectory_banks(cfg)
    basis, eigenvalues = _graph_basis(cfg)
    rng = np.random.default_rng(cfg.random_seed)
    records_by_arm: dict[ArmName, list[dict[str, Any]]] = {
        "placebo_null": [],
        "positive_control": [],
        "placebo_on_positive": [],
    }
    for _ in range(cfg.simulation_count):
        panel = _make_panel(rng, cfg, baseline, positive_bank, basis)
        records_by_arm["placebo_null"].append(
            _evaluate_arm(
                panel,
                "null",
                placebo_bank,
                cfg.actuation_gain_grid,
                baseline,
                basis,
                eigenvalues,
                cfg,
            )
        )
        records_by_arm["positive_control"].append(
            _evaluate_arm(
                panel,
                "positive",
                positive_bank,
                cfg.actuation_gain_grid,
                baseline,
                basis,
                eigenvalues,
                cfg,
            )
        )
        records_by_arm["placebo_on_positive"].append(
            _evaluate_arm(
                panel,
                "positive",
                placebo_bank,
                cfg.actuation_gain_grid,
                baseline,
                basis,
                eigenvalues,
                cfg,
            )
        )
    summaries = {
        arm: _rate_summary(records, cfg) for arm, records in records_by_arm.items()
    }
    placebo_upper = summaries["placebo_null"]["full_eligibility_wilson_95"][1]
    power_lower = summaries["positive_control"]["full_eligibility_wilson_95"][0]
    specificity_upper = summaries["placebo_on_positive"]["full_eligibility_wilson_95"][
        1
    ]
    checks = {
        "placebo_null_full_gate_upper_below_5_percent": bool(
            placebo_upper <= cfg.maximum_placebo_false_positive_rate
        ),
        "positive_control_full_gate_lower_above_80_percent": bool(
            power_lower >= cfg.minimum_positive_control_power
        ),
        "wrong_family_on_positive_upper_below_5_percent": bool(
            specificity_upper <= cfg.maximum_placebo_false_positive_rate
        ),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "MechanismGateControlEvidence",
        "claim_boundary": (
            "Controlled operating-point evidence only; no released or prospective "
            "physical outcome is used, and no real-world false-positive guarantee is claimed."
        ),
        "design": {
            "source_sessions": 12,
            "crossfit": "three folds with 8 fit and 4 held out",
            "prefix_frames": cfg.prefix_frame_count,
            "positive_control": (
                "known 10 percent actuation-gain loss, fitted on a scalar grid"
            ),
            "placebo": (
                "the exact scalar actuation-response bank rotated 90 degrees, "
                "preserving candidate norms and the identical nine-value grid"
            ),
            "common_noise_draws_across_arms": True,
            "heldout_readout_correction_refit": True,
            "future_used_for_heldout_correction_fit": False,
        },
        "config": cfg.as_dict(),
        "arms": summaries,
        "threshold_sensitivity": _threshold_sensitivity(records_by_arm),
        "acceptance_checks": checks,
        "frozen_v3_gate_supported_in_controlled_benchmark": bool(all(checks.values())),
    }
    result = _canonicalize_report_floats(result)
    result["result_sha256"] = mechanism_gate_control_sha256(result)
    return result
