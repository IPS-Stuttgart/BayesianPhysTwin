"""Forecast-oriented, fixed-budget queries for an opened DEFORM development study.

The planning covariance is a local model, not a calibrated predictive posterior.
Schedules depend on native simulated responses, never on candidate observations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform_multiobject_restart import load_protocol as load_parent_protocol
from .deform_state_restart import (
    RestartConfig,
    file_digest,
    interpolate_material_residual,
    sparse_state_increments,
)


@dataclass(frozen=True)
class SensingConfig:
    anchor_frame: int = 25
    query_frames: tuple[int, ...] = (25, 33, 41, 49)
    position_std_m: float = 0.01
    velocity_std_m_s: float = 0.1
    shared_bias_std_m: float = 0.005
    measurement_std_m: float = 0.001
    finite_difference_fraction: float = 0.1
    maximum_position_increment_m: float = 0.03
    maximum_velocity_increment_m_s: float = 0.3
    budgets: tuple[int, ...] = (4, 8, 12, 16)
    random_repetitions: int = 4
    random_seed: int = 260831
    temporal_time_constants_s: tuple[float, ...] = (0.1, 0.3, 1.0)

    def __post_init__(self) -> None:
        positive = (
            self.position_std_m,
            self.velocity_std_m_s,
            self.shared_bias_std_m,
            self.measurement_std_m,
            self.finite_difference_fraction,
            self.maximum_position_increment_m,
            self.maximum_velocity_increment_m_s,
            *self.temporal_time_constants_s,
        )
        if any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("scales must be positive and finite")
        if (
            self.query_frames != tuple(sorted(set(self.query_frames)))
            or self.query_frames[0] != self.anchor_frame
            or self.query_frames[-1] != 49
            or self.anchor_frame < 0
        ):
            raise ValueError("query times must be unique and within the fixed prefix")
        if self.budgets != (4, 8, 12, 16) or self.random_repetitions < 1:
            raise ValueError("registered observation budgets differ")


def load_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(path.read_text())
    required = {
        "schema": "deform-forecast-aware-sensing-v1",
        "scope": "exploratory-only-on-the-already-open-three-object-roster",
        "primary_arm": "forecast_8",
        "comparison_arm": "uniform_8",
        "all_predictions_sealed_before_new_metrics": True,
        "protected_data_access": False,
        "new_official_evaluation": False,
        "checkpoint_or_readout_refitting": False,
        "automatic_promotion": False,
        "future_free_node_truth_is_model_input": False,
        "measurement_values_used_in_schedule_selection": False,
        "predictive_covariance_calibration_claim": False,
    }
    if any(value.get(k) != v for k, v in required.items()):
        raise ValueError("sensing protocol scope or primary method changed")
    parent_spec = value["parent_protocol"]
    parent_path = root / parent_spec["path"]
    if file_digest(parent_path) != parent_spec["sha256"]:
        raise ValueError("parent cohort, checkpoints, or incumbent contract changed")
    parent = load_parent_protocol(parent_path)
    if value["sensing"] != config_record(SensingConfig()):
        raise ValueError("frozen sensing configuration changed")
    if value["noise"] != {
        "conditions": ["independent_1mm", "independent_1mm_shared_5mm"],
        "repetitions": 8,
        "seed": 260832,
        "independent_std_m": 0.001,
        "shared_std_m": 0.005,
    }:
        raise ValueError("noise conditions changed")
    return value, parent


def config_record(config: SensingConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(config)))


def native_arm_names(sensing: SensingConfig) -> tuple[str, ...]:
    return tuple(
        [
            f"{policy}_{budget}"
            for budget in sensing.budgets
            for policy in ("uniform", "forecast")
        ]
        + ["current_8"]
        + [f"random_8_seed{i}" for i in range(sensing.random_repetitions)]
    )


def temporal_arm_names(sensing: SensingConfig) -> tuple[str, ...]:
    return tuple(
        ["incumbent", "temporal_static", "temporal_linear"]
        + [
            f"temporal_{kind}_{int(round(tau * 1000))}ms"
            for tau in sensing.temporal_time_constants_s
            for kind in ("damped_velocity", "decay")
        ]
    )


def clean_arm_names(sensing: SensingConfig) -> tuple[str, ...]:
    return (
        temporal_arm_names(sensing) + ("previous_paired_8",) + native_arm_names(sensing)
    )


def noise_arm_names(sensing: SensingConfig) -> tuple[str, ...]:
    return temporal_arm_names(sensing) + (
        "previous_paired_8",
        "uniform_8",
        "forecast_8",
    )


def query_pairs(
    rod: RestartConfig, sensing: SensingConfig
) -> tuple[tuple[int, int], ...]:
    if rod.prefix_length != 50 or sensing.query_frames[-1] >= rod.prefix_length:
        raise ValueError("only causal prefix queries are admitted")
    return tuple((t, n) for t in sensing.query_frames for n in rod.observed_nodes)


def material_basis(rod: RestartConfig) -> np.ndarray:
    rank = len(rod.observed_nodes) * 3
    values = np.eye(rank).reshape(rank, len(rod.observed_nodes), 3)
    return interpolate_material_residual(values, rod)


def planning_matrices(
    response: np.ndarray, rod: RestartConfig, sensing: SensingConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return query Jacobians and future/current shape objectives in metres^2."""
    rank = len(rod.observed_nodes) * 6
    expected = (rod.forecast_end - sensing.anchor_frame, rod.node_count, 3, rank)
    value = np.asarray(response, dtype=np.float64)
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError(
            "response time, identity, physical rank, or finiteness differs"
        )
    pairs = query_pairs(rod, sensing)
    physical = np.stack([value[t - sensing.anchor_frame, n] for t, n in pairs])
    nuisance = np.broadcast_to(
        sensing.shared_bias_std_m * np.eye(3), (len(pairs), 3, 3)
    )
    design = np.concatenate((physical, nuisance), axis=-1)

    def objective(points: np.ndarray) -> np.ndarray:
        flat = points.reshape(-1, rank)
        result = np.zeros((rank + 3, rank + 3))
        # Average squared Euclidean error per point, not per coordinate.
        result[:rank, :rank] = flat.T @ flat / (len(flat) / 3)
        return result

    future = value[rod.prefix_length - sensing.anchor_frame :][:, rod.hidden_nodes]
    free = [i for i in range(rod.node_count) if i not in rod.clamped_nodes]
    current = value[rod.prefix_length - 1 - sensing.anchor_frame, free]
    return design, objective(future), objective(current)


def covariance_update(
    covariance: np.ndarray, observation: np.ndarray, variance: float
) -> tuple[np.ndarray, np.ndarray]:
    p, h = np.asarray(covariance), np.asarray(observation)
    if (
        p.ndim != 2
        or p.shape[0] != p.shape[1]
        or h.shape != (3, p.shape[0])
        or not np.isfinite(p).all()
        or not np.isfinite(h).all()
        or not np.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError("invalid linear Gaussian update")
    innovation = h @ p @ h.T + variance * np.eye(3)
    gain = np.linalg.solve(innovation, h @ p).T
    transition = np.eye(len(p)) - gain @ h
    # Joseph form preserves PSD when observations are much sharper than the prior.
    posterior = transition @ p @ transition.T + variance * gain @ gain.T
    return 0.5 * (posterior + posterior.T), gain


def validate_schedule(
    schedule: Sequence[int], count: int, budget: int
) -> tuple[int, ...]:
    indices = tuple(schedule)
    if (
        len(indices) != budget
        or len(set(indices)) != len(indices)
        or any(type(i) not in (int, np.int64, np.int32) for i in indices)
        or any(i < 0 or i >= count for i in indices)
    ):
        raise ValueError("query budget, uniqueness, or admissibility differs")
    return tuple(sorted(int(i) for i in indices))


def greedy_schedule(
    design: np.ndarray, objective: np.ndarray, budget: int, measurement_std_m: float
) -> tuple[int, ...]:
    """Preplan using expected forecast-variance reduction, without observations."""
    h, weight = np.asarray(design), np.asarray(objective)
    if (
        h.ndim != 3
        or h.shape[1] != 3
        or weight.shape != (h.shape[-1], h.shape[-1])
        or not 0 <= budget <= len(h)
        or not np.isfinite(h).all()
        or not np.isfinite(weight).all()
        or not np.allclose(weight, weight.T, atol=1e-12)
        or np.linalg.eigvalsh(weight).min() < -1e-10
    ):
        raise ValueError("invalid target-free planning problem")
    covariance = np.eye(h.shape[-1])
    remaining = list(range(len(h)))
    selected: list[int] = []
    for _ in range(budget):
        proposals = [
            covariance_update(covariance, h[i], measurement_std_m**2)[0]
            for i in remaining
        ]
        costs = [float(np.sum(weight * p.T)) for p in proposals]
        # Remaining indices are ordered, so exact ties use the first query.
        best = int(np.argmin(costs))
        selected.append(remaining.pop(best))
        covariance = proposals[best]
    return validate_schedule(selected, len(h), budget)


def schedules_for_case(
    response: np.ndarray, rod: RestartConfig, sensing: SensingConfig, *, seed: int
) -> tuple[dict[str, tuple[int, ...]], np.ndarray, np.ndarray]:
    design, future, current = planning_matrices(response, rod, sensing)
    result: dict[str, tuple[int, ...]] = {}
    for budget in sensing.budgets:
        result[f"uniform_{budget}"] = tuple(range(len(design) - budget, len(design)))
        result[f"forecast_{budget}"] = greedy_schedule(
            design, future, budget, sensing.measurement_std_m
        )
    result["current_8"] = greedy_schedule(design, current, 8, sensing.measurement_std_m)
    rng = np.random.default_rng(seed)
    for repeat in range(sensing.random_repetitions):
        result[f"random_8_seed{repeat}"] = validate_schedule(
            rng.choice(len(design), size=8, replace=False).tolist(), len(design), 8
        )
    return result, design, future


class LockedQueryBank:
    """Reveal only a precommitted schedule, in chronological point/time order."""

    def __init__(
        self,
        points: np.ndarray,
        pairs: Sequence[tuple[int, int]],
        schedule: Sequence[int],
    ):
        value = np.asarray(points, dtype=np.float64)
        if value.shape != (len(pairs), 3) or not np.isfinite(value).all():
            raise ValueError("candidate bank must contain only finite prefix points")
        if tuple(pairs) != tuple(sorted(set(pairs))):
            raise ValueError("query identities must be ordered and unique")
        self.__schedule = validate_schedule(schedule, len(pairs), len(schedule))
        self.__values = value[list(self.__schedule)].copy()
        self.__pairs = tuple(pairs[i] for i in self.__schedule)
        self.__cursor = 0
        self.access_log: list[tuple[int, int]] = []

    def reveal(self, frame: int, node: int) -> np.ndarray:
        pair = (frame, node)
        if self.__cursor >= len(self.__pairs) or self.__pairs[self.__cursor] != pair:
            raise ValueError("unplanned, duplicated, hidden, or out-of-order query")
        value = self.__values[self.__cursor].copy()
        self.__cursor += 1
        self.access_log.append(pair)
        return value


def infer_coefficients(
    design: np.ndarray,
    reference: np.ndarray,
    bank: LockedQueryBank,
    pairs: Sequence[tuple[int, int]],
    schedule: Sequence[int],
    measurement_std_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(design, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if (
        h.ndim != 3
        or h.shape[:2] != (len(pairs), 3)
        or ref.shape != (len(pairs), 3)
        or not np.isfinite(h).all()
        or not np.isfinite(ref).all()
    ):
        raise ValueError("query design and incumbent prefix must align")
    selected = validate_schedule(schedule, len(pairs), len(schedule))
    mean, covariance = np.zeros(h.shape[-1]), np.eye(h.shape[-1])
    for i in selected:
        value = bank.reveal(*pairs[i])
        covariance, gain = covariance_update(covariance, h[i], measurement_std_m**2)
        mean += gain @ (value - ref[i] - h[i] @ mean)
    return mean, covariance


def bounded_increments(
    coefficients: np.ndarray, rod: RestartConfig, sensing: SensingConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = material_basis(rod)
    rank = len(basis)
    value = np.asarray(coefficients, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[1] != rank * 2 + 3
        or not np.isfinite(value).all()
    ):
        raise ValueError("physical and shared-bias coefficient contract differs")
    pose = sensing.position_std_m * np.einsum("bk,knc->bnc", value[:, :rank], basis)
    velocity = sensing.velocity_std_m_s * np.einsum(
        "bk,knc->bnc", value[:, rank : 2 * rank], basis
    )
    pose_scale = np.linalg.norm(pose, axis=-1).max(axis=-1)
    velocity_scale = np.linalg.norm(velocity, axis=-1).max(axis=-1)
    denominator = np.maximum(
        1.0,
        np.maximum(
            pose_scale / sensing.maximum_position_increment_m,
            velocity_scale / sensing.maximum_velocity_increment_m_s,
        ),
    )
    gain = 1.0 / denominator
    return pose * gain[:, None, None], velocity * gain[:, None, None], gain


def temporal_controls(
    incumbent: np.ndarray,
    observed: np.ndarray,
    rod: RestartConfig,
    sensing: SensingConfig,
) -> dict[str, np.ndarray]:
    if incumbent.ndim != 4 or incumbent.shape[1:] != (
        rod.forecast_end,
        rod.node_count,
        3,
    ):
        raise ValueError("full frozen incumbent trajectory required")
    pose, velocity = sparse_state_increments(
        incumbent[:, : rod.prefix_length], observed, rod
    )
    base = incumbent[:, rod.prefix_length :]
    time = np.arange(1, base.shape[1] + 1)[None, :, None, None] * rod.dt_s
    result = {
        "incumbent": base,
        "temporal_static": base + pose[:, None],
        "temporal_linear": base + pose[:, None] + time * velocity[:, None],
    }
    for tau in sensing.temporal_time_constants_s:
        label = f"{int(round(tau * 1000))}ms"
        decay = np.exp(-time / tau)
        result[f"temporal_damped_velocity_{label}"] = (
            base + pose[:, None] + tau * (1 - decay) * velocity[:, None]
        )
        result[f"temporal_decay_{label}"] = base + decay * (
            pose[:, None] + time * velocity[:, None]
        )
    return result


def query_noise(shape: tuple[int, ...], *, seed: int, shared: bool) -> np.ndarray:
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError("noise requires a case/query/coordinate bank")
    rng = np.random.default_rng(seed)
    independent = rng.normal(0, 0.001, shape)
    bias = rng.normal(0, 0.005, (shape[0], 1, 3))
    return independent + bias if shared else independent


def primary_decision(results: Mapping[str, Any]) -> dict[str, Any]:
    if set(results) != {"DLO1", "DLO2", "DLO3"}:
        raise ValueError("all opened objects must remain in the result")
    checks: dict[str, Any] = {}
    for name in ("DLO1", "DLO3"):
        summary = results[name]["clean"]["summaries"]
        candidate, base, uniform = [
            summary[k] for k in ("forecast_8", "incumbent", "uniform_8")
        ]
        metrics = ("coordinate_l1_mm", "point_rmse_mm")
        temporal = [v for k, v in summary.items() if k.startswith("temporal_")]
        checks[name] = {
            "both_metrics_improve_over_incumbent": all(
                candidate[k] < base[k] for k in metrics
            ),
            "both_metrics_improve_over_uniform": all(
                candidate[k] < uniform[k] for k in metrics
            ),
            "at_least_2percent_rmse_gain_over_uniform": candidate[metrics[1]]
            <= 0.98 * uniform[metrics[1]],
            "beats_all_frozen_temporal_controls_on_both": all(
                candidate[k] < control[k] for control in temporal for k in metrics
            ),
            "late_rmse_nonincreasing_over_incumbent": candidate["late"][metrics[1]]
            <= base["late"][metrics[1]],
            "at_least_5_of_8_joint_wins": candidate["joint_wins"] >= 5,
        }
    passed = all(all(v.values()) for v in checks.values())
    return {
        "primary_arm": "forecast_8",
        "checks": checks,
        "development_advancement_gate_passed": passed,
        "independent_evaluation_recommended": passed,
        "automatic_target_authorization": False,
        "incumbent_promoted": False,
        "exploratory_only": True,
    }
