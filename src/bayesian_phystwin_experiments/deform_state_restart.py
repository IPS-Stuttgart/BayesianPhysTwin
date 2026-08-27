"""Sparse prefix state updates for an isolated, already-open DEFORM experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RestartConfig:
    schema: str = "deform-sparse-state-restart-dev-v1"
    prefix_length: int = 50
    forecast_end: int = 170
    observation_frames: tuple[int, int] = (41, 49)
    observed_nodes: tuple[int, ...] = (2, 4, 6, 8)
    hidden_nodes: tuple[int, ...] = (3, 5, 7, 9)
    clamped_nodes: tuple[int, ...] = (0, 1, 10, 11)
    node_count: int = 12
    dt_s: float = 0.01
    design_case: str = "103.pkl"
    bootstrap_replicates: int = 10000
    seed: int = 260829

    def __post_init__(self) -> None:
        if self.schema != "deform-sparse-state-restart-dev-v1":
            raise ValueError("unrecognized state-restart schema")
        if not 0 <= self.observation_frames[0] < self.observation_frames[1]:
            raise ValueError("observations must be ordered causal frames")
        if self.observation_frames[1] != self.prefix_length - 1:
            raise ValueError("last measurement must be at the prefix endpoint")
        if self.forecast_end <= self.prefix_length or self.dt_s <= 0:
            raise ValueError("invalid time contract")
        if not np.isfinite(self.dt_s):
            raise ValueError("dt must be finite")
        groups = (self.observed_nodes, self.hidden_nodes, self.clamped_nodes)
        for group in groups:
            if not group or len(set(group)) != len(group):
                raise ValueError("node groups must be nonempty and unique")
            if any(n < 0 or n >= self.node_count for n in group):
                raise ValueError("node index outside geometry")
        if any(
            set(a) & set(b)
            for a, b in (
                (groups[0], groups[1]),
                (groups[0], groups[2]),
                (groups[1], groups[2]),
            )
        ):
            raise ValueError("observed, hidden, and clamped nodes must be disjoint")
        if tuple(sorted(self.observed_nodes)) != self.observed_nodes:
            raise ValueError("observed nodes must be sorted")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap replicate count must be positive")


def array_digest(array: np.ndarray) -> str:
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def interpolate_material_residual(
    residual: np.ndarray,
    config: RestartConfig,
) -> np.ndarray:
    """Piecewise-linear identity interpolation with exact zero clamped updates."""
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(config.observed_nodes), 3):
        raise ValueError("residual must have shape (batch, observed nodes, 3)")
    if not np.isfinite(values).all():
        raise ValueError("observed residual must be finite")
    knots = sorted((*config.observed_nodes, *config.clamped_nodes))
    knot_values = np.zeros((len(values), len(knots), 3), dtype=np.float64)
    for index, node in enumerate(config.observed_nodes):
        knot_values[:, knots.index(node)] = values[:, index]
    result = np.empty((len(values), config.node_count, 3), dtype=np.float64)
    for batch in range(len(values)):
        for axis in range(3):
            result[batch, :, axis] = np.interp(
                np.arange(config.node_count),
                knots,
                knot_values[batch, :, axis],
            )
    result[:, config.clamped_nodes] = 0.0
    return result


def sparse_state_increments(
    reference_prefix: np.ndarray,
    observed_positions: np.ndarray,
    config: RestartConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer pose and residual-velocity increments from eight prefix identities.

    Velocity is the slope of the position residual over the fixed two-frame
    measurement interval, added to (not substituted for) the simulator velocity.
    """
    reference = np.asarray(reference_prefix, dtype=np.float64)
    observations = np.asarray(observed_positions, dtype=np.float64)
    if reference.ndim != 4 or reference.shape[1:] != (
        config.prefix_length,
        config.node_count,
        3,
    ):
        raise ValueError("only the exact permitted reference prefix is accepted")
    expected = (len(reference), 2, len(config.observed_nodes), 3)
    if observations.shape != expected or not np.isfinite(observations).all():
        raise ValueError("invalid sparse prefix observations")
    if not np.isfinite(reference).all():
        raise ValueError("reference prefix must be finite")
    selected = reference[:, config.observation_frames][:, :, config.observed_nodes]
    residual = observations - selected
    pose = interpolate_material_residual(residual[:, -1], config)
    duration = (
        config.observation_frames[1] - config.observation_frames[0]
    ) * config.dt_s
    velocity = interpolate_material_residual(
        (residual[:, 1] - residual[:, 0]) / duration, config
    )
    return pose, velocity


def paired_physical_readout(
    incumbent: np.ndarray,
    nominal_physical: np.ndarray,
    updated_physical: np.ndarray,
) -> np.ndarray:
    """Keep the frozen readout and add only the paired dynamical response."""
    if (
        incumbent.shape != nominal_physical.shape
        or incumbent.shape != updated_physical.shape
    ):
        raise ValueError("paired rollout shapes differ")
    if not all(
        np.isfinite(x).all() for x in (incumbent, nominal_physical, updated_physical)
    ):
        raise ValueError("paired rollouts must be finite")
    if array_digest(nominal_physical) == array_digest(updated_physical):
        return incumbent
    return incumbent + (updated_physical.astype(np.float64) - nominal_physical)


@dataclass(frozen=True)
class RodState:
    positions: Any
    velocity: Any
    previous_positions: Any
    material_u0: Any
    theta: Any
    prediction_index: int

    def clone(self) -> RodState:
        return RodState(
            self.positions.detach().clone(),
            self.velocity.detach().clone(),
            self.previous_positions.detach().clone(),
            self.material_u0.detach().clone(),
            self.theta.detach().clone(),
            self.prediction_index,
        )


def update_rod_state(
    state: RodState,
    pose_increment: Any,
    velocity_increment: Any,
    *,
    gain: float,
    clamped_nodes: tuple[int, ...],
) -> RodState:
    if not np.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError("gain must be finite and in [0, 1]")
    if (
        pose_increment.shape != state.positions.shape
        or velocity_increment.shape != state.velocity.shape
    ):
        raise ValueError("state increment shape differs")
    if not pose_increment.isfinite().all() or not velocity_increment.isfinite().all():
        raise ValueError("state increments must be finite")
    if (
        pose_increment[:, clamped_nodes].count_nonzero()
        or velocity_increment[:, clamped_nodes].count_nonzero()
    ):
        raise ValueError("state updates cannot alter registered actuator nodes")
    if gain == 0 or (
        not pose_increment.count_nonzero() and not velocity_increment.count_nonzero()
    ):
        return state.clone()
    return RodState(
        state.positions + gain * pose_increment,
        state.velocity + gain * velocity_increment,
        state.previous_positions.clone(),
        state.material_u0.clone(),
        state.theta.clone(),
        state.prediction_index,
    )


def prediction_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if (
        prediction.shape != truth.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 3
    ):
        raise ValueError("metrics require aligned (time, identity, 3) arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("cannot silently drop nonfinite predictions")
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {
        "coordinate_l1_mm": float(1000 * np.abs(error).mean()),
        "point_rmse_mm": float(1000 * np.sqrt(np.sum(error * error, axis=-1).mean())),
        "fde_mm": float(1000 * np.linalg.norm(error[-1], axis=-1).mean()),
    }


def aggregate_paired_metrics(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    names: list[str],
    config: RestartConfig,
) -> dict[str, Any]:
    if "incumbent" not in predictions or len(names) != len(truth):
        raise ValueError("missing baseline or mismatched case count")
    expected = (
        len(names),
        config.forecast_end - config.prefix_length,
        config.node_count,
        3,
    )
    if truth.shape != expected or any(
        x.shape != expected for x in predictions.values()
    ):
        raise ValueError("forecast shape differs from protocol")
    keep = [i for i, name in enumerate(names) if name != config.design_case]
    if len(keep) < 2:
        raise ValueError(
            "paired aggregation needs at least two non-design trajectories"
        )
    rng = np.random.default_rng(config.seed)
    draws = rng.integers(0, len(keep), size=(config.bootstrap_replicates, len(keep)))
    per_case: dict[str, list[dict[str, Any]]] = {}
    for arm, points in predictions.items():
        rows = []
        for index, name in enumerate(names):
            pred = points[index][:, config.hidden_nodes]
            target = truth[index][:, config.hidden_nodes]
            row: dict[str, Any] = {"case": name, **prediction_metrics(pred, target)}
            for label, frames in zip(
                ("early", "middle", "late"),
                np.array_split(np.arange(len(pred)), 3),
                strict=True,
            ):
                row[label] = prediction_metrics(pred[frames], target[frames])
            rows.append(row)
        per_case[arm] = rows
    summaries: dict[str, Any] = {}
    baseline = per_case["incumbent"]
    for arm, rows in per_case.items():
        summary: dict[str, Any] = {"case_count": len(keep)}
        for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm"):
            values = np.array([rows[i][metric] for i in keep])
            reference = np.array([baseline[i][metric] for i in keep])
            delta = values - reference
            summary[metric] = float(values.mean())
            summary[metric + "_change_percent"] = float(
                100 * (values.mean() / reference.mean() - 1)
            )
            summary[metric + "_delta_ci95"] = np.quantile(
                delta[draws].mean(axis=1), [0.025, 0.975]
            ).tolist()
            summary[metric + "_wins"] = int(np.sum(delta < -1e-10))
        summary["joint_wins"] = sum(
            rows[i]["coordinate_l1_mm"] < baseline[i]["coordinate_l1_mm"]
            and rows[i]["point_rmse_mm"] < baseline[i]["point_rmse_mm"]
            for i in keep
        )
        for label in ("early", "middle", "late"):
            summary[label] = {
                metric: float(np.mean([rows[i][label][metric] for i in keep]))
                for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm")
            }
        summaries[arm] = summary
    return {
        "scope": "exploratory-already-open-trajectories-one-physical-object",
        "design_case_excluded_from_aggregate": config.design_case,
        "per_case": per_case,
        "summaries": summaries,
        "bootstrap_unit": "whole-trajectory-not-coordinate-or-frame",
        "uncertainty_or_fresh_confirmation_claim": False,
    }
