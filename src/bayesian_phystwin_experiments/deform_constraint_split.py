"""Constraint-separated readout screen on frozen, already-open DEFORM rollouts.

The projector constrains an incremental displacement, not the nonlinear rod or
the observation mechanism. No native state or upstream checkpoint is changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .deform_state_restart import prediction_metrics

SCHEMA = "deform-constraint-split-source-v1"
ARMS = (
    "incumbent",
    "paired",
    "readout",
    "half_blend",
    "tangent_only",
    "constraint_split",
)
PRIMARY = "constraint_split"
CONTROLS = ("incumbent", "paired", "readout", "half_blend")
HORIZONS = {"all": (0, 120), "early": (0, 40), "middle": (40, 80), "late": (80, 120)}


@dataclass(frozen=True)
class SplitConfig:
    node_count: int = 12
    clamped_nodes: tuple[int, ...] = (0, 1, 10, 11)
    hidden_nodes: tuple[int, ...] = (3, 5, 7, 9)
    minimum_edge_length_m: float = 1e-8
    relative_svd_tolerance: float = 1e-10
    linear_constraint_tolerance_m: float = 1e-9
    constant_readout_tolerance_m: float = 1e-12
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 260913

    def __post_init__(self) -> None:
        if self.node_count < 3 or self.bootstrap_replicates < 1:
            raise ValueError("nonempty geometry and bootstrap required")
        for group in (self.clamped_nodes, self.hidden_nodes):
            if not group or len(group) != len(set(group)):
                raise ValueError("unique nonempty identity groups required")
            if any(i < 0 or i >= self.node_count for i in group):
                raise ValueError("identity outside the registered geometry")
        if set(self.clamped_nodes) & set(self.hidden_nodes):
            raise ValueError("hidden identities cannot be clamps")
        if len(self.clamped_nodes) == self.node_count:
            raise ValueError("the rod must have free identities")
        scales = (
            self.minimum_edge_length_m,
            self.relative_svd_tolerance,
            self.linear_constraint_tolerance_m,
            self.constant_readout_tolerance_m,
        )
        if any(not np.isfinite(x) or x <= 0 for x in scales):
            raise ValueError("positive finite tolerances required")


def config_record(config: SplitConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config)))


def content_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class ProjectionUnavailable(ValueError):
    """A geometry-only technical failure that retains the paired comparator."""


def constraint_rows(
    geometry: np.ndarray, config: SplitConfig
) -> tuple[np.ndarray, tuple[int, ...]]:
    points = np.asarray(geometry, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (config.node_count, 3):
        raise ValueError("geometry must have shape (time, identity, xyz)")
    if not len(points) or not np.isfinite(points).all():
        raise ProjectionUnavailable("nonfinite_or_empty_nominal_geometry")
    edges = np.diff(points, axis=1)
    length = np.linalg.norm(edges, axis=-1)
    if np.any(length < config.minimum_edge_length_m):
        raise ProjectionUnavailable("degenerate_nominal_edge")
    direction = edges / length[..., None]
    free = tuple(i for i in range(config.node_count) if i not in config.clamped_nodes)
    rows = np.zeros((len(points), config.node_count - 1, 3 * len(free)))
    for column, identity in enumerate(free):
        if identity:
            rows[:, identity - 1, 3 * column : 3 * column + 3] = direction[
                :, identity - 1
            ]
        if identity < config.node_count - 1:
            rows[:, identity, 3 * column : 3 * column + 3] = -direction[:, identity]
    return rows, free


def tangent_project(
    displacement: np.ndarray, geometry: np.ndarray, config: SplitConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    delta = np.asarray(displacement, dtype=np.float64)
    if delta.shape != geometry.shape or not np.isfinite(delta).all():
        raise ValueError("finite aligned displacement required")
    if np.any(delta[:, config.clamped_nodes]):
        raise ValueError("a correction cannot move prescribed clamps")
    rows, free = constraint_rows(geometry, config)
    try:
        _, singular, directions = np.linalg.svd(rows, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ProjectionUnavailable("nominal_constraint_svd_failed") from exc
    active = singular > singular[:, :1] * config.relative_svd_tolerance
    flattened = delta[:, free].reshape(len(delta), -1)
    normal_coefficients = np.einsum("tij,tj->ti", directions, flattened) * active
    normal = np.einsum("tij,ti->tj", directions, normal_coefficients)
    projected = flattened - normal
    residual = np.einsum("tij,tj->ti", rows, projected)
    maximum = float(np.max(np.abs(residual)))
    if (
        not np.isfinite(projected).all()
        or maximum > config.linear_constraint_tolerance_m
    ):
        raise ProjectionUnavailable("linear_constraint_certificate_failed")
    result = np.zeros_like(delta)
    result[:, free] = projected.reshape(len(delta), len(free), 3)
    energy = float(np.sum(flattened**2))
    return result, {
        "constraint_rank_min": int(np.min(active.sum(axis=-1))),
        "constraint_rank_max": int(np.max(active.sum(axis=-1))),
        "maximum_linear_constraint_residual_m": maximum,
        "input_squared_norm_m2": energy,
        "tangent_squared_norm_m2": float(np.sum(projected**2)),
        "normal_squared_norm_m2": float(np.sum(normal**2)),
    }


def split_forecast(
    incumbent: np.ndarray,
    paired: np.ndarray,
    readout: np.ndarray,
    nominal: np.ndarray,
    config: SplitConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shape = (120, config.node_count, 3)
    for value in (incumbent, paired, readout):
        if value.shape != shape or value.dtype != np.dtype("float64"):
            raise ValueError(
                "frozen means require float64, 120 frames, and exact identities"
            )
        if not np.isfinite(value).all():
            raise ValueError("the frozen mean is not a valid fallback")
    if nominal.shape != shape:
        raise ValueError("nominal time or identity alignment differs")
    dynamic = paired - incumbent
    readout_series = readout - incumbent
    offset = np.broadcast_to(readout_series[:1], shape)
    if not np.allclose(
        readout_series, offset, atol=config.constant_readout_tolerance_m, rtol=0
    ):
        raise ValueError(
            "readout comparator is not the frozen persistent prefix offset"
        )
    if np.any(dynamic[:, config.clamped_nodes]) or np.any(
        offset[:, config.clamped_nodes]
    ):
        raise ValueError("registered comparator changes a clamp")
    arms = {
        "incumbent": incumbent,
        "paired": paired,
        "readout": readout,
        "half_blend": incumbent + 0.5 * dynamic + 0.5 * offset,
    }
    try:
        tangent_dynamic, dynamic_info = tangent_project(dynamic, nominal, config)
        tangent_offset, offset_info = tangent_project(offset, nominal, config)
    except ProjectionUnavailable as exc:
        arms.update(tangent_only=paired, constraint_split=paired)
        return arms, {
            "ordinary_success": False,
            "exact_fallback": True,
            "reason": str(exc),
        }
    remainder = offset - tangent_offset
    arms["tangent_only"] = incumbent + tangent_dynamic
    arms[PRIMARY] = incumbent + tangent_dynamic + remainder
    if not np.any(dynamic) and not np.any(offset):
        arms["tangent_only"] = incumbent
        arms[PRIMARY] = incumbent
    return arms, {
        "ordinary_success": True,
        "exact_fallback": False,
        "dynamic": dynamic_info,
        "readout": offset_info,
        "remainder_rms_m": float(np.sqrt(np.mean(remainder**2))),
    }


def score_arrays(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    names: list[str],
    ordinary_success: list[bool],
    config: SplitConfig,
) -> dict[str, Any]:
    if set(predictions) != set(ARMS) or len(names) != 14 or len(set(names)) != 14:
        raise ValueError("the entire fixed roster and arm bank must be scored")
    if names.count("103.pkl") != 1 or len(ordinary_success) != len(names):
        raise ValueError("only the registered design case can be excluded")
    shape = (14, 120, config.node_count, 3)
    if truth.shape != shape or not np.isfinite(truth).all():
        raise ValueError("finite, aligned source truth required")
    if any(a.shape != shape or not np.isfinite(a).all() for a in predictions.values()):
        raise ValueError("missing or invalid predictions cannot be dropped")
    keep = [i for i, name in enumerate(names) if name != "103.pkl"]
    metrics: dict[str, Any] = {}
    for arm, value in predictions.items():
        by_case = {}
        for i in keep:
            by_case[names[i]] = {
                horizon: prediction_metrics(
                    value[i, start:end][:, config.hidden_nodes],
                    truth[i, start:end][:, config.hidden_nodes],
                )
                for horizon, (start, end) in HORIZONS.items()
            }
        metrics[arm] = {
            "cases": by_case,
            "mean": {
                horizon: {
                    metric: float(
                        np.mean([case[horizon][metric] for case in by_case.values()])
                    )
                    for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm")
                }
                for horizon in HORIZONS
            },
        }
    rng = np.random.default_rng(config.bootstrap_seed)
    indices = rng.integers(0, len(keep), (config.bootstrap_replicates, len(keep)))
    contrasts: dict[str, Any] = {}
    for control in CONTROLS:
        difference = np.asarray(
            [
                metrics[PRIMARY]["cases"][names[i]]["all"]["point_rmse_mm"]
                - metrics[control]["cases"][names[i]]["all"]["point_rmse_mm"]
                for i in keep
            ]
        )
        contrasts[control] = {
            "rmse_difference_mm": float(difference.mean()),
            "trajectory_bootstrap_ci95_mm": np.quantile(
                difference[indices].mean(axis=-1), (0.025, 0.975)
            ).tolist(),
        }
    means = {arm: metrics[arm]["mean"] for arm in ARMS}
    joint_wins = sum(
        all(
            metrics[PRIMARY]["cases"][names[i]]["all"][key]
            < metrics["paired"]["cases"][names[i]]["all"][key]
            for key in ("coordinate_l1_mm", "point_rmse_mm")
        )
        for i in keep
    )
    checks = {
        "all_14_ordinary_predictions": all(ordinary_success),
        "at_least_2_percent_rmse_gain_over_every_control": all(
            means[PRIMARY]["all"]["point_rmse_mm"]
            <= 0.98 * means[arm]["all"]["point_rmse_mm"]
            for arm in CONTROLS
        ),
        "lower_l1_than_every_control": all(
            means[PRIMARY]["all"]["coordinate_l1_mm"]
            < means[arm]["all"]["coordinate_l1_mm"]
            for arm in CONTROLS
        ),
        "at_least_9_of_13_joint_wins_over_paired": joint_wins >= 9,
        "late_rmse_no_worse_than_incumbent_and_paired": all(
            means[PRIMARY]["late"]["point_rmse_mm"]
            <= means[arm]["late"]["point_rmse_mm"]
            for arm in ("incumbent", "paired")
        ),
        "every_case_rmse_at_most_1_05_times_incumbent": all(
            metrics[PRIMARY]["cases"][names[i]]["all"]["point_rmse_mm"]
            <= 1.05 * metrics["incumbent"]["cases"][names[i]]["all"]["point_rmse_mm"]
            for i in keep
        ),
        "rmse_difference_upper_bound_below_zero_against_every_control": all(
            contrast["trajectory_bootstrap_ci95_mm"][1] < 0
            for contrast in contrasts.values()
        ),
    }
    return {
        "metrics": metrics,
        "contrasts": contrasts,
        "joint_wins_over_paired": joint_wins,
        "analysis_count": len(keep),
        "gate": {"passed": all(checks.values()), "checks": checks},
        "new_transfer_or_target_execution_authorized": False,
        "fresh_confirmation": False,
    }
