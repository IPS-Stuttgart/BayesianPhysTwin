"""Frozen topology-supported sparse observations for the native DEFT source pilot.

This is an information-placement and paired-response test, not a new rod model
or an assertion that four measured points identify the full dynamical state.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .deform_state_restart import paired_physical_readout, prediction_metrics
from .deft_cross_branch_source import (
    _NumericUnpickler,
    branch_increments,
    pack_branched_world,
)
from .deft_native_restart import PARENT_CLAMPS, NativeDeft, update_deft_state

OBSERVED = ((0, 4), (0, 8), (1, 4), (2, 3))
PARENT_OBSERVED = ((0, 2), (0, 4), (0, 6), (0, 8))
HIDDEN = {
    "parent": (0, (3, 5, 7, 9, 10)),
    "child1": (1, (1, 2, 3)),
    "child2": (2, (1, 2)),
}
PRIMARY = "topology_paired_pose_velocity"
ARMS = (
    "native_full",
    "topology_readout_persistence",
    "topology_readout_linear_velocity",
    "topology_paired_pose",
    PRIMARY,
    "topology_direct_full_pose_velocity",
    "topology_native_periodic_pose",
    "parent_paired_pose_velocity",
)
COMPARATORS = (
    "native_full",
    "topology_readout_persistence",
    "topology_readout_linear_velocity",
    "topology_native_periodic_pose",
    "parent_paired_pose_velocity",
)
CASE_IDS = ("robot-track1324", "robot-track1342", "robot-track24242")


def load_training_case(path: Path, spec: Mapping[str, str]) -> np.ndarray:
    payload = path.read_bytes()
    git_blob = hashlib.sha1(
        b"blob " + str(len(payload)).encode() + b"\0" + payload
    ).hexdigest()
    if (
        git_blob != spec["git_blob"]
        or hashlib.sha256(payload).hexdigest() != spec["sha256"]
    ):
        raise ValueError("selected public training bytes changed")
    value = np.asarray(_NumericUnpickler(io.BytesIO(payload)).load())
    if value.dtype.kind not in "fi" or value.size != 30000:
        raise ValueError("released BDLO1 training layout changed")
    return pack_branched_world(
        value.astype(np.float64).reshape(3, 500, 20).transpose(1, 2, 0)
    )


def observe(
    points: np.ndarray, nodes: tuple[tuple[int, int], ...] = OBSERVED
) -> np.ndarray:
    array = np.asarray(points)
    if array.shape[-3:] != (3, 13, 3):
        raise ValueError("observation operator requires padded BDLO1 identity order")
    return np.stack([array[..., branch, node, :] for branch, node in nodes], axis=-2)


def permitted_inputs(trajectory: np.ndarray) -> dict[str, np.ndarray]:
    if trajectory.shape != (500, 3, 13, 3):
        raise ValueError("public training trajectory shape changed")
    result = {
        "initial_two": trajectory[:2].copy(),
        "clamps": trajectory[2:172, 0][:, PARENT_CLAMPS].copy(),
        "topology_observations": observe(trajectory[[43, 51]]).copy(),
        "parent_control_observations": observe(
            trajectory[[43, 51]], PARENT_OBSERVED
        ).copy(),
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("permitted inputs contain missing/nonfinite observations")
    return result


def topology_basis() -> np.ndarray:
    """Four scalar Dirichlet coefficients: two junctions and two child tips."""
    basis = np.zeros((3, 13, 4), dtype=np.float64)
    knots = (0, 1, 4, 8, 11, 12)
    for column, knot in enumerate((4, 8)):
        basis[0, :, column] = np.interp(
            np.arange(13), knots, [float(n == knot) for n in knots]
        )
    for branch, (junction, length) in enumerate(((4, 5), (8, 4)), start=1):
        fraction = np.linspace(0.0, 1.0, length)
        basis[branch, :length] = (1 - fraction[:, None]) * basis[0, junction]
        basis[branch, :length, branch + 1] += fraction
    return basis


def interpolate_topology(residual: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    if residual.shape != (4, 3) or not np.isfinite(residual).all():
        raise ValueError("exactly four finite metric topology residuals are required")
    return np.einsum("bnk,kd->bnd", topology_basis(), residual)


def topology_increments(
    prefix: np.ndarray, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if prefix.shape != (50, 3, 13, 3) or observations.shape != (2, 4, 3):
        raise ValueError(
            "only the frozen 50-step prefix and eight observations are allowed"
        )
    residual = observations - observe(prefix[[41, 49]])
    return interpolate_topology(residual[-1]), interpolate_topology(
        (residual[-1] - residual[0]) / 0.08
    )


def synthetic_qualification() -> dict[str, Any]:
    basis = topology_basis()
    distributed = np.stack([basis[b, n] for b, n in OBSERVED])
    parent = np.stack([basis[b, n] for b, n in PARENT_OBSERVED])
    pose = np.arange(12, dtype=np.float64).reshape(4, 3) * 0.001
    velocity = np.arange(12, dtype=np.float64).reshape(4, 3) * 0.01 - 0.04
    observations = np.stack((pose - 0.08 * velocity, pose))
    dx, dv = topology_increments(np.zeros((50, 3, 13, 3)), observations)
    zero = topology_increments(np.zeros((50, 3, 13, 3)), np.zeros((2, 4, 3)))
    checks = {
        "topology_scalar_rank_four": int(np.linalg.matrix_rank(distributed)) == 4,
        "parent_scalar_rank_two_on_same_basis": int(np.linalg.matrix_rank(parent)) == 2,
        "parent_only_has_exact_child_tip_null_columns": bool(
            np.array_equal(parent[:, 2:], np.zeros((4, 2)))
        ),
        "synthetic_pose_recovered": bool(
            np.allclose(dx, interpolate_topology(pose), atol=1e-14, rtol=0)
        ),
        "synthetic_velocity_recovered": bool(
            np.allclose(dv, interpolate_topology(velocity), atol=1e-14, rtol=0)
        ),
        "zero_observations_exact_zero_increments": bool(
            all(np.array_equal(x, np.zeros_like(x)) for x in zero)
        ),
    }
    return {
        "schema": "deft-topology-observer-synthetic-qualification-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "Rank and recovery on the declared linear interpolation space, not full dynamical observability",
        "source_trajectory_decoded": False,
        "protected_data_read": False,
    }


def predict_topology(
    full: NativeDeft, shadow: NativeDeft, inputs: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    expected = {
        "initial_two",
        "clamps",
        "topology_observations",
        "parent_control_observations",
    }
    if set(inputs) != expected:
        raise ValueError(
            "unexpected input channel; future free-node truth is forbidden"
        )
    initial, actions = inputs["initial_two"], inputs["clamps"]
    observed, parent_observed = (
        inputs["topology_observations"],
        inputs["parent_control_observations"],
    )
    if actions.shape != (170, 4, 3) or any(
        x.shape != (2, 4, 3) for x in (observed, parent_observed)
    ):
        raise ValueError("frozen horizon or observation budget changed")
    if not all(np.isfinite(x).all() for x in inputs.values()):
        raise ValueError("permitted inputs must be finite")
    early, _, state41 = full.rollout(initial, actions[:42])
    tail, _, state49 = full.rollout(initial, actions[:50], state41)
    prefix = np.concatenate((early, tail))
    incumbent, _, _ = full.rollout(initial, actions, state49)
    _, _, shadow49 = shadow.rollout(initial, actions[:50])
    physical, _, physical_final = shadow.rollout(initial, actions, shadow49)
    zero_state = update_deft_state(shadow49, np.zeros((3, 13, 3)), np.zeros((3, 13, 3)))
    zero_future, _, zero_final = shadow.rollout(initial, actions, zero_state)
    if (
        not np.array_equal(physical, zero_future)
        or physical_final.digests() != zero_final.digests()
    ):
        raise ValueError("source-case exact-zero continuation failed")
    dx, dv = topology_increments(prefix, observed)
    parent_dx, parent_dv = branch_increments(
        prefix, parent_observed, extend_children=True
    )
    arrays = {
        "native_full": incumbent,
        "topology_readout_persistence": incumbent + dx,
        "topology_readout_linear_velocity": incumbent
        + dx
        + np.arange(1, 121)[:, None, None, None] * 0.01 * dv,
    }
    for arm, pose, velocity in (
        ("topology_paired_pose", dx, np.zeros_like(dv)),
        (PRIMARY, dx, dv),
        ("parent_paired_pose_velocity", parent_dx, parent_dv),
    ):
        corrected, _, _ = shadow.rollout(
            initial, actions, update_deft_state(shadow49, pose, velocity)
        )
        arrays[arm] = paired_physical_readout(incumbent, physical, corrected)
    direct, _, _ = full.rollout(initial, actions, update_deft_state(state49, dx, dv))
    arrays["topology_direct_full_pose_velocity"] = direct
    # Matched periodic control consumes each observation only when it arrives.
    first_delta = interpolate_topology(observed[0] - observe(early[-1]))
    periodic_tail, _, periodic49 = full.rollout(
        initial,
        actions[:50],
        update_deft_state(state41, first_delta, np.zeros_like(dv)),
    )
    last_delta = interpolate_topology(observed[1] - observe(periodic_tail[-1]))
    periodic, _, _ = full.rollout(
        initial, actions, update_deft_state(periodic49, last_delta, np.zeros_like(dv))
    )
    arrays["topology_native_periodic_pose"] = periodic
    if set(arrays) != set(ARMS) or any(
        x.shape != (120, 3, 13, 3) or not np.isfinite(x).all() for x in arrays.values()
    ):
        raise ValueError("frozen prediction family is incomplete or nonfinite")
    return arrays, {
        "zero_update_byte_identical": True,
        "full_model_id": full.model_id,
        "shadow_model_id": shadow.model_id,
        "point_observations_per_corrected_arm": 8,
        "topology_pose_rms_m": float(np.sqrt(np.mean(dx * dx))),
        "topology_velocity_rms_m_s": float(np.sqrt(np.mean(dv * dv))),
    }


def score_case(
    predictions: Mapping[str, np.ndarray], truth: np.ndarray
) -> dict[str, Any]:
    if set(predictions) != set(ARMS) or truth.shape != (120, 3, 13, 3):
        raise ValueError("case arm/identity/horizon contract changed")
    if not np.isfinite(truth).all() or any(
        x.shape != truth.shape or not np.isfinite(x).all() for x in predictions.values()
    ):
        raise ValueError("all fixed predictions and outcomes must be finite")
    result = {}
    for arm in ARMS:
        rows = {}
        for name, (branch, nodes) in HIDDEN.items():
            predicted, target = (
                predictions[arm][:, branch][:, nodes],
                truth[:, branch][:, nodes],
            )
            rows[name] = prediction_metrics(predicted, target)
            rows[name]["late_rmse_mm"] = prediction_metrics(
                predicted[80:], target[80:]
            )["point_rmse_mm"]
        rows["equal_child_branch"] = {
            key: float(np.mean([rows[name][key] for name in ("child1", "child2")]))
            for key in rows["child1"]
        }
        result[arm] = rows
    return result


def score_study(cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(cases) != set(CASE_IDS):
        raise ValueError("all three original recordings must remain in the denominator")
    aggregate = {
        arm: {
            region: {
                metric: float(
                    np.mean([cases[case][arm][region][metric] for case in CASE_IDS])
                )
                for metric in (
                    "point_rmse_mm",
                    "coordinate_l1_mm",
                    "late_rmse_mm",
                    "fde_mm",
                )
            }
            for region in (*HIDDEN, "equal_child_branch")
        }
        for arm in ARMS
    }
    checks = {}
    wins = {}
    for comparator in COMPARATORS:
        wins[comparator] = sum(
            cases[case][PRIMARY]["equal_child_branch"]["point_rmse_mm"]
            < cases[case][comparator]["equal_child_branch"]["point_rmse_mm"]
            and cases[case][PRIMARY]["equal_child_branch"]["coordinate_l1_mm"]
            <= cases[case][comparator]["equal_child_branch"]["coordinate_l1_mm"]
            for case in CASE_IDS
        )
        checks[f"at_least_two_recording_joint_wins_vs_{comparator}"] = (
            wins[comparator] >= 2
        )
        for child in ("child1", "child2"):
            candidate, base = aggregate[PRIMARY][child], aggregate[comparator][child]
            checks[f"{child}_rmse_gain_5pct_vs_{comparator}"] = (
                base["point_rmse_mm"] > 0
                and candidate["point_rmse_mm"] <= 0.95 * base["point_rmse_mm"]
            )
            for metric in ("coordinate_l1_mm", "late_rmse_mm"):
                checks[f"{child}_{metric}_nonincreasing_vs_{comparator}"] = (
                    candidate[metric] <= base[metric]
                )
    checks["no_recording_more_than_10pct_worse_than_native"] = all(
        cases[case][PRIMARY]["equal_child_branch"]["point_rmse_mm"]
        <= 1.10 * cases[case]["native_full"]["equal_child_branch"]["point_rmse_mm"]
        for case in CASE_IDS
    )
    return {
        "per_recording": dict(cases),
        "equal_recording_mean": aggregate,
        "recording_joint_wins": wins,
        "checks": checks,
        "source_gate_passed": all(checks.values()),
    }
