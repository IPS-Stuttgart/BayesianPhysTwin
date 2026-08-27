"""Locked training-only test of sparse state transfer through DEFT junctions."""

from __future__ import annotations

import hashlib
import io
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    RestartConfig,
    file_digest,
    paired_physical_readout,
    prediction_metrics,
    sparse_state_increments,
)
from bayesian_phystwin_experiments.deft_native_restart import (
    BRANCH_LENGTHS,
    JUNCTIONS,
    PARENT_CLAMPS,
    NativeDeft,
    update_deft_state,
)

SOURCE_FILE_SHA256 = "f4fad890394c69a772532d3130f2a35c857a01b8bb4dc524842e567a7af1a347"
ARMS = (
    "native_full",
    "native_physics_only",
    "readout_persistence",
    "readout_linear_velocity",
    "paired_physics_pose",
    "paired_physics_pose_velocity",
    "paired_parent_only_pose_velocity",
    "native_full_pose_velocity",
)
PRIMARY = "paired_physics_pose_velocity"
PARENT_CONFIG = RestartConfig(
    node_count=13,
    clamped_nodes=PARENT_CLAMPS,
    hidden_nodes=(3, 5, 7, 9, 10),
)


class _NumericUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            from numpy._core import multiarray, numeric
        except ImportError:
            from numpy.core import multiarray, numeric

        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): multiarray.scalar,
            ("numpy.core.numeric", "_frombuffer"): numeric._frombuffer,
            ("numpy._core.numeric", "_frombuffer"): numeric._frombuffer,
        }
        if (module, name) not in allowed:
            raise ValueError("public source pickle contains a nonnumeric global")
        return allowed[(module, name)]


def load_numeric_training_source(path: Path) -> np.ndarray:
    if file_digest(path) != SOURCE_FILE_SHA256:
        raise ValueError("training source differs from the metadata-selected file")
    value = _NumericUnpickler(io.BytesIO(path.read_bytes())).load()
    raw = np.asarray(value)
    if raw.dtype.kind not in "fi" or raw.size != 3 * 500 * 20:
        raise ValueError(
            "BDLO1 source layout differs from the released loader contract"
        )
    # This is the exact .view(3, 500, -1).permute(1, 2, 0) upstream contract.
    points = raw.astype(np.float64).reshape(3, 500, 20).transpose(1, 2, 0)
    return pack_branched_world(points)


def pack_branched_world(raw_points: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_points, dtype=np.float64)
    if raw.shape != (500, 20, 3):
        raise ValueError("raw BDLO1 material identity layout changed")
    transformed = np.stack((-raw[..., 2], -raw[..., 0], raw[..., 1]), axis=-1)
    result = np.zeros((500, 3, 13, 3), dtype=np.float64)
    result[:, 0] = transformed[:, :13]
    result[:, 1, 0] = transformed[:, 4]
    result[:, 1, 1:5] = transformed[:, 13:17]
    result[:, 2, 0] = transformed[:, 8]
    result[:, 2, 1:4] = transformed[:, 17:20]
    return result


def permitted_inputs(trajectory: np.ndarray) -> dict[str, np.ndarray]:
    if trajectory.shape != (500, 3, 13, 3):
        raise ValueError("training trajectory shape changed")
    result = {
        "initial_two": trajectory[:2].copy(),
        "clamps": trajectory[2:172, 0][:, PARENT_CLAMPS].copy(),
        "sparse_parent_observations": trajectory[[43, 51], 0][
            :, PARENT_CONFIG.observed_nodes
        ].copy(),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("permitted source inputs are nonfinite")
    return result


def branch_increments(
    reference_prefix: np.ndarray, observations: np.ndarray, *, extend_children: bool
) -> tuple[np.ndarray, np.ndarray]:
    if reference_prefix.shape != (50, 3, 13, 3) or observations.shape != (2, 4, 3):
        raise ValueError(
            "updates accept only the declared prefix and eight parent measurements"
        )
    parent_dx, parent_dv = sparse_state_increments(
        reference_prefix[None, :, 0], observations[None], PARENT_CONFIG
    )
    result = []
    for parent in (parent_dx[0], parent_dv[0]):
        delta = np.zeros((3, 13, 3))
        delta[0] = parent
        for branch, junction in enumerate(JUNCTIONS, start=1):
            count = BRANCH_LENGTHS[branch] if extend_children else 1
            delta[branch, :count] = parent[junction]
        result.append(delta)
    return result[0], result[1]


def physics_shadow(native: NativeDeft) -> None:
    """Only a freshly constructed, private shadow disables its learned residual."""
    with native.torch.no_grad():
        native.model.learning_weight.zero_()
    native.model_id = hashlib.sha256(
        (native.model_id + ":physics-shadow-no-gnn-v1").encode()
    ).hexdigest()
    native.model._bpt_model_id = native.model_id


def predict_cross_branch(
    full: NativeDeft, shadow: NativeDeft, inputs: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if set(inputs) != {"initial_two", "clamps", "sparse_parent_observations"}:
        raise ValueError(
            "unexpected input channel; future free-node truth is forbidden"
        )
    initial = inputs["initial_two"]
    actions = inputs["clamps"]
    observed = inputs["sparse_parent_observations"]
    if actions.shape != (170, 4, 3) or observed.shape != (2, 4, 3):
        raise ValueError("source horizon or observation budget changed")
    prefix, _, full_state = full.rollout(initial, actions[:50])
    incumbent, _, full_final = full.rollout(initial, actions, full_state)
    shadow_prefix, _, shadow_state = shadow.rollout(initial, actions[:50])
    physical, _, physical_final = shadow.rollout(initial, actions, shadow_state)
    zero = update_deft_state(shadow_state, np.zeros((3, 13, 3)), np.zeros((3, 13, 3)))
    zero_future, _, zero_final = shadow.rollout(initial, actions, zero)
    if (
        not np.array_equal(zero_future, physical)
        or physical_final.digests() != zero_final.digests()
    ):
        raise ValueError("source-case zero update violates exact fallback")
    dx, dv = branch_increments(prefix, observed, extend_children=True)
    parent_dx, parent_dv = branch_increments(prefix, observed, extend_children=False)
    arrays = {
        "native_full": incumbent,
        "native_physics_only": physical,
        "readout_persistence": incumbent + dx[None],
        "readout_linear_velocity": incumbent
        + dx[None]
        + np.arange(1, 121)[:, None, None, None] * 0.01 * dv[None],
    }
    for name, pose, velocity in (
        ("paired_physics_pose", dx, np.zeros_like(dv)),
        ("paired_physics_pose_velocity", dx, dv),
        ("paired_parent_only_pose_velocity", parent_dx, parent_dv),
    ):
        state = update_deft_state(shadow_state, pose, velocity)
        updated, _, _ = shadow.rollout(initial, actions, state)
        arrays[name] = paired_physical_readout(incumbent, physical, updated)
    updated_full, _, _ = full.rollout(
        initial, actions, update_deft_state(full_state, dx, dv)
    )
    arrays["native_full_pose_velocity"] = updated_full
    if set(arrays) != set(ARMS) or any(
        array.shape != (120, 3, 13, 3) or not np.isfinite(array).all()
        for array in arrays.values()
    ):
        raise ValueError("registered prediction family is incomplete or nonfinite")
    return arrays, {
        "zero_update_byte_identical": True,
        "full_model_id": full.model_id,
        "shadow_model_id": shadow.model_id,
        "full_final_internal_state": full_final.digests(),
        "physical_final_internal_state": physical_final.digests(),
        "prefix_state_fields": len(full_state.fields),
        "prefix_physical_finite": bool(np.isfinite(shadow_prefix).all()),
        "pose_increment_rms_m": float(np.sqrt(np.mean(dx * dx))),
        "velocity_increment_rms_m_s": float(np.sqrt(np.mean(dv * dv))),
    }


def score_cross_branch(
    predictions: Mapping[str, np.ndarray], truth: np.ndarray
) -> dict[str, Any]:
    if set(predictions) != set(ARMS) or truth.shape != (120, 3, 13, 3):
        raise ValueError("prediction family or future alignment differs from lock")
    rows: dict[str, Any] = {}
    for arm in ARMS:
        points = predictions[arm]
        if (
            points.shape != truth.shape
            or not np.isfinite(points).all()
            or not np.isfinite(truth).all()
        ):
            raise ValueError("all registered predictions and outcomes must be finite")
        branches = {}
        for branch, end in ((1, 5), (2, 4)):
            pred, target = points[:, branch, 1:end], truth[:, branch, 1:end]
            branches[f"child{branch}"] = prediction_metrics(pred, target)
            branches[f"child{branch}"]["late_rmse_mm"] = prediction_metrics(
                pred[80:], target[80:]
            )["point_rmse_mm"]
        rows[arm] = {
            **branches,
            "equal_child_branch": {
                metric: float(
                    np.mean([branches[name][metric] for name in ("child1", "child2")])
                )
                for metric in branches["child1"]
            },
            "parent_hidden": prediction_metrics(
                points[:, 0][:, PARENT_CONFIG.hidden_nodes],
                truth[:, 0][:, PARENT_CONFIG.hidden_nodes],
            ),
        }
    checks = {}
    for branch_name in ("child1", "child2"):
        candidate = rows[PRIMARY][branch_name]
        for comparator in (
            "native_full",
            "readout_persistence",
            "readout_linear_velocity",
        ):
            base = rows[comparator][branch_name]
            checks[f"{branch_name}_rmse_at_least_5pct_better_than_{comparator}"] = (
                base["point_rmse_mm"] > 0
                and candidate["point_rmse_mm"] <= 0.95 * base["point_rmse_mm"]
            )
            checks[f"{branch_name}_l1_nonworsening_vs_{comparator}"] = (
                candidate["coordinate_l1_mm"] <= base["coordinate_l1_mm"]
            )
            checks[f"{branch_name}_late_nonworsening_vs_{comparator}"] = (
                candidate["late_rmse_mm"] <= base["late_rmse_mm"]
            )
    return {
        "per_arm": rows,
        "primary_arm": PRIMARY,
        "checks": checks,
        "source_pilot_gate_passed": all(checks.values()),
        "recording_count": 1,
        "independent_confirmation": False,
        "checkpoint_training_exposure": True,
        "inferential_confidence_interval": None,
        "automatic_next_dataset_authorization": False,
    }
