"""Native matched-reset execution for task-aware value-of-information study."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .deform_state_restart import array_digest
from .dlolab_native import DloLabConfig, DloLabRuntime, native_state_digests
from .dlolab_task_aware_voi import (
    ACTION_ROOT_Z_M,
    FIXED_PROBE_INDEX,
    NULL_PROBE_INDEX,
    PROBE_NAMES,
    action_commands,
    particle_parameters,
    probe_commands,
    truth_partition,
)


def _qualify(
    trajectory: NDArray[Any], commands: NDArray[Any]
) -> dict[str, float | bool]:
    value = np.asarray(trajectory, dtype=np.float64)
    control = np.asarray(commands, dtype=np.float64)
    if (
        value.ndim != 4
        or value.shape[0] != control.shape[1]
        or value.shape[1] != control.shape[0]
        or value.shape[2:] != (DloLabConfig().node_count, 3)
        or control.shape[2:] != (2, 3)
        or not np.isfinite(value).all()
        or not np.isfinite(control).all()
    ):
        raise ValueError("complete native task-aware branch required")
    clamp_error = float(
        np.max(np.abs(value[:, :, :2] - control.transpose(1, 0, 2, 3)))
    )
    segment = np.linalg.norm(np.diff(value, axis=2), axis=-1)
    ratio = segment / DloLabConfig().interval_m
    result = {
        "ordinary": bool(np.isfinite(value).all()),
        "maximum_clamp_error_m": clamp_error,
        "minimum_segment_ratio": float(ratio.min()),
        "maximum_segment_ratio": float(ratio.max()),
        "minimum_height_m": float(value[..., 2].min()),
    }
    result["passed"] = bool(
        result["ordinary"]
        and clamp_error <= 1e-10
        and result["minimum_segment_ratio"] >= 0.85
        and result["maximum_segment_ratio"] <= 1.15
        and result["minimum_height_m"] >= -0.05
    )
    return result


def _bind_twisting(runtime: DloLabRuntime, twisting: NDArray[Any]) -> str:
    values = np.asarray(twisting, dtype=np.float64)
    if values.shape != (runtime.batch_size,) or np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("one positive twisting modulus per world required")
    import torch

    runtime.rod.set_twisting_stiffness(torch.as_tensor(values))
    actual = runtime.rod.get_all_twisting_stiffness_tc().detach().cpu().numpy()
    if not np.array_equal(actual, values):
        raise RuntimeError("native twisting bank does not match its model identity")
    identity = hashlib.sha256(
        json.dumps(
            {
                "base_model_id": runtime.model_id,
                "twisting": array_digest(values),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    runtime.model_id = identity
    return identity


def _branches(
    upstream: Path,
    bending: NDArray[Any],
    twisting: NDArray[Any],
    branches: tuple[tuple[str, int], ...],
) -> tuple[list[NDArray[Any]], dict[str, Any]]:
    bend = np.asarray(bending, dtype=np.float64)
    twist = np.asarray(twisting, dtype=np.float64)
    if bend.shape != twist.shape or bend.ndim != 1:
        raise ValueError("aligned one-dimensional material arrays required")
    started = time.monotonic()
    runtime = DloLabRuntime(
        upstream,
        DloLabConfig(),
        batch_size=len(bend),
        bending_moduli=bend,
        lateral_velocities=np.zeros(len(bend), dtype=np.float64),
    )
    try:
        model_id = _bind_twisting(runtime, twist)
        snapshot = runtime.capture()
        initial = runtime.positions()
        initial_state = native_state_digests(snapshot.native_state)
        trajectories: list[NDArray[Any]] = []
        qa: list[dict[str, float | bool]] = []
        restores: list[dict[str, str]] = []
        command_digests: list[str] = []
        branch_names: list[str] = []
        for kind, index in branches:
            runtime.restore(snapshot)
            restored = native_state_digests(runtime.scene.get_state())
            if restored != initial_state or runtime.step_index != snapshot.step_index:
                raise RuntimeError("matched-reset task-aware native state changed")
            if kind == "probe":
                controls = probe_commands(initial[:, :2], index)
            elif kind == "action":
                controls = action_commands(initial[:, :2], index)
            else:
                raise ValueError("unknown task-aware native branch kind")
            trajectory = runtime.rollout(controls).transpose(1, 0, 2, 3)
            branch_qa = _qualify(trajectory, controls)
            trajectories.append(trajectory)
            qa.append(branch_qa)
            restores.append(restored)
            command_digests.append(array_digest(controls))
            branch_names.append(f"{kind}-{index}")
        snapshot.validate(runtime.config, runtime.model_id)
        return trajectories, {
            "model_id": model_id,
            "bending_sha256": array_digest(bend),
            "twisting_sha256": array_digest(twist),
            "initial_position_sha256": array_digest(initial),
            "initial_state_sha256": initial_state,
            "restore_state_sha256": restores,
            "command_sha256": command_digests,
            "branch_names": branch_names,
            "branch_qa": qa,
            "all_branches_qualified": all(row["passed"] for row in qa),
            "branch_count": len(trajectories),
            "world_count": len(bend),
            "wall_seconds": time.monotonic() - started,
            "device": "cpu",
            "runtime_camera_rendered": False,
            "native_source_modified": False,
            "probe_and_task_state_coupled": False,
        }
    finally:
        runtime.close()


def generate_particle_bank(
    upstream: Path,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    parameters = particle_parameters()
    branches = tuple(("probe", index) for index in range(len(PROBE_NAMES))) + tuple(
        ("action", index) for index in range(len(ACTION_ROOT_Z_M))
    )
    trajectories, native = _branches(
        upstream,
        parameters["bending"],
        parameters["twisting"],
        branches,
    )
    probes = np.stack(trajectories[: len(PROBE_NAMES)])
    actions = np.stack(trajectories[len(PROBE_NAMES) :])
    return {
        **parameters,
        "initial_position_m": probes[0, :, 0],
        "probe_trajectory_m": probes,
        "action_trajectory_m": actions,
    }, {**native, "matched_initial_state": True}


def generate_truth_probes(
    upstream: Path,
    generic_mi_probe: int,
    task_aware_probe: int,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    if (
        type(generic_mi_probe) is not int
        or type(task_aware_probe) is not int
        or generic_mi_probe not in range(len(PROBE_NAMES))
        or task_aware_probe not in range(len(PROBE_NAMES))
        or generic_mi_probe == NULL_PROBE_INDEX
        or task_aware_probe in (NULL_PROBE_INDEX, generic_mi_probe)
    ):
        raise ValueError("distinct sealed nonnull selector probes required")
    truth = truth_partition()
    indices = (NULL_PROBE_INDEX, FIXED_PROBE_INDEX, generic_mi_probe, task_aware_probe)
    trajectories, meta = _branches(
        upstream,
        truth["bending"],
        truth["twisting"],
        tuple(("probe", index) for index in indices),
    )
    trajectory_array = np.stack(trajectories)
    return {
        **truth,
        "probe_indices": np.asarray(indices, dtype=np.int64),
        "initial_position_m": trajectory_array[0, :, 0],
        "probe_trajectory_m": trajectory_array,
    }, meta


def generate_truth_futures(
    upstream: Path,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    truth = truth_partition()
    trajectories, meta = _branches(
        upstream,
        truth["bending"],
        truth["twisting"],
        tuple(("action", index) for index in range(len(ACTION_ROOT_Z_M))),
    )
    trajectory_array = np.stack(trajectories)
    return {
        **truth,
        "initial_position_m": trajectory_array[0, :, 0],
        "action_trajectory_m": trajectory_array,
    }, meta
