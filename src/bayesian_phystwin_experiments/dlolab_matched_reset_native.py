"""Native CPU execution for the matched-reset dual-control source study."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .deform_state_restart import array_digest
from .dlolab_matched_reset_dual_control import (
    ACTION_AMPLITUDES_M,
    FIXED_CONTROL_PROBE_INDEX,
    PROBE_NAMES,
    action_commands,
    particle_bending,
    probe_commands,
    truth_partition,
)
from .dlolab_native import DloLabConfig, DloLabRuntime, native_state_digests


def _qualify(trajectory: np.ndarray, commands: np.ndarray) -> dict[str, float | bool]:
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
        raise ValueError("complete native branch required")
    clamp_error = float(np.max(np.abs(value[:, :, :2] - control.transpose(1, 0, 2, 3))))
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


def _branches(
    upstream: Path,
    bending: np.ndarray,
    branches: tuple[tuple[str, int], ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(bending, dtype=np.float64)
    started = time.monotonic()
    runtime = DloLabRuntime(
        upstream,
        DloLabConfig(),
        batch_size=len(values),
        bending_moduli=values,
        lateral_velocities=np.zeros(len(values), dtype=np.float64),
    )
    try:
        snapshot = runtime.capture()
        initial = runtime.positions()
        initial_state = native_state_digests(snapshot.native_state)
        trajectories = []
        qa = []
        restores = []
        command_digests = []
        branch_names = []
        for kind, index in branches:
            runtime.restore(snapshot)
            restored = native_state_digests(runtime.scene.get_state())
            if restored != initial_state or runtime.step_index != snapshot.step_index:
                raise RuntimeError("matched-reset native state changed")
            if kind == "probe":
                controls = probe_commands(initial[:, :2], index)
            elif kind == "action":
                controls = action_commands(initial[:, :2], index)
            else:
                raise ValueError("unknown native branch kind")
            trajectory = runtime.rollout(controls).transpose(1, 0, 2, 3)
            branch_qa = _qualify(trajectory, controls)
            trajectories.append(trajectory)
            qa.append(branch_qa)
            restores.append(restored)
            command_digests.append(array_digest(controls))
            branch_names.append(f"{kind}-{index}")
        snapshot.validate(runtime.config, runtime.model_id)
        return np.stack(trajectories), {
            "model_id": runtime.model_id,
            "initial_position_sha256": array_digest(initial),
            "initial_state_sha256": initial_state,
            "restore_state_sha256": restores,
            "command_sha256": command_digests,
            "branch_names": branch_names,
            "branch_qa": qa,
            "all_branches_qualified": all(row["passed"] for row in qa),
            "branch_count": len(trajectories),
            "world_count": len(values),
            "wall_seconds": time.monotonic() - started,
            "device": "cpu",
            "runtime_camera_rendered": False,
            "native_source_modified": False,
            "probe_and_task_state_coupled": False,
        }
    finally:
        runtime.close()


def generate_particle_bank(upstream: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bending = particle_bending()
    branches = tuple(("probe", index) for index in range(len(PROBE_NAMES))) + tuple(
        ("action", index) for index in range(len(ACTION_AMPLITUDES_M))
    )
    trajectories, native = _branches(
        upstream,
        bending,
        branches,
    )
    probes = trajectories[: len(PROBE_NAMES)]
    actions = trajectories[len(PROBE_NAMES) :]
    return {
        "bending": bending,
        "initial_position_m": probes[0, :, 0],
        "probe_trajectory_m": probes,
        "action_trajectory_m": actions,
    }, {**native, "matched_initial_state": True}


def generate_truth_probes(
    upstream: Path, selected_probe: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if type(selected_probe) is not int or selected_probe not in range(len(PROBE_NAMES)):
        raise ValueError("sealed selected probe required")
    truth = truth_partition()
    indices = (0, FIXED_CONTROL_PROBE_INDEX, selected_probe)
    trajectories, meta = _branches(
        upstream,
        truth["bending"],
        tuple(("probe", index) for index in indices),
    )
    if selected_probe in (0, FIXED_CONTROL_PROBE_INDEX):
        raise ValueError("active source study requires a nonnull selected probe")
    return {
        **truth,
        "probe_indices": np.asarray(indices, dtype=np.int64),
        "initial_position_m": trajectories[0, :, 0],
        "probe_trajectory_m": trajectories,
    }, meta


def generate_truth_futures(upstream: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    truth = truth_partition()
    trajectories, meta = _branches(
        upstream,
        truth["bending"],
        tuple(("action", index) for index in range(len(ACTION_AMPLITUDES_M))),
    )
    return {
        **truth,
        "initial_position_m": trajectories[0, :, 0],
        "action_trajectory_m": trajectories,
    }, meta
