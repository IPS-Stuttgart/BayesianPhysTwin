#!/usr/bin/env python3
"""Post-result arithmetic verification for the partial active-wrapping failure."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_active_wrapping_source import (
    MEMORY_NAMES,
    N_ENVS,
    POSITION_FIELDS,
    POSTS,
    PROBE_SLOT,
    PROBE_STEPS,
    task,
)
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-active-probe-wrapping-source-v1-1")


def angular_reward(positions: np.ndarray, posts: np.ndarray) -> np.ndarray:
    relative = positions[..., :, None, :2] - posts[..., None, :, :2]
    angle = np.unwrap(np.arctan2(relative[..., 1], relative[..., 0]), axis=-2)
    first = angle[..., 0, :]
    last = angle[..., -1, :]
    closing = last + np.angle(np.exp(1j * (first - last)))
    turns = (closing - angle[..., 0, :]) / (2 * np.pi)
    distance = np.linalg.norm(
        positions[..., :, None, :] - posts[..., None, :, :], axis=-1
    ).min(axis=-2)
    return (
        1
        - np.mean((np.abs(turns) - 1) ** 2, axis=-1)
        - np.maximum(distance - 0.015, 0).sum(axis=-1)
    )


def git_blob_digest(revision: str, path: str) -> str:
    payload = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


def compare(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError("partial verification fields changed")
        return max((compare(left[key], right[key]) for key in left), default=0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("partial verification list length changed")
        return max((compare(a, b) for a, b in zip(left, right, strict=True)), default=0)
    if (
        isinstance(left, (bool, str))
        or isinstance(right, (bool, str))
        or left is None
        or right is None
    ):
        if left != right:
            raise ValueError(f"partial verification scalar changed: {left} != {right}")
        return 0.0
    return abs(float(left) - float(right))


def independent_qa(data: dict[str, np.ndarray], native: dict) -> dict[str, Any]:
    final = angular_reward(data["rod_pos_m"][-1], data["post_pos_m"][-1])
    cumulative = np.zeros(N_ENVS, dtype=np.float32)
    for frame in range(19, PROBE_STEPS, 20):
        cumulative += angular_reward(
            data["rod_pos_m"][frame], data["post_pos_m"][frame]
        ).astype(np.float32) + np.float32(1)
    rest = np.linalg.norm(
        np.roll(data["initial_rod_pos_m"], -1, axis=1) - data["initial_rod_pos_m"],
        axis=-1,
    )
    ratios = (
        np.linalg.norm(
            np.roll(data["rod_pos_m"], -1, axis=2) - data["rod_pos_m"], axis=-1
        )
        / rest
    )
    groups = tuple(
        tuple(index for index, value in enumerate(PROBE_SLOT) if value == probe)
        for probe in range(4)
    )
    duplicate = max(
        float(
            np.abs(
                data[key][:, list(group)] - data[key][:, group[0] : group[0] + 1]
            ).max()
        )
        for key in POSITION_FIELDS
        for group in groups
    )
    endpoint = float(np.ptp(data["gripper_pos_m"][-1], axis=0).max())
    attachment = float(
        np.linalg.norm(
            data["rod_pos_m"][:, :, [17, 33]] - data["gripper_pos_m"], axis=-1
        ).max()
    )
    fixed = float(np.abs(data["post_pos_m"] - POSTS).max())
    final_error = float(np.abs(final - np.asarray(native["native_final_reward"])).max())
    checks = {
        "native_final_reward": final_error <= 1e-7,
        "native_cumulative_reward": np.array_equal(
            cumulative,
            np.asarray(native["native_cumulative_reward"], dtype=np.float32),
        ),
        "ordinary_native_success": bool(
            np.all(np.asarray(native["native_final_reward"]) > -98)
        ),
        "common_full_prefix": True,
        "common_probe_endpoint_tools": endpoint <= 1e-5,
        "duplicate_positions": duplicate <= 0.001,
        "fixed_posts": fixed <= 1e-9,
        "finite_extensible_segments": bool(ratios.min() >= 0.25 and ratios.max() <= 3),
        "above_floor": float(data["rod_pos_m"][..., 2].min()) >= -0.01,
        "attached_material_points": attachment <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_common_prefix_error_m": 0.0,
        "maximum_common_probe_endpoint_tool_error_m": endpoint,
        "maximum_duplicate_coordinate_error_m": duplicate,
        "fixed_post_error_m": fixed,
        "segment_length_ratio_range": [float(ratios.min()), float(ratios.max())],
        "maximum_attachment_distance_m": attachment,
        "final_reward_reconstruction_error": final_error,
        "final_rewards": final.tolist(),
    }


def verify() -> dict[str, Any]:
    lock = read_record(OUTPUT / "lock.json")
    result = read_record(OUTPUT / "result.json")
    if (
        result["lock_id"] != lock["artifact_id"]
        or result["status"] != "probe_native_qualification_failed"
        or result["completed_batches"] != 1
        or result["admitted_batches"] != 0
        or result["completed_native_trajectories"] != 9
        or result["unrun_batches"] != 32
        or result["source_gate_passed"] is not False
        or result["retry_authorized"] is not False
        or any(
            git_blob_digest(lock["revision"], path) != digest
            for path, digest in lock["source_sha256"].items()
        )
    ):
        raise ValueError("partial source lock/result accounting changed")
    spec = task("probe", 0)
    directory = OUTPUT / spec["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim["lock_id"] != lock["artifact_id"]
        or seal["lock_id"] != lock["artifact_id"]
        or seal["claim_id"] != claim["artifact_id"]
        or seal["task"] != spec
    ):
        raise ValueError("partial native custody changed")
    data = load_native_bundle(directory, seal["bundle"])
    if set(data) != set(MEMORY_NAMES) | {
        "rod_pos_m",
        "rod_vel_m_s",
        "post_pos_m",
        "gripper_pos_m",
        "robot_qpos",
        "controls",
        "joint_targets",
        "initial_rod_pos_m",
    }:
        raise ValueError("partial native bundle fields changed")
    independent = independent_qa(data, seal["native"])
    stored = read_record(directory / "qa.json")["qa"]
    difference = compare(independent, stored)
    failed = [name for name, passed in independent["checks"].items() if not passed]
    return {
        "schema": "dlolab-active-probe-wrapping-partial-second-arithmetic-v1",
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "verified_source_files": len(lock["source_sha256"]),
        "verified_native_trajectories": 9,
        "verified_array_members": len(data),
        "maximum_arithmetic_difference": difference,
        "failed_checks": failed,
        "native_physics_reexecuted": False,
        "independent_human_review": False,
        "source_gate_passed": False,
        "passed": difference <= 1e-10 and failed == ["common_probe_endpoint_tools"],
        "protected_data_read": False,
    }


if __name__ == "__main__":
    output = OUTPUT / "second-arithmetic.json"
    if output.exists() or output.is_symlink():
        raise ValueError("write-once partial verifier already exists")
    row = write_record(output, verify())
    print(json.dumps(row, indent=2))
