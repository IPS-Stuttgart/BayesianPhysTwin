"""Bounded source-only task competence, before a Bayesian comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dlolab_benchmark import slingshot_actions
from .dlolab_native import array_digest, file_digest
from .dlolab_regret_artifacts import read_record
from .dlolab_slingshot_process import load_native_bundle, qualify


def action_bank() -> tuple[np.ndarray, list[str]]:
    actions = [x.copy() for x in slingshot_actions()]
    names = ["zero", "continuous_pull_120mm"]
    for mode in ("simultaneous", "lift_then_pull"):
        for lift in (-0.01, 0.02, 0.05):
            for pull in (0.03, 0.06, 0.09):
                action = np.zeros((3, 6), dtype=np.float64)
                if mode == "simultaneous":
                    action[:, 1] = -pull / 3
                    action[:, 2] = lift / 3
                else:
                    action[0, 2] = lift
                    action[1:, 1] = -pull / 2
                actions.append(action)
                names.append(f"{mode}_lift_{lift:.2f}_pull_{pull:.2f}")
    rng = np.random.default_rng(260828)
    for index in range(4):
        action = np.zeros((3, 6), dtype=np.float64)
        action[:, :3] = rng.normal(0, 0.04, size=(3, 3))
        action[:, 1] = -np.abs(action[:, 1])
        norm = np.linalg.norm(action[:, :3], axis=1, keepdims=True)
        action[:, :3] *= np.minimum(1.0, 0.1 / np.maximum(norm, 1e-12))
        action[:, 3:] = np.clip(rng.normal(0, 0.2, size=(3, 3)), -1, 1)
        actions.append(action)
        names.append(f"fixed_random_pull_{index}")
    return np.stack(actions), names


def protocol() -> dict[str, Any]:
    actions, names = action_bank()
    return {
        "schema": "dlolab-slingshot-task-competence-v1",
        "role": "development_action_bank_not_method_comparison",
        "task": "unchanged_native_slingshot",
        "execution": "fresh_process_per_action",
        "candidate_count": len(actions),
        "action_names": names,
        "actions_sha256": array_digest(actions),
        "actions": actions.tolist(),
        "native_steps": 900,
        "controller_substeps": 10,
        "translation_limit_per_stage_m": 0.1,
        "rotation_limit_per_component_rad": 1.0,
        "minimum_cube_forward_progress_m": 0.01,
        "minimum_sphere_forward_progress_m": 0.01,
        "minimum_native_reward_gain": 0.01,
        "minimum_gripper_cube_separation_m": 0.08,
        "required_complete_candidates": len(actions),
        "selection": "maximum_native_cumulative_reward_then_lowest_index",
        "baseline": "exact_native_zero_action",
        "native_reward_unchanged": True,
        "native_physics_unchanged": True,
        "adaptive_tuning": False,
        "retry_authorized": False,
        "method_evaluation_authorized": False,
        "protected_data_read": False,
        "new_recordings": False,
        "gpu_work": False,
        "claim_boundary": "bounded_source_competence_not_published_controller_parity",
    }


def verify_qualification(path: Path, source_root: Path) -> dict[str, Any]:
    if (
        file_digest(path)
        != "4028e4af2db6a1b76ace1dca4ef0a2a94c5247ca6b4cba1e33c1e409aa953bf4"
    ):
        raise ValueError("exact passing native qualification required")
    result = read_record(path)
    lock = read_record(path.parent / "lock.json")
    if result["lock_id"] != lock["artifact_id"]:
        raise ValueError("qualification lock mismatch")
    for name, digest in lock["source_sha256"].items():
        if file_digest(source_root / name) != digest:
            raise ValueError("qualified native implementation changed")
    rows = []
    for index, identity in enumerate(result["rollout_seals"]):
        directory = path.parent / f"run-{index}"
        seal = read_record(directory / "seal.json")
        claim = read_record(directory / "claim.json")
        if (
            seal["artifact_id"] != identity
            or seal["lock_id"] != lock["artifact_id"]
            or seal["claim_id"] != claim["artifact_id"]
            or seal["index"] != index
        ):
            raise ValueError("qualification rollout binding changed")
        rows.append(load_native_bundle(directory, seal["bundle"]))
    computed = qualify(rows)
    if not computed["native_qualification_passed"] or any(
        result[key] != value for key, value in computed.items()
    ):
        raise ValueError("native qualification does not replay")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_digest(path),
        "result_id": result["artifact_id"],
        "native_source": lock["native_source"],
        "runtime": lock["runtime"],
    }


def native_reward_from_trace(cube_positions: np.ndarray) -> float:
    cube = np.asarray(cube_positions)
    if cube.shape != (900, 1, 3) or not np.isfinite(cube).all():
        raise ValueError("complete native cube trace required")
    frames = list(range(119, 700, 20))
    frames[-1] = 899  # The final native reward is read after the release rollout.
    total = np.float32(0)
    for frame in frames:
        total += np.float32(np.clip(cube[frame, 0, 1], 0, 5))
    return float(total)


def candidate_metrics(
    arrays: dict[str, np.ndarray], index: int, native_reward: float
) -> dict[str, Any]:
    actions, names = action_bank()
    if type(index) is not int or not 0 <= index < len(actions):
        raise ValueError("invalid candidate index")
    if array_digest(arrays["controls"]) != array_digest(actions[index][None]):
        raise ValueError("candidate commands changed")
    reward = native_reward_from_trace(arrays["cube_pos_m"])
    if not np.isfinite(native_reward) or reward != native_reward:
        raise ValueError("native reward does not reproduce from its trace")
    sphere = arrays["sphere_pos_m"]
    cube = arrays["cube_pos_m"]
    gripper = arrays["gripper_pos_m"]
    if sphere.shape != cube.shape or gripper.shape != cube.shape:
        raise ValueError("native identity/time axes changed")
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("nonfinite native candidate")
    return {
        "index": index,
        "name": names[index],
        "native_reward": reward,
        "cube_forward_progress_m": float(cube[-1, 0, 1] - cube[99, 0, 1]),
        "sphere_forward_progress_m": float(sphere[-1, 0, 1] - sphere[99, 0, 1]),
        "minimum_gripper_cube_separation_m": float(
            np.min(np.linalg.norm(gripper - cube, axis=-1))
        ),
    }


def summarize(rows: list[dict[str, Any]], failures: list[int]) -> dict[str, Any]:
    count = protocol()["candidate_count"]
    indices = [row["index"] for row in rows]
    if len(set(indices + failures)) != count or sorted(indices + failures) != list(
        range(count)
    ):
        raise ValueError("the full frozen action denominator is required")
    baseline = next((row for row in rows if row["index"] == 0), None)
    capable = []
    for row in rows:
        if baseline is not None and (
            row["cube_forward_progress_m"] >= 0.01
            and row["sphere_forward_progress_m"] >= 0.01
            and row["minimum_gripper_cube_separation_m"] >= 0.08
            and row["native_reward"] - baseline["native_reward"] >= 0.01
        ):
            capable.append(row)
    best = (
        min(capable, key=lambda row: (-row["native_reward"], row["index"]))
        if capable
        else None
    )
    return {
        "candidate_count": count,
        "ordinary_success_count": len(rows),
        "retained_failure_count": len(failures),
        "failed_indices": failures,
        "capable_candidate_count": len(capable),
        "best_capable_candidate": best,
        "baseline": baseline,
        "candidates": sorted(rows, key=lambda row: row["index"]),
        "task_competence_passed": not failures and best is not None,
        "method_evaluation_authorized": False,
        "protected_data_read": False,
        "published_controller_parity": False,
        "bayesian_gain": False,
    }
