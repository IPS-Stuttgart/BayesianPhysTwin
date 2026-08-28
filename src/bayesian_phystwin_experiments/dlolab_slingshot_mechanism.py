"""Source-only task-readout replay and rod-to-projectile contact audit."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .dlolab_native import array_digest, file_digest
from .dlolab_regret_artifacts import read_record
from .dlolab_slingshot_batch import split_batch
from .dlolab_slingshot_cmaes import final_checks, task_metrics, verify_inputs
from .dlolab_slingshot_process import load_native_bundle, run_native

ARMS = ("native_repeat_0", "native_repeat_1", "sphere_rod_contact_disabled")


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-mechanism-source-v1",
        "arms": list(ARMS),
        "selected_control": "frozen_cmaes_source_candidate_29",
        "native_evaluations": 3,
        "physics_intervention": "sphere_collision_geometry_needs_coup_false_only",
        "rigid_rigid_contact_unchanged": True,
        "native_robot_commands_unchanged": True,
        "unchanged_repeats": 2,
        "observable_position_atol_m": 1e-6,
        "native_reward_atol": 1e-5,
        "minimum_sphere_progress_m": 0.01,
        "minimum_cube_progress_m": 0.01,
        "maximum_contact_disabled_progress_ratio": 0.2,
        "full_memory_gate_from_parent": "retained_failed_not_relaxed",
        "state_restart_authorized": False,
        "uncertainty_method_evaluation_authorized": False,
        "independent_target_confirmation": False,
        "retry_authorized": False,
        "protected_data_read": False,
        "new_recordings": False,
        "gpu_work": False,
    }


def verify_controller(
    path: Path, root: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if (
        file_digest(path)
        != "8918c51ba2a073dfdb37145058c6950eee8046a275587e57ebde5fe1b7744398"
    ):
        raise ValueError("exact retained controller result required")
    result = read_record(path)
    lock = read_record(path.parent / "lock.json")
    selection = read_record(path.parent / "selection.json")
    if (
        result["lock_id"] != lock["artifact_id"]
        or result["selection_id"] != selection["artifact_id"]
    ):
        raise ValueError("controller result binding changed")
    for name, expected in lock["source_sha256"].items():
        if file_digest(root / name) != expected:
            raise ValueError("frozen optimizer source changed")
    qualification = Path(lock["verified"]["qualification"]["path"])
    evidence = qualification.parent.parent
    verified, _, warm = verify_inputs(
        evidence / "batch-qualification-v1/result.json",
        evidence / "task-competence-v1/result.json",
        root,
    )
    if verified != lock["verified"]:
        raise ValueError("optimizer upstream source evidence changed")
    best: dict[str, Any] = {"source": "warm_start", "index": -1, **task_metrics(warm)}
    best_array = warm
    entries = []
    for index in range(8):
        directory = path.parent / f"batch-{index:02d}"
        seal = read_record(directory / "output/seal.json")
        plan = read_record(directory / "plan.json")
        if (
            seal["lock_id"] != lock["artifact_id"]
            or seal["plan_id"] != plan["artifact_id"]
        ):
            raise ValueError("optimizer batch binding changed")
        values = load_native_bundle(directory / "output", seal["bundle"])
        inputs = load_native_bundle(directory / "input", plan["input_bundle"])
        if array_digest(values["controls"]) != array_digest(inputs["controls"]):
            raise ValueError("optimizer action identity changed")
        for local, row in enumerate(split_batch(values, 8)):
            item = {
                "source": "optimizer",
                "index": 8 * index + local,
                "batch_index": index,
                "local_index": local,
                **task_metrics(row),
            }
            if (
                item["native_reward"]
                != seal["native"]["native_cumulative_reward"][local]
            ):
                raise ValueError("native optimizer reward changed")
            entries.append(item)
            if item["native_reward"] > best["native_reward"]:
                best, best_array = item, row
    if (
        entries != selection["evaluations"]
        or best != result["best"]
        or best["index"] != 29
    ):
        raise ValueError("selected source control changed")
    directory = path.parent / "best-replay/output"
    seal = read_record(directory / "seal.json")
    reference = load_native_bundle(directory, seal["bundle"])
    recomputed = final_checks(best_array, reference, verified["zero_reward"])
    if (
        any(result[key] != value for key, value in recomputed.items())
        or result["controller_competence_passed"]
    ):
        raise ValueError("parent full-memory failure was changed")
    if any(
        not value
        for key, value in result["checks"].items()
        if key != "replay_all_memory"
    ):
        raise ValueError("the source controller lacks registered task-readout support")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_digest(path),
        "artifact_id": result["artifact_id"],
        "runtime": verified["qualification"]["runtime"],
        "native_source": verified["qualification"]["native_source"],
        "reference_bundle": seal["bundle"],
        "reference_root": str(directory),
        "controls_sha256": array_digest(reference["controls"]),
        "parent_full_memory_gate_passed": False,
    }, reference


def change_projectile_coupling(env: Any, disable: bool) -> dict[str, Any]:
    """Use the pinned pre-build hook; do not change rigid contact filtering."""
    sphere_geoms = list(env.sphere.geoms)
    if len(sphere_geoms) != 1 or sphere_geoms[0].needs_coup is not True:
        raise ValueError("native sphere coupling contract changed")
    geoms = list(env.scene.sim.rigid_solver.geoms)
    before = [bool(geom.needs_coup) for geom in geoms]
    index = sphere_geoms[0].idx
    if geoms[index] is not sphere_geoms[0]:
        raise ValueError("native sphere geometry index changed")
    if disable:
        sphere_geoms[0]._needs_coup = False
    after = [bool(geom.needs_coup) for geom in geoms]
    expected = before.copy()
    expected[index] = not disable
    if after != expected:
        raise ValueError("unregistered geometry coupling change")
    return {
        "sphere_geom_index": index,
        "before": before,
        "after": after,
        "disabled": disable,
        "rigid_collision_filters_changed": False,
    }


def run_arm(upstream: Path, output: Path, controls: np.ndarray, arm: str):
    if arm not in ARMS:
        raise ValueError("unregistered mechanism arm")
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    module = importlib.import_module("envs.env_slingshot")
    from envs.env_slingshot import Train_Env_Slingshot

    captured: dict[str, Any] = {}

    class ContactAudit(Train_Env_Slingshot):
        def construct_extra_cameras(self):
            super().construct_extra_cameras()
            captured.update(change_projectile_coupling(self, arm == ARMS[2]))

        def __init__(self, config):
            super().__init__(config)
            actual = self.scene.sim.rigid_solver.geoms_info.needs_coup.to_numpy()
            if actual.astype(bool).tolist() != captured["after"]:
                raise ValueError(
                    "built native coupling flags do not match intervention"
                )
            captured["built_flags_verified"] = True

    with patch.object(module, "Train_Env_Slingshot", ContactAudit):
        arrays, native = run_native(upstream, output, controls)
    if task_metrics(arrays)["native_reward"] != native["native_cumulative_reward"][0]:
        raise ValueError("mechanism native reward does not reproduce")
    return arrays, {**native, "contact_intervention": captured}


def assess(
    rows: list[dict[str, np.ndarray]], reference: dict[str, np.ndarray]
) -> dict[str, Any]:
    if len(rows) != 3:
        raise ValueError("all three mechanism arms are required")
    for row in rows:
        if set(row) != set(reference) or array_digest(row["controls"]) != array_digest(
            reference["controls"]
        ):
            raise ValueError("mechanism comparison action/layout changed")
        if any(
            row[key].shape != reference[key].shape
            or row[key].dtype != reference[key].dtype
            or not np.isfinite(row[key]).all()
            for key in row
        ):
            raise ValueError("incomplete or nonfinite mechanism trajectory")
    metrics = [task_metrics(row) for row in rows]
    reference_metrics = task_metrics(reference)
    errors = [
        max(
            float(np.abs(row[name] - reference[name]).max())
            for name in ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
        )
        for row in rows[:2]
    ]
    ratios = {
        key: max(0.0, metrics[2][key]) / min(metrics[i][key] for i in range(2))
        if min(metrics[i][key] for i in range(2)) > 0
        else None
        for key in ("cube_progress_m", "sphere_progress_m")
    }
    checks = {
        "unchanged_repeats_within_1um": max(errors) <= 1e-6,
        "unchanged_native_rewards": all(
            abs(row["native_reward"] - reference_metrics["native_reward"]) <= 1e-5
            for row in metrics[:2]
        ),
        "native_cube_progress_at_least_10mm": all(
            row["cube_progress_m"] >= 0.01 for row in metrics[:2]
        ),
        "native_sphere_progress_at_least_10mm": all(
            row["sphere_progress_m"] >= 0.01 for row in metrics[:2]
        ),
        "projectile_contact_removal_reduces_cube_progress_at_least_80pct": ratios[
            "cube_progress_m"
        ]
        is not None
        and ratios["cube_progress_m"] <= 0.2,
        "projectile_contact_removal_reduces_sphere_progress_at_least_80pct": ratios[
            "sphere_progress_m"
        ]
        is not None
        and ratios["sphere_progress_m"] <= 0.2,
    }
    return {
        "checks": checks,
        "arm_metrics": metrics,
        "position_replay_errors_m": errors,
        "contact_disabled_progress_ratios": ratios,
        "mechanism_audit_passed": all(checks.values()),
        "parent_full_memory_gate_passed": False,
        "state_restart_authorized": False,
        "bayesian_gain": False,
        "protected_data_read": False,
    }
