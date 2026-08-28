"""Numerical replay variation of known controls, not a control-value study."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from . import dlolab_slingshot_grip as grip
from .dlolab_slingshot_contact import COUPLINGS, POSITION_FIELDS

LAYOUTS = {"a": (2, 6, 5, 2, 6, 5, 5, 5), "b": (6, 2, 6, 2, 5, 5, 5, 5)}
REPETITIONS = (("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1))
POLICIES = (2, 6, 5)


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(15):
        raise ValueError("unregistered numerical-audit task")
    contact = (2, 0, 1)[index // 5]
    layout, repetition = REPETITIONS[index % 5]
    return {
        "index": index,
        "name": f"coupling-{contact}-{layout}-{repetition}",
        "contact_index": contact,
        "coupling": COUPLINGS[contact],
        "layout": layout,
        "repetition": repetition,
        "grip_source_indices": list(LAYOUTS[layout]),
    }


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-numerical-repeatability-v1",
        "role": "source_engineering_audit_not_controller_evaluation",
        "tasks": [task(i) for i in range(15)],
        "native_batch_count": 15,
        "native_trajectory_count": 120,
        "native_steps": 900,
        "fresh_process_per_batch": True,
        "native_seed": 0,
        "known_grip_policy_indices": list(POLICIES),
        "new_recovery_actions_run": False,
        "x_offset_m": 0.0,
        "bending_E": 100000.0,
        "stretching_K": 800000.0,
        "native_force_limits_N": [-30.0, 30.0],
        "native_force_frames": list(grip.FORCE_FRAMES),
        "prefix_force_N": -3.0,
        "reset_force_N": -1.0,
        "force_branch_native_step": 300,
        "release_native_step": 700,
        "release_finger_position_m": 0.08,
        "measurement_admission": "finite_complete_native_arrays_exact_inputs_material_and_force_binding_fixed_endpoints",
        "duplicate_and_cross_process_differences_are_outcomes_not_admission": True,
        "minimum_scientific_reward_gain": 0.005,
        "observed_reward_span_budget": 0.00025,
        "observed_paired_regret_span_budget": 0.0005,
        "observed_coordinate_span_budget_m": 0.001,
        "paired_regret": "per_batch_policy_mean_minus_same_batch_fallback_mean",
        "covariance_is_descriptive_not_calibrated": True,
        "numerical_repeats_not_independent_physical_worlds": True,
        "earlier_gates_unchanged_and_closed": True,
        "new_controller_evaluation_authorized": False,
        "protected_data_read": False,
        "new_recordings": False,
        "robot_execution": False,
        "gpu_work": False,
        "retry_authorized": False,
    }


def controls(source: np.ndarray, index: int) -> np.ndarray:
    return grip.controls(source)[task(index)["grip_source_indices"]].copy()


def forces(index: int) -> tuple[float, ...]:
    return tuple(-3.0 if p == 5 else -24.0 for p in task(index)["grip_source_indices"])


def validate_force_record(record: dict[str, Any], index: int) -> None:
    with patch.object(grip, "FORCES_N", forces(index)):
        grip.validate_force_record(record)


def run_repeat(
    upstream: Path, output: Path, commands: np.ndarray, index: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with patch.object(grip, "FORCES_N", forces(index)):
        return grip.run_grip_world(
            upstream, output, commands, task(index)["contact_index"]
        )


def _span(values: list[np.ndarray]) -> float:
    return float(np.ptp(np.stack(values), axis=0).max())


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 15 or [r["task"] for r in records] != [
        task(i) for i in range(15)
    ]:
        raise ValueError("complete ordered numerical-audit denominator required")
    for row in records:
        if not row["measurement_admitted"]:
            raise ValueError("inadmissible native batch cannot enter numerical summary")
        if len(row["reward"]) != 8 or not np.isfinite(row["reward"]).all():
            raise ValueError("complete finite native reward vector required")
        for name in POSITION_FIELDS:
            expected = (900, 8, 12, 3) if name == "rod_pos_m" else (900, 8, 3)
            if (
                row["arrays"][name].shape != expected
                or not np.isfinite(row["arrays"][name]).all()
            ):
                raise ValueError("complete finite native positions required")
    worlds = []
    for world in range(3):
        rows = [r for r in records if r["task"]["contact_index"] == world]
        summaries = []
        batch_means = np.empty((5, 3))
        for column, policy in enumerate(POLICIES):
            observed: list[tuple[dict[str, Any], int]] = []
            all_rewards: list[float] = []
            within_reward, within_position = 0.0, 0.0
            for i, row in enumerate(rows):
                slots = [
                    j
                    for j, value in enumerate(row["task"]["grip_source_indices"])
                    if value == policy
                ]
                selected = np.asarray(row["reward"])[slots]
                batch_means[i, column] = selected.mean()
                all_rewards.extend(selected.tolist())
                observed.extend((row, slot) for slot in slots)
                within_reward = max(within_reward, float(np.ptp(selected)))
                within_position = max(
                    within_position,
                    *(
                        float(np.ptp(row["arrays"][k][:, slots], axis=1).max())
                        for k in POSITION_FIELDS
                    ),
                )
            ranges = {
                k: _span([r["arrays"][k][:, slot] for r, slot in observed])
                for k in POSITION_FIELDS
            }
            process_reward, process_position = 0.0, 0.0
            layout_means = {}
            for layout in LAYOUTS:
                matched = [r for r in rows if r["task"]["layout"] == layout]
                slots = [
                    s for s, value in enumerate(LAYOUTS[layout]) if value == policy
                ]
                layout_means[layout] = float(
                    np.mean([np.mean(np.asarray(r["reward"])[slots]) for r in matched])
                )
                for slot in slots:
                    process_reward = max(
                        process_reward,
                        float(np.ptp([r["reward"][slot] for r in matched])),
                    )
                    process_position = max(
                        process_position,
                        *(
                            _span([r["arrays"][k][:, slot] for r in matched])
                            for k in POSITION_FIELDS
                        ),
                    )
            summaries.append(
                {
                    "grip_policy_index": policy,
                    "trajectory_count": len(observed),
                    "reward_span": float(np.ptp(all_rewards)),
                    "coordinate_spans_m": ranges,
                    "within_batch_reward_span": within_reward,
                    "within_batch_coordinate_span_m": within_position,
                    "same_layout_same_slot_process_reward_span": process_reward,
                    "same_layout_same_slot_process_coordinate_span_m": process_position,
                    "layout_a_minus_b_mean_reward": layout_means["a"]
                    - layout_means["b"],
                }
            )
        regret = batch_means[:, :2] - batch_means[:, 2:3]
        covariance = np.cov(batch_means, rowvar=False, ddof=1)
        contrast = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
        worlds.append(
            {
                "contact_index": world,
                "coupling": COUPLINGS[world],
                "batch_count": len(rows),
                "policies": summaries,
                "paired_regret_spans": np.ptp(regret, axis=0).tolist(),
                "descriptive_reward_covariance": covariance.tolist(),
                "descriptive_paired_regret_covariance": (
                    contrast @ covariance @ contrast.T
                ).tolist(),
            }
        )
    maximum_reward = max(p["reward_span"] for w in worlds for p in w["policies"])
    maximum_position = max(
        value
        for w in worlds
        for p in w["policies"]
        for value in p["coordinate_spans_m"].values()
    )
    maximum_regret = max(value for w in worlds for value in w["paired_regret_spans"])
    checks = {
        "observed_reward_span_within_0_00025": maximum_reward <= 0.00025,
        "observed_regret_span_within_0_0005": maximum_regret <= 0.0005,
        "observed_coordinate_span_within_1mm": maximum_position <= 0.001,
    }
    return {
        "worlds": worlds,
        "maximum_reward_span": maximum_reward,
        "maximum_paired_regret_span": maximum_regret,
        "maximum_coordinate_span_m": maximum_position,
        "checks": checks,
        "observed_numerical_budget_passed": all(checks.values()),
        "population_repeatability_bound_established": False,
        "earlier_failed_path_study_reopened": False,
        "controller_value_or_posterior_computed": False,
        "new_controller_evaluation_authorized": False,
    }
