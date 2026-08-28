"""Source decision-value audit with fixed native worlds and a common prefix."""

from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .dlolab_native import array_digest, file_digest
from .dlolab_regret_artifacts import read_record
from .dlolab_slingshot_batch import run_batch, split_batch
from .dlolab_slingshot_cmaes import task_metrics
from .dlolab_slingshot_mechanism import ARMS, assess, verify_controller
from .dlolab_slingshot_process import load_native_bundle

ACTION_NAMES = (
    "incumbent",
    "lateral_minus",
    "lateral_plus",
    "pull_0_8",
    "pull_1_2",
    "yaw_plus",
    "yaw_minus",
    "incumbent_duplicate",
)


def worlds() -> list[dict[str, float | int]]:
    values = [{"index": 0, "x_offset_m": 0.0, "bending_E": 1e5, "stretching_K": 8e5}]
    for index, (offset, bending, stretching) in enumerate(
        itertools.product((-0.02, 0.02), (0.5, 2.0), (0.5, 2.0)), 1
    ):
        values.append(
            {
                "index": index,
                "x_offset_m": offset,
                "bending_E": 1e5 * bending,
                "stretching_K": 8e5 * stretching,
            }
        )
    return values


def action_bank(incumbent: np.ndarray) -> np.ndarray:
    if (
        incumbent.shape != (1, 3, 6)
        or incumbent.dtype != np.float64
        or not np.isfinite(incumbent).all()
    ):
        raise ValueError("exact native source control required")
    bank = np.repeat(incumbent, 8, axis=0)
    bank[1, 1, 0] -= 0.03
    bank[2, 1, 0] += 0.03
    bank[3, 1:, 1] *= 0.8
    bank[4, 1:, 1] *= 1.2
    bank[5, 2, 5] += 0.2
    bank[6, 2, 5] -= 0.2
    # Same component and translation-ball projection as the official helper.
    for index in range(1, 7):
        suffix = np.clip(
            bank[index, 1:],
            -np.asarray([0.1] * 3 + [1.0] * 3),
            np.asarray([0.1] * 3 + [1.0] * 3),
        )
        norm = np.linalg.norm(suffix[:, :3], axis=1, keepdims=True)
        scale = np.ones_like(norm)
        scale[norm > 0.1] = 0.1 / (norm[norm > 0.1] + 1e-12)
        suffix[:, :3] *= scale
        bank[index, 1:] = suffix
    if any(
        array_digest(row[None, :1]) != array_digest(incumbent[:, :1]) for row in bank
    ):
        raise ValueError("common prefix changed")
    return bank


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-decision-value-source-v1",
        "role": "post_source_diagnostic_not_confirmatory_method_evaluation",
        "worlds": worlds(),
        "action_names": list(ACTION_NAMES),
        "unique_actions": 7,
        "batch_slots": 8,
        "native_evaluations": 72,
        "first_stage_shared": True,
        "branch_after_native_frame": 299,
        "observation_prefix_last_frame": 299,
        "fresh_process_per_world": True,
        "hidden_state_restart": False,
        "point_numeric_envelope_m": 0.0005,
        "reward_numeric_envelope": 0.001,
        "numeric_envelope_is_coverage_guarantee": False,
        "native_reward_unchanged": True,
        "minimum_blind_reward_gain_over_zero": 0.01,
        "minimum_oracle_reward_headroom_after_numeric_margin": 0.01,
        "minimum_relative_oracle_headroom": 0.1,
        "minimum_worlds_beating_best_blind_by_0_01": 3,
        "minimum_distinct_best_actions": 2,
        "prior_replay_gates": "retained_failed_not_reclassified",
        "bayesian_method_fit_authorized": False,
        "protected_data_read": False,
        "retry_authorized": False,
        "new_recordings": False,
        "gpu_work": False,
    }


def verify_source(controller: Path, mechanism: Path, root: Path):
    verified, reference = verify_controller(controller, root)
    if (
        file_digest(mechanism)
        != "3a61484640fd0afe314c1c77fbbd97374fa5a6696cc51107a1db2107a5ca746d"
    ):
        raise ValueError("exact retained mechanism result required")
    lock = read_record(mechanism.parent / "lock.json")
    result = read_record(mechanism)
    if result["lock_id"] != lock["artifact_id"]:
        raise ValueError("mechanism binding changed")
    for name, expected in lock["source_sha256"].items():
        if file_digest(root / name) != expected:
            raise ValueError("mechanism source changed")
    rows = []
    for index, arm in enumerate(ARMS):
        seal = read_record(mechanism.parent / arm / "seal.json")
        if seal["artifact_id"] != result["seals"][index]:
            raise ValueError("mechanism seal changed")
        rows.append(load_native_bundle(mechanism.parent / arm, seal["bundle"]))
    computed = assess(rows, reference)
    if (
        any(result[key] != value for key, value in computed.items())
        or result["mechanism_audit_passed"]
    ):
        raise ValueError("retained mechanism disposition changed")
    needed = (
        "projectile_contact_removal_reduces_cube_progress_at_least_80pct",
        "projectile_contact_removal_reduces_sphere_progress_at_least_80pct",
    )
    if (
        not all(result["checks"][key] for key in needed)
        or max(result["position_replay_errors_m"])
        >= protocol()["point_numeric_envelope_m"]
    ):
        raise ValueError(
            "source numerical/mechanism evidence insufficient for this new audit"
        )
    return {
        "controller": verified,
        "mechanism": {
            "path": str(mechanism.resolve()),
            "sha256": file_digest(mechanism),
            "artifact_id": result["artifact_id"],
            "original_gate_passed": False,
        },
    }, reference


def run_world(upstream: Path, output: Path, controls: np.ndarray, world: dict):
    if world not in worlds() or world != worlds()[int(world["index"])]:
        raise ValueError("unregistered native world")
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    module = importlib.import_module("envs.env_slingshot")
    import genesis as gs
    import torch
    from envs.env_slingshot import Train_Env_Slingshot

    captured = {}

    class SourceWorld(Train_Env_Slingshot):
        def _randomize_bending_stiffness(self, objects, envs_idx=None):
            if envs_idx is not None or objects != [self.rope]:
                raise ValueError("full native reset required")
            self.rope.set_bending_stiffness(
                torch.full((self.n_envs,), float(world["bending_E"]), dtype=gs.tc_float)
            )
            actual = self.scene.sim.rod_solver.rods_bending_stiffness.to_numpy()
            if actual.shape != (1, 8) or not np.all(actual == world["bending_E"]):
                raise ValueError("native bending parameter binding failed")
            captured["bending_E"] = actual.tolist()

        def _randomize_stretching_stiffness(self, objects, envs_idx=None):
            if envs_idx is not None or objects != [self.rope]:
                raise ValueError("full native reset required")
            self.rope.set_stretching_stiffness(
                torch.full(
                    (self.n_envs,), float(world["stretching_K"]), dtype=gs.tc_float
                )
            )
            actual = self.scene.sim.rod_solver.rods_stretching_stiffness.to_numpy()
            if actual.shape != (1, 8) or not np.all(actual == world["stretching_K"]):
                raise ValueError("native stretching parameter binding failed")
            captured["stretching_K"] = actual.tolist()

        def _randomize_sphere_and_cube_positions(self, envs_idx=None):
            if envs_idx is not None:
                raise ValueError("full native reset required")
            displacement = torch.tensor(
                [[world["x_offset_m"], 0.0, 0.0]] * self.n_envs, dtype=gs.tc_float
            )
            for name in ("sphere", "cube"):
                entity = getattr(self, name)
                before = entity.get_pos().detach().cpu().numpy().copy()
                entity.set_pos(displacement, relative=True)
                after = entity.get_pos().detach().cpu().numpy().copy()
                if not np.allclose(
                    after - before, displacement.numpy(), rtol=0, atol=1e-15
                ):
                    raise ValueError("native placement binding failed")
                captured[f"{name}_initial_position_m"] = after.tolist()

    with patch.object(module, "Train_Env_Slingshot", SourceWorld):
        values, native = run_batch(upstream, output, controls)
    if set(captured) != {
        "bending_E",
        "stretching_K",
        "sphere_initial_position_m",
        "cube_initial_position_m",
    }:
        raise ValueError("incomplete native world realization")
    return values, {**native, "world_realization": captured}


def world_metrics(values, native, world, reference):
    rows = split_batch(values, 8)
    if array_digest(values["controls"]) != array_digest(
        action_bank(reference["controls"])
    ):
        raise ValueError("registered source action bank changed")
    metrics = [task_metrics(row) for row in rows]
    if [row["native_reward"] for row in metrics] != native["native_cumulative_reward"]:
        raise ValueError("source rewards do not reproduce")
    fields = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
    prefix_error = max(
        float(np.abs(row[name][:300] - rows[0][name][:300]).max())
        for row in rows
        for name in fields
    )
    duplicate_error = max(
        float(np.abs(rows[0][name] - rows[7][name]).max()) for name in fields
    )
    reference_error = (
        max(float(np.abs(rows[0][name] - reference[name]).max()) for name in fields)
        if world["index"] == 0
        else None
    )
    fixed_error = max(
        float(
            np.abs(
                row["rod_pos_m"][:, 0, node] - reference["rod_pos_m"][0, 0, node]
            ).max()
        )
        for row in rows
        for node in (0, 1, 10, 11)
    )
    checks = {
        "common_prefix_within_numeric_envelope": prefix_error <= 0.0005,
        "duplicate_positions_within_numeric_envelope": duplicate_error <= 0.0005,
        "duplicate_reward_within_numeric_envelope": abs(
            metrics[0]["native_reward"] - metrics[7]["native_reward"]
        )
        <= 0.001,
        "nominal_reference_within_numeric_envelope": reference_error is None
        or reference_error <= 0.0005,
        "fixed_endpoints": fixed_error <= 1e-9,
    }
    return {
        "world": world,
        "metrics": metrics,
        "checks": checks,
        "maximum_common_prefix_error_m": prefix_error,
        "maximum_duplicate_position_error_m": duplicate_error,
        "nominal_reference_position_error_m": reference_error,
        "world_qa_passed": all(checks.values()),
    }


def decision_value(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 9 or [row["world"] for row in rows] != worlds():
        raise ValueError("all nine registered worlds required")
    reward = np.asarray(
        [[metric["native_reward"] for metric in row["metrics"][:7]] for row in rows]
    )
    if reward.shape != (9, 7) or not np.isfinite(reward).all():
        raise ValueError("complete finite action/world rewards required")
    mean = reward.mean(axis=0)
    blind = int(np.argmax(mean))
    oracle = np.argmax(reward, axis=1)
    raw = float(reward.max(axis=1).mean() - mean[blind])
    conservative = raw - 2 * protocol()["reward_numeric_envelope"]
    blind_gain = float(mean[blind] - 6.900000095367432)
    relative = conservative / max(0.01, blind_gain)
    improved = int(np.count_nonzero(reward.max(axis=1) - reward[:, blind] > 0.01))
    checks = {
        "all_world_qa": all(row["world_qa_passed"] for row in rows),
        "best_blind_is_nontrivial": blind_gain >= 0.01,
        "oracle_gain_at_least_0_01_after_numeric_margin": conservative >= 0.01,
        "relative_oracle_gain_at_least_10pct": relative >= 0.1,
        "at_least_three_worlds_benefit": improved >= 3,
        "distinct_best_actions": len(set(oracle.tolist())) >= 2,
    }
    return {
        "checks": checks,
        "source_decision_value_passed": all(checks.values()),
        "best_world_blind_action": blind,
        "world_best_actions": oracle.tolist(),
        "mean_action_rewards": mean.tolist(),
        "raw_oracle_reward_gain": raw,
        "numeric_margin_adjusted_oracle_gain": conservative,
        "best_blind_reward_gain_over_zero": blind_gain,
        "relative_oracle_gain": relative,
        "worlds_with_gain_above_0_01": improved,
        "bayesian_gain": False,
        "protected_data_read": False,
    }
