"""Contracts for a bounded, standard CMA-ES native controller baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dlolab_native import array_digest, file_digest
from .dlolab_regret_artifacts import read_record
from .dlolab_slingshot_batch import compare, split_batch
from .dlolab_slingshot_controls import (
    candidate_metrics,
    native_reward_from_trace,
    summarize,
    verify_qualification,
)
from .dlolab_slingshot_process import load_native_bundle


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-cmaes-source-v1",
        "role": "nominal_controller_development_not_bayesian_test",
        "optimizer": "cma.CMAEvolutionStrategy",
        "cma_version": "4.4.4",
        "population": 16,
        "generations": 4,
        "batch_size": 8,
        "seed": 260829,
        "sigma0": 0.02,
        "warm_start": "highest_native_reward_in_frozen_24_action_source_bank",
        "projection": "official_trajopt_cmaes_project_deltas",
        "translation_limit_m": 0.1,
        "angle_component_limit_rad": 1.0,
        "optimization_score": "negative_unchanged_native_cumulative_reward",
        "evaluation_count": 64,
        "additional_selected_isolated_replay_count": 1,
        "minimum_cube_progress_m": 0.01,
        "minimum_sphere_progress_m": 0.01,
        "minimum_reward_gain_over_zero": 0.01,
        "minimum_gripper_cube_separation_m": 0.08,
        "replay_position_atol_m": 1e-6,
        "replay_memory_rtol": 1e-6,
        "replay_memory_atol": 1e-9,
        "replay_reward_atol": 1e-5,
        "retry_authorized": False,
        "protected_data_read": False,
        "method_evaluation_authorized": False,
        "published_controller_parity": False,
        "new_recordings": False,
        "gpu_work": False,
        "elastic_launch_mechanism_validated": False,
        "separate_rod_mechanism_audit_required": True,
    }


def verify_inputs(
    batch_result: Path, source_result: Path, root: Path
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    if (
        file_digest(batch_result)
        != "21c31c97882b80c8f191cf266aeaaf9620be81324a55d4bcca011d617bcef3cf"
        or file_digest(source_result)
        != "fcbf55f54f71269cd48ac211df051820cf1e902f5f03fe0eaa5408fe4f3e6f29"
    ):
        raise ValueError("exact frozen source/batch inputs required")
    result = read_record(batch_result)
    batch_lock = read_record(batch_result.parent / "lock.json")
    generation = read_record(batch_result.parent / "generation.json")
    if (
        result["lock_id"] != batch_lock["artifact_id"]
        or result["generation_id"] != generation["artifact_id"]
    ):
        raise ValueError("batch binding changed")
    for path, expected in batch_lock["source_sha256"].items():
        if file_digest(root / path) != expected:
            raise ValueError("qualified batch source changed")
    qualification = Path(batch_lock["qualification"]["path"])
    verified = verify_qualification(qualification, root)
    references = []
    for index in range(2):
        directory = qualification.parent / f"run-{index}"
        references.append(
            load_native_bundle(
                directory, read_record(directory / "seal.json")["bundle"]
            )
        )
    batch_arrays = load_native_bundle(batch_result.parent, generation["bundle"])
    computed = compare(
        split_batch(batch_arrays, 8),
        references,
        generation["native"]["native_cumulative_reward"],
    )
    if not computed["batch_qualification_passed"] or any(
        result[key] != value for key, value in computed.items()
    ):
        raise ValueError("batch qualification does not replay")
    source = read_record(source_result)
    source_lock = read_record(source_result.parent / "lock.json")
    if (
        source["lock_id"] != source_lock["artifact_id"]
        or source["ordinary_success_count"] != 24
        or source["retained_failure_count"] != 0
    ):
        raise ValueError("complete retained source bank required")
    for path, expected in source_lock["source_sha256"].items():
        if file_digest(root / path) != expected:
            raise ValueError("retained source implementation changed")
    rows, values = [], []
    for index in range(24):
        directory = source_result.parent / f"candidate-{index:02d}"
        seal = read_record(directory / "seal.json")
        if seal["lock_id"] != source_lock["artifact_id"] or seal["index"] != index:
            raise ValueError("source candidate identity changed")
        data = load_native_bundle(directory, seal["bundle"])
        values.append(data)
        rows.append(
            candidate_metrics(
                data, index, seal["summary"]["native_cumulative_reward"][0]
            )
        )
    recomputed = summarize(rows, [])
    if any(source[key] != value for key, value in recomputed.items()):
        raise ValueError("source bank does not replay")
    best = min(rows, key=lambda row: (-row["native_reward"], row["index"]))
    selected = values[best["index"]]
    return (
        {
            "qualification": verified,
            "batch_result_id": result["artifact_id"],
            "batch_result_sha256": file_digest(batch_result),
            "source_result_id": source["artifact_id"],
            "source_result_sha256": file_digest(source_result),
            "warm_start": best,
            "warm_start_controls_sha256": array_digest(selected["controls"]),
            "zero_reward": rows[0]["native_reward"],
        },
        selected["controls"][0].copy(),
        selected,
    )


def task_metrics(row: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "native_reward": native_reward_from_trace(row["cube_pos_m"]),
        "cube_progress_m": float(
            row["cube_pos_m"][-1, 0, 1] - row["cube_pos_m"][99, 0, 1]
        ),
        "sphere_progress_m": float(
            row["sphere_pos_m"][-1, 0, 1] - row["sphere_pos_m"][99, 0, 1]
        ),
        "minimum_gripper_cube_separation_m": float(
            np.min(np.linalg.norm(row["gripper_pos_m"] - row["cube_pos_m"], axis=-1))
        ),
    }


def final_checks(
    selected: dict[str, np.ndarray], replay: dict[str, np.ndarray], zero_reward: float
) -> dict[str, Any]:
    if set(selected) != set(replay) or array_digest(
        selected["controls"]
    ) != array_digest(replay["controls"]):
        raise ValueError("selected replay identity changed")
    for name in selected:
        if (
            selected[name].shape != replay[name].shape
            or selected[name].dtype != replay[name].dtype
        ):
            raise ValueError("selected replay layout changed")
    memory = [name for name in selected if name.startswith("memory_")]
    if len(memory) != 23:
        raise ValueError("complete selected replay memory required")
    position_error = max(
        float(np.max(np.abs(selected[name] - replay[name])))
        for name in ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
    )
    memory_ok = all(
        np.allclose(selected[name], replay[name], rtol=1e-6, atol=1e-9)
        for name in memory
    )
    metrics = task_metrics(replay)
    checks = {
        "replay_positions": position_error <= 1e-6,
        "replay_all_memory": bool(memory_ok),
        "replay_reward": abs(
            task_metrics(selected)["native_reward"] - metrics["native_reward"]
        )
        <= 1e-5,
        "cube_progress": metrics["cube_progress_m"] >= 0.01,
        "sphere_progress": metrics["sphere_progress_m"] >= 0.01,
        "reward_progress": metrics["native_reward"] - zero_reward >= 0.01,
        "gripper_cube_separation": metrics["minimum_gripper_cube_separation_m"] >= 0.08,
    }
    return {
        "checks": checks,
        "selected_replay_metrics": metrics,
        "maximum_position_replay_error_m": position_error,
        "controller_competence_passed": all(checks.values()),
        "bayesian_gain": False,
        "method_evaluation_authorized": False,
        "protected_data_read": False,
    }
