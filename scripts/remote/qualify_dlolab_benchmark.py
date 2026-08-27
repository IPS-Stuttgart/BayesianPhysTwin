"""Source-only native slingshot qualification with no task or solver rewrites."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import file_digest
from bayesian_phystwin_experiments.dlolab_benchmark import (
    fixed_endpoint_error,
    memory_comparison,
    native_memory,
    protocol,
    slingshot_actions,
    source_identity,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    runtime_identity,
    write_record,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "scripts/remote/qualify_dlolab_benchmark.py",
    "tests/test_dlolab_benchmark.py",
    "docs/dlolab_native_benchmark_source_v1.md",
    "docs/dlolab_native_benchmark_reporter_amendment_v1_1.md",
)


def observe(env: Any) -> dict[str, np.ndarray]:
    values = {
        "rod_pos_m": env.rope.get_all_verts(),
        "rod_vel_m_s": env.rope.get_all_vels(),
        "sphere_pos_m": env.sphere.get_pos(),
        "sphere_vel_m_s": env.sphere.get_vel(),
        "cube_pos_m": env.cube.get_pos(),
        "cube_vel_m_s": env.cube.get_vel(),
        "gripper_pos_m": env.c1.ef.get_pos(),
        "robot_qpos": env.franka1.get_qpos(),
    }
    result = {}
    for key, value in values.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result[key] = np.array(value, copy=True, order="C")
        if not np.isfinite(result[key]).all():
            raise RuntimeError(f"nonfinite native observable {key}")
    return result


def run(output: Path, assets: Path, parent_failure: Path) -> dict[str, Any]:
    revision = clean_revision(ROOT)
    upstream, mushroom = assets / "upstream", assets / "mushroom-rl"
    source = source_identity(upstream, mushroom, assets / "dlo-lab.zip")
    runtime = runtime_identity()
    runtime["benchmark_packages"] = {
        p: importlib.metadata.version(p)
        for p in (
            "pin",
            "pin-pink",
            "qpsolvers",
            "proxsuite",
            "quadprog",
            "mushroom-rl",
            "omegaconf",
        )
    }
    if (
        file_digest(parent_failure)
        != "aecc4225c9e8c06998d4e339df28a53d32ca12ae26fde8a42f9bd34680819db3"
    ):
        raise ValueError("the retained v1 reporting failure must remain unchanged")
    parent = read_record(parent_failure)
    parent_attempt = read_record(parent_failure.parent / "attempt.json")
    old_protocol = dict(parent_attempt["protocol"])
    old_protocol["schema"] = protocol()["schema"]
    old_protocol["reporter_repair_only"] = True
    old_protocol["seal_each_rollout_before_analysis"] = True
    if (
        parent["attempt_id"] != parent_attempt["artifact_id"]
        or parent_attempt["native_source"] != source
        or parent_attempt["runtime"] != runtime
        or old_protocol != protocol()
        or parent["protected_data_read"] is not False
        or parent["method_evaluation_authorized"] is not False
    ):
        raise ValueError("the reporter repair cannot change the native experiment")
    output.mkdir(parents=True, exist_ok=False)
    attempt = write_record(
        output / "attempt.json",
        {
            "schema": "dlolab-native-benchmark-attempt-v1",
            "source_revision": revision,
            "source_sha256": {p: file_digest(ROOT / p) for p in SOURCES},
            "native_source": source,
            "runtime": runtime,
            "protocol": protocol(),
            "retained_parent_failure": {
                "artifact_id": parent["artifact_id"],
                "file_sha256": file_digest(parent_failure),
                "path": str(parent_failure.resolve()),
            },
            "output_root": str(output.resolve()),
            "retry_authorized": False,
            "protected_data_read": False,
            "method_evaluation_authorized": False,
        },
    )
    gs: Any = None
    stage = "imports"
    completed_seals: list[str] = []
    started = time.monotonic()
    try:
        sys.path.insert(0, str(upstream))
        sys.path.insert(0, str(upstream / "experiments"))
        gs = importlib.import_module("genesis")
        import torch
        from envs.env_slingshot import Train_Env_Slingshot
        from omegaconf import DictConfig

        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        gs.init(
            seed=0,
            precision="64",
            logging_level="error",
            backend=gs.cpu,
            performance_mode=True,
            theme="dumb",
        )
        stage = "scene-construction"
        env = Train_Env_Slingshot(
            DictConfig(
                {
                    "task": "slingshot",
                    "log_dir": str(output / "native-log"),
                    "n_envs": 1,
                    "GUI": False,
                    "camera": False,
                    "raytracer": False,
                    "requires_grad": False,
                }
            )
        )
        env.init_cmaes_env(n_steps_sub=10)
        trace: list[dict[str, np.ndarray]] = []
        original_step = env.scene.step

        def traced_step(*args: Any, **kwargs: Any) -> Any:
            value = original_step(*args, **kwargs)
            trace.append(observe(env))
            return value

        env.scene.step = traced_step
        arrays: dict[str, np.ndarray] = {}
        summaries = []
        memories = []
        for run_index, action_index in enumerate((0, 1, 1)):
            stage = f"native-rollout-{run_index}"
            trace.clear()
            controls = slingshot_actions()[action_index][None]
            print(f"starting native benchmark rollout {run_index + 1}/3", flush=True)
            native = env.eval_traj(controls)
            if len(trace) != 900:
                raise RuntimeError(f"native step contract changed: {len(trace)} != 900")
            for name in trace[0]:
                arrays[f"run_{run_index}_{name}"] = np.stack([x[name] for x in trace])
            arrays[f"run_{run_index}_controls"] = controls
            arrays[f"run_{run_index}_joint_targets"] = env.qpos_seq.copy()
            memories.append(native_memory(env.scene.get_state()))
            for name, value in memories[-1].items():
                arrays[f"run_{run_index}_memory_{name}"] = value
            summaries.append(
                {
                    "action_index": action_index,
                    "native_cumulative_reward": np.asarray(
                        native["cum_reward"]
                    ).tolist(),
                    "native_forward_seconds": float(native["forward_time"]),
                    "native_steps": len(trace),
                }
            )
            stage = f"seal-native-rollout-{run_index}"
            run_directory = output / f"run-{run_index}"
            run_directory.mkdir(exist_ok=False)
            run_bundle = write_native_bundle(
                run_directory,
                {
                    name: value
                    for name, value in arrays.items()
                    if name.startswith(f"run_{run_index}_")
                },
            )
            run_seal = write_record(
                run_directory / "seal.json",
                {
                    "schema": "dlolab-native-benchmark-rollout-seal-v1-1",
                    "attempt_id": attempt["artifact_id"],
                    "run_index": run_index,
                    "summary": summaries[-1],
                    "bundle": run_bundle,
                    "protected_data_read": False,
                    "method_evaluation_authorized": False,
                },
            )
            completed_seals.append(run_seal["artifact_id"])
            print(f"completed native benchmark rollout {run_index + 1}/3", flush=True)
        env.scene.step = original_step
        stage = "seal-all-native-rollouts"
        bundle = write_native_bundle(output, arrays)
        generation = write_record(
            output / "generation.json",
            {
                "schema": "dlolab-native-benchmark-generation-v1-1",
                "attempt_id": attempt["artifact_id"],
                "rollout_seals": completed_seals,
                "bundle": bundle,
                "rollouts": summaries,
                "protected_data_read": False,
                "method_evaluation_authorized": False,
            },
        )
        stage = "replay-analysis"
        replay_error = max(
            float(np.max(np.abs(arrays[f"run_1_{key}"] - arrays[f"run_2_{key}"])))
            for key in ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
        )
        memory = memory_comparison(memories[1], memories[2])
        gripper = arrays["run_1_gripper_pos_m"]
        rod = arrays["run_1_rod_pos_m"]
        gripper_motion = float(
            np.linalg.norm(gripper[699] - gripper[99], axis=-1).max()
        )
        band_motion = float(
            np.linalg.norm(rod[699, :, 6] - rod[99, :, 6], axis=-1).max()
        )
        fixed_error = fixed_endpoint_error(
            [arrays[f"run_{i}_rod_pos_m"] for i in range(3)]
        )
        checks = {
            "all_three_native_rollouts_complete": len(summaries) == 3,
            "finite_native_arrays": all(np.isfinite(x).all() for x in arrays.values()),
            "gripper_motion_at_least_10mm": gripper_motion >= 0.01,
            "band_motion_at_least_10mm": band_motion >= 0.01,
            "fixed_endpoints_unchanged": fixed_error <= 1e-9,
            "position_replay_within_1um": replay_error <= 1e-6,
            "memory_replay_within_tolerance": memory["within_tolerance"],
        }
        stage = "seal"
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-native-benchmark-qualification-result-v1",
                "attempt_id": attempt["artifact_id"],
                "generation_id": generation["artifact_id"],
                "bundle": bundle,
                "rollouts": summaries,
                "checks": checks,
                "native_qualification_passed": all(checks.values()),
                "gripper_motion_m": gripper_motion,
                "band_motion_m": band_motion,
                "maximum_fixed_endpoint_error_m": fixed_error,
                "maximum_position_replay_error_m": replay_error,
                "memory_replay": memory,
                "wall_seconds": time.monotonic() - started,
                "source_only": True,
                "method_comparison": False,
                "method_evaluation_authorized": False,
                "protected_data_read": False,
            },
        )
        print(
            f"qualification={result['native_qualification_passed']}; id={result['artifact_id']}"
        )
        return result
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-native-benchmark-failure-v1",
                "attempt_id": attempt["artifact_id"],
                "terminal_stage": stage,
                "completed_rollout_seals": completed_seals,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "method_evaluation_authorized": False,
                "protected_data_read": False,
            },
        )
        raise
    finally:
        if gs is not None and gs._initialized:
            gs.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--parent-failure", required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.assets, args.parent_failure)
