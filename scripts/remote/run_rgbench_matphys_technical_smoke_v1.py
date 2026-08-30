#!/usr/bin/env python3
"""Run one target-closed RGBench-to-MatPhys technical source smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.rgbench_matphys_protocol_v1 import (
    load_rgbench_matphys_preaccess_amendment_v1,
)
from bayesian_phystwin.rgbench_matphys_source_v1 import (
    build_rgbench_matphys_graph_v1,
    load_episode_world_points_v1,
    load_rgbench_source_episode_index_v1,
    spring_graph_component_count_v1,
)

PLAN_SCHEMA: Final = "bayesian-phystwin.rgbench-matphys-technical-smoke-execution"
PLAN_VERSION: Final = 1
RESULT_SCHEMA: Final = "bayesian-phystwin.rgbench-matphys-technical-smoke-result"
RESULT_VERSION: Final = 1
EXPECTED_RUNTIME: Final = {
    "python_version": "3.10.12",
    "numpy_version": "1.26.4",
    "torch_version": "2.4.0+cu121",
    "torch_cuda_version": "12.1",
    "warp_version": "1.15.0",
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_revision(repository: Path, revision: str, *, label: str) -> None:
    _require(repository.is_dir(), f"{label} repository is missing")
    _require(_git_revision(repository) == revision, f"{label} revision changed")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status, f"{label} repository is dirty")


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "execution plan is invalid")
    _require(_file_sha256(path) == expected_sha256, "execution plan SHA-256 changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    plan = _mapping(value, name="execution plan")
    _require(plan.get("schema") == PLAN_SCHEMA, "execution plan schema changed")
    _require(
        plan.get("schema_version") == PLAN_VERSION, "execution plan version changed"
    )
    identity = dict(plan)
    declared = identity.pop("plan_id", None)
    _require(declared == content_id(identity), "execution plan identity changed")
    return plan


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _consume_attempt(path: Path, *, plan_id: str, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "schema": "bayesian-phystwin.rgbench-matphys-technical-smoke-attempt",
                "schema_version": 1,
                "plan_id": plan_id,
                "output_root": str(output_root),
                "attempt_consumed": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _state_numpy(simulator: Any, warp: Any) -> np.ndarray:
    return (
        warp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
        .detach()
        .cpu()
        .numpy()
        .copy()
    )


def _runtime_identity(torch: Any) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": importlib.metadata.version("torch"),
        "torch_cuda_version": str(torch.version.cuda),
        "warp_version": importlib.metadata.version("warp-lang"),
    }


def _run(plan: dict[str, Any], output_root: Path) -> dict[str, Any]:
    implementation = _mapping(plan.get("implementation"), name="implementation")
    execution_repository = Path(implementation["repository_path"]).resolve(strict=True)
    execution_revision = str(implementation["revision"])
    _require_clean_revision(
        execution_repository,
        execution_revision,
        label="BayesianPhysTwin implementation",
    )
    runner_path = execution_repository / str(implementation["runner_relative_path"])
    _require(
        runner_path.resolve(strict=True) == Path(__file__).resolve(strict=True),
        "execution runner path changed",
    )
    _require(
        _file_sha256(runner_path) == implementation["runner_sha256"],
        "execution runner SHA-256 changed",
    )

    upstreams = _mapping(plan.get("upstreams"), name="upstreams")
    matphys_repository = Path(upstreams["matphys_repository_path"]).resolve(strict=True)
    _require_clean_revision(
        matphys_repository,
        str(upstreams["matphys_revision"]),
        label="MatPhys",
    )
    simulator_path = matphys_repository / str(upstreams["simulator_relative_path"])
    _require(
        _file_sha256(simulator_path) == upstreams["simulator_sha256"],
        "MatPhys simulator source changed",
    )
    rgbench_repository = Path(upstreams["rgbench_repository_path"]).resolve(strict=True)
    _require_clean_revision(
        rgbench_repository,
        str(upstreams["rgbench_revision"]),
        label="RGBench",
    )

    cohort = _mapping(plan.get("cohort"), name="cohort")
    protocol_path = Path(cohort["protocol_path"]).resolve(strict=True)
    amendment_path = Path(cohort["amendment_path"]).resolve(strict=True)
    _require(
        _file_sha256(protocol_path) == cohort["protocol_sha256"], "protocol changed"
    )
    _require(
        _file_sha256(amendment_path) == cohort["amendment_sha256"], "amendment changed"
    )
    amended = load_rgbench_matphys_preaccess_amendment_v1(
        protocol_path,
        amendment_path,
    )
    cell = _mapping(cohort.get("source_cell"), name="source_cell")
    episode = load_rgbench_source_episode_index_v1(
        amended,
        Path(cohort["dataset_root"]),
        garment_id=str(cell["garment_id"]),
        action=str(cell["action"]),
        sample_id=str(cell["sample_id"]),
        camera_delay_s=float(cohort["camera_delay_s"]),
    )
    _require(
        episode.cell.data_subfolder == cell["data_subfolder"],
        "source cell path changed",
    )

    method = _mapping(plan.get("method"), name="method")
    frame_count = int(method["frame_count"])
    _require(
        3 <= frame_count <= len(episode.pcd_paths), "technical frame count is invalid"
    )
    initial = load_episode_world_points_v1(episode, 0)
    graph = build_rgbench_matphys_graph_v1(
        initial,
        episode.controller_points_m[0],
        node_count=int(method["node_count"]),
        total_mass_kg=float(method["total_mass_kg"]),
        object_radius_m=float(method["object_radius_m"]),
        object_max_neighbours=int(method["object_max_neighbours"]),
        controller_radius_m=float(method["controller_radius_m"]),
        controller_max_neighbours=int(method["controller_max_neighbours"]),
    )
    frame_times = episode.frame_times_s[:frame_count]
    frame_deltas = np.diff(frame_times)
    median_frame_delta = float(np.median(frame_deltas))
    integration_dt = float(method["integration_dt_s"])
    num_substeps = int(round(median_frame_delta / integration_dt))
    integrated_frame_delta = integration_dt * num_substeps
    _require(num_substeps > 0, "technical integration schedule is invalid")
    _require(
        float(np.max(np.abs(frame_deltas - integrated_frame_delta)))
        <= float(method["maximum_frame_interval_error_s"]),
        "PCD frame spacing exceeds the technical smoke tolerance",
    )

    runtime = _mapping(plan.get("runtime"), name="runtime")
    expected_python = Path(runtime["python_path"]).resolve(strict=True)
    _require(
        Path(sys.executable).resolve(strict=True) == expected_python,
        "technical Python executable changed",
    )

    import torch
    import warp as wp

    from bayesian_phystwin._phystwin_warp_backend import (
        load_official_spring_mass_module,
    )

    runtime_identity = _runtime_identity(torch)
    _require(runtime_identity == EXPECTED_RUNTIME, "technical runtime changed")
    _require(
        runtime.get("expected_versions") == EXPECTED_RUNTIME, "plan runtime changed"
    )
    device = str(runtime["device"])
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.cuda.set_device(torch.device(device))
    np.random.seed(int(method["seed"]))
    torch.manual_seed(int(method["seed"]))

    runtime_cfg = SimpleNamespace(
        device=device,
        use_graph=True,
        data_type="real",
        collision_learn=False,
        chamfer_weight=0.0,
        track_weight=0.0,
        acc_weight=0.0,
    )
    official = load_official_spring_mass_module(
        matphys_repository,
        runtime_config=runtime_cfg,
    )

    def tensor(values: np.ndarray, dtype: Any):
        return torch.as_tensor(values, dtype=dtype, device=device).contiguous()

    node_count = int(method["node_count"])
    initial_nodes = np.asarray(graph.vertices[:node_count], dtype=np.float32)
    controls = np.asarray(episode.controller_points_m[:frame_count], dtype=np.float32)
    gt_points = np.repeat(initial_nodes[None], frame_count, axis=0)
    gt_visible = np.ones((frame_count, node_count), dtype=np.int32)
    gt_motion = np.ones((frame_count - 1, node_count), dtype=np.int32)
    build_started = time.perf_counter()
    simulator = official.SpringMassSystemWarp(
        tensor(graph.vertices, torch.float32),
        tensor(graph.springs, torch.int32),
        tensor(graph.rest_lengths, torch.float32),
        tensor(graph.masses, torch.float32),
        dt=integration_dt,
        num_substeps=num_substeps,
        spring_Y=float(method["object_spring_y"]),
        collide_elas=float(method["ground_elasticity"]),
        collide_fric=float(method["ground_friction"]),
        dashpot_damping=float(method["dashpot_damping"]),
        drag_damping=float(method["drag_damping"]),
        collide_object_elas=float(method["object_elasticity"]),
        collide_object_fric=float(method["object_friction"]),
        collision_dist=float(method["collision_distance_m"]),
        num_object_points=node_count,
        num_surface_points=node_count,
        num_original_points=node_count,
        controller_points=tensor(controls, torch.float32),
        reverse_z=False,
        spring_Y_min=0.0,
        spring_Y_max=float(method["maximum_spring_y"]),
        gt_object_points=tensor(gt_points, torch.float32),
        gt_object_visibilities=tensor(gt_visible, torch.int32),
        gt_object_motions_valid=tensor(gt_motion, torch.int32),
        self_collision=bool(method["self_collision"]),
        disable_backward=True,
    )
    spring_y = np.full(
        len(graph.springs),
        float(method["object_spring_y"]),
        dtype=np.float32,
    )
    spring_y[graph.num_object_springs :] = float(method["controller_spring_y"])
    simulator.set_spring_Y(torch.log(tensor(spring_y, torch.float32)))
    simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)
    wp.synchronize()
    build_seconds = time.perf_counter() - build_started

    trajectory = [initial_nodes.copy()]
    rollout_started = time.perf_counter()
    for frame in range(1, frame_count):
        simulator.set_controller_target(frame, pure_inference=True)
        if simulator.object_collision_flag:
            simulator.update_collision_graph()
        wp.capture_launch(simulator.forward_graph)
        wp.synchronize()
        trajectory.append(_state_numpy(simulator, wp))
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
            pure_inference=True,
        )
    rollout_seconds = time.perf_counter() - rollout_started
    positions = np.asarray(trajectory, dtype=np.float32)
    _require(
        positions.shape == (frame_count, node_count, 3),
        "technical trajectory shape changed",
    )
    _require(np.all(np.isfinite(positions)), "technical trajectory is non-finite")
    displacement = np.linalg.norm(positions - positions[0:1], axis=2)
    maximum_displacement = float(np.max(displacement))
    _require(
        maximum_displacement >= float(method["minimum_motion_m"]),
        "technical trajectory has no measurable motion",
    )
    _require(
        maximum_displacement <= float(method["maximum_motion_m"]),
        "technical trajectory exceeds its stability bound",
    )
    _require(
        float(np.min(positions[..., 2])) >= float(method["minimum_z_m"]),
        "technical trajectory crossed the ground tolerance",
    )

    trajectory_path = output_root / "trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        vertices_m=positions,
        controller_points_m=controls,
        frame_times_s=frame_times,
    )
    input_files = {
        "protocol": protocol_path,
        "amendment": amendment_path,
        "initial_pcd": episode.pcd_paths[0],
        "calibration": episode.episode_dir
        / "calibration"
        / "world_to_camera_transform.json",
        "left_pose": episode.episode_dir
        / "joints"
        / "left_arm_joint_states_and_end_pose.csv",
        "right_pose": episode.episode_dir
        / "joints"
        / "right_arm_joint_states_and_end_pose.csv",
        "runner": runner_path,
        "matphys_simulator": simulator_path,
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_VERSION,
        "plan_id": plan["plan_id"],
        "passed": True,
        "technical_smoke_only": True,
        "scientific_source_gate_passed": False,
        "source_competence_claim_authorized": False,
        "target_authorized": False,
        "source_cell": dict(cell),
        "information_boundary": {
            "decoded_source_frame_indices": [0],
            "source_future_outcomes_scored": False,
            "target_payload_read": False,
            "target_outcomes_opened": False,
            "held_v8_accessed": False,
            "dlo4_dlo5_accessed": False,
        },
        "runtime": {
            **runtime_identity,
            "device": device,
            "cuda_device_name": torch.cuda.get_device_name(torch.device(device)),
            "build_seconds": build_seconds,
            "rollout_seconds": rollout_seconds,
        },
        "schedule": {
            "frame_count": frame_count,
            "median_frame_delta_s": median_frame_delta,
            "integration_dt_s": integration_dt,
            "num_substeps": num_substeps,
            "integrated_frame_delta_s": integrated_frame_delta,
            "maximum_frame_interval_error_s": float(
                np.max(np.abs(frame_deltas - integrated_frame_delta))
            ),
        },
        "graph": {
            "object_node_count": node_count,
            "object_spring_count": graph.num_object_springs,
            "controller_spring_count": len(graph.springs) - graph.num_object_springs,
            "object_component_count": spring_graph_component_count_v1(graph),
        },
        "trajectory": {
            "shape": list(positions.shape),
            "minimum_z_m": float(np.min(positions[..., 2])),
            "maximum_z_m": float(np.max(positions[..., 2])),
            "maximum_displacement_m": maximum_displacement,
            "trajectory_file_sha256": _file_sha256(trajectory_path),
        },
        "input_file_sha256": {
            name: _file_sha256(path) for name, path in input_files.items()
        },
        "pcd_filename_roster_sha256": _canonical_sha256(
            [path.name for path in episode.pcd_paths]
        ),
    }
    result["result_id"] = content_id(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan_path = args.execution_plan.resolve(strict=True)
    plan = _load_plan(plan_path, args.expected_plan_sha256)
    custody = _mapping(plan.get("custody"), name="custody")
    attempt_path = Path(custody["attempt_ledger_path"])
    output_root = Path(custody["output_root"])
    _require(not output_root.exists(), "technical smoke output root already exists")
    _consume_attempt(
        attempt_path,
        plan_id=str(plan["plan_id"]),
        output_root=output_root,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(plan, output_root)
    except Exception as error:
        failure = {
            "schema": RESULT_SCHEMA,
            "schema_version": RESULT_VERSION,
            "plan_id": plan["plan_id"],
            "passed": False,
            "technical_smoke_only": True,
            "scientific_source_gate_passed": False,
            "source_competence_claim_authorized": False,
            "target_authorized": False,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "traceback": traceback.format_exc(),
            "information_boundary": {
                "source_future_outcomes_scored": False,
                "target_payload_read": False,
                "target_outcomes_opened": False,
                "held_v8_accessed": False,
                "dlo4_dlo5_accessed": False,
            },
        }
        failure["result_id"] = content_id(failure)
        _write_json(output_root / "failure.json", failure)
        raise
    _write_json(output_root / "technical_smoke.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
