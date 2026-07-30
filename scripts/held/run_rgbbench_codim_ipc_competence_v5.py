#!/usr/bin/env python3
"""Run the target-free RGBench Codim-IPC numerical competence gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.rgbench_codim_ipc import (
    CodimIPCClothParameters,
    run_codim_ipc_fling,
)
from bayesian_phystwin.rgbench_libuipc import (
    FlingPinController,
    load_rgbbench_position_trajectory,
    transform_vertices_wxyz,
    triangle_mesh_area_m2,
)
from bayesian_phystwin.rgbench_online_belief import (
    load_obj_triangles,
    sha256_file,
)

SUPPORTED_PROTOCOLS = {
    "rgbbench-codim-ipc-competence-v5": "RGBenchCodimIPCCompetenceProtocol",
    "rgbbench-codim-ipc-cholmod-v6": "RGBenchCodimIPCCholmodProtocol",
    "rgbbench-codim-ipc-full-horizon-v7": (
        "RGBenchCodimIPCFullHorizonProtocol"
    ),
}
SOURCE_DIGEST_KEYS = {
    "mesh": "mesh_sha256",
    "left": "left_trajectory_sha256",
    "right": "right_trajectory_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single")
    single.add_argument("--protocol", type=Path, required=True)
    single.add_argument("--rgbbench-root", type=Path, required=True)
    single.add_argument("--dataset-root", type=Path, required=True)
    single.add_argument("--codim-root", type=Path, required=True)
    single.add_argument("--replay-index", type=int, choices=(1, 2), required=True)
    single.add_argument("--output", type=Path, required=True)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--protocol", type=Path, required=True)
    gate.add_argument("--rgbbench-root", type=Path, required=True)
    gate.add_argument("--dataset-root", type=Path, required=True)
    gate.add_argument("--codim-root", type=Path, required=True)
    gate.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol_id = payload.get("protocol_id")
    _require(
        isinstance(payload, dict)
        and SUPPORTED_PROTOCOLS.get(protocol_id) == payload.get("artifact_kind"),
        "competence protocol identity changed",
    )
    gate = _gate_spec(payload)
    _require(
        int(gate["independent_replays"]) == 2,
        "Codim-IPC protocol must require two replays",
    )
    return payload


def _case_spec(protocol: dict[str, Any]) -> dict[str, Any]:
    case = protocol.get("qualification_case", protocol.get("competence_case"))
    _require(isinstance(case, dict), "case specification is missing")
    return case


def _gate_spec(protocol: dict[str, Any]) -> dict[str, Any]:
    gate = protocol.get("qualification_gate", protocol.get("competence_gate"))
    _require(isinstance(gate, dict), "gate specification is missing")
    return gate


def _duration_s(protocol: dict[str, Any]) -> float:
    case = _case_spec(protocol)
    values = [
        case[key]
        for key in ("smoke_duration_s", "full_horizon_duration_s")
        if key in case
    ]
    _require(len(values) == 1, "case must bind exactly one simulation duration")
    duration = float(values[0])
    _require(math.isfinite(duration) and duration > 0.0, "invalid duration")
    return duration


def _artifact_prefix(protocol: dict[str, Any]) -> str:
    prefix = protocol.get(
        "result_artifact_prefix", "RGBenchCodimIPCCompetence"
    )
    _require(isinstance(prefix, str) and prefix, "invalid result artifact prefix")
    return prefix


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_source(
    protocol: dict[str, Any],
    *,
    rgbbench_root: Path,
    dataset_root: Path,
    codim_root: Path,
) -> dict[str, Path]:
    upstream = protocol["upstream"]
    case = _case_spec(protocol)
    implementation_root = Path(__file__).resolve().parents[2]
    for relative_path, expected_sha256 in upstream.get(
        "implementation_artifact_sha256s", {}
    ).items():
        artifact_path = implementation_root / relative_path
        _require(
            artifact_path.is_file(),
            f"missing implementation artifact: {artifact_path}",
        )
        _require(
            sha256_file(artifact_path) == expected_sha256,
            f"implementation artifact changed: {relative_path}",
        )
    for absolute_path, expected_sha256 in upstream.get(
        "runtime_dependency_sha256s", {}
    ).items():
        dependency_path = Path(absolute_path)
        _require(
            dependency_path.is_file(),
            f"missing runtime dependency: {dependency_path}",
        )
        _require(
            sha256_file(dependency_path) == expected_sha256,
            f"runtime dependency changed: {dependency_path}",
        )
    _require(
        _git_head(rgbbench_root) == upstream["rgbbench_commit"],
        "RGBench checkout commit changed",
    )
    _require(
        _git_head(codim_root) == upstream["codim_ipc_commit"],
        "Codim-IPC checkout commit changed",
    )
    main_config = rgbbench_root / "configs" / "main.yaml"
    cloth_config = (
        rgbbench_root / "configs" / "cloth_params" / "green_tshirt_10k.yaml"
    )
    _require(
        sha256_file(main_config) == upstream["rgbbench_main_config_sha256"],
        "RGBench main configuration changed",
    )
    _require(
        sha256_file(cloth_config) == upstream["rgbbench_cloth_config_sha256"],
        "RGBench cloth configuration changed",
    )
    patched_boundary = codim_root / "Library" / "FEM" / "BOUNDARY_CONDITION.h"
    _require(
        sha256_file(patched_boundary)
        == upstream["codim_patched_boundary_condition_sha256"],
        "Codim-IPC boundary patch changed",
    )
    for relative_path, expected_sha256 in upstream.get(
        "codim_source_sha256s", {}
    ).items():
        source_path = codim_root / relative_path
        _require(source_path.is_file(), f"missing Codim-IPC source: {source_path}")
        _require(
            sha256_file(source_path) == expected_sha256,
            f"Codim-IPC source changed: {relative_path}",
        )
    module_root = codim_root / upstream.get(
        "codim_runtime_module_relative_directory", "build"
    )
    build_module = list(module_root.glob("JGSL*.so"))
    _require(len(build_module) == 1, "Codim-IPC runtime module is missing or ambiguous")
    expected_module_sha256 = upstream.get("codim_runtime_module_sha256")
    if expected_module_sha256 is not None:
        _require(
            sha256_file(build_module[0]) == expected_module_sha256,
            "Codim-IPC runtime module changed",
        )
    capture = dataset_root / case["data_subfolder"]
    paths = {
        "mesh": dataset_root / "meshes" / case["mesh_relative_path"],
        "left": capture / case["left_trajectory_relative_path"],
        "right": capture / case["right_trajectory_relative_path"],
    }
    for name, path in paths.items():
        _require(path.is_file(), f"missing frozen source {name}: {path}")
        _require(
            sha256_file(path) == case[SOURCE_DIGEST_KEYS[name]],
            f"frozen source {name} changed",
        )
    paths["module_root"] = module_root
    return paths


def _parameters(payload: dict[str, Any]) -> CodimIPCClothParameters:
    physics = payload["physics"]
    return CodimIPCClothParameters(
        timestep_s=float(physics["timestep_s"]),
        youngs_modulus_pa=float(physics["youngs_modulus_pa"]),
        poisson_ratio=float(physics["poisson_ratio"]),
        volume_density_kg_m3=float(physics["volume_density_kg_m3"]),
        thickness_m=float(physics["thickness_m"]),
        bending_stiffness_multiplier=float(
            physics["bending_stiffness_multiplier"]
        ),
        newton_tolerance=float(physics["newton_tolerance"]),
        contact_thickness_m=float(physics["contact_thickness_m"]),
        collision_enabled=bool(physics["collision_enabled"]),
    )


def _load_case(
    protocol: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, FlingPinController]:
    case = _case_spec(protocol)
    vertices, triangles = load_obj_triangles(paths["mesh"])
    area = triangle_mesh_area_m2(vertices, triangles)
    _require(
        math.isclose(
            area,
            float(case["mesh_surface_area_m2"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "bound source mesh area changed",
    )
    implied_thickness = float(case["source_mass_kg"]) / (
        float(protocol["physics"]["volume_density_kg_m3"]) * area
    )
    _require(
        math.isclose(
            implied_thickness,
            float(protocol["physics"]["thickness_m"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "shell thickness no longer matches source mass and density",
    )
    vertices = transform_vertices_wxyz(
        vertices,
        tuple(float(value) for value in case["initial_pose_xyz_wxyz"]),
    )
    pin_indices = tuple(int(value) for value in case["pin_indices"])
    left = load_rgbbench_position_trajectory(
        paths["left"],
        base_translation_m=tuple(
            float(value) for value in case["left_base_translation_m"]
        ),
    )
    right = load_rgbbench_position_trajectory(
        paths["right"],
        base_translation_m=tuple(
            float(value) for value in case["right_base_translation_m"]
        ),
    )
    return (
        vertices,
        triangles,
        FlingPinController(
            pin_indices=pin_indices,
            initial_positions_m=vertices[np.asarray(pin_indices)],
            left=left,
            right=right,
            prepare_time_s=float(case["prepare_time_s"]),
            wait_time_s=float(case["wait_time_s"]),
        ),
    )


def _single(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    rgbbench_root = args.rgbbench_root.resolve()
    dataset_root = args.dataset_root.resolve()
    codim_root = args.codim_root.resolve()
    paths = _verify_source(
        protocol,
        rgbbench_root=rgbbench_root,
        dataset_root=dataset_root,
        codim_root=codim_root,
    )
    vertices, triangles, controller = _load_case(protocol, paths)
    duration_s = _duration_s(protocol)
    output = args.output.resolve()
    _require(output.suffix == ".npy", "single replay output must be .npy")
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rollout = run_codim_ipc_fling(
        vertices_m=vertices,
        triangles=triangles,
        controller=controller,
        parameters=_parameters(protocol),
        duration_s=duration_s,
        workspace=output.with_suffix(".workspace"),
        module_root=paths["module_root"],
        python_root=codim_root / "Python",
        expected_linear_solver_backend=protocol["upstream"].get(
            "expected_linear_solver_backend"
        ),
    )
    np.save(output, rollout.final_vertices_m, allow_pickle=False)
    displacement = np.linalg.norm(rollout.final_vertices_m - vertices, axis=1)
    _write_json_once(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact_kind": f"{_artifact_prefix(protocol)}Replay",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_commit": _git_head(Path(__file__).resolve().parents[2]),
            "rgbbench_commit": _git_head(rgbbench_root),
            "codim_ipc_commit": _git_head(codim_root),
            "codim_runtime_module_sha256": sha256_file(
                next(paths["module_root"].glob("JGSL*.so"))
            ),
            "linear_solver_backend": protocol["upstream"].get(
                "expected_linear_solver_backend",
                protocol["upstream"]["linear_solver"],
            ),
            "replay_index": int(args.replay_index),
            "duration_s": duration_s,
            "step_count": rollout.step_count,
            "total_newton_iterations": rollout.total_newton_iterations,
            "vertex_count": int(len(rollout.final_vertices_m)),
            "face_count": int(len(triangles)),
            "all_vertices_finite": bool(
                np.all(np.isfinite(rollout.final_vertices_m))
            ),
            "maximum_pin_target_error_m": (
                rollout.maximum_pin_target_error_m
            ),
            "mean_vertex_displacement_m": float(np.mean(displacement)),
            "maximum_vertex_displacement_m": float(np.max(displacement)),
            "final_vertices_sha256": sha256_file(output),
            "point_cloud_filenames_read": False,
            "point_cloud_coordinates_read": False,
            "source_accuracy_outcomes_read": False,
            "future_object_outcomes_read": False,
            "known_future_actuator_trajectory_read": True,
        },
    )
    return 0


def _gate(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    output_root = args.output_root.resolve()
    _require(not output_root.exists(), f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    outputs = [output_root / f"replay_{index}.npy" for index in (1, 2)]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(protocol["upstream"]["omp_num_threads"]),
            "OMP_PROC_BIND": str(
                protocol["upstream"].get("omp_proc_bind", "false")
            ),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return_codes: list[int] = []
    elapsed_seconds: list[float] = []
    for index, output in enumerate(outputs, start=1):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "single",
            "--protocol",
            str(protocol_path),
            "--rgbbench-root",
            str(args.rgbbench_root.resolve()),
            "--dataset-root",
            str(args.dataset_root.resolve()),
            "--codim-root",
            str(args.codim_root.resolve()),
            "--replay-index",
            str(index),
            "--output",
            str(output),
        ]
        with (output_root / f"replay_{index}.log").open(
            "x", encoding="utf-8"
        ) as log:
            started = time.monotonic()
            completed = subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            elapsed_seconds.append(float(time.monotonic() - started))
        return_codes.append(int(completed.returncode))
        if completed.returncode != 0:
            break
    complete = len(return_codes) == 2 and all(code == 0 for code in return_codes)
    if not complete:
        _write_json_once(
            output_root / "gate.json",
            {
                "schema_version": 1,
                "artifact_kind": f"{_artifact_prefix(protocol)}Gate",
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": sha256_file(protocol_path),
                "status": "technical_failure",
                "competence_gate_passed": False,
                "replay_return_codes": return_codes,
                "replay_elapsed_seconds": elapsed_seconds,
                "point_cloud_filenames_read": False,
                "point_cloud_coordinates_read": False,
                "source_accuracy_outcomes_read": False,
                "future_object_outcomes_read": False,
            },
        )
        return 3
    arrays = [np.load(path, allow_pickle=False) for path in outputs]
    metadata = [
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        for path in outputs
    ]
    byte_identical = sha256_file(outputs[0]) == sha256_file(outputs[1])
    value_identical = np.array_equal(arrays[0], arrays[1])
    finite = all(np.all(np.isfinite(array)) for array in arrays)
    count_preserved = arrays[0].shape == arrays[1].shape
    maximum_pin_error = max(
        float(item["maximum_pin_target_error_m"]) for item in metadata
    )
    minimum_motion = min(
        float(item["mean_vertex_displacement_m"]) for item in metadata
    )
    gate = _gate_spec(protocol)
    checks = {
        "both_complete": complete,
        "all_vertices_finite": finite,
        "vertex_count_preserved": count_preserved,
        "value_identical_final_vertices": value_identical,
        "byte_identical_final_vertices": byte_identical,
        "pin_tracking_within_limit": (
            maximum_pin_error <= float(gate["maximum_pin_target_error_m"])
        ),
        "nontrivial_cloth_motion": (
            minimum_motion >= float(gate["minimum_mean_vertex_displacement_m"])
        ),
    }
    if "expected_step_count" in gate:
        checks["expected_step_count"] = all(
            int(item["step_count"]) == int(gate["expected_step_count"])
            for item in metadata
        )
    if "maximum_replay_elapsed_s" in gate:
        checks["runtime_within_limit"] = max(elapsed_seconds) <= float(
            gate["maximum_replay_elapsed_s"]
        )
    passed = all(checks.values())
    _write_json_once(
        output_root / "gate.json",
        {
            "schema_version": 1,
            "artifact_kind": f"{_artifact_prefix(protocol)}Gate",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_commit": _git_head(Path(__file__).resolve().parents[2]),
            "status": "passed" if passed else "gate_failed",
            "competence_gate_passed": passed,
            "checks": checks,
            "replay_return_codes": return_codes,
            "replay_elapsed_seconds": elapsed_seconds,
            "maximum_replay_elapsed_s": max(elapsed_seconds),
            "maximum_pin_target_error_m": maximum_pin_error,
            "minimum_mean_vertex_displacement_m": minimum_motion,
            "replay_sha256s": [sha256_file(path) for path in outputs],
            "replay_metadata_sha256s": [
                sha256_file(path.with_suffix(".json")) for path in outputs
            ],
            "passed_gate_action": protocol["information_boundary"][
                "passed_gate_action"
            ],
            "failed_gate_action": protocol["information_boundary"][
                "failed_gate_action"
            ],
            "point_cloud_filenames_read": False,
            "point_cloud_coordinates_read": False,
            "source_accuracy_outcomes_read": False,
            "future_object_outcomes_read": False,
        },
    )
    return 0 if passed else 2


def main() -> None:
    args = _parse_args()
    if args.command == "single":
        raise SystemExit(_single(args))
    raise SystemExit(_gate(args))


if __name__ == "__main__":
    main()
