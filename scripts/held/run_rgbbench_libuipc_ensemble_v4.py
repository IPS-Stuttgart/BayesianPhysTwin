#!/usr/bin/env python3
"""Run target-free full-horizon RGBench LibuIPC qualification."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.rgbench_libuipc import (
    FlingPinController,
    LibuIPCClothParameters,
    load_rgbbench_position_trajectory,
    run_libuipc_fling,
    summarize_independent_replays,
    transform_vertices_wxyz,
    triangle_mesh_area_m2,
)
from bayesian_phystwin.rgbench_online_belief import (
    load_obj_triangles,
    sha256_file,
)

PROTOCOL_ID = "rgbbench-libuipc-ensemble-v4"
ARTIFACT_KIND = "RGBenchLibuIPCEnsembleQualificationProtocol"
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
    single.add_argument("--replay-index", type=int, required=True)
    single.add_argument("--output", type=Path, required=True)
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--protocol", type=Path, required=True)
    qualify.add_argument("--rgbbench-root", type=Path, required=True)
    qualify.add_argument("--dataset-root", type=Path, required=True)
    qualify.add_argument("--physical-gpu-index", type=int, required=True)
    qualify.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("artifact_kind") == ARTIFACT_KIND
        and payload.get("protocol_id") == PROTOCOL_ID,
        "ensemble qualification protocol identity changed",
    )
    _require(
        int(payload["qualification_gate"]["independent_replays"]) >= 3,
        "ensemble qualification requires at least three replays",
    )
    return payload


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
) -> dict[str, Path]:
    upstream = protocol["upstream"]
    case = protocol["qualification_case"]
    _require(
        _git_head(rgbbench_root) == upstream["rgbbench_commit"],
        "RGBench checkout commit changed",
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
    _require(
        importlib.metadata.version("pyuipc") == upstream["pyuipc_version"],
        "pyuipc version changed",
    )
    return paths


def _parameters(payload: dict[str, Any]) -> LibuIPCClothParameters:
    physics = payload["physics"]
    return LibuIPCClothParameters(
        timestep_s=float(physics["timestep_s"]),
        youngs_modulus_pa=float(physics["youngs_modulus_pa"]),
        poisson_ratio=float(physics["poisson_ratio"]),
        volume_density_kg_m3=float(physics["volume_density_kg_m3"]),
        thickness_m=float(physics["thickness_m"]),
        bending_stiffness=float(physics["bending_stiffness"]),
        friction_coefficient=float(physics["friction_coefficient"]),
        contact_distance_m=float(physics["contact_distance_m"]),
        contact_resistance=float(physics["contact_resistance"]),
        constraint_strength_ratio=float(physics["constraint_strength_ratio"]),
    )


def _load_case(
    protocol: dict[str, Any],
    sources: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, FlingPinController]:
    case = protocol["qualification_case"]
    vertices, triangles = load_obj_triangles(sources["mesh"])
    mesh_area = triangle_mesh_area_m2(vertices, triangles)
    _require(
        math.isclose(
            mesh_area,
            float(case["mesh_surface_area_m2"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "bound source mesh area changed",
    )
    implied_thickness = float(case["source_mass_kg"]) / (
        float(protocol["physics"]["volume_density_kg_m3"]) * mesh_area
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
        sources["left"],
        base_translation_m=tuple(
            float(value) for value in case["left_base_translation_m"]
        ),
    )
    right = load_rgbbench_position_trajectory(
        sources["right"],
        base_translation_m=tuple(
            float(value) for value in case["right_base_translation_m"]
        ),
    )
    controller = FlingPinController(
        pin_indices=pin_indices,
        initial_positions_m=vertices[np.asarray(pin_indices)],
        left=left,
        right=right,
        prepare_time_s=float(case["prepare_time_s"]),
        wait_time_s=float(case["wait_time_s"]),
    )
    return vertices, triangles, controller


def _single(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    rgbbench_root = args.rgbbench_root.resolve()
    dataset_root = args.dataset_root.resolve()
    sources = _verify_source(
        protocol,
        rgbbench_root=rgbbench_root,
        dataset_root=dataset_root,
    )
    vertices, triangles, controller = _load_case(protocol, sources)
    case = protocol["qualification_case"]
    duration_s = float(case["full_horizon_duration_s"])
    output = args.output.resolve()
    _require(output.suffix == ".npy", "single replay output must be .npy")
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = output.with_suffix(".workspace")
    start_time = time.monotonic()
    final = run_libuipc_fling(
        vertices_m=vertices,
        triangles=triangles,
        controller=controller,
        parameters=_parameters(protocol),
        duration_s=duration_s,
        workspace=workspace,
    )
    elapsed_s = time.monotonic() - start_time
    np.save(output, final, allow_pickle=False)
    pins = np.asarray(controller.pin_indices)
    target = controller.targets_at(duration_s)
    pin_errors = np.linalg.norm(final[pins] - target, axis=1)
    displacement = np.linalg.norm(final - vertices, axis=1)
    _write_json_once(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchLibuIPCEnsembleReplay",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_commit": _git_head(
                Path(__file__).resolve().parents[2]
            ),
            "rgbbench_commit": _git_head(rgbbench_root),
            "dataset_revision": protocol["upstream"]["dataset_revision"],
            "libuipc_commit": protocol["upstream"]["libuipc_commit"],
            "pyuipc_version": importlib.metadata.version("pyuipc"),
            "replay_index": int(args.replay_index),
            "duration_s": duration_s,
            "elapsed_s": elapsed_s,
            "vertex_count": int(len(final)),
            "face_count": int(len(triangles)),
            "all_vertices_finite": bool(np.all(np.isfinite(final))),
            "pin_target_errors_m": pin_errors.tolist(),
            "maximum_pin_target_error_m": float(np.max(pin_errors)),
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


def _gpu_compute_pids(index: int) -> set[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        int(line.strip())
        for line in output.splitlines()
        if line.strip().isdigit()
    }


def _gpu_uuid(index: int) -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def _run_monitored_replay(
    command: list[str],
    *,
    gpu_index: int,
    log_path: Path,
) -> tuple[int, int]:
    _require(
        not _gpu_compute_pids(gpu_index),
        f"physical GPU {gpu_index} is not exclusive before replay",
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    contaminating_pids: set[int] = set()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            time.sleep(0.25)
            contaminating_pids.update(
                _gpu_compute_pids(gpu_index) - {process.pid}
            )
        return_code = int(process.wait())
    contaminating_pids.update(_gpu_compute_pids(gpu_index))
    return return_code, len(contaminating_pids)


def _qualification(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    output_root = args.output_root.resolve()
    _require(not output_root.exists(), f"refusing to overwrite {output_root}")
    _require(args.physical_gpu_index >= 0, "physical GPU index must be nonnegative")
    _require(
        not _gpu_compute_pids(args.physical_gpu_index),
        f"physical GPU {args.physical_gpu_index} is not initially exclusive",
    )
    output_root.mkdir(parents=True)
    replay_count = int(protocol["qualification_gate"]["independent_replays"])
    outputs: list[Path] = []
    contamination_counts: list[int] = []
    return_codes: list[int] = []
    for replay_index in range(1, replay_count + 1):
        output = output_root / f"replay_{replay_index}.npy"
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
            "--replay-index",
            str(replay_index),
            "--output",
            str(output),
        ]
        return_code, contamination_count = _run_monitored_replay(
            command,
            gpu_index=args.physical_gpu_index,
            log_path=output_root / f"replay_{replay_index}.log",
        )
        return_codes.append(return_code)
        contamination_counts.append(contamination_count)
        if return_code != 0 or contamination_count != 0:
            break
        outputs.append(output)

    complete = len(outputs) == replay_count
    exclusive = all(count == 0 for count in contamination_counts)
    if not complete or not exclusive:
        _write_json_once(
            output_root / "qualification.json",
            {
                "schema_version": 1,
                "artifact_kind": "RGBenchLibuIPCEnsembleQualification",
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": sha256_file(protocol_path),
                "status": "technical_failure",
                "qualification_passed": False,
                "physical_gpu_index": args.physical_gpu_index,
                "gpu_uuid": _gpu_uuid(args.physical_gpu_index),
                "replay_return_codes": return_codes,
                "contaminating_process_counts": contamination_counts,
                "completed_replays": len(outputs),
                "required_replays": replay_count,
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
    summary = summarize_independent_replays(arrays)
    ensemble_mean = output_root / "ensemble_endpoint_mean.npy"
    ensemble_variance = output_root / "ensemble_endpoint_variance_m2.npy"
    np.save(ensemble_mean, summary.mean_vertices_m, allow_pickle=False)
    np.save(ensemble_variance, summary.variance_m2, allow_pickle=False)
    gate = protocol["qualification_gate"]
    maximum_pin_error = max(
        float(item["maximum_pin_target_error_m"]) for item in metadata
    )
    minimum_mean_displacement = min(
        float(item["mean_vertex_displacement_m"]) for item in metadata
    )
    checks = {
        "exclusive_gpu_observation": exclusive,
        "all_replays_complete": complete,
        "all_vertices_finite": all(
            bool(item["all_vertices_finite"]) for item in metadata
        ),
        "vertex_count_preserved": len({item["vertex_count"] for item in metadata})
        == 1,
        "pairwise_rmse_within_limit": (
            summary.maximum_pairwise_rmse_m
            <= float(gate["maximum_pairwise_endpoint_rmse_m"])
        ),
        "coordinate_difference_within_limit": (
            summary.maximum_pairwise_coordinate_difference_m
            <= float(gate["maximum_pairwise_coordinate_difference_m"])
        ),
        "pin_tracking_within_limit": (
            maximum_pin_error <= float(gate["maximum_pin_target_error_m"])
        ),
        "nontrivial_cloth_motion": (
            minimum_mean_displacement
            >= float(gate["minimum_mean_vertex_displacement_m"])
        ),
    }
    passed = all(checks.values())
    _write_json_once(
        output_root / "qualification.json",
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchLibuIPCEnsembleQualification",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_commit": _git_head(
                Path(__file__).resolve().parents[2]
            ),
            "status": "passed" if passed else "gate_failed",
            "qualification_passed": passed,
            "physical_gpu_index": args.physical_gpu_index,
            "gpu_uuid": _gpu_uuid(args.physical_gpu_index),
            "replay_count": replay_count,
            "checks": checks,
            "maximum_pairwise_endpoint_rmse_m": (
                summary.maximum_pairwise_rmse_m
            ),
            "maximum_pairwise_coordinate_difference_m": (
                summary.maximum_pairwise_coordinate_difference_m
            ),
            "maximum_pin_target_error_m": maximum_pin_error,
            "minimum_mean_vertex_displacement_m": minimum_mean_displacement,
            "mean_replay_variance_m2": float(np.mean(summary.variance_m2)),
            "maximum_replay_variance_m2": float(np.max(summary.variance_m2)),
            "ensemble_mean_sha256": sha256_file(ensemble_mean),
            "ensemble_variance_sha256": sha256_file(ensemble_variance),
            "replay_sha256s": [sha256_file(path) for path in outputs],
            "replay_metadata_sha256s": [
                sha256_file(path.with_suffix(".json")) for path in outputs
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
    raise SystemExit(_qualification(args))


if __name__ == "__main__":
    main()
