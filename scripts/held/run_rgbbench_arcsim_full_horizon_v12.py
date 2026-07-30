#!/usr/bin/env python3
"""Run the target-free RGBench ARCSim full-horizon qualification."""

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

from bayesian_phystwin.rgbench_arcsim import run_arcsim_fling
from bayesian_phystwin.rgbench_online_belief import sha256_file
from scripts.held.run_rgbbench_arcsim_competence_v8 import (
    _git_head,
    _load_case,
    _parameters,
    _require,
    _verify_source,
    _write_json_once,
)

PROTOCOL_ID = "rgbbench-arcsim-dirichlet-full-horizon-v12"
ARTIFACT_KIND = "RGBenchARCSimFullHorizonProtocol"
RESULT_PREFIX = "RGBenchARCSimFullHorizonQualification"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("single", "gate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--protocol", type=Path, required=True)
        subparser.add_argument("--rgbbench-root", type=Path, required=True)
        subparser.add_argument("--dataset-root", type=Path, required=True)
        subparser.add_argument("--arcsim-root", type=Path, required=True)
        subparser.add_argument("--arcsim-archive", type=Path, required=True)
        if command == "single":
            subparser.add_argument(
                "--replay-index",
                type=int,
                choices=(1, 2),
                required=True,
            )
            subparser.add_argument("--output", type=Path, required=True)
        else:
            subparser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("artifact_kind") == ARTIFACT_KIND,
        "ARCSim full-horizon protocol identity changed",
    )
    case = payload.get("qualification_case")
    gate = payload.get("qualification_gate")
    _require(isinstance(case, dict), "qualification case is missing")
    _require(isinstance(gate, dict), "qualification gate is missing")
    _require(
        int(gate["independent_replays"]) == 2,
        "ARCSim full-horizon protocol must require two independent replays",
    )
    duration_s = float(case["full_horizon_duration_s"])
    timestep_s = float(payload["physics"]["timestep_s"])
    step_count = int(round(duration_s / timestep_s))
    _require(
        math.isclose(duration_s / timestep_s, step_count, abs_tol=1e-12),
        "full horizon must contain an integer number of solver steps",
    )
    _require(
        step_count == int(gate["expected_step_count"]),
        "full-horizon step contract changed",
    )
    return payload


def _competence_shape(protocol: dict[str, Any]) -> dict[str, Any]:
    """Map the qualification schema onto the frozen source-verification helpers."""

    mapped = dict(protocol)
    mapped["competence_case"] = protocol["qualification_case"]
    mapped["competence_gate"] = protocol["qualification_gate"]
    return mapped


def _single(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    mapped = _competence_shape(protocol)
    rgbbench_root = args.rgbbench_root.resolve()
    dataset_root = args.dataset_root.resolve()
    arcsim_root = args.arcsim_root.resolve()
    paths = _verify_source(
        mapped,
        rgbbench_root=rgbbench_root,
        dataset_root=dataset_root,
        arcsim_root=arcsim_root,
        arcsim_archive=args.arcsim_archive.resolve(),
    )
    vertices, triangles, controller = _load_case(mapped, paths)
    case = protocol["qualification_case"]
    duration_s = float(case["full_horizon_duration_s"])
    output = args.output.resolve()
    _require(output.suffix == ".npy", "single replay output must be .npy")
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rollout = run_arcsim_fling(
        source_mesh_path=paths["mesh"],
        initial_vertices_m=vertices,
        controller=controller,
        parameters=_parameters(mapped),
        duration_s=duration_s,
        initial_pose_xyz_wxyz=tuple(
            float(value) for value in case["initial_pose_xyz_wxyz"]
        ),
        workspace=output.with_suffix(".workspace"),
        arcsim_root=arcsim_root,
        timeout_s=float(protocol["qualification_gate"]["maximum_replay_elapsed_s"]),
    )
    np.save(output, rollout.final_vertices_m, allow_pickle=False)
    displacement = np.linalg.norm(rollout.final_vertices_m - vertices, axis=1)
    _write_json_once(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact_kind": f"{RESULT_PREFIX}Replay",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_commit": _git_head(Path(__file__).resolve().parents[2]),
            "rgbbench_commit": _git_head(rgbbench_root),
            "arcsim_release": protocol["upstream"]["arcsim_release"],
            "arcsim_executable_sha256": sha256_file(paths["executable"]),
            "replay_index": int(args.replay_index),
            "duration_s": duration_s,
            "step_count": rollout.step_count,
            "elapsed_s": rollout.elapsed_s,
            "vertex_count": int(len(rollout.final_vertices_m)),
            "face_count": int(len(triangles)),
            "all_vertices_finite": bool(np.all(np.isfinite(rollout.final_vertices_m))),
            "maximum_pin_target_error_m": rollout.maximum_pin_target_error_m,
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
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": str(protocol["upstream"]["omp_proc_bind"]),
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
            "--arcsim-root",
            str(args.arcsim_root.resolve()),
            "--arcsim-archive",
            str(args.arcsim_archive.resolve()),
            "--replay-index",
            str(index),
            "--output",
            str(output),
        ]
        with (output_root / f"replay_{index}.log").open("x", encoding="utf-8") as log:
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
                "artifact_kind": f"{RESULT_PREFIX}Gate",
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": sha256_file(protocol_path),
                "status": "technical_failure",
                "qualification_gate_passed": False,
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
    gate = protocol["qualification_gate"]
    expected_shape = (int(gate["expected_vertex_count"]), 3)
    maximum_pin_error = max(
        float(item["maximum_pin_target_error_m"]) for item in metadata
    )
    minimum_motion = min(float(item["mean_vertex_displacement_m"]) for item in metadata)
    maximum_elapsed = max(float(item["elapsed_s"]) for item in metadata)
    checks = {
        "both_complete": complete,
        "all_vertices_finite": all(
            bool(np.all(np.isfinite(array))) for array in arrays
        ),
        "vertex_count_preserved": all(
            tuple(array.shape) == expected_shape for array in arrays
        ),
        "value_identical_final_vertices": np.array_equal(arrays[0], arrays[1]),
        "byte_identical_final_vertices": (
            sha256_file(outputs[0]) == sha256_file(outputs[1])
        ),
        "expected_step_count": all(
            int(item["step_count"]) == int(gate["expected_step_count"])
            for item in metadata
        ),
        "pin_tracking_within_limit": (
            maximum_pin_error <= float(gate["maximum_pin_target_error_m"])
        ),
        "nontrivial_cloth_motion": (
            minimum_motion >= float(gate["minimum_mean_vertex_displacement_m"])
        ),
        "runtime_within_limit": (
            maximum_elapsed <= float(gate["maximum_replay_elapsed_s"])
        ),
    }
    passed = all(checks.values())
    _write_json_once(
        output_root / "gate.json",
        {
            "schema_version": 1,
            "artifact_kind": f"{RESULT_PREFIX}Gate",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_commit": _git_head(Path(__file__).resolve().parents[2]),
            "status": "passed" if passed else "gate_failed",
            "qualification_gate_passed": passed,
            "checks": checks,
            "replay_return_codes": return_codes,
            "replay_elapsed_seconds": elapsed_seconds,
            "maximum_replay_elapsed_s": maximum_elapsed,
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
