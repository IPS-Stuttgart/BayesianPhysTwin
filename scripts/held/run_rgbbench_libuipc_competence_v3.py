#!/usr/bin/env python3
"""Run the source-only RGBench LibuIPC competence gate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.rgbench_libuipc import (
    FlingPinController,
    LibuIPCClothParameters,
    load_rgbbench_position_trajectory,
    run_libuipc_fling,
    transform_vertices_wxyz,
)
from bayesian_phystwin.rgbench_online_belief import (
    load_obj_triangles,
    sha256_file,
)

PROTOCOL_ID = "rgbbench-libuipc-competence-v3"
ARTIFACT_KIND = "RGBenchLibuIPCCompetenceProtocol"
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
    single.add_argument("--replay-index", type=int, choices=(1, 2), required=True)
    single.add_argument("--output", type=Path, required=True)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--protocol", type=Path, required=True)
    gate.add_argument("--rgbbench-root", type=Path, required=True)
    gate.add_argument("--dataset-root", type=Path, required=True)
    gate.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("artifact_kind") == ARTIFACT_KIND
        and payload.get("protocol_id") == PROTOCOL_ID,
        "competence protocol identity changed",
    )
    _require(
        payload["competence_gate"]["independent_replays"] == 2,
        "competence protocol must require two replays",
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
    case = protocol["competence_case"]
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
    case = protocol["competence_case"]
    vertices, triangles = load_obj_triangles(sources["mesh"])
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
    output = args.output.resolve()
    _require(output.suffix == ".npy", "single replay output must be .npy")
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = output.with_suffix(".workspace")
    final = run_libuipc_fling(
        vertices_m=vertices,
        triangles=triangles,
        controller=controller,
        parameters=_parameters(protocol),
        duration_s=float(case["smoke_duration_s"]),
        workspace=workspace,
    )
    np.save(output, final, allow_pickle=False)
    _write_json_once(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchLibuIPCCompetenceReplay",
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
            "vertex_count": int(len(final)),
            "face_count": int(len(triangles)),
            "all_vertices_finite": bool(np.all(np.isfinite(final))),
            "final_vertices_sha256": sha256_file(output),
            "point_cloud_filenames_read": False,
            "point_cloud_coordinates_read": False,
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
    for index, output in enumerate(outputs, start=1):
        subprocess.run(
            [
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
                str(index),
                "--output",
                str(output),
            ],
            check=True,
        )
    arrays = [np.load(path, allow_pickle=False) for path in outputs]
    byte_identical = sha256_file(outputs[0]) == sha256_file(outputs[1])
    value_identical = np.array_equal(arrays[0], arrays[1])
    finite = all(np.all(np.isfinite(array)) for array in arrays)
    count_preserved = arrays[0].shape == arrays[1].shape
    passed = bool(byte_identical and value_identical and finite and count_preserved)
    _write_json_once(
        output_root / "gate.json",
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchLibuIPCCompetenceGate",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": sha256_file(protocol_path),
            "replay_count": len(outputs),
            "both_complete": True,
            "all_vertices_finite": finite,
            "vertex_count_preserved": count_preserved,
            "value_identical_final_vertices": value_identical,
            "byte_identical_final_vertices": byte_identical,
            "competence_gate_passed": passed,
            "failed_gate_action": protocol["information_boundary"][
                "failed_gate_action"
            ],
            "replay_sha256s": [sha256_file(path) for path in outputs],
            "point_cloud_filenames_read": False,
            "point_cloud_coordinates_read": False,
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
