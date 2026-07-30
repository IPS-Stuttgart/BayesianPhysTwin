#!/usr/bin/env python3
"""Run target-free RGBench v2 physical-backend preflight."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.rgbench_isotropic_mesh import (
    RGBenchIsotropicMeshArtifact,
    RGBenchIsotropicMeshManifest,
    load_isotropic_mesh_manifest,
    write_json_once,
)
from bayesian_phystwin.rgbench_online_belief import (
    evaluation_pcd_paths,
    force_pybullet_direct_connection,
    load_obj_triangles,
    sha256_file,
)
from bayesian_phystwin.rgbench_protocol import (
    ACTIONS,
    PAPER_GARMENTS,
    RGBENCH_COMMIT,
)

METHOD_ID = "rgbbench-isotropic-physical-v2"
SIMULATOR = "pybullet"
MODE = "fixed_point"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--benchmark-root", type=Path, required=True)
    simulate.add_argument("--dataset-root", type=Path, required=True)
    simulate.add_argument("--dataset-manifest", type=Path, required=True)
    simulate.add_argument("--mesh-manifest", type=Path, required=True)
    simulate.add_argument("--case-id", required=True)
    simulate.add_argument("--replay-index", type=int, choices=(1, 2), required=True)
    simulate.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--mesh-manifest", type=Path, required=True)
    verify.add_argument("--metadata", type=Path, nargs="+", required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def _load_dataset_manifest(
    path: Path,
    *,
    mesh_manifest_file_sha256: str,
    mesh_manifest_artifact_sha256: str,
) -> dict[str, Any]:
    payload = _load_json(path)
    _require(
        payload.get("artifact_kind") == "RGBenchDatasetManifest"
        and payload.get("rgbbench_commit") == RGBENCH_COMMIT,
        "dataset manifest provenance changed",
    )
    _require(
        sha256_file(path) == mesh_manifest_file_sha256
        and payload.get("artifact_sha256") == mesh_manifest_artifact_sha256,
        "dataset manifest differs from the mesh-selection input",
    )
    return payload


def _case_descriptor(
    manifest: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    matches = [case for case in manifest["cases"] if case["case_id"] == case_id]
    _require(len(matches) == 1, f"manifest does not contain exactly one {case_id}")
    case = matches[0]
    _require(case["sample"] == "01", "physical preflight is locked to sample 01")
    return case


def _mesh_artifact(
    manifest: RGBenchIsotropicMeshManifest,
    garment: str,
) -> RGBenchIsotropicMeshArtifact:
    matches = [
        artifact for artifact in manifest.artifacts if artifact.garment == garment
    ]
    _require(len(matches) == 1, f"mesh manifest does not contain one {garment}")
    return matches[0]


def _capture_root(dataset_root: Path, case: dict[str, Any]) -> Path:
    capture = dataset_root / case["data_subfolder"]
    _require(capture.is_dir(), f"capture does not exist: {capture}")
    return capture


def _case_pcd_paths(
    dataset_root: Path,
    case: dict[str, Any],
) -> tuple[Path, ...]:
    return evaluation_pcd_paths(
        _capture_root(dataset_root, case),
        master_start_time_s=float(case["master_start_time_s"]),
        camera_delay_s=float(case["camera_delay_s"]),
        start_calculate_time_s=float(case["start_calculate_time_s"]),
        end_calculate_time_s=float(case["end_calculate_time_s"]),
        expected_count=int(case["evaluation_frame_count"]),
        expected_name_sha256=str(case["point_cloud_name_sha256"]),
    )


def _compose_config(
    benchmark_root: Path,
    dataset_root: Path,
    case: dict[str, Any],
    mesh_path: Path,
    mesh_artifact: RGBenchIsotropicMeshArtifact,
) -> object:
    sys.path.insert(0, str(benchmark_root))
    hydra = importlib.import_module("hydra")
    OmegaConf = importlib.import_module("omegaconf").OmegaConf
    overrides = [
        f"params.cloth_name={case['garment']}",
        f"params.action_type={case['action']}",
        f"params.sample_index={case['sample']}",
        f"params.sim_environment={SIMULATOR}",
        f"params.sim_mode={MODE}",
        f"cloth_params={case['garment']}",
        f"env={SIMULATOR}",
        f"dataset_path={dataset_root}",
        f"cloth_model_path={dataset_root / 'meshes'}",
        f"project_root={benchmark_root}",
        f"output_path={benchmark_root / 'outputs'}",
        "active_run.visualization.vis_sim=false",
        "active_run.visualization.save_gifs=false",
        "active_run.visualization.save_sim_pcd=false",
        "active_run.visualization.save_target_pcd=false",
    ]
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str((benchmark_root / "configs").resolve()),
    ):
        config = hydra.compose(config_name="main", overrides=overrides)
    OmegaConf.resolve(config)
    active = config.active_run
    OmegaConf.set_readonly(active, False)
    active.cloth.model_path = str(mesh_path)
    active.cloth_params.shoulder_index = list(
        mesh_artifact.derived_fling_pin_indices
    )
    return active


def _simulate(args: argparse.Namespace) -> int:
    benchmark = args.benchmark_root.resolve()
    git_head = subprocess.check_output(
        ["git", "-C", str(benchmark), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    _require(git_head == RGBENCH_COMMIT, "RGBench checkout commit changed")
    mesh_manifest_path = args.mesh_manifest.resolve()
    mesh_manifest = load_isotropic_mesh_manifest(mesh_manifest_path)
    _require(
        mesh_manifest.rgbbench_commit == RGBENCH_COMMIT,
        "mesh manifest RGBench commit changed",
    )
    dataset_manifest_path = args.dataset_manifest.resolve()
    dataset_manifest = _load_dataset_manifest(
        dataset_manifest_path,
        mesh_manifest_file_sha256=mesh_manifest.dataset_manifest_file_sha256,
        mesh_manifest_artifact_sha256=(
            mesh_manifest.dataset_manifest_artifact_sha256
        ),
    )
    case = _case_descriptor(dataset_manifest, args.case_id)
    mesh_artifact = _mesh_artifact(mesh_manifest, str(case["garment"]))
    mesh_path = (
        mesh_manifest_path.parent / mesh_artifact.derived_mesh_relative_path
    )
    _require(
        mesh_path.is_file()
        and sha256_file(mesh_path) == mesh_artifact.derived_mesh_sha256,
        "derived physical mesh changed",
    )
    dataset = args.dataset_root.resolve()
    paths = _case_pcd_paths(dataset, case)
    config = _compose_config(
        benchmark,
        dataset,
        case,
        mesh_path,
        mesh_artifact,
    )
    get_env = importlib.import_module("rgbench.envs").get_env
    pybullet = importlib.import_module("pybullet")
    with force_pybullet_direct_connection(pybullet):
        environment = get_env(config)
    connection_info = pybullet.getConnectionInfo(
        physicsClientId=environment.physics_client
    )
    _require(
        connection_info["connectionMethod"] == pybullet.DIRECT,
        "PyBullet connection was not forced to DIRECT",
    )
    try:
        master_start = float(environment.get_master_start_time())
        _require(
            abs(master_start - float(case["master_start_time_s"])) <= 1e-6,
            "simulator and manifest master start times differ",
        )
        preparation_time = (
            float(config.action.fling_prepare_time)
            + float(config.action.fling_wait_time)
            if case["action"] == "fling"
            else 0.0
        )
        vertices: list[np.ndarray] = []
        target_times: list[float] = []
        for path in paths:
            absolute_time = float(
                path.name.removeprefix("pointcloud_").removesuffix(
                    "_segmented.pcd"
                )
            )
            target_time = absolute_time - master_start
            environment.step_to_time(
                target_time
                + float(case["camera_delay_s"])
                + preparation_time
            )
            frame = np.asarray(environment.get_sim_vertices(), dtype=np.float64)
            _require(
                frame.ndim == 2
                and frame.shape[1] == 3
                and np.all(np.isfinite(frame)),
                "simulator returned invalid vertices",
            )
            vertices.append(frame.copy())
            target_times.append(target_time)
        anchor_indices = (
            int(environment.left_anchor_vertex),
            int(environment.right_anchor_vertex),
        )
    finally:
        environment.close()

    _, faces = load_obj_triangles(mesh_path)
    node_count = len(vertices[0])
    _require(
        node_count == mesh_artifact.derived_vertex_count
        and all(len(frame) == node_count for frame in vertices),
        "simulator node count differs from the derived mesh",
    )
    _require(
        int(np.max(faces)) < node_count
        and len(faces) == mesh_artifact.derived_face_count,
        "derived faces do not index the simulator vertices",
    )
    if case["action"] == "fling":
        _require(
            anchor_indices == mesh_artifact.derived_fling_pin_indices,
            "simulator did not use the bound fling contacts",
        )
    output = args.output.resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices_m=np.stack(vertices),
        faces=faces,
        target_times_s=np.asarray(target_times, dtype=np.float64),
    )
    write_json_once(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchIsotropicPhysicalPreflight",
            "method_id": METHOD_ID,
            "case_id": case["case_id"],
            "garment": case["garment"],
            "action": case["action"],
            "sample": case["sample"],
            "split": case["split"],
            "replay_index": args.replay_index,
            "rgbbench_commit": RGBENCH_COMMIT,
            "dataset_revision": dataset_manifest["dataset_revision"],
            "simulator": SIMULATOR,
            "mode": MODE,
            "pybullet_connection_mode": "DIRECT",
            "upstream_gui_request_overridden": True,
            "node_count": node_count,
            "face_count": len(faces),
            "anchor_indices": list(anchor_indices),
            "mesh_mode": mesh_artifact.mode,
            "mesh_artifact_sha256": mesh_artifact.artifact_sha256,
            "mesh_manifest_sha256": sha256_file(mesh_manifest_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "evaluation_frame_count": len(vertices),
            "npz_path": str(output),
            "npz_sha256": sha256_file(output),
            "point_cloud_filenames_read": True,
            "point_cloud_coordinates_read": False,
            "known_future_actuator_trajectory_read": True,
            "future_object_outcomes_read": False,
        },
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    mesh_manifest_path = args.mesh_manifest.resolve()
    mesh_manifest = load_isotropic_mesh_manifest(mesh_manifest_path)
    metadata = [_load_json(path.resolve()) for path in args.metadata]
    _require(
        all(
            item.get("artifact_kind") == "RGBenchIsotropicPhysicalPreflight"
            and item.get("method_id") == METHOD_ID
            and item.get("mesh_manifest_sha256")
            == sha256_file(mesh_manifest_path)
            for item in metadata
        ),
        "preflight metadata provenance changed",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in metadata:
        grouped.setdefault(str(item["case_id"]), []).append(item)
    expected_cases = {
        f"{garment}/{action}/01"
        for garment in PAPER_GARMENTS
        for action in ACTIONS
    }
    _require(set(grouped) == expected_cases, "physical preflight case set changed")
    replay_records: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        entries = sorted(grouped[case_id], key=lambda item: item["replay_index"])
        _require(
            [item["replay_index"] for item in entries] == [1, 2],
            f"{case_id} does not have replays 1 and 2",
        )
        npz_paths = [Path(str(item["npz_path"])) for item in entries]
        _require(
            all(
                path.is_file()
                and sha256_file(path) == item["npz_sha256"]
                for path, item in zip(npz_paths, entries, strict=True)
            ),
            f"{case_id} preflight output changed",
        )
        _require(
            entries[0]["npz_sha256"] == entries[1]["npz_sha256"],
            f"{case_id} physical replays are not byte-identical",
        )
        replay_records.append(
            {
                "case_id": case_id,
                "node_count": entries[0]["node_count"],
                "face_count": entries[0]["face_count"],
                "anchor_indices": entries[0]["anchor_indices"],
                "mesh_mode": entries[0]["mesh_mode"],
                "npz_sha256": entries[0]["npz_sha256"],
            }
        )
    output = args.output.resolve()
    write_json_once(
        output,
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchIsotropicPhysicalPreflightGate",
            "method_id": METHOD_ID,
            "rgbbench_commit": RGBENCH_COMMIT,
            "mesh_manifest_artifact_sha256": mesh_manifest.artifact_sha256,
            "mesh_manifest_file_sha256": sha256_file(mesh_manifest_path),
            "case_count": len(replay_records),
            "replay_count": len(metadata),
            "all_replays_byte_identical": True,
            "physical_preflight_passed": True,
            "cases": replay_records,
            "information_boundary": {
                "point_cloud_coordinates_read": False,
                "future_object_outcomes_read": False,
                "known_future_actuator_trajectory_read": True,
            },
        },
    )
    return 0


def main() -> None:
    args = _parse_args()
    if args.command == "simulate":
        raise SystemExit(_simulate(args))
    raise SystemExit(_verify(args))


if __name__ == "__main__":
    main()
