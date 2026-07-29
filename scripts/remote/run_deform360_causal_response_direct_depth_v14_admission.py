#!/usr/bin/env python3
"""Build one frame-zero carrier and hash-only V14 source preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    AdaptiveCausalResponseQueryConfig,
    build_adaptive_causal_response_query_schedule,
    write_adaptive_causal_response_query_artifacts,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_admission_v14 import (
    CARRIER_DIRECTORY,
    PREFLIGHT_FILENAME,
    aggregate_source_sha256,
    load_v14_admission_prelock_protocol,
    write_v14_admission_report,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_physical import (
    PHYSICAL_ARCHIVE_FILENAME,
    PHYSICAL_MANIFEST_FILENAME,
    load_v14_physical_prelock_protocol,
    v14_physical_case_record,
    validate_v14_physical_artifacts,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_physical_runtime_v2 import (
    load_v14_physical_runtime_v2,
    validate_v14_physical_action_v2,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    AdaptiveDirectDepthSourcePreflightConfigV14,
    evaluate_adaptive_direct_depth_source_preflight_v14,
    write_adaptive_direct_depth_source_preflight_v14,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    CausalResponseSourceCameraRecord,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_admission_v2 import (
    load_complete_camera_geometry,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def _canonical_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_output(repository, "rev-parse", "HEAD")
    _require(
        not _git_output(repository, "status", "--porcelain", "--untracked-files=normal"),
        "V14 admission repository is dirty",
    )
    return revision


def _verify_parent(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    role: str,
    semantic_digest: str,
) -> None:
    expected = protocol["parent_artifacts"][role]
    _require(
        expected["config_or_queue_sha256"] == semantic_digest
        and expected["file_sha256"] == file_sha256(path),
        f"V14 admission parent changed: {role}",
    )


def _read_frame_zero_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as stream:
        _require(
            "data" in stream and stream["data"].ndim == 3 and len(stream["data"]) >= 1,
            f"invalid V14 frame-zero stream: {path}",
        )
        return np.asarray(stream["data"][0])


def _load_frame_zero_camera_inputs(
    episode: Path,
    camera_names: tuple[str, ...],
    *,
    depth_scale_to_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for camera in camera_names:
        directory = episode / camera
        encoded = _read_frame_zero_h5(directory / "rendered_depth.h5")
        mask = _read_frame_zero_h5(directory / "mask_refined.h5").astype(
            bool,
            copy=False,
        )
        _require(
            encoded.shape == mask.shape,
            f"V14 frame-zero depth and mask differ: {camera}",
        )
        depth.append(encoded.astype(np.float32) * depth_scale_to_m)
        masks.append(mask)
    _require(
        len({values.shape for values in depth}) == 1,
        "V14 frame-zero camera shapes differ",
    )
    return np.stack(depth), np.stack(masks)


def _camera_records(
    geometry_manifest: Mapping[str, Any],
) -> tuple[CausalResponseSourceCameraRecord, ...]:
    by_camera = {
        str(record["camera"]): record
        for record in geometry_manifest["camera_records"]
    }
    support = geometry_manifest["frame_zero_projected_support_by_camera"]
    calibration_valid = bool(geometry_manifest["calibration_valid"])
    return tuple(
        CausalResponseSourceCameraRecord(
            camera_id=camera,
            depth_frame_count=int(by_camera.get(camera, {}).get("depth_frame_count", 0)),
            mask_frame_count=int(by_camera.get(camera, {}).get("mask_frame_count", 0)),
            calibration_valid=calibration_valid and camera in by_camera,
            frame_zero_projected_support_count=int(support.get(camera, 0)),
        )
        for camera in REGISTERED_CAMERA_IDS
    )


def _source_sha256(
    *,
    candidate: Mapping[str, Any],
    geometry_manifest: Mapping[str, Any],
    known_action: Path,
    staged_episode: Path,
    processed_episode: Path,
    stage_result: Mapping[str, Any],
) -> tuple[dict[str, str], int, int]:
    outputs = geometry_manifest["outputs_sha256"]
    camera_names = tuple(map(str, geometry_manifest["cameras"]))
    source = {
        "metadata": str(candidate["metadata_sha256"]),
        "robot": file_sha256(known_action),
        "physical_geometry": str(outputs["frame_zero_points"]),
    }
    tactile_hashes: dict[str, str] = {}
    tactile_counts: list[int] = []
    for record in stage_result["tactile_records"]:
        sensor = str(record["sensor"])
        path = staged_episode / sensor / "synced_tactile.npy"
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        digest = file_sha256(path)
        _require(
            digest == record["array_sha256"],
            f"V14 staged tactile source changed: {sensor}",
        )
        tactile_hashes[sensor] = digest
        tactile_counts.append(int(values.shape[0]))
    source["tactile"] = aggregate_source_sha256("tactile", tactile_hashes)
    intrinsics = str(outputs["intrinsics"])
    extrinsics = str(outputs["extrinsics"])
    for camera in camera_names:
        depth_path = processed_episode / camera / "rendered_depth.h5"
        mask_path = processed_episode / camera / "mask_refined.h5"
        _require(
            file_sha256(depth_path) == outputs["depth_by_camera"][camera]
            and file_sha256(mask_path) == outputs["mask_by_camera"][camera],
            f"V14 causal camera source changed: {camera}",
        )
        source[f"depth/{camera}"] = file_sha256(depth_path)
        source[f"mask/{camera}"] = file_sha256(mask_path)
        source[f"calibration/{camera}"] = aggregate_source_sha256(
            f"calibration/{camera}",
            {
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
            },
        )
    robot_archive = np.load(known_action, allow_pickle=False, mmap_mode="r")
    try:
        robot_count = int(len(robot_archive["actions"]))
    finally:
        robot_archive.close()
    _require(tactile_counts, "V14 staged tactile panel is empty")
    return source, robot_count, min(tactile_counts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--admission-prelock", type=Path, required=True)
    parser.add_argument("--method-protocol", type=Path, required=True)
    parser.add_argument("--prefix-assets", type=Path, required=True)
    parser.add_argument("--physical-prelock", type=Path, required=True)
    parser.add_argument("--physical-runtime-v2", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-rank", type=int, required=True)
    parser.add_argument("--window-stage-result", type=Path, required=True)
    parser.add_argument("--staged-episode-dir", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repo.resolve()
    revision = _require_clean_repository(repository)
    admission_path = args.admission_prelock.resolve()
    admission = load_v14_admission_prelock_protocol(admission_path)
    implementation_paths = {
        "admission_module": (
            repository
            / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_admission_v14.py"
        ),
        "admission_runner": Path(__file__).resolve(),
        "preflight_module": (
            repository
            / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_preflight.py"
        ),
    }
    _require(
        all(
            file_sha256(path)
            == admission["implementation"]["file_sha256"][name]
            for name, path in implementation_paths.items()
        )
        and _git_output(
            repository,
            "merge-base",
            "--is-ancestor",
            admission["implementation"]["parent_commit"],
            revision,
        )
        == "",
        "V14 admission implementation changed",
    )

    method_path = args.method_protocol.resolve()
    method = _read_json(method_path)
    _require(
        method.get("protocol_id")
        == "deform360-causal-response-direct-depth-v14-source"
        and method.get("config_sha256") == _canonical_config_sha256(method),
        "V14 method protocol changed",
    )
    assets_path = args.prefix_assets.resolve()
    assets = _read_json(assets_path)
    physical_prelock_path = args.physical_prelock.resolve()
    physical_prelock = load_v14_physical_prelock_protocol(physical_prelock_path)
    physical_runtime_path = args.physical_runtime_v2.resolve()
    physical_runtime = load_v14_physical_runtime_v2(
        physical_runtime_path,
        parent_prelock_path=physical_prelock_path,
    )
    queue_path = args.queue.resolve()
    queue = validate_v14_staging_queue(queue_path)
    for path, role, semantic in (
        (method_path, "method_protocol", method["config_sha256"]),
        (assets_path, "prefix_assets", assets["config_sha256"]),
        (
            physical_prelock_path,
            "physical_prelock",
            physical_prelock["config_sha256"],
        ),
        (
            physical_runtime_path,
            "physical_runtime_v2",
            physical_runtime["config_sha256"],
        ),
        (queue_path, "staging_queue", queue["queue_sha256"]),
    ):
        _verify_parent(path, protocol=admission, role=role, semantic_digest=semantic)

    rank = int(args.queue_rank)
    _require(1 <= rank <= len(queue["candidates"]), "V14 queue rank is invalid")
    candidate = queue["candidates"][rank - 1]
    case_record = v14_physical_case_record(
        physical_prelock,
        queue,
        queue_rank=rank,
    )
    physical_dir = args.physical_dir.resolve()
    physical_manifest, physical = validate_v14_physical_artifacts(
        physical_dir,
        prelock_protocol_path=physical_prelock_path,
    )
    _require(
        all(
            physical_manifest[key] == case_record[key]
            for key in (
                "queue_rank",
                "object_hash",
                "case_hash",
                "metadata_sha256",
                "physical_node_count",
            )
        ),
        "V14 admission physical artifact uses another case",
    )

    geometry_path = args.geometry_manifest.resolve()
    geometry_manifest = _read_json(geometry_path)
    _require(
        geometry_manifest.get("artifact_sha256")
        == case_record["geometry_manifest_artifact_sha256"]
        and file_sha256(geometry_path)
        == case_record["geometry_manifest_file_sha256"]
        and geometry_manifest.get("object_hash") == case_record["object_hash"]
        and geometry_manifest.get("case_hash") == case_record["case_hash"]
        and physical_manifest["inputs_sha256"]["geometry_manifest"]
        == file_sha256(geometry_path),
        "V14 admission geometry differs from the physical carrier",
    )
    processed = args.processed_episode_dir.resolve()
    geometry = load_complete_camera_geometry(
        processed,
        candidate_camera_names=REGISTERED_CAMERA_IDS,
        minimum_complete_camera_count=int(
            admission["numerical_contract"]["minimum_complete_camera_count"]
        ),
        frame_count=1,
    )
    depth, masks = _load_frame_zero_camera_inputs(
        processed,
        geometry.camera_names,
        depth_scale_to_m=float(admission["numerical_contract"]["depth_scale_to_m"]),
    )
    query_config = AdaptiveCausalResponseQueryConfig(**method["adaptive_carrier"])
    _require(
        query_config == AdaptiveCausalResponseQueryConfig(),
        "V14 adaptive carrier differs from the frozen implementation",
    )
    schedule = build_adaptive_causal_response_query_schedule(
        physical["physical_prediction_m"][0],
        physical["graph_basis"],
        physical["action_support"],
        geometry.intrinsics,
        geometry.camera_to_world,
        depth,
        masks,
        camera_ids=geometry.camera_names,
        config=query_config,
    )

    stage_path = args.window_stage_result.resolve()
    stage = _read_json(stage_path)
    staged_episode = args.staged_episode_dir.resolve()
    known_action = staged_episode / "robot" / "robot.npz"
    source_sha256, raw_robot_count, raw_tactile_count = _source_sha256(
        candidate=candidate,
        geometry_manifest=geometry_manifest,
        known_action=known_action,
        staged_episode=staged_episode,
        processed_episode=processed,
        stage_result=stage,
    )
    validate_v14_physical_action_v2(
        physical_runtime,
        queue_rank=rank,
        object_hash=case_record["object_hash"],
        case_hash=case_record["case_hash"],
        window_stage_result_path=stage_path,
        known_action_path=known_action,
        staged_frame_count=raw_robot_count,
    )
    prediction_count = int(stage["prediction_frame_count"])
    _require(
        prediction_count
        == admission["numerical_contract"]["physical_robot_tactile_frame_count"]
        and raw_robot_count >= prediction_count
        and raw_tactile_count >= prediction_count
        and physical["physical_prediction_m"].shape[0] == prediction_count,
        "V14 predictive stream lengths changed",
    )
    preflight_values = dict(method["source_preflight"])
    preflight_values["registered_camera_ids"] = tuple(
        preflight_values["registered_camera_ids"]
    )
    preflight_config = AdaptiveDirectDepthSourcePreflightConfigV14(
        **preflight_values
    )

    output = args.output_dir.resolve()
    _require(not output.exists(), "V14 admission output already exists")
    scratch = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "V14 admission scratch already exists")
    scratch.mkdir(parents=True)
    carrier_report = write_adaptive_causal_response_query_artifacts(
        scratch / CARRIER_DIRECTORY,
        schedule,
        case_id=case_record["case_hash"],
        repository_revision=revision,
        protocol_path=method_path,
        physical_manifest_path=physical_dir / PHYSICAL_MANIFEST_FILENAME,
        physical_archive_path=physical_dir / PHYSICAL_ARCHIVE_FILENAME,
        camera_certificate_sha256=geometry.artifact_sha256,
    )
    preflight = evaluate_adaptive_direct_depth_source_preflight_v14(
        object_id=str(candidate["object_id"]),
        episode_id=int(candidate["episode_id"]),
        category=str(candidate["category"]),
        bimanual_value=str(candidate["bimanual"]),
        episode_frame_count=prediction_count,
        robot_frame_count=prediction_count,
        tactile_frame_count=prediction_count,
        physical_node_count=int(physical_manifest["physical_node_count"]),
        camera_records=_camera_records(geometry_manifest),
        carrier=schedule,
        source_sha256=source_sha256,
        config=preflight_config,
    )
    write_adaptive_direct_depth_source_preflight_v14(
        scratch / PREFLIGHT_FILENAME,
        preflight,
    )
    report = write_v14_admission_report(
        scratch,
        queue_rank=rank,
        object_hash=case_record["object_hash"],
        case_hash=case_record["case_hash"],
        repository_revision=revision,
        admission_protocol=admission,
        physical_artifact_sha256=physical_manifest["artifact_sha256"],
        geometry_artifact_sha256=geometry_manifest["artifact_sha256"],
        carrier_result_sha256=carrier_report["result_sha256"],
        carrier_artifact_sha256=schedule.artifact_sha256,
        preflight_artifact_sha256=preflight.artifact_sha256,
        admitted=preflight.admitted,
        input_files={
            "admission_prelock": admission_path,
            "geometry_manifest": geometry_path,
            "known_action": known_action,
            "method_protocol": method_path,
            "physical_archive": physical_dir / PHYSICAL_ARCHIVE_FILENAME,
            "physical_manifest": physical_dir / PHYSICAL_MANIFEST_FILENAME,
            "physical_prelock": physical_prelock_path,
            "physical_runtime_v2": physical_runtime_path,
            "prefix_assets": assets_path,
            "staging_queue": queue_path,
            "window_stage_result": stage_path,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch.rename(output)
    print(
        json.dumps(
            {
                "queue_rank": rank,
                "status": report["status"],
                "carrier_arm": schedule.arm,
                "complete_camera_count": len(preflight.complete_camera_ids),
                "selected_entity_count": len(schedule.query_schedule.entity_ids),
                "admission_artifact_sha256": report["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
