#!/usr/bin/env python3
"""Run the source-only tactile-prompted carrier on an already-open case."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_metric_object_carrier import (
    cover_resize_mask_nearest,
)
from bayesian_phystwin.deform360_tactile_metric_gauge import SimilarityTransform
from bayesian_phystwin.deform360_tactile_prompted_carrier import (
    TACTILE_PROMPTED_CARRIER_POLICY,
    PromptedCandidateGeometry,
    build_bias_aware_metric_carrier,
    build_dense_point_candidates,
    build_tactile_prompt_assignments,
    evaluate_prompted_mask,
    load_tactile_prompted_carrier_validation_lock,
    project_prompt_assignment,
    select_crossview_candidate_pair,
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_ancestor(repository: Path, expected: str) -> str:
    head = _git_head(repository)
    result = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", expected, head],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(result.returncode == 0, "runtime lacks the frozen implementation")
    return head


def _load_selector(source: Path):
    package_name = "causal4d_public"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(source.parent)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    name = "causal4d_public.deform360_object_sam2"
    spec = importlib.util.spec_from_file_location(name, source)
    _require(spec is not None and spec.loader is not None, "cannot load SAM2 selector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.DeformableObjectSam2VideoPredictor


def _frame_rgb(video: Path, source_frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, bgr = capture.read()
        observed = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    finally:
        capture.release()
    _require(
        ok and observed == source_frame, f"cannot decode frame {source_frame}: {video}"
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _transform(record: dict[str, object]) -> SimilarityTransform:
    value = record["similarity_transform"]
    _require(isinstance(value, dict), "missing similarity transform")
    return SimilarityTransform(
        scale=float(value["scale"]),
        rotation=np.asarray(value["rotation"], dtype=np.float64),
        translation=np.asarray(value["translation_m"], dtype=np.float64),
    )


def _validation_providers(
    *,
    lock: dict[str, object],
    job_manifest_path: Path,
    provider_root: Path,
    provider_run_report_path: Path,
    processed_episode: Path,
) -> dict[str, dict[str, object]]:
    bindings = lock["bindings"]
    source = lock["source_case"]
    _require(isinstance(bindings, dict), "validation bindings are missing")
    _require(isinstance(source, dict), "validation source is missing")
    _require(
        _sha256(job_manifest_path)
        == bindings["motioncrafter_job_manifest_file_sha256"],
        "MotionCrafter job manifest changed",
    )
    _require(
        _sha256(provider_run_report_path)
        == bindings["motioncrafter_stage1_run_report_sha256"],
        "MotionCrafter provider run report changed",
    )
    job_manifest = _json(job_manifest_path)
    _require(
        job_manifest.get("manifest_sha256")
        == bindings["motioncrafter_job_manifest_id"],
        "MotionCrafter job manifest identity changed",
    )
    run_report = _json(provider_run_report_path)
    _require(run_report.get("status") == "complete", "provider run is incomplete")
    _require(
        run_report.get("job_manifest_sha256")
        == bindings["motioncrafter_job_manifest_file_sha256"],
        "provider run used a different job manifest",
    )
    completed = {
        str(row["job_id"]): row for row in run_report.get("completed_jobs", [])
    }
    jobs = {str(row["job_id"]): row for row in job_manifest.get("jobs", [])}
    result: dict[str, dict[str, object]] = {}
    for frozen in bindings["motioncrafter_jobs"]:
        _require(isinstance(frozen, dict), "invalid frozen provider job")
        job_id = str(frozen["job_id"])
        _require(job_id in jobs and job_id in completed, "provider job is unavailable")
        job = jobs[job_id]
        camera = str(frozen["camera"])
        _require(
            job.get("camera") == camera
            and job.get("object_id") == source["object_id"]
            and job.get("episode")
            == f"episode_{int(source['processing_episode_index']):04d}"
            and job.get("source_frame_start") == frozen["source_frame_start"]
            and job.get("source_frame_stop_exclusive")
            == frozen["source_frame_stop_exclusive"]
            and job.get("source_video", {}).get("sha256") == frozen["video_sha256"],
            f"provider job changed: {camera}",
        )
        directory = provider_root / str(job["output_relative_path"])
        prediction_manifest_path = directory / "predictions.json"
        _require(
            prediction_manifest_path.is_file()
            and _sha256(prediction_manifest_path)
            == completed[job_id]["prediction_manifest_sha256"],
            f"prediction manifest changed: {camera}",
        )
        prediction = _json(prediction_manifest_path)
        integrity = prediction.get("artifact_integrity")
        _require(isinstance(integrity, dict), f"missing provider integrity: {camera}")
        members = {
            str(row["path"]): row for row in integrity.get("members", [])
        }
        windows: list[dict[str, object]] = []
        for row in prediction.get("overlap_windows", []):
            path = str(row["path"])
            member = members.get(path)
            _require(
                isinstance(member, dict)
                and member.get("kind") == "independently_decoded_overlap_window",
                f"unbound provider window: {camera}/{path}",
            )
            member_path = directory / path
            _require(
                member_path.is_file() and _sha256(member_path) == member["sha256"],
                f"provider window changed: {camera}/{path}",
            )
            windows.append(
                {
                    "path": str(member_path),
                    "sha256": member["sha256"],
                    "start_frame": int(row["start_frame"]),
                    "stop_frame": int(row["stop_frame"]),
                }
            )
        video_path = processed_episode / camera / "undistorted.mp4"
        _require(
            video_path.is_file() and _sha256(video_path) == frozen["video_sha256"],
            f"provider video changed: {camera}",
        )
        result[camera] = {
            "camera": camera,
            "job_id": job_id,
            "prediction_manifest_path": str(prediction_manifest_path),
            "prediction_manifest_sha256": _sha256(prediction_manifest_path),
            "video_path": str(video_path),
            "video_sha256": frozen["video_sha256"],
            "windows": windows,
        }
    _require(
        tuple(result) == tuple(source["camera_panel"]),
        "provider camera order changed",
    )
    return result


def _provider_window(
    provider: dict[str, object],
    *,
    prompt_frame: int,
) -> Path:
    if "window_path" in provider:
        return Path(str(provider["window_path"]))
    windows = provider.get("windows")
    _require(isinstance(windows, list), "provider windows are missing")
    candidates = [
        row
        for row in windows
        if int(row["start_frame"]) <= prompt_frame < int(row["stop_frame"])
    ]
    _require(candidates, "prompt frame is absent from every provider window")
    selected = max(candidates, key=lambda row: int(row["start_frame"]))
    return Path(str(selected["path"]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--parent-carrier-lock", type=Path)
    mode.add_argument("--validation-lock", type=Path)
    parser.add_argument("--motioncrafter-job-manifest", type=Path)
    parser.add_argument("--provider-root", type=Path)
    parser.add_argument("--provider-run-report", type=Path)
    parser.add_argument("--metric-gauge-result", type=Path, required=True)
    parser.add_argument("--robot-prefix", type=Path, required=True)
    parser.add_argument("--tactile-geometry", type=Path, required=True)
    parser.add_argument("--processed-episode", type=Path, required=True)
    parser.add_argument("--selector-source", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repository.resolve()
    parent_lock_path = (
        args.parent_carrier_lock.resolve()
        if args.parent_carrier_lock is not None
        else None
    )
    validation_lock_path = (
        args.validation_lock.resolve() if args.validation_lock is not None else None
    )
    metric_result_path = args.metric_gauge_result.resolve()
    robot_prefix_path = args.robot_prefix.resolve()
    tactile_geometry_path = args.tactile_geometry.resolve()
    processed_episode = args.processed_episode.resolve()
    selector_source = args.selector_source.resolve()
    sam2_repository = args.sam2_repository.resolve()
    sam2_checkpoint = args.sam2_checkpoint.resolve()
    output = args.output.resolve()
    _require(not output.exists(), "output already exists")
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status, "implementation checkout is dirty")
    required_paths = [
        (metric_result_path, "metric result"),
        (robot_prefix_path, "robot prefix"),
        (tactile_geometry_path, "tactile geometry"),
        (selector_source, "SAM2 selector"),
        (sam2_checkpoint, "SAM2 checkpoint"),
    ]
    if parent_lock_path is not None:
        required_paths.append((parent_lock_path, "parent carrier lock"))
    else:
        _require(validation_lock_path is not None, "validation lock is missing")
        required_paths.append((validation_lock_path, "validation lock"))
        for value, label in (
            (args.motioncrafter_job_manifest, "MotionCrafter job manifest"),
            (args.provider_run_report, "provider run report"),
        ):
            _require(value is not None, f"missing {label}")
            required_paths.append((value.resolve(), label))
        _require(args.provider_root is not None, "missing provider root")
    for path, label in required_paths:
        _require(path.is_file(), f"missing {label}: {path}")
    validation: dict[str, object] | None = None
    if parent_lock_path is not None:
        parent = _json(parent_lock_path)
        object_id = str(parent["source_case"]["object_id"])
        _require(object_id == "026-sock-cloth", "development case changed")
        cameras = tuple(str(camera) for camera in parent["cameras"])
        providers = {str(row["camera"]): row for row in parent["providers"]}
    else:
        assert validation_lock_path is not None
        validation = dict(
            load_tactile_prompted_carrier_validation_lock(validation_lock_path)
        )
        implementation = validation["implementation"]
        _require(isinstance(implementation, dict), "validation implementation missing")
        _require_ancestor(repository, str(implementation["revision"]))
        source = validation["source_case"]
        _require(isinstance(source, dict), "validation source missing")
        object_id = str(source["object_id"])
        cameras = tuple(str(camera) for camera in source["camera_panel"])
        providers = _validation_providers(
            lock=validation,
            job_manifest_path=args.motioncrafter_job_manifest.resolve(),
            provider_root=args.provider_root.resolve(),
            provider_run_report_path=args.provider_run_report.resolve(),
            processed_episode=processed_episode,
        )
    metric = _json(metric_result_path)
    _require(metric.get("status") == "admitted", "metric gauge is not admitted")
    gate = metric.get("gate")
    _require(
        isinstance(gate, dict) and gate.get("metric_gauge_authorized") is True,
        "metric gauge is unauthorized",
    )
    _require(
        isinstance(gate, dict) and gate.get("contact_anchor_authorized") is False,
        "contact geometry was already promoted to an anchor",
    )
    _require(set(cameras) == set(providers), "provider camera panel changed")
    for camera in cameras:
        path = Path(str(providers[camera]["video_path"]))
        _require(
            path.is_file() and _sha256(path) == providers[camera]["video_sha256"],
            f"provider input changed: {camera}/video_path",
        )
        if "window_path" in providers[camera]:
            window = Path(str(providers[camera]["window_path"]))
            _require(
                window.is_file()
                and _sha256(window) == providers[camera]["window_sha256"],
                f"provider input changed: {camera}/window_path",
            )

    with np.load(robot_prefix_path, allow_pickle=False) as archive:
        robot = {name: archive[name] for name in archive.files}
    with np.load(tactile_geometry_path, allow_pickle=False) as archive:
        tactile = {name: archive[name] for name in archive.files}
    assignments = build_tactile_prompt_assignments(
        tactile_source_frame_ids=tactile["source_frame_ids"],
        tactile_values=tactile["tactile_values"],
        finger_side_indices=tactile["finger_side_indices"],
        world_points_hypotheses_m=tactile["world_points_hypotheses_m"],
        gripper_indices_hypotheses=tactile["gripper_indices_hypotheses"],
        robot_source_frame_ids=robot["source_frame_ids"],
        robot_world_from_gripper=robot["T_worlds"],
        offset_m=float(TACTILE_PROMPTED_CARRIER_POLICY["prompt_offset_m"]),
    )
    prompt_frame = assignments[0].source_frame_id
    _require(
        all(item.source_frame_id == prompt_frame for item in assignments),
        "prompt frames differ",
    )
    if validation is not None:
        source = validation["source_case"]
        _require(isinstance(source, dict), "validation source missing")
        window = source["causal_window"]
        _require(isinstance(window, dict), "validation causal window missing")
        _require(
            int(window["contact_start_frame"])
            <= prompt_frame
            < int(window["causal_frame_stop"]),
            "prompt frame is outside the locked contact prefix",
        )
    intrinsics = np.load(
        processed_episode / "undistorted_intrinsics.npy", allow_pickle=True
    ).item()
    extrinsics = np.load(processed_episode / "extrinsics.npy", allow_pickle=True).item()
    _require(
        isinstance(intrinsics, dict) and isinstance(extrinsics, dict),
        "invalid calibration",
    )
    camera_metric = {str(row["camera"]): row for row in metric["camera_results"]}

    selector_class = _load_selector(selector_source)
    selector = selector_class(
        sam2_repository,
        sam2_checkpoint,
        device=args.device,
    )
    rgb_by_camera: dict[str, np.ndarray] = {}
    annotations_by_camera: dict[str, list[dict[str, object]]] = {}
    provider_by_camera: dict[str, dict[str, np.ndarray]] = {}
    try:
        for camera in cameras:
            rgb = _frame_rgb(Path(str(providers[camera]["video_path"])), prompt_frame)
            rgb_by_camera[camera] = rgb
            annotations_by_camera[camera] = list(selector._automatic_annotations(rgb))
            with np.load(
                _provider_window(providers[camera], prompt_frame=prompt_frame),
                allow_pickle=False,
            ) as archive:
                frame_ids = np.asarray(archive["frame_indices"], dtype=np.int64)
                rows = np.flatnonzero(frame_ids == prompt_frame)
                _require(len(rows) == 1, f"prompt frame absent from provider: {camera}")
                row = int(rows[0])
                provider_by_camera[camera] = {
                    "point_map": np.asarray(archive["point_map"][row]),
                    "valid_mask": np.asarray(archive["valid_mask"][row]),
                    "deform_mask": np.asarray(archive["deform_mask"][row]),
                }
    finally:
        selector.close()

    branch_records: list[dict[str, object]] = []
    branch_carriers: dict[int, object] = {}
    selected_masks: dict[tuple[int, str], np.ndarray] = {}
    admitted_status = (
        "admitted-independent-calibration-carrier"
        if validation is not None
        else "admitted-development-carrier"
    )
    for assignment in assignments:
        candidates_by_camera: dict[str, list[PromptedCandidateGeometry]] = {}
        camera_diagnostics: dict[str, object] = {}
        for camera in cameras:
            rgb = rgb_by_camera[camera]
            prompts = project_prompt_assignment(
                assignment,
                intrinsics=np.asarray(intrinsics[camera]),
                world_from_camera=np.asarray(extrinsics[camera]),
                image_shape=rgb.shape[:2],
            )
            hypotheses = camera_metric[camera]["assignment_hypotheses"]
            metric_hypothesis = hypotheses[assignment.assignment_index]
            transform = _transform(metric_hypothesis)
            covariance = np.asarray(
                metric_hypothesis["covariance_m2"], dtype=np.float64
            )
            candidates: list[PromptedCandidateGeometry] = []
            prompt_rows: list[dict[str, object]] = []
            for candidate_index, annotation in enumerate(annotations_by_camera[camera]):
                mask = np.asarray(annotation["segmentation"], dtype=bool)
                prompt = evaluate_prompted_mask(
                    mask,
                    prompts,
                    predicted_iou=float(annotation["predicted_iou"]),
                    stability_score=float(annotation["stability_score"]),
                    minimum_positive_hits=int(
                        TACTILE_PROMPTED_CARRIER_POLICY["minimum_positive_prompt_hits"]
                    ),
                    maximum_negative_hits=int(
                        TACTILE_PROMPTED_CARRIER_POLICY["maximum_negative_prompt_hits"]
                    ),
                    minimum_area_fraction=float(
                        TACTILE_PROMPTED_CARRIER_POLICY["minimum_mask_area_fraction"]
                    ),
                    maximum_area_fraction=float(
                        TACTILE_PROMPTED_CARRIER_POLICY["maximum_mask_area_fraction"]
                    ),
                )
                prompt_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "eligible": prompt.eligible,
                        "positive_hits": prompt.positive_hits,
                        "positive_visible": prompt.positive_visible,
                        "negative_hits": prompt.negative_hits,
                        "negative_visible": prompt.negative_visible,
                        "area_fraction": prompt.area_fraction,
                        "predicted_iou": float(annotation["predicted_iou"]),
                        "stability_score": float(annotation["stability_score"]),
                    }
                )
                if not prompt.eligible:
                    continue
                target_mask = cover_resize_mask_nearest(
                    mask,
                    target_shape=tuple(provider_by_camera[camera]["valid_mask"].shape),
                )
                try:
                    dense = build_dense_point_candidates(
                        provider_by_camera[camera]["point_map"],
                        provider_by_camera[camera]["valid_mask"],
                        target_mask,
                        provider_by_camera[camera]["deform_mask"],
                        transform=transform,
                        gauge_covariance_m2=covariance,
                        block_size_px=int(
                            TACTILE_PROMPTED_CARRIER_POLICY["block_size_px"]
                        ),
                        minimum_mask_pixels=int(
                            TACTILE_PROMPTED_CARRIER_POLICY[
                                "minimum_mask_pixels_per_block"
                            ]
                        ),
                        minimum_valid_fraction=float(
                            TACTILE_PROMPTED_CARRIER_POLICY[
                                "minimum_valid_fraction_per_block"
                            ]
                        ),
                        full_reliability_deform_fraction=float(
                            TACTILE_PROMPTED_CARRIER_POLICY[
                                "minimum_deform_fraction_for_full_reliability"
                            ]
                        ),
                        covariance_floor_m=float(
                            TACTILE_PROMPTED_CARRIER_POLICY["local_covariance_floor_m"]
                        ),
                    )
                except ValueError:
                    continue
                candidates.append(
                    PromptedCandidateGeometry(
                        candidate_index=candidate_index,
                        predicted_iou=float(annotation["predicted_iou"]),
                        stability_score=float(annotation["stability_score"]),
                        prompt=prompt,
                        dense=dense,
                    )
                )
            candidates_by_camera[camera] = candidates
            camera_diagnostics[camera] = {
                "automatic_candidate_count": len(annotations_by_camera[camera]),
                "prompt_eligible_candidate_count": sum(
                    int(row["eligible"]) for row in prompt_rows
                ),
                "metric_candidate_count": len(candidates),
                "prompt_candidates": prompt_rows,
            }
        try:
            pair = select_crossview_candidate_pair(
                candidates_by_camera,
                assignment_index=assignment.assignment_index,
                camera_order=cameras,
                maximum_distance_m=float(
                    TACTILE_PROMPTED_CARRIER_POLICY["maximum_cross_view_distance_m"]
                ),
                minimum_mutual_matches=int(
                    TACTILE_PROMPTED_CARRIER_POLICY["minimum_mutual_block_matches"]
                ),
                maximum_percentile_90_m=float(
                    TACTILE_PROMPTED_CARRIER_POLICY[
                        "maximum_cross_view_percentile_90_m"
                    ]
                ),
            )
            carrier = build_bias_aware_metric_carrier(
                pair,
                node_count=int(TACTILE_PROMPTED_CARRIER_POLICY["carrier_node_count"]),
                maximum_distance_m=float(
                    TACTILE_PROMPTED_CARRIER_POLICY["maximum_cross_view_distance_m"]
                ),
                shared_bias_floor_m=float(
                    TACTILE_PROMPTED_CARRIER_POLICY["shared_bias_floor_m"]
                ),
                unsupported_node_floor_m=float(
                    TACTILE_PROMPTED_CARRIER_POLICY["unsupported_node_floor_m"]
                ),
                unsupported_reliability_scale=float(
                    TACTILE_PROMPTED_CARRIER_POLICY[
                        "unsupported_node_reliability_scale"
                    ]
                ),
            )
            branch_carriers[assignment.assignment_index] = carrier
            selected_masks[(assignment.assignment_index, pair.reference_camera)] = (
                np.asarray(
                    annotations_by_camera[pair.reference_camera][
                        pair.reference.candidate_index
                    ]["segmentation"],
                    dtype=bool,
                )
            )
            selected_masks[(assignment.assignment_index, pair.support_camera)] = (
                np.asarray(
                    annotations_by_camera[pair.support_camera][
                        pair.support.candidate_index
                    ]["segmentation"],
                    dtype=bool,
                )
            )
            branch_records.append(
                {
                    "assignment_index": assignment.assignment_index,
                    "status": admitted_status,
                    "reference_camera": pair.reference_camera,
                    "support_camera": pair.support_camera,
                    "reference_candidate_index": pair.reference.candidate_index,
                    "support_candidate_index": pair.support.candidate_index,
                    "mutual_block_match_count": pair.mutual_block_match_count,
                    "median_block_distance_m": pair.median_block_distance_m,
                    "percentile_90_block_distance_m": pair.percentile_90_block_distance_m,
                    "represented_node_count": len(carrier.points_world_m),
                    "effective_reference_information_clusters": int(
                        len(np.unique(carrier.information_cluster_id))
                    ),
                    "supported_node_count": int(
                        np.count_nonzero(carrier.support_indices >= 0)
                    ),
                    "estimated_cross_view_bias_m": carrier.estimated_cross_view_bias_m.tolist(),
                    "camera_diagnostics": camera_diagnostics,
                }
            )
        except ValueError as error:
            branch_records.append(
                {
                    "assignment_index": assignment.assignment_index,
                    "status": "exact-fallback",
                    "reason": str(error),
                    "camera_diagnostics": camera_diagnostics,
                }
            )

    assignment_count = len(assignments)
    node_count = int(TACTILE_PROMPTED_CARRIER_POLICY["carrier_node_count"])
    mask_shape = rgb_by_camera[cameras[0]].shape[:2]
    _require(
        all(rgb_by_camera[camera].shape[:2] == mask_shape for camera in cameras),
        "camera image shapes differ",
    )
    packed_mask_width = (mask_shape[0] * mask_shape[1] + 7) // 8
    mask_available = np.zeros((assignment_count, len(cameras)), dtype=bool)
    packed_masks = np.zeros(
        (assignment_count, len(cameras), packed_mask_width), dtype=np.uint8
    )
    for (assignment_index, camera), mask in selected_masks.items():
        camera_index = cameras.index(camera)
        mask_available[assignment_index, camera_index] = True
        packed = np.packbits(mask.reshape(-1), bitorder="little")
        packed_masks[assignment_index, camera_index, : len(packed)] = packed
    arrays: dict[str, np.ndarray] = {
        "assignment_admitted": np.asarray(
            [index in branch_carriers for index in range(assignment_count)],
            dtype=bool,
        ),
        "assignment_prior_probability": np.asarray(
            tactile["assignment_prior_probability"], dtype=np.float64
        ),
        "prompt_source_frame": np.asarray(prompt_frame, dtype=np.int64),
        "selected_mask_shape": np.asarray(mask_shape, dtype=np.int64),
        "selected_mask_available": mask_available,
        "selected_mask_packed_little": packed_masks,
        "positive_prompt_world_m": np.stack(
            [item.positive_world_m for item in assignments]
        ),
        "negative_prompt_world_m": np.stack(
            [item.negative_world_m for item in assignments]
        ),
        "points_world_m": np.full((assignment_count, node_count, 3), np.nan),
        "local_covariance_m2": np.full((assignment_count, node_count, 3, 3), np.nan),
        "shared_bias_covariance_m2": np.full((assignment_count, 3, 3), np.nan),
        "marginal_covariance_m2": np.full((assignment_count, node_count, 3, 3), np.nan),
        "prior_reliability": np.zeros((assignment_count, node_count)),
        "reference_pixel_xy": np.full((assignment_count, node_count, 2), np.nan),
        "information_cluster_id": np.full(
            (assignment_count, node_count), -1, dtype=np.int64
        ),
        "support_indices": np.full((assignment_count, node_count), -1, dtype=np.int64),
        "support_distance_m": np.full((assignment_count, node_count), np.inf),
    }
    for assignment_index, carrier in branch_carriers.items():
        arrays["points_world_m"][assignment_index] = carrier.points_world_m
        arrays["local_covariance_m2"][assignment_index] = carrier.local_covariance_m2
        arrays["shared_bias_covariance_m2"][assignment_index] = (
            carrier.shared_bias_covariance_m2
        )
        arrays["marginal_covariance_m2"][assignment_index] = (
            carrier.marginal_covariance_m2
        )
        arrays["prior_reliability"][assignment_index] = carrier.prior_reliability
        arrays["reference_pixel_xy"][assignment_index] = carrier.reference_pixel_xy
        arrays["information_cluster_id"][assignment_index] = (
            carrier.information_cluster_id
        )
        arrays["support_indices"][assignment_index] = carrier.support_indices
        arrays["support_distance_m"][assignment_index] = carrier.support_distance_m

    output.mkdir(parents=True)
    arrays_path = output / "tactile_prompted_carrier.npz"
    _deterministic_npz(arrays_path, arrays)
    processing_episode_index = (
        validation["source_case"]["processing_episode_index"]
        if validation is not None
        else parent["source_case"]["processing_episode_index"]
    )
    input_records: dict[str, object] = {
        "metric_gauge_result_sha256": _sha256(metric_result_path),
        "robot_prefix_sha256": _sha256(robot_prefix_path),
        "tactile_geometry_sha256": _sha256(tactile_geometry_path),
        "selector_source_sha256": _sha256(selector_source),
        "sam2_repository_revision": _git_head(sam2_repository),
        "sam2_checkpoint_sha256": _sha256(sam2_checkpoint),
    }
    if validation is None:
        assert parent_lock_path is not None
        input_records["parent_carrier_lock_sha256"] = _sha256(parent_lock_path)
    else:
        assert validation_lock_path is not None
        input_records.update(
            {
                "validation_lock_id": validation["artifact_id"],
                "validation_lock_sha256": _sha256(validation_lock_path),
                "motioncrafter_job_manifest_sha256": _sha256(
                    args.motioncrafter_job_manifest.resolve()
                ),
                "provider_run_report_sha256": _sha256(
                    args.provider_run_report.resolve()
                ),
            }
        )
    descriptor: dict[str, object] = {
        "schema": (
            "bayesian-phystwin.deform360-tactile-prompted-carrier-validation"
            if validation is not None
            else "bayesian-phystwin.deform360-tactile-prompted-carrier-development"
        ),
        "schema_version": 1,
        "status": (
            "source-only-independent-calibration-complete"
            if validation is not None
            else "source-only-development-complete"
        ),
        "implementation_revision": _git_head(repository),
        "source_case": {
            "object_id": object_id,
            "processing_episode_index": processing_episode_index,
            "prompt_source_frame": prompt_frame,
        },
        "policy": TACTILE_PROMPTED_CARRIER_POLICY,
        "branches": branch_records,
        "inputs": input_records,
        "outputs": {
            "arrays": arrays_path.name,
            "arrays_sha256": _sha256(arrays_path),
        },
        "information_boundary": {
            "calibration_scores_opened": False,
            "confirmation_payloads_opened": False,
            "future_camera_frames_used": False,
            "future_tactile_values_used": False,
            "held_v8_accessed": False,
            "physical_state_residual_used_for_reliability": False,
            "target_outcomes_used": False,
        },
        "claim_boundary": (
            "Independent calibration-object source-only carrier admission. This "
            "does not authorize a state update, score access, confirmation access, "
            "or a SOTA claim. Each failed tactile-assignment branch remains an exact "
            "fallback with its original prior mass."
            if validation is not None
            else "Exploratory source-only carrier feasibility on an already-open "
            "case. This does not authorize a state update, score access, "
            "confirmation access, or a SOTA claim. Each failed tactile-assignment "
            "branch remains an exact fallback with its original prior mass."
        ),
    }
    result = {"artifact_id": content_id(descriptor), **descriptor}
    (output / "tactile_prompted_carrier_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
