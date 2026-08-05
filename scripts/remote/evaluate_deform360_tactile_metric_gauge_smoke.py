#!/usr/bin/env python3
"""Evaluate the frozen source-only tactile metric-gauge feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_official_hub_motioncrafter_jobs import (
    load_deform360_motioncrafter_job_manifest,
)
from bayesian_phystwin.deform360_tactile_contact_geometry import (
    verify_tactile_contact_geometry_artifact,
)
from bayesian_phystwin.deform360_tactile_metric_gauge import (
    covariance_intersection_equal_weight,
    fit_robust_similarity,
    held_frame_gauge_quality,
    load_tactile_metric_gauge_lock,
    project_world_points_to_target,
    sample_point_map_bilinear,
    unknown_correlation_covariance_union,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_clean(root: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("Bayesian-PhysTwin checkout is dirty")
    return head


def _require_ancestor(root: Path, ancestor: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("checkout does not contain the frozen provider implementation")


def _safe_member(root: Path, relative: str) -> Path:
    candidate = (root.resolve() / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes provider root: {relative}") from error
    return candidate


def _load_content_addressed_report(path: Path, *, id_field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load run report {path}") from error
    if not isinstance(value, dict) or not isinstance(value.get(id_field), str):
        raise ValueError(f"invalid run report {path}")
    descriptor = dict(value)
    declared = descriptor.pop(id_field)
    if content_id(descriptor) != declared:
        raise ValueError(f"run report identity changed: {path}")
    return value


def _load_camera_dictionary(path: Path) -> dict[str, np.ndarray]:
    values = np.load(path, allow_pickle=True)
    if (
        not isinstance(values, np.ndarray)
        or values.shape != ()
        or values.dtype != object
    ):
        raise ValueError(f"unexpected camera archive {path}")
    mapping = values.item()
    if not isinstance(mapping, dict):
        raise ValueError(f"invalid camera dictionary {path}")
    return {str(name): np.asarray(value) for name, value in mapping.items()}


def _validate_prediction_payload(
    payload: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> Path:
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("prediction manifest lacks config")
    expected_config = {
        "model_type": configuration["model_type"],
        "height": configuration["height"],
        "width": configuration["width"],
        "window_size": configuration["window_size"],
        "overlap": configuration["overlap"],
        "num_inference_steps": configuration["num_inference_steps"],
        "guidance_scale": configuration["guidance_scale"],
        "decode_chunk_size": configuration["decode_chunk_size"],
        "seed": configuration["seed"],
        "seed_policy": configuration["seed_policy"],
        "low_memory_usage": configuration["low_memory_usage"],
        "frame_start": job["source_frame_start"],
        "frame_stop": job["source_frame_stop_exclusive"],
        "frame_stride": configuration["frame_stride"],
        "model_source_set_sha256": configuration["model_source_set_sha256"],
    }
    if {name: config.get(name) for name in expected_config} != expected_config:
        raise ValueError("prediction configuration changed")
    if payload.get("stochastic_seed_schedule", {}).get("calls") != job["seed_schedule"]:
        raise ValueError("prediction seed schedule changed")
    expected_windows = [
        {
            "window_id": item["window_id"],
            "path": f"windows/{item['window_id']}.npz",
            "start_frame": item["source_frame_start"],
            "stop_frame": item["source_frame_stop_exclusive"],
        }
        for item in job["windows"]
    ]
    if payload.get("overlap_windows") != expected_windows:
        raise ValueError("prediction windows changed")
    return Path(expected_windows[-1]["path"])


def _transform_record(transform: Any) -> dict[str, Any]:
    return {
        "scale": float(transform.scale),
        "rotation": np.asarray(transform.rotation).tolist(),
        "translation_m": np.asarray(transform.translation).tolist(),
    }


def _hypothesis_record(
    *,
    source_points: np.ndarray,
    target_points_m: np.ndarray,
    frame_ids: np.ndarray,
    projection_visible: np.ndarray,
    provider_support: np.ndarray,
    quality_gate: Mapping[str, Any],
) -> dict[str, Any]:
    support = np.asarray(projection_visible, dtype=bool) & np.asarray(
        provider_support, dtype=bool
    )
    reasons: list[str] = []
    if not np.all(projection_visible):
        reasons.append("projection-support-incomplete")
    if not np.all(provider_support):
        reasons.append("provider-support-incomplete")
    quality = None
    transform = None
    if len(np.unique(frame_ids[support])) < 3:
        reasons.append("fewer-than-three-supported-frames")
    else:
        quality = held_frame_gauge_quality(
            source_points[support],
            target_points_m[support],
            frame_ids[support],
            huber_delta_m=float(quality_gate["huber_delta_m"]),
            covariance_floor_m=float(quality_gate["covariance_floor_m"]),
            maximum_median_error_m=float(
                quality_gate["maximum_median_held_frame_error_m"]
            ),
            maximum_percentile_90_error_m=float(
                quality_gate["maximum_percentile_90_held_frame_error_m"]
            ),
        )
        reasons.extend(quality.reason_codes)
        try:
            transform = fit_robust_similarity(
                source_points[support],
                target_points_m[support],
                huber_delta_m=float(quality_gate["huber_delta_m"]),
            )
        except ValueError:
            reasons.append("final-fit-degenerate")
    admitted = not reasons and quality is not None and transform is not None
    return {
        "admitted": admitted,
        "reason_codes": sorted(set(reasons)),
        "row_count": int(len(frame_ids)),
        "supported_row_count": int(np.sum(support)),
        "supported_frame_count": int(len(np.unique(frame_ids[support]))),
        "median_held_frame_error_m": (
            None if quality is None else quality.median_error_m
        ),
        "percentile_90_held_frame_error_m": (
            None if quality is None else quality.percentile_90_error_m
        ),
        "maximum_held_frame_error_m": (
            None if quality is None else quality.maximum_error_m
        ),
        "covariance_m2": (None if quality is None else quality.covariance_m2.tolist()),
        "similarity_transform": (
            None if transform is None else _transform_record(transform)
        ),
    }


def _gate_summary(
    camera_records: Sequence[Mapping[str, Any]],
    *,
    quality_gate: Mapping[str, Any],
    assignment_probabilities: Sequence[float],
) -> dict[str, Any]:
    minimum_admitted_cameras = int(quality_gate["minimum_admitted_cameras"])
    correlation_policy = str(quality_gate["cross_view_correlation"])
    joint = [
        record
        for record in camera_records
        if all(item["admitted"] for item in record["assignment_hypotheses"])
    ]
    assignment_records: list[dict[str, Any]] = []
    for hypothesis, probability in enumerate(assignment_probabilities):
        covariances = np.asarray(
            [
                record["assignment_hypotheses"][hypothesis]["covariance_m2"]
                for record in joint
            ],
            dtype=np.float64,
        )
        record: dict[str, Any] = {
            "assignment_hypothesis": hypothesis,
            "prior_probability": float(probability),
            "admitted_camera_count": len(joint),
        }
        if len(covariances) < minimum_admitted_cameras:
            record["covariance_intersection_m2"] = None
        elif correlation_policy == "unknown-equal-weight-covariance-intersection":
            record["covariance_intersection_m2"] = (
                covariance_intersection_equal_weight(covariances).tolist()
            )
        elif correlation_policy == "unknown-no-precision-gain-covariance-union":
            record["covariance_intersection_m2"] = None
            record["conservative_covariance_union_m2"] = (
                unknown_correlation_covariance_union(
                    covariances,
                    shared_bias_floor_m=float(quality_gate["shared_bias_floor_m"]),
                ).tolist()
            )
        else:
            raise ValueError("unsupported cross-view covariance policy")
        assignment_records.append(record)
    admitted = len(joint) >= minimum_admitted_cameras
    return {
        "metric_gauge_authorized": admitted,
        "contact_anchor_authorized": False,
        "jointly_admitted_cameras": [record["camera"] for record in joint],
        "assignment_mixture": assignment_records,
        "reason_codes": (
            [] if admitted else ["fewer-than-required-jointly-admitted-cameras"]
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--parent-job-manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--parent-provider-root", type=Path, required=True)
    parser.add_argument("--supplemental-provider-root", type=Path, required=True)
    parser.add_argument("--tactile-manifest", type=Path, required=True)
    parser.add_argument("--prob4d-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository_root = args.repository_root.resolve()
    runtime_revision = _require_clean(repository_root)
    evaluator_source = Path(__file__).resolve()
    try:
        evaluator_source.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("evaluator is outside the repository") from error
    lock_path = args.lock.resolve()
    lock = load_tactile_metric_gauge_lock(lock_path)
    implementation = lock["implementation"]
    assert isinstance(implementation, Mapping)
    _require_ancestor(repository_root, str(implementation["revision"]))

    parent_manifest_path = args.parent_job_manifest.resolve()
    parent_manifest = load_deform360_motioncrafter_job_manifest(parent_manifest_path)
    parent_record = lock["parents"]["motioncrafter_job_manifest"]
    if (
        _sha256(parent_manifest_path) != parent_record["sha256"]
        or parent_manifest["manifest_sha256"] != parent_record["artifact_id"]
    ):
        raise ValueError("parent job manifest changed")
    tactile_path = args.tactile_manifest.resolve()
    tactile = verify_tactile_contact_geometry_artifact(tactile_path)
    tactile_record = lock["parents"]["tactile_contact_geometry"]
    if (
        _sha256(tactile_path) != tactile_record["sha256"]
        or tactile["artifact_id"] != tactile_record["artifact_id"]
    ):
        raise ValueError("tactile geometry changed")

    source = lock["source_case"]
    camera_policy = lock["camera_selection"]["policy"]
    object_id = str(source["object_id"])
    episode_index = int(source["processing_episode_index"])
    episode_dir = (
        args.processed_root.resolve() / object_id / f"episode_{episode_index:04d}"
    )
    intrinsics_path = episode_dir / "undistorted_intrinsics.npy"
    extrinsics_path = episode_dir / "extrinsics.npy"
    if (
        _sha256(intrinsics_path) != lock["calibration"]["undistorted_intrinsics_sha256"]
        or _sha256(extrinsics_path) != lock["calibration"]["extrinsics_sha256"]
    ):
        raise ValueError("camera calibration changed")
    intrinsics = _load_camera_dictionary(intrinsics_path)
    extrinsics = _load_camera_dictionary(extrinsics_path)

    tactile_archive_path = tactile_path.parent / str(tactile["archive"]["path"])
    with np.load(tactile_archive_path, allow_pickle=False) as archive:
        world_points = np.asarray(
            archive["world_points_hypotheses_m"], dtype=np.float64
        )
        frame_ids = np.asarray(archive["source_frame_ids"], dtype=np.int64)
        assignment_probabilities = np.asarray(
            archive["assignment_prior_probability"], dtype=np.float64
        )

    sys.path.insert(0, str(args.prob4d_root.resolve() / "src"))
    from prob4d.motioncrafter_integrity import (  # noqa: PLC0415
        verify_motioncrafter_prediction_manifest,
    )

    parent_jobs = {
        str(job["camera"]): job
        for job in parent_manifest["jobs"]
        if job.get("object_id") == object_id
        and job.get("episode") == f"episode_{episode_index:04d}"
    }
    supplemental_jobs = {str(job["camera"]): job for job in lock["supplemental_jobs"]}
    selected = list(lock["camera_selection"]["selected_cameras"])
    reused = set(lock["camera_selection"]["reused_provider_cameras"])
    supplemental = set(lock["camera_selection"]["supplemental_provider_cameras"])
    parent_report_path = args.parent_provider_root.resolve() / "run_report.json"
    parent_report = _load_content_addressed_report(
        parent_report_path,
        id_field="run_sha256",
    )
    if (
        parent_report.get("status") != "complete"
        or parent_report.get("job_manifest_sha256")
        != parent_manifest["manifest_sha256"]
    ):
        raise ValueError("parent provider run is incomplete or differently bound")
    supplemental_report_path: Path | None = None
    if supplemental_jobs:
        supplemental_report_path = (
            args.supplemental_provider_root.resolve() / "run_report.json"
        )
        supplemental_report = _load_content_addressed_report(
            supplemental_report_path,
            id_field="run_id",
        )
        completed_job_ids = {
            str(item["job_id"])
            for item in supplemental_report.get("completed_jobs", ())
        }
        if (
            supplemental_report.get("status") != "complete"
            or supplemental_report.get("lock_id") != lock["artifact_id"]
            or completed_job_ids
            != {str(job["job_id"]) for job in lock["supplemental_jobs"]}
        ):
            raise ValueError(
                "supplemental provider run is incomplete or differently bound"
            )
    configuration = lock["provider"]["run_configuration"]
    quality_gate = lock["provider"]["quality_gate"]
    camera_records: list[dict[str, Any]] = []
    for camera in selected:
        if camera in reused:
            provider_root = args.parent_provider_root.resolve()
            job = parent_jobs[camera]
            provider_kind = "reused-parent"
        elif camera in supplemental:
            provider_root = args.supplemental_provider_root.resolve()
            job = supplemental_jobs[camera]
            provider_kind = "supplemental"
        else:
            raise ValueError("selected camera lacks a provider partition")
        output_dir = _safe_member(provider_root, str(job["output_relative_path"]))
        prediction_path = output_dir / "predictions.json"
        verification = verify_motioncrafter_prediction_manifest(
            prediction_path,
            verify_hashes=True,
        )
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        window_relative = _validate_prediction_payload(
            payload,
            job=job,
            configuration=configuration,
        )
        window_path = _safe_member(output_dir, window_relative.as_posix())
        with np.load(window_path, allow_pickle=False) as archive:
            point_map = np.asarray(archive["point_map"])
            valid_mask = np.asarray(archive["valid_mask"], dtype=bool)
            provider_frames = np.asarray(archive["frame_indices"], dtype=np.int64)
            hypotheses: list[dict[str, Any]] = []
            for hypothesis in range(world_points.shape[1]):
                target_points = world_points[:, hypothesis]
                xy, _, visible = project_world_points_to_target(
                    target_points,
                    intrinsics=intrinsics[camera],
                    world_from_camera=extrinsics[camera],
                    source_shape=tuple(camera_policy["source_shape"]),
                    target_shape=tuple(camera_policy["target_shape"]),
                )
                sampled, provider_support = sample_point_map_bilinear(
                    point_map,
                    valid_mask,
                    provider_frames,
                    frame_ids,
                    xy,
                )
                hypotheses.append(
                    _hypothesis_record(
                        source_points=sampled,
                        target_points_m=target_points,
                        frame_ids=frame_ids,
                        projection_visible=visible,
                        provider_support=provider_support,
                        quality_gate=quality_gate,
                    )
                )
        camera_records.append(
            {
                "camera": camera,
                "provider_kind": provider_kind,
                "prediction_manifest_sha256": _sha256(prediction_path),
                "window_sha256": _sha256(window_path),
                "verified_member_count": int(verification["member_count"]),
                "assignment_hypotheses": hypotheses,
            }
        )

    gate = _gate_summary(
        camera_records,
        quality_gate=quality_gate,
        assignment_probabilities=assignment_probabilities,
    )
    source_artifacts = {
        "lock_sha256": _sha256(lock_path),
        "parent_job_manifest_sha256": _sha256(parent_manifest_path),
        "tactile_manifest_sha256": _sha256(tactile_path),
        "tactile_archive_sha256": _sha256(tactile_archive_path),
        "intrinsics_sha256": _sha256(intrinsics_path),
        "extrinsics_sha256": _sha256(extrinsics_path),
        "parent_provider_run_report_sha256": _sha256(parent_report_path),
    }
    if supplemental_report_path is not None:
        source_artifacts["supplemental_provider_run_report_sha256"] = _sha256(
            supplemental_report_path
        )
    descriptor = {
        "schema": "bayesian-phystwin.deform360-tactile-metric-gauge-result",
        "schema_version": 1,
        "status": "admitted" if gate["metric_gauge_authorized"] else "fallback",
        "lock_id": lock["artifact_id"],
        "implementation": {
            "runtime_revision": runtime_revision,
            "evaluator_source_sha256": _sha256(evaluator_source),
        },
        "source_artifacts": source_artifacts,
        "sampling_policy": {
            "coordinate_mapping": "motioncrafter-cover-resize-center-crop",
            "point_map_sampling": "bilinear-all-four-neighbors-valid",
            "validation_partition": "leave-one-complete-source-frame-out",
            "row_support": "all-projected-contact-rows-required",
        },
        "camera_results": camera_records,
        "gate": gate,
        "information_boundary": lock["information_boundary"],
        "claim_boundary": (
            "Source-only metric-gauge feasibility. Even admission does not authorize "
            "a tactile object-state update, open calibration scores, or support a "
            "confirmation or SOTA claim."
        ),
    }
    result = {"artifact_id": content_id(descriptor), **descriptor}
    write_atomic_json(result, args.output, overwrite=args.overwrite)
    print(
        f"artifact_id={result['artifact_id']} status={result['status']} "
        f"admitted_cameras={len(gate['jointly_admitted_cameras'])}"
    )


if __name__ == "__main__":
    main()
