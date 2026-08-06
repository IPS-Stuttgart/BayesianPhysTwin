"""Build, seal, fuse, and score a multiview TAPIP3D competence control."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..phystwin_tapip3d_competence import (
    IdentityTrajectory,
    identity_trajectory_metrics,
    load_canonical_tapip3d_prediction,
    load_tapip3d_prediction,
    save_canonical_tapip3d_prediction,
    shared_support_displacement_metrics,
    validate_tapip3d_prediction_contract,
)
from ..phystwin_tapip3d_multiview import (
    evaluate_multiview_tapip3d_gates,
    expand_tapip3d_view,
    fuse_tapip3d_views,
    load_multiview_tapip3d_prediction,
    multiview_identity_trajectory,
    multiview_support_diagnostics,
    save_multiview_tapip3d_prediction,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported multiview TAPIP3D protocol schema")
    if protocol.get("protocol_id") != "phystwin-tapip3d-multiview-competence-v1":
        raise ValueError("unexpected multiview TAPIP3D protocol")
    boundary = protocol.get("information_boundary", {})
    if boundary.get("held_v8_access") is not False:
        raise ValueError("protocol must explicitly forbid held-v8 access")
    if boundary.get("future_after_prefix_used") is not False:
        raise ValueError("protocol must explicitly forbid future observations")
    return protocol


def _load_query_points(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if "query_point" not in archive.files:
            raise ValueError("query source lacks query_point")
        query_points = np.asarray(archive["query_point"], dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 4:
        raise ValueError("query_point must have shape (N, 4)")
    if not np.all(np.isfinite(query_points)):
        raise ValueError("query_point must be finite")
    if not np.all(query_points[:, 0] == 0.0):
        raise ValueError("only frame-zero query points are permitted")
    return query_points


def frame_zero_depth_eligibility(
    query_points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    depth_m: np.ndarray,
    *,
    maximum_depth_error_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determine view eligibility using frame-zero geometry only."""

    queries = np.asarray(query_points_world_m, dtype=np.float64)
    calibration = np.asarray(world_to_camera, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("query_points_world_m must have shape (N, 3)")
    if camera_matrix.shape != (3, 3) or calibration.shape != (4, 4):
        raise ValueError("invalid camera calibration shape")
    if depth.ndim != 2:
        raise ValueError("depth_m must be a two-dimensional image")
    if maximum_depth_error_m <= 0.0:
        raise ValueError("maximum_depth_error_m must be positive")
    homogeneous = np.concatenate(
        [queries, np.ones((len(queries), 1), dtype=np.float64)],
        axis=1,
    )
    camera = homogeneous @ calibration.T
    projected = camera[:, :3] @ camera_matrix.T
    pixels = projected[:, :2] / projected[:, 2:3]
    rounded = np.rint(pixels).astype(np.int64)
    inside = (
        (camera[:, 2] > 0.0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < depth.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < depth.shape[0])
    )
    sampled_depth = np.full(len(queries), np.nan, dtype=np.float64)
    indices = np.flatnonzero(inside)
    sampled_depth[indices] = depth[rounded[indices, 1], rounded[indices, 0]]
    depth_error = np.abs(sampled_depth - camera[:, 2])
    eligible = (
        inside
        & np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & (depth_error <= maximum_depth_error_m)
    )
    return eligible, pixels, depth_error


def _decode_video(path: Path, frame_count: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised on vision runtime
        raise RuntimeError("build-view-input requires the vision extra") from exc
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(np.asarray(frame[..., ::-1], dtype=np.uint8))
    finally:
        capture.release()
    if len(frames) != frame_count:
        raise ValueError(
            f"video supplied {len(frames)} frames, expected {frame_count}"
        )
    return np.stack(frames)


def _aggregate_depth_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for frame_index, path in enumerate(paths):
        digest.update(f"{frame_index}\0{_sha256(path)}\n".encode())
    return digest.hexdigest()


def build_view_input(args: argparse.Namespace) -> int:
    """Build one official TAPIP3D input without reading later identities."""

    case_dir = Path(args.case_dir)
    camera_index = int(args.camera_index)
    frame_count = int(args.prefix_frame_count)
    query_source = Path(args.query_source_npz)
    query_points = _load_query_points(query_source)
    metadata_path = case_dir / "metadata.json"
    calibration_path = case_dir / "calibrate.pkl"
    video_path = case_dir / "color" / f"{camera_index}.mp4"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with calibration_path.open("rb") as handle:
        camera_to_world = np.asarray(pickle.load(handle), dtype=np.float64)
    intrinsics_all = np.asarray(metadata["intrinsics"], dtype=np.float64)
    if not 0 <= camera_index < len(intrinsics_all):
        raise ValueError("camera_index is outside the metadata camera panel")
    if camera_to_world.shape != (len(intrinsics_all), 4, 4):
        raise ValueError("calibration camera panel differs from metadata")
    depth_paths = [
        case_dir / "depth" / str(camera_index) / f"{index}.npy"
        for index in range(frame_count)
    ]
    missing = [str(path) for path in depth_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing depth frames: {missing[:3]}")
    depths = np.stack(
        [np.asarray(np.load(path), dtype=np.float32) / 1000.0 for path in depth_paths]
    )
    video = _decode_video(video_path, frame_count)
    intrinsics = intrinsics_all[camera_index]
    world_to_camera = np.linalg.inv(camera_to_world[camera_index])
    eligible, pixels, depth_error = frame_zero_depth_eligibility(
        query_points[:, 1:],
        intrinsics,
        world_to_camera,
        depths[0],
        maximum_depth_error_m=float(args.maximum_frame_zero_depth_error_m),
    )
    selected_indices = np.flatnonzero(eligible)
    if len(selected_indices) == 0:
        raise ValueError("camera has no eligible frame-zero query identities")
    selected_queries = query_points[selected_indices]
    repeated_intrinsics = np.repeat(
        intrinsics.astype(np.float32)[None], frame_count, axis=0
    )
    repeated_extrinsics = np.repeat(
        world_to_camera.astype(np.float32)[None], frame_count, axis=0
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        video=video,
        depths=depths,
        intrinsics=repeated_intrinsics,
        extrinsics=repeated_extrinsics,
        query_point=selected_queries.astype(np.float32),
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "TAPIP3DPhysTwinMultiviewFrameZeroInputV1",
        "created_at_utc": _utc_now(),
        "case": args.case,
        "camera_index": camera_index,
        "frame_range": [0, frame_count],
        "global_query_count": len(query_points),
        "selected_query_count": len(selected_indices),
        "selected_global_query_indices": selected_indices.tolist(),
        "selected_frame_zero_pixels": pixels[selected_indices].tolist(),
        "selected_frame_zero_depth_error_m": depth_error[selected_indices].tolist(),
        "maximum_frame_zero_depth_error_m": float(
            args.maximum_frame_zero_depth_error_m
        ),
        "query_source": {
            "path": str(query_source),
            "sha256": _sha256(query_source),
            "field_read": "query_point only",
        },
        "input_npz": {"path": str(output), "sha256": _sha256(output)},
        "raw_inputs": {
            "video": {"path": str(video_path), "sha256": _sha256(video_path)},
            "metadata": {
                "path": str(metadata_path),
                "sha256": _sha256(metadata_path),
            },
            "calibration": {
                "path": str(calibration_path),
                "sha256": _sha256(calibration_path),
            },
            "depth_frame_count": len(depth_paths),
            "depth_aggregate_sha256": _aggregate_depth_digest(depth_paths),
        },
        "depth_units": "metres",
        "extrinsics_convention": "world-to-camera",
        "manual_identity_role": "frame-zero query coordinates only",
        "later_manual_trajectory_loaded": False,
        "future_after_prefix_loaded": False,
        "held_v8_access": False,
    }
    manifest_path = Path(args.manifest)
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _expected_view(protocol: dict[str, Any], camera_index: int) -> dict[str, Any]:
    for view in protocol["views"]:
        if int(view["camera_index"]) == camera_index:
            return view
    raise ValueError(f"camera {camera_index} is not in the locked view panel")


def _verify_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def seal_view_prediction(args: argparse.Namespace) -> int:
    """Bind one official result to its locked view before scoring."""

    protocol_path = Path(args.protocol)
    protocol = _load_protocol(protocol_path)
    camera_index = int(args.camera_index)
    expected = _expected_view(protocol, camera_index)
    input_manifest_path = Path(args.input_manifest)
    _verify_hash(
        input_manifest_path,
        expected["input_manifest_sha256"],
        "view input manifest",
    )
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if int(input_manifest["camera_index"]) != camera_index:
        raise ValueError("input manifest camera differs from the requested camera")
    input_path = Path(input_manifest["input_npz"]["path"])
    _verify_hash(input_path, expected["input_npz_sha256"], "view input")
    checkpoint = Path(protocol["tapip3d"]["checkpoint"]["path"])
    _verify_hash(
        checkpoint,
        protocol["tapip3d"]["checkpoint"]["sha256"],
        "TAPIP3D checkpoint",
    )
    result_path = Path(args.tapip3d_result)
    prediction = load_tapip3d_prediction(result_path)
    expected_queries = _load_query_points(input_path)
    validate_tapip3d_prediction_contract(
        prediction,
        expected_queries,
        expected_frame_count=int(protocol["prefix_frame_count"]),
        query_tolerance_m=float(protocol["query_tolerance_m"]),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    carrier_path = output_dir / "prediction.npz"
    save_canonical_tapip3d_prediction(carrier_path, prediction)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTapip3dMultiviewViewPredictionV1",
        "created_at_utc": _utc_now(),
        "case": protocol["case"],
        "camera_index": camera_index,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "input_manifest": {
            "path": str(input_manifest_path),
            "sha256": _sha256(input_manifest_path),
        },
        "source_result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "prediction": {"path": str(carrier_path), "sha256": _sha256(carrier_path)},
        "implementation_commit": _git_revision(),
        "frame_count": int(prediction.coords_world_m.shape[0]),
        "query_count": int(prediction.coords_world_m.shape[1]),
        "visible_fraction_before_target_intersection": float(np.mean(prediction.valid)),
        "later_manual_trajectory_loaded": False,
        "future_after_prefix_loaded": False,
        "held_v8_access": False,
    }
    manifest_path = output_dir / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _load_view_manifest(
    path: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != "PhysTwinTapip3dMultiviewViewPredictionV1":
        raise ValueError("unexpected view prediction artifact")
    if manifest["protocol"]["sha256"] != _sha256(protocol_path):
        raise ValueError("view prediction was not sealed under this protocol")
    for field in (
        "later_manual_trajectory_loaded",
        "future_after_prefix_loaded",
        "held_v8_access",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"view prediction violates boundary: {field}")
    prediction_path = Path(manifest["prediction"]["path"])
    _verify_hash(prediction_path, manifest["prediction"]["sha256"], "view carrier")
    return manifest, prediction_path


def seal_fusion(args: argparse.Namespace) -> int:
    """Fuse all sealed view predictions without score targets."""

    protocol_path = Path(args.protocol)
    protocol = _load_protocol(protocol_path)
    supplied: dict[int, tuple[Path, dict[str, Any], Path]] = {}
    for value in args.prediction_manifest:
        path = Path(value)
        manifest, prediction_path = _load_view_manifest(path, protocol_path)
        camera_index = int(manifest["camera_index"])
        if camera_index in supplied:
            raise ValueError(f"duplicate camera prediction: {camera_index}")
        supplied[camera_index] = (path, manifest, prediction_path)
    expected_cameras = {int(view["camera_index"]) for view in protocol["views"]}
    if set(supplied) != expected_cameras:
        raise ValueError("sealed camera panel differs from the locked protocol")
    predictions = [
        load_canonical_tapip3d_prediction(supplied[index][2])
        for index in sorted(supplied)
    ]
    global_queries = _load_query_points(Path(protocol["global_query_source"]["path"]))
    _verify_hash(
        Path(protocol["global_query_source"]["path"]),
        protocol["global_query_source"]["sha256"],
        "global query source",
    )
    fusion = protocol["fusion"]
    fused = fuse_tapip3d_views(
        predictions,
        global_queries,
        minimum_view_count=int(fusion["minimum_view_count"]),
        maximum_pairwise_disagreement_m=float(
            fusion["maximum_pairwise_disagreement_m"]
        ),
        shared_bias_floor_m=float(fusion["shared_bias_floor_m"]),
        query_tolerance_m=float(protocol["query_tolerance_m"]),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    carrier_path = output_dir / "multiview_prediction.npz"
    save_multiview_tapip3d_prediction(carrier_path, fused)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTapip3dMultiviewFusionPredictionV1",
        "created_at_utc": _utc_now(),
        "case": protocol["case"],
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "view_predictions": [
            {
                "camera_index": index,
                "manifest_path": str(supplied[index][0]),
                "manifest_sha256": _sha256(supplied[index][0]),
                "prediction_sha256": supplied[index][1]["prediction"]["sha256"],
            }
            for index in sorted(supplied)
        ],
        "prediction": {"path": str(carrier_path), "sha256": _sha256(carrier_path)},
        "target_free_support": multiview_support_diagnostics(fused),
        "implementation_commit": _git_revision(),
        "later_manual_trajectory_loaded": False,
        "future_after_prefix_loaded": False,
        "held_v8_access": False,
    }
    manifest_path = output_dir / "fusion_manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _verify_fusion_manifest(
    path: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != "PhysTwinTapip3dMultiviewFusionPredictionV1":
        raise ValueError("unexpected fusion prediction artifact")
    if manifest["protocol"]["sha256"] != _sha256(protocol_path):
        raise ValueError("fusion prediction was not sealed under this protocol")
    for field in (
        "later_manual_trajectory_loaded",
        "future_after_prefix_loaded",
        "held_v8_access",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"fusion prediction violates boundary: {field}")
    prediction_path = Path(manifest["prediction"]["path"])
    _verify_hash(prediction_path, manifest["prediction"]["sha256"], "fusion carrier")
    return manifest, prediction_path


def _covariance_diagnostics(
    coords: np.ndarray,
    valid: np.ndarray,
    covariance: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int | None]:
    finite_target = np.all(np.isfinite(target), axis=2)
    selected = valid & finite_target
    if not np.any(selected):
        return {"count": 0, "mean_nees": None, "ellipsoid_90_coverage": None}
    errors = coords[selected] - target[selected]
    selected_covariance = covariance[selected]
    solved = np.linalg.solve(selected_covariance, errors[..., None])[..., 0]
    nees = np.sum(errors * solved, axis=1)
    return {
        "count": int(len(nees)),
        "mean_nees": float(np.mean(nees)),
        "ellipsoid_90_coverage": float(np.mean(nees <= 6.251388631170325)),
    }


def score_fusion(args: argparse.Namespace) -> int:
    """Score the already-sealed association-oracle source prediction."""

    protocol_path = Path(args.protocol)
    protocol = _load_protocol(protocol_path)
    fusion_manifest_path = Path(args.fusion_manifest)
    fusion_manifest, prediction_path = _verify_fusion_manifest(
        fusion_manifest_path, protocol_path
    )
    fused = load_multiview_tapip3d_prediction(prediction_path)
    manual_path = Path(args.case_dir) / "gt_track_3d.pkl"
    _verify_hash(manual_path, protocol["score_inputs"]["gt_track_sha256"], "manual track")
    with manual_path.open("rb") as handle:
        manual = np.asarray(pickle.load(handle), dtype=np.float64)[
            : int(protocol["prefix_frame_count"])
        ]
    if manual.shape != fused.coords_world_m.shape:
        raise ValueError("manual trajectory shape differs from the fused prediction")
    if np.max(np.linalg.norm(manual[0] - fused.query_points[:, 1:], axis=1)) > float(
        protocol["query_tolerance_m"]
    ):
        raise ValueError("locked global queries differ from manual frame-zero identities")
    trajectory = multiview_identity_trajectory(fused)
    metrics = identity_trajectory_metrics(trajectory, manual)
    late_start = 2 * len(manual) // 3
    late_metrics = identity_trajectory_metrics(
        trajectory, manual, frame_start=late_start
    )
    view_metrics: dict[str, Any] = {}
    view_trajectories: dict[int, IdentityTrajectory] = {}
    for entry in fusion_manifest["view_predictions"]:
        camera_index = int(entry["camera_index"])
        view_manifest_path = Path(entry["manifest_path"])
        _verify_hash(
            view_manifest_path,
            entry["manifest_sha256"],
            f"camera {camera_index} manifest",
        )
        view_manifest, view_prediction_path = _load_view_manifest(
            view_manifest_path, protocol_path
        )
        if view_manifest["prediction"]["sha256"] != entry["prediction_sha256"]:
            raise ValueError("fusion manifest view lineage mismatch")
        view_prediction = load_canonical_tapip3d_prediction(view_prediction_path)
        view_trajectory = expand_tapip3d_view(
            view_prediction,
            fused.query_points,
            query_tolerance_m=float(protocol["query_tolerance_m"]),
        )
        view_trajectories[camera_index] = view_trajectory
        view_metrics[str(camera_index)] = identity_trajectory_metrics(
            view_trajectory, manual
        )
    reference_camera = int(protocol["competence_gate"]["reference_camera_index"])
    reference_shared = shared_support_displacement_metrics(
        trajectory,
        view_trajectories[reference_camera],
        manual,
    )
    gate = protocol["competence_gate"]
    gates = evaluate_multiview_tapip3d_gates(
        metrics,
        late_metrics,
        reference_shared["first_relative_improvement_fraction"],
        minimum_support_fraction=float(gate["minimum_support_fraction"]),
        maximum_displacement_rmse_m=float(gate["maximum_displacement_rmse_m"]),
        maximum_frame_zero_anchor_rmse_m=float(
            gate["maximum_frame_zero_anchor_rmse_m"]
        ),
        minimum_late_support_fraction=float(gate["minimum_late_support_fraction"]),
        maximum_late_displacement_rmse_m=float(
            gate["maximum_late_displacement_rmse_m"]
        ),
        minimum_best_single_shared_improvement_fraction=float(
            gate["minimum_reference_shared_improvement_fraction"]
        ),
    )
    result = {
        "schema_version": 1,
        "study_id": protocol["protocol_id"],
        "status": (
            "opened one-case association-oracle multiview observation competence "
            "control; not prediction, transfer, confirmation, or SOTA evidence"
        ),
        "case": protocol["case"],
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "fusion_manifest": {
            "path": str(fusion_manifest_path),
            "sha256": _sha256(fusion_manifest_path),
        },
        "multiview": metrics,
        "multiview_late_third": late_metrics,
        "per_view": view_metrics,
        "reference_camera_shared_comparison": reference_shared,
        "covariance_diagnostic_only": _covariance_diagnostics(
            fused.coords_world_m,
            fused.valid,
            fused.observation_covariance_m2,
            manual,
        ),
        "gates": gates,
        "recommendation": (
            "Proceed to a locked automatic frame-zero query source study only."
            if gates["competence_gate_passed"]
            else "Stop this multiview TAPIP3D feeder without tuning on the opened case."
        ),
        "information_boundary": {
            "rgbd_frames_used_by_prediction": [0, int(protocol["prefix_frame_count"])],
            "manual_identity_role": (
                "frame-zero coordinates are prediction queries; the sealed prefix "
                "trajectory is score-only"
            ),
            "future_after_prefix_used": False,
            "held_v8_access": False,
        },
        "scored_at_utc": _utc_now(),
    }
    output = Path(args.output)
    _write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-view-input")
    build.add_argument("--case", required=True)
    build.add_argument("--case-dir", required=True)
    build.add_argument("--query-source-npz", required=True)
    build.add_argument("--camera-index", required=True, type=int)
    build.add_argument("--prefix-frame-count", required=True, type=int)
    build.add_argument("--maximum-frame-zero-depth-error-m", type=float, default=0.01)
    build.add_argument("--output", required=True)
    build.add_argument("--manifest", required=True)
    build.set_defaults(function=build_view_input)

    seal_view = subparsers.add_parser("seal-view-prediction")
    seal_view.add_argument("--protocol", required=True)
    seal_view.add_argument("--camera-index", required=True, type=int)
    seal_view.add_argument("--input-manifest", required=True)
    seal_view.add_argument("--tapip3d-result", required=True)
    seal_view.add_argument("--output-dir", required=True)
    seal_view.set_defaults(function=seal_view_prediction)

    seal_all = subparsers.add_parser("seal-fusion")
    seal_all.add_argument("--protocol", required=True)
    seal_all.add_argument("--prediction-manifest", action="append", required=True)
    seal_all.add_argument("--output-dir", required=True)
    seal_all.set_defaults(function=seal_fusion)

    score = subparsers.add_parser("score")
    score.add_argument("--protocol", required=True)
    score.add_argument("--fusion-manifest", required=True)
    score.add_argument("--case-dir", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(function=score_fusion)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
