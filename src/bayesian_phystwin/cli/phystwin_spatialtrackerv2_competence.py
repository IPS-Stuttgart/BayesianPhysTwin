"""Seal and score a prefix-only SpatialTrackerV2 PhysTwin competence control."""

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

from ..phystwin_cotracker3_cues import (
    load_cotracker3_multiview_observations,
)
from ..phystwin_spatialtrackerv2_competence import (
    load_canonical_spatialtrackerv2_prediction,
    load_spatialtrackerv2_prediction,
    project_world_queries_to_pixels,
    save_canonical_spatialtrackerv2_prediction,
    validate_spatialtrackerv2_prediction_contract,
)
from ..phystwin_tapip3d_competence import (
    IdentityTrajectory,
    build_same_query_cotracker3_trajectory,
    evaluate_tapip3d_competence_gates,
    identity_trajectory_metrics,
    shared_support_displacement_metrics,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path: str | Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported SpatialTrackerV2 competence protocol schema")
    if protocol.get("protocol_id") != "phystwin-spatialtrackerv2-competence-v1":
        raise ValueError("unexpected SpatialTrackerV2 competence protocol")
    if protocol.get("information_boundary", {}).get("held_v8_access") is not False:
        raise ValueError("protocol must explicitly forbid held-v8 access")
    return protocol


def _load_input_contract(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        required = {"query_point", "intrinsics", "extrinsics"}
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"locked RGB-D input lacks fields: {names}")
        queries = np.asarray(archive["query_point"])
        intrinsics = np.asarray(archive["intrinsics"])
        extrinsics = np.asarray(archive["extrinsics"])
    query_pixels, _ = project_world_queries_to_pixels(
        queries[:, 1:],
        intrinsics[0],
        extrinsics[0],
    )
    return queries, query_pixels


def seal_prediction(args: argparse.Namespace) -> int:
    """Seal output before any trajectory labels or comparator cues are opened."""

    protocol_path = Path(args.protocol)
    protocol = _load_protocol(protocol_path)
    input_manifest_path = Path(args.input_manifest)
    expected_manifest = protocol["inputs"]["input_manifest"]
    _verify_hash(
        input_manifest_path,
        expected_manifest["sha256"],
        "input manifest",
    )
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    input_path = Path(input_manifest["input_npz"]["path"])
    if (
        input_manifest["input_npz"]["sha256"]
        != protocol["inputs"]["input_npz"]["sha256"]
    ):
        raise ValueError("input manifest and protocol disagree on input SHA-256")
    _verify_hash(
        input_path,
        protocol["inputs"]["input_npz"]["sha256"],
        "SpatialTrackerV2 input",
    )
    checkpoint = protocol["spatialtrackerv2"]["checkpoint"]
    _verify_hash(
        checkpoint["model_path"],
        checkpoint["model_sha256"],
        "SpatialTrackerV2 model",
    )
    _verify_hash(
        checkpoint["config_path"],
        checkpoint["config_sha256"],
        "SpatialTrackerV2 config",
    )

    result_path = Path(args.spatialtrackerv2_result)
    prediction = load_spatialtrackerv2_prediction(result_path)
    expected_queries, expected_pixels = _load_input_contract(input_path)
    validate_spatialtrackerv2_prediction_contract(
        prediction,
        expected_queries,
        expected_pixels,
        expected_frame_count=int(protocol["prefix_frame_count"]),
        world_tolerance_m=float(protocol["query_tolerance_m"]),
        pixel_tolerance=float(protocol["pixel_tolerance"]),
    )
    if prediction.coords_world_m.shape[1] != int(protocol["query_count"]):
        raise ValueError("SpatialTrackerV2 query count differs from protocol")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    carrier_path = output / "prediction.npz"
    save_canonical_spatialtrackerv2_prediction(carrier_path, prediction)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinSpatialTrackerV2CompetencePredictionV1",
        "created_at_utc": _utc_now(),
        "case": protocol["case"],
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
        },
        "input_manifest": {
            "path": str(input_manifest_path),
            "sha256": _sha256(input_manifest_path),
        },
        "spatialtrackerv2_result": {
            "path": str(result_path),
            "sha256": _sha256(result_path),
        },
        "prediction": {
            "path": str(carrier_path),
            "sha256": _sha256(carrier_path),
        },
        "implementation_commit": _git_revision(),
        "frame_count": int(prediction.coords_world_m.shape[0]),
        "query_count": int(prediction.coords_world_m.shape[1]),
        "visible_fraction_before_target_intersection": float(np.mean(prediction.valid)),
        "manual_trajectory_loaded": False,
        "future_object_observation_loaded": False,
        "cotracker3_cues_loaded": False,
        "held_v8_access": False,
    }
    manifest_path = output / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "prediction_manifest": str(manifest_path),
                "prediction_manifest_sha256": _sha256(manifest_path),
                "prediction_sha256": _sha256(carrier_path),
                "source_result_sha256": _sha256(result_path),
                "visible_fraction_before_target_intersection": manifest[
                    "visible_fraction_before_target_intersection"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _verify_prediction_manifest(
    path: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != (
        "PhysTwinSpatialTrackerV2CompetencePredictionV1"
    ):
        raise ValueError("unexpected SpatialTrackerV2 prediction artifact")
    if manifest["protocol"]["sha256"] != _sha256(protocol_path):
        raise ValueError("prediction was not sealed under this protocol")
    for field in (
        "manual_trajectory_loaded",
        "future_object_observation_loaded",
        "cotracker3_cues_loaded",
        "held_v8_access",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"prediction manifest violates boundary: {field}")
    prediction_path = Path(manifest["prediction"]["path"])
    _verify_hash(
        prediction_path,
        manifest["prediction"]["sha256"],
        "canonical prediction",
    )
    return manifest, prediction_path


def score_prediction(args: argparse.Namespace) -> int:
    """Score an already sealed prediction on the opened source prefix."""

    protocol_path = Path(args.protocol)
    protocol = _load_protocol(protocol_path)
    manifest_path = Path(args.prediction_manifest)
    _, prediction_path = _verify_prediction_manifest(
        manifest_path,
        protocol_path,
    )
    prediction = load_canonical_spatialtrackerv2_prediction(prediction_path)
    prefix_frame_count = int(protocol["prefix_frame_count"])
    validate_spatialtrackerv2_prediction_contract(
        prediction,
        prediction.query_points,
        prediction.query_pixels_xyt,
        expected_frame_count=prefix_frame_count,
        world_tolerance_m=float(protocol["query_tolerance_m"]),
        pixel_tolerance=float(protocol["pixel_tolerance"]),
    )

    case_dir = Path(args.case_dir)
    final_data_path = case_dir / "final_data.pkl"
    manual_tracks_path = case_dir / "gt_track_3d.pkl"
    cues_path = Path(args.cues)
    expected = protocol["score_inputs"]
    _verify_hash(
        final_data_path,
        expected["final_data_sha256"],
        "final_data",
    )
    _verify_hash(
        manual_tracks_path,
        expected["gt_track_sha256"],
        "manual trajectory",
    )
    _verify_hash(cues_path, expected["cues_sha256"], "CoTracker3 cues")

    data = _load_pickle(final_data_path)
    initial_nodes = np.asarray(data["object_points"], dtype=float)[0]
    manual = np.asarray(_load_pickle(manual_tracks_path), dtype=float)[
        :prefix_frame_count
    ]
    query_positions = prediction.query_points[:, 1:]
    if manual.shape != prediction.coords_world_m.shape:
        raise ValueError("manual prefix shape differs from SpatialTrackerV2 prediction")
    frame_zero_difference = np.linalg.norm(
        manual[0] - query_positions,
        axis=1,
    )
    if np.max(frame_zero_difference) > float(protocol["query_tolerance_m"]):
        raise ValueError("sealed queries do not match manual frame-zero identities")

    comparator_config = protocol["cotracker3_comparator"]
    cotracker = load_cotracker3_multiview_observations(
        cues_path,
        initial_nodes,
        train_end_frame=prefix_frame_count,
        minimum_view_quality=float(comparator_config["minimum_view_quality"]),
        maximum_reprojection_error_px=float(
            comparator_config["maximum_reprojection_error_px"]
        ),
        maximum_cycle_error_px=float(comparator_config["maximum_cycle_error_px"]),
        minimum_camera_count=int(comparator_config["minimum_camera_count"]),
    )
    cotracker_trajectory, association = build_same_query_cotracker3_trajectory(
        cotracker.points_world_m,
        cotracker.valid,
        initial_nodes,
        query_positions,
    )
    tracker_trajectory = IdentityTrajectory(
        coords_world_m=prediction.coords_world_m,
        valid=prediction.valid,
    )
    late_start = 2 * prefix_frame_count // 3
    tracker_metrics = identity_trajectory_metrics(
        tracker_trajectory,
        manual,
    )
    cotracker_metrics = identity_trajectory_metrics(
        cotracker_trajectory,
        manual,
    )
    late_metrics = identity_trajectory_metrics(
        tracker_trajectory,
        manual,
        frame_start=late_start,
    )
    shared_metrics = shared_support_displacement_metrics(
        tracker_trajectory,
        cotracker_trajectory,
        manual,
    )
    gate = protocol["competence_gate"]
    gates = evaluate_tapip3d_competence_gates(
        tracker_metrics,
        late_metrics,
        shared_metrics,
        minimum_support_fraction=float(gate["minimum_support_fraction"]),
        minimum_shared_rmse_improvement_fraction=float(
            gate["minimum_shared_rmse_improvement_fraction"]
        ),
        maximum_displacement_rmse_m=float(gate["maximum_displacement_rmse_m"]),
        maximum_frame_zero_anchor_rmse_m=float(
            gate["maximum_frame_zero_anchor_rmse_m"]
        ),
        minimum_late_support_fraction=float(gate["minimum_late_support_fraction"]),
        maximum_late_displacement_rmse_m=float(
            gate["maximum_late_displacement_rmse_m"]
        ),
    )
    result = {
        "schema_version": 1,
        "study_id": protocol["protocol_id"],
        "status": (
            "opened one-case association-oracle observation competence control; "
            "not prediction, transfer, confirmation, or SOTA evidence"
        ),
        "case": protocol["case"],
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
        },
        "prediction_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "query_frame_zero_match_max_m": float(np.max(frame_zero_difference)),
        "frame_zero_association": {
            "node_indices": association.node_indices.tolist(),
            "distance_m": association.distance_m.tolist(),
            "maximum_distance_m": float(np.max(association.distance_m)),
            "mean_distance_m": float(np.mean(association.distance_m)),
        },
        "spatialtrackerv2": tracker_metrics,
        "spatialtrackerv2_late_third": late_metrics,
        "strict_three_view_cotracker3": cotracker_metrics,
        "shared_support_comparison": shared_metrics,
        "gates": gates,
        "recommendation": (
            "Freeze this competence result and design an automatic frame-zero "
            "graph-query source study; do not open a fresh target."
            if gates["competence_gate_passed"]
            else "Stop SpatialTrackerV2 as the PhysTwin sparse-identity feeder "
            "without tuning on this opened source trajectory."
        ),
        "information_boundary": {
            "rgbd_frames_used_by_prediction": [0, prefix_frame_count],
            "manual_identity_role": (
                "frame-zero coordinates are prediction queries; the sealed "
                "source prefix trajectory is score-only"
            ),
            "future_after_prefix_used": False,
            "held_v8_access": False,
        },
        "scored_at_utc": _utc_now(),
    }
    output_path = Path(args.output)
    _write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal-prediction")
    seal.add_argument("--protocol", required=True)
    seal.add_argument("--input-manifest", required=True)
    seal.add_argument("--spatialtrackerv2-result", required=True)
    seal.add_argument("--output-dir", required=True)
    seal.set_defaults(function=seal_prediction)

    score = subparsers.add_parser("score")
    score.add_argument("--protocol", required=True)
    score.add_argument("--prediction-manifest", required=True)
    score.add_argument("--case-dir", required=True)
    score.add_argument("--cues", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(function=score_prediction)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
