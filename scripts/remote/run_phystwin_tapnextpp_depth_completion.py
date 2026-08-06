#!/usr/bin/env python3
"""Complete a sealed TAPNext++ carrier with target-free single-view RGB-D rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.tapnextpp_depth_completion import (
    TAPNextPPDepthCompletionConfig,
    complete_strict_multiview_carrier,
    lift_per_camera_rgbd_tracks,
)

PREDICTION_FILENAME = "tapnextpp_depth_completion_prediction.npz"
REPORT_FILENAME = "tapnextpp_depth_completion_prediction_report.json"
SEAL_FILENAME = "tapnextpp_depth_completion_prediction_seal.json"
SOURCE_SMOKE_PROTOCOL_ID = "phystwin-tapnextpp-depth-completion-source-v1"
TRANSFER_PANEL_PROTOCOL_ID = "phystwin-tapnextpp-depth-completion-transfer-v1"
TRACKER_PROTOCOL_ID = "phystwin-tapnextpp-prefix-competence-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    content = dict(payload)
    content.pop("result_sha256", None)
    encoded = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--protocol", type=Path, required=True)
    predict.add_argument("--strict-prediction", type=Path, required=True)
    predict.add_argument("--prediction-input", type=Path, required=True)
    predict.add_argument("--raw-case-dir", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--prediction-dir", type=Path, required=True)
    evaluate.add_argument("--withheld-prefix", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = asdict(TAPNextPPDepthCompletionConfig())
    protocol_id = protocol.get("protocol_id")
    if protocol_id == SOURCE_SMOKE_PROTOCOL_ID:
        _require(
            protocol.get("status") == "frozen-source-smoke",
            "protocol is not frozen for the source smoke",
        )
        _require(
            protocol.get("method_config") == expected,
            "method config changed",
        )
    elif protocol_id == TRACKER_PROTOCOL_ID:
        _require(
            protocol.get("status") == "locked-before-tapnextpp-prediction",
            "transfer case is not prediction-locked",
        )
        _require(
            protocol.get("source_panel_protocol_id")
            == TRANSFER_PANEL_PROTOCOL_ID,
            "transfer case does not bind the source panel",
        )
        _require(
            protocol.get("depth_completion_config") == expected,
            "depth-completion config changed",
        )
        _require(
            isinstance(protocol.get("depth_completion_gates"), dict),
            "transfer case omits depth-completion gates",
        )
    else:
        raise ValueError("protocol ID changed")
    return protocol


def _depth_completion_gates(protocol: dict[str, Any]) -> dict[str, Any]:
    return (
        protocol["gates"]
        if protocol["protocol_id"] == SOURCE_SMOKE_PROTOCOL_ID
        else protocol["depth_completion_gates"]
    )


def _validate_strict_prediction_seal(
    strict_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    report_path = strict_path.with_name("tapnextpp_prediction_report.json")
    seal_path = strict_path.with_name("tapnextpp_prediction_seal.json")
    _require(report_path.is_file(), "strict prediction report is missing")
    _require(seal_path.is_file(), "strict prediction seal is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        report.get("result_sha256") == _canonical_sha256(report),
        "strict prediction report hash changed",
    )
    _require(
        seal.get("result_sha256") == _canonical_sha256(seal),
        "strict prediction seal hash changed",
    )
    _require(
        seal.get("prediction_archive_sha256") == _file_sha256(strict_path)
        and seal.get("prediction_report_sha256") == _file_sha256(report_path),
        "strict prediction differs from its seal",
    )
    _require(
        report.get("protocol_id") == protocol["protocol_id"]
        and report.get("case") == protocol.get("case")
        and seal.get("protocol_id") == protocol["protocol_id"]
        and seal.get("case") == protocol.get("case"),
        "strict prediction provenance differs from the transfer protocol",
    )
    return {
        "strict_prediction_report_sha256": _file_sha256(report_path),
        "strict_prediction_seal_sha256": _file_sha256(seal_path),
    }


def _load_inputs(
    strict_path: Path,
    prediction_input_path: Path,
    protocol: dict[str, Any],
) -> dict[str, np.ndarray]:
    with np.load(strict_path, allow_pickle=False) as stored:
        strict = {
            "strict_points": np.asarray(stored["anchored_tracker_m"], np.float64),
            "strict_support": np.asarray(stored["accepted_support"], bool),
            "strict_reliability": np.asarray(
                stored["observation_reliability"], np.float64
            ),
            "strict_covariance": np.asarray(
                stored["observation_covariance_m2"], np.float64
            ),
            "tracks": np.asarray(stored["per_camera_tracks_xy"], np.float64),
            "visibility": np.asarray(
                stored["per_camera_visibility_probability"], np.float64
            ),
            "strict_identity_ids": np.asarray(stored["identity_ids"], np.int64),
        }
    with np.load(prediction_input_path, allow_pickle=False) as stored:
        source = {
            "frame_zero": np.asarray(stored["query_points_world_m"], np.float64),
            "identity_ids": np.asarray(stored["identity_ids"], np.int64),
            "masks": np.asarray(stored["object_masks"], bool),
            "cameras": np.asarray(stored["selected_cameras"], np.int64),
            "source_frame": np.asarray(stored["source_frame"], np.int64),
            "train_end": np.asarray(
                stored["train_end_frame_exclusive"], np.int64
            ),
        }
    result = {**strict, **source}
    _require(
        np.array_equal(result["strict_identity_ids"], result["identity_ids"]),
        "strict carrier and prediction input identities differ",
    )
    _require(
        result["strict_points"].shape == (*result["strict_support"].shape, 3),
        "strict carrier shape changed",
    )
    _require(
        result["tracks"].shape[:3]
        == (
            len(result["cameras"]),
            *result["strict_support"].shape,
        ),
        "per-camera track shape changed",
    )
    _require(
        int(result["source_frame"]) == int(protocol["source_frame_start"]),
        "source frame changed",
    )
    _require(
        result["strict_support"].shape[0] == int(protocol["prefix_frame_count"]),
        "prefix frame count changed",
    )
    _require(
        int(result["source_frame"]) + len(result["strict_support"])
        <= int(result["train_end"]),
        "source smoke crosses the training boundary",
    )
    return result


def _load_metric_frames(
    raw_case_dir: Path,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    metadata_path = raw_case_dir / "metadata.json"
    calibration_path = raw_case_dir / "calibrate.pkl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    all_intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    with calibration_path.open("rb") as stream:
        all_poses = np.asarray(pickle.load(stream), dtype=np.float64)
    cameras = inputs["cameras"]
    intrinsics = all_intrinsics[cameras]
    poses = all_poses[cameras]
    start = int(inputs["source_frame"])
    frame_count = len(inputs["strict_support"])
    depths = []
    frame_hashes: dict[str, list[dict[str, Any]]] = {}
    for camera in cameras:
        camera_depths = []
        hashes = []
        for frame in range(start, start + frame_count):
            path = raw_case_dir / "depth" / str(int(camera)) / f"{frame}.npy"
            camera_depths.append(np.asarray(np.load(path), dtype=np.float64) * 0.001)
            hashes.append({"frame": frame, "depth_sha256": _file_sha256(path)})
        depths.append(np.stack(camera_depths))
        frame_hashes[str(int(camera))] = hashes
    provenance = {
        "metadata_sha256": _file_sha256(metadata_path),
        "calibration_sha256": _file_sha256(calibration_path),
        "depth_frame_sha256": frame_hashes,
    }
    return np.stack(depths), intrinsics, poses, provenance


def _competence_json(value: Any) -> dict[str, Any]:
    def finite_or_none(number: float) -> float | None:
        return float(number) if np.isfinite(number) else None

    return {
        "camera_index": value.camera_index,
        "accepted": value.accepted,
        "reason": value.reason,
        "overlap_rows": value.overlap_rows,
        "overlap_fraction": value.overlap_fraction,
        "centered_median_m": finite_or_none(value.centered_median_m),
        "centered_p90_m": finite_or_none(value.centered_p90_m),
        "penalized_agreement_m": finite_or_none(value.penalized_agreement_m),
        "carrier_offset_m": value.carrier_offset_m.tolist(),
        "residual_covariance_m2": value.residual_covariance_m2.tolist(),
    }


def _predict(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    strict_path = args.strict_prediction.resolve()
    input_path = args.prediction_input.resolve()
    raw_case_dir = args.raw_case_dir.resolve()
    output = args.output_dir.resolve()
    _require(not output.exists(), "prediction output already exists")
    protocol = _load_protocol(protocol_path)
    strict_seal_provenance = (
        _validate_strict_prediction_seal(strict_path, protocol)
        if protocol["protocol_id"] == TRACKER_PROTOCOL_ID
        else {}
    )
    inputs = _load_inputs(strict_path, input_path, protocol)
    depths, intrinsics, poses, depth_provenance = _load_metric_frames(
        raw_case_dir,
        inputs,
    )
    config = TAPNextPPDepthCompletionConfig()
    observations = lift_per_camera_rgbd_tracks(
        inputs["tracks"],
        inputs["visibility"],
        depths,
        inputs["masks"],
        intrinsics,
        poses,
        inputs["frame_zero"],
        config=config,
    )
    result = complete_strict_multiview_carrier(
        inputs["strict_points"],
        inputs["strict_support"],
        inputs["strict_reliability"],
        inputs["strict_covariance"],
        observations,
        config=config,
    )
    output.mkdir(parents=True)
    archive_path = output / PREDICTION_FILENAME
    np.savez_compressed(
        archive_path,
        completed_points_world_m=result.points_world_m.astype(np.float32),
        completed_support=result.support,
        completed_prior_reliability=result.prior_reliability.astype(np.float32),
        completed_covariance_m2=result.covariance_m2.astype(np.float32),
        source_camera=result.source_camera,
        selected_camera=np.asarray(
            -1 if result.selected_camera is None else result.selected_camera,
            dtype=np.int16,
        ),
        strict_points_world_m=inputs["strict_points"].astype(np.float32),
        strict_support=inputs["strict_support"],
        identity_ids=inputs["identity_ids"],
    )
    strict_count = int(np.sum(inputs["strict_support"]))
    completed_count = int(np.sum(result.support))
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPDepthCompletionPrediction",
        "protocol_id": protocol["protocol_id"],
        "case": protocol["case"],
        "method_config": asdict(config),
        "decision": {
            "accepted": result.accepted,
            "reason": result.reason,
            "selected_camera_local_index": result.selected_camera,
            "selected_camera_released_index": (
                None
                if result.selected_camera is None
                else int(inputs["cameras"][result.selected_camera])
            ),
        },
        "support": {
            "strict_rows": strict_count,
            "completed_rows": completed_count,
            "added_rows": completed_count - strict_count,
            "total_rows": int(result.support.size),
        },
        "camera_competence": [
            _competence_json(value) for value in result.camera_competence
        ],
        "inputs": {
            "protocol_sha256": _file_sha256(protocol_path),
            "strict_prediction_sha256": _file_sha256(strict_path),
            "prediction_input_sha256": _file_sha256(input_path),
            **strict_seal_provenance,
            **depth_provenance,
        },
        "information_boundary": {
            "withheld_prefix_read": False,
            "manual_future_read": False,
            "future_rgbd_read": False,
            "physical_state_innovation_used_for_reliability": False,
            "strict_carrier_rows_used_for_camera_competence_only": True,
            "correlated_camera_precision_accumulation": False,
            "held_v8_accessed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    report["result_sha256"] = _canonical_sha256(report)
    report_path = output / REPORT_FILENAME
    _write_json(report_path, report)
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPDepthCompletionPredictionSeal",
        "prediction_archive_sha256": _file_sha256(archive_path),
        "prediction_report_sha256": _file_sha256(report_path),
    }
    seal["result_sha256"] = _canonical_sha256(seal)
    _write_json(output / SEAL_FILENAME, seal)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def _radial_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
) -> float | None:
    if not np.any(selected):
        return None
    squared = np.sum(np.square(prediction[selected] - target[selected]), axis=1)
    return float(np.sqrt(np.mean(squared)))


def _evaluate(args: argparse.Namespace) -> None:
    protocol = _load_protocol(args.protocol.resolve())
    gate_config = _depth_completion_gates(protocol)
    prediction = args.prediction_dir.resolve()
    output = args.output.resolve()
    _require(not output.exists(), "evaluation output already exists")
    archive_path = prediction / PREDICTION_FILENAME
    report_path = prediction / REPORT_FILENAME
    seal_path = prediction / SEAL_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("result_sha256") == _canonical_sha256(seal),
        "prediction seal hash changed",
    )
    _require(
        seal.get("prediction_archive_sha256") == _file_sha256(archive_path)
        and seal.get("prediction_report_sha256") == _file_sha256(report_path),
        "sealed prediction files changed",
    )
    withheld_path = args.withheld_prefix.resolve()
    with np.load(archive_path, allow_pickle=False) as stored:
        candidate = np.asarray(stored["completed_points_world_m"], np.float64)
        support = np.asarray(stored["completed_support"], bool)
        strict = np.asarray(stored["strict_points_world_m"], np.float64)
        strict_support = np.asarray(stored["strict_support"], bool)
        identity_ids = np.asarray(stored["identity_ids"], np.int64)
    with np.load(withheld_path, allow_pickle=False) as stored:
        target = np.asarray(stored["target_tracks_world_m"], np.float64)
        target_ids = np.asarray(stored["identity_ids"], np.int64)
        source_start = int(stored["source_frame_start"])
        source_end = int(stored["source_frame_end_exclusive"])
    _require(candidate.shape == strict.shape == target.shape, "target shape changed")
    _require(np.array_equal(identity_ids, target_ids), "target identities changed")
    _require(source_start == int(protocol["source_frame_start"]), "target start changed")
    _require(
        source_end - source_start == int(protocol["prefix_frame_count"]),
        "target interval changed",
    )
    target_valid = np.all(np.isfinite(target), axis=-1)
    scored_frames = np.arange(len(target)) > 0
    eligible = target_valid & scored_frames[:, None]
    candidate_rows = eligible & support
    strict_rows = eligible & strict_support
    fallback_rows = candidate_rows & ~strict_support
    persistence = np.broadcast_to(target[:1], target.shape)
    endpoint = np.arange(len(target)) >= (
        len(target) - int(gate_config["endpoint_frame_count"])
    )
    candidate_rmse = _radial_rmse(candidate, target, candidate_rows)
    persistence_rmse = _radial_rmse(persistence, target, candidate_rows)
    endpoint_rmse = _radial_rmse(candidate, target, candidate_rows & endpoint[:, None])
    relative_gain = (
        None
        if candidate_rmse is None or persistence_rmse in (None, 0.0)
        else (persistence_rmse - candidate_rmse) / persistence_rmse
    )
    supported_fraction = float(np.sum(candidate_rows) / max(np.sum(eligible), 1))
    gates = {
        "supported_fraction": supported_fraction
        >= float(gate_config["minimum_supported_fraction"]),
        "relative_gain_over_persistence": relative_gain is not None
        and relative_gain
        >= float(gate_config["minimum_relative_gain_over_persistence"]),
        "identity_rmse": candidate_rmse is not None
        and candidate_rmse <= float(gate_config["maximum_identity_rmse_m"]),
        "endpoint_rmse": endpoint_rmse is not None
        and endpoint_rmse <= float(gate_config["maximum_endpoint_rmse_m"]),
    }
    is_source_smoke = protocol["protocol_id"] == SOURCE_SMOKE_PROTOCOL_ID
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": (
            "PhysTwinTAPNextPPDepthCompletionSourceResult"
            if is_source_smoke
            else "PhysTwinTAPNextPPDepthCompletionTransferCaseResult"
        ),
        "protocol_id": protocol["protocol_id"],
        "case": protocol["case"],
        "metrics": {
            "eligible_rows": int(np.sum(eligible)),
            "strict_supported_rows": int(np.sum(strict_rows)),
            "completed_supported_rows": int(np.sum(candidate_rows)),
            "fallback_rows": int(np.sum(fallback_rows)),
            "supported_fraction": supported_fraction,
            "strict_identity_rmse_m": _radial_rmse(strict, target, strict_rows),
            "candidate_identity_rmse_m": candidate_rmse,
            "fallback_identity_rmse_m": _radial_rmse(
                candidate, target, fallback_rows
            ),
            "persistence_identity_rmse_m": persistence_rmse,
            "relative_gain_over_persistence": relative_gain,
            "candidate_endpoint_rmse_m": endpoint_rmse,
        },
        "gates": gates,
        (
            "source_smoke_passed"
            if is_source_smoke
            else "provider_gate_passed"
        ): all(gates.values()),
        "decision": (
            (
                "freeze-separate-opened-cohort-transfer-protocol"
                if all(gates.values())
                else "stop-depth-completion-route"
            )
            if is_source_smoke
            else (
                "record-case-provider-gate-for-frozen-cohort"
                if all(gates.values())
                else "retain-failed-case-without-replacement"
            )
        ),
        "inputs": {
            "prediction_seal_sha256": _file_sha256(seal_path),
            "withheld_prefix_sha256": _file_sha256(withheld_path),
        },
        "information_boundary": {
            "prediction_sealed_before_this_evaluation": True,
            "future_simulator_outcome_read": False,
            "held_v8_accessed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_sha256"] = _canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def main() -> None:
    args = _parse_args()
    if args.operation == "predict":
        _predict(args)
    else:
        _evaluate(args)


if __name__ == "__main__":
    main()
