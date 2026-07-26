"""Lock, seal, and score a source-only recursive gauge-RBF smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..phystwin_comparison import official_metrics_by_frame
from ..phystwin_official_evaluation import _nearest_distances
from ..phystwin_recursive_gauge_rbf_source import (
    PhysTwinRecursiveGaugeRbfPrediction,
    PhysTwinRecursiveGaugeRbfSourceConfig,
    run_recursive_gauge_rbf_source_prediction,
    score_recursive_gauge_rbf_prediction,
)
from ..phystwin_sparse_identity_observation import (
    SparseIdentityObservationConfig,
    SparseIdentityObservations,
    load_cotracker3_sparse_identity_observations,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _strict_sparse_config() -> SparseIdentityObservationConfig:
    return SparseIdentityObservationConfig(
        minimum_view_quality=0.5,
        maximum_cycle_error_px=5.0,
        maximum_reprojection_error_px=3.0,
        minimum_camera_count=3,
        redundant_camera_count=3,
        pixel_noise_std=2.0,
        prior_std_m=0.10,
        # Shared bias is represented explicitly by the gauge-aware solver.
        shared_bias_std_m=1e-6,
        two_view_extra_std_m=0.010,
        boundary_scale_px=8.0,
        two_view_reliability_multiplier=0.5,
        minimum_ray_angle_degrees=0.5,
    )


def _method_descriptor() -> dict[str, Any]:
    return {
        "source_config": asdict(
            PhysTwinRecursiveGaugeRbfSourceConfig()
        ),
        "sparse_observation_config": asdict(_strict_sparse_config()),
    }


def _verify_hash(path: str | Path, expected: str, name: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _save_sparse(
    output: Path,
    observations: SparseIdentityObservations,
    *,
    observed_points_prefix_m: np.ndarray,
    visibility_prefix: np.ndarray,
    motion_valid_prefix: np.ndarray,
) -> None:
    np.savez_compressed(
        output,
        observed_points_prefix_m=observed_points_prefix_m,
        visibility_prefix=visibility_prefix,
        motion_valid_prefix=motion_valid_prefix,
        sparse_points_world_m=observations.points_world_m,
        sparse_observation_covariance_m2=(
            observations.observation_covariance_m2
        ),
        sparse_observation_variance_m2=(
            observations.observation_variance_m2
        ),
        sparse_prior_reliability=observations.prior_reliability,
        sparse_valid=observations.valid,
        sparse_raw_camera_count=observations.raw_camera_count,
        sparse_effective_camera_count=(
            observations.effective_camera_count
        ),
        sparse_reprojection_error_px=(
            observations.reprojection_error_px
        ),
        sparse_redundant_view_disagreement_m=(
            observations.redundant_view_disagreement_m
        ),
        sparse_two_view_fallback=observations.two_view_fallback,
    )


def _load_sparse_prefix(
    path: Path,
) -> tuple[dict[str, np.ndarray], SparseIdentityObservations]:
    with np.load(path) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    observations = SparseIdentityObservations(
        points_world_m=values["sparse_points_world_m"],
        observation_covariance_m2=values[
            "sparse_observation_covariance_m2"
        ],
        observation_variance_m2=values[
            "sparse_observation_variance_m2"
        ],
        prior_reliability=values["sparse_prior_reliability"],
        valid=values["sparse_valid"],
        raw_camera_count=values["sparse_raw_camera_count"],
        effective_camera_count=values["sparse_effective_camera_count"],
        reprojection_error_px=values["sparse_reprojection_error_px"],
        redundant_view_disagreement_m=values[
            "sparse_redundant_view_disagreement_m"
        ],
        two_view_fallback=values["sparse_two_view_fallback"],
    )
    return values, observations


def prepare_prefix(args: argparse.Namespace) -> int:
    """Materialize a prefix-only carrier before the protocol is locked."""

    case_dir = Path(args.case_dir)
    baseline_path = Path(args.baseline)
    cues_path = Path(args.cues)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    split_path = case_dir / "split.json"
    final_data_path = case_dir / "final_data.pkl"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    future_end = int(split["test"][1])
    data = _load_pickle(final_data_path)
    observed = np.asarray(data["object_points"], dtype=np.float64)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    baseline = np.asarray(_load_pickle(baseline_path))
    if len(baseline) != future_end:
        raise ValueError("baseline frame count does not match split")
    if observed.shape[1] > baseline.shape[1]:
        raise ValueError("baseline does not cover original object identities")

    sparse = load_cotracker3_sparse_identity_observations(
        cues_path,
        observed[0],
        train_end_frame=train_end,
        config=_strict_sparse_config(),
    )
    prefix_path = output / "prefix_only.npz"
    _save_sparse(
        prefix_path,
        sparse,
        observed_points_prefix_m=observed[:train_end],
        visibility_prefix=visible[:train_end],
        motion_valid_prefix=motion_valid[: train_end - 1],
    )
    supported = np.any(sparse.valid, axis=0)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinRecursiveGaugeRbfPrefixV1",
        "created_at_utc": _utc_now(),
        "case": args.case,
        "fit_end_frame_exclusive": int(args.fit_end),
        "train_end_frame_exclusive": train_end,
        "future_end_frame_exclusive": future_end,
        "original_point_count": int(observed.shape[1]),
        "num_surface_points": int(
            observed.shape[1] + len(np.asarray(data["surface_points"]))
        ),
        "inputs": {
            "final_data": {
                "path": str(final_data_path),
                "sha256": _sha256(final_data_path),
            },
            "split": {
                "path": str(split_path),
                "sha256": _sha256(split_path),
            },
            "baseline": {
                "path": str(baseline_path),
                "sha256": _sha256(baseline_path),
            },
            "cues": {
                "path": str(cues_path),
                "sha256": _sha256(cues_path),
            },
        },
        "prefix_artifact": {
            "path": str(prefix_path),
            "sha256": _sha256(prefix_path),
        },
        "target_free_diagnostics": {
            "strict_three_view_identity_count": int(np.sum(supported)),
            "strict_three_view_valid_fraction": float(
                np.mean(sparse.valid)
            ),
            "median_effective_camera_count": (
                None
                if not np.any(sparse.valid)
                else float(
                    np.median(
                        sparse.effective_camera_count[sparse.valid]
                    )
                )
            ),
            "median_observation_std_m": (
                None
                if not np.any(sparse.valid)
                else float(
                    np.median(
                        np.sqrt(
                            sparse.observation_variance_m2[sparse.valid]
                        )
                    )
                )
            ),
            "mean_prior_reliability": (
                0.0
                if not np.any(sparse.valid)
                else float(
                    np.mean(sparse.prior_reliability[sparse.valid])
                )
            ),
            "two_view_fraction": (
                0.0
                if not np.any(sparse.valid)
                else float(
                    np.mean(sparse.two_view_fallback[sparse.valid])
                )
            ),
        },
        "method_descriptor": _method_descriptor(),
        "information_boundary": {
            "future_object_observations_written": False,
            "manual_tracks_read": False,
            "future_rgb_read": False,
            "held_v8_access": False,
        },
    }
    manifest_path = output / "prefix_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "prefix_manifest": str(manifest_path),
                "prefix_manifest_sha256": _sha256(manifest_path),
                "prefix_artifact_sha256": _sha256(prefix_path),
                "target_free_diagnostics": manifest[
                    "target_free_diagnostics"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported recursive gauge-RBF protocol")
    if protocol.get("method_descriptor") != _method_descriptor():
        raise ValueError(
            "locked method descriptor differs from the implemented arm"
        )
    return protocol


def _save_prediction(
    output: Path,
    prediction: PhysTwinRecursiveGaugeRbfPrediction,
) -> None:
    np.savez_compressed(
        output,
        dense_baseline=prediction.dense_baseline,
        candidate=prediction.candidate,
        correction_mean_m=prediction.correction_mean_m,
        correction_covariance_m2=(
            prediction.correction_covariance_m2
        ),
        center_ids=prediction.center_ids,
    )


def seal_prediction(args: argparse.Namespace) -> int:
    """Create a target-free future carrier under an already locked protocol."""

    protocol_path = Path(args.protocol)
    protocol = _load_protocol(protocol_path)
    inputs = protocol["inputs"]
    prefix_path = Path(inputs["prefix_artifact"]["path"])
    baseline_path = Path(inputs["baseline"]["path"])
    _verify_hash(
        prefix_path,
        inputs["prefix_artifact"]["sha256"],
        "prefix artifact",
    )
    _verify_hash(
        baseline_path,
        inputs["baseline"]["sha256"],
        "baseline",
    )
    values, observations = _load_sparse_prefix(prefix_path)
    baseline = np.asarray(_load_pickle(baseline_path))
    prediction = run_recursive_gauge_rbf_source_prediction(
        baseline,
        values["observed_points_prefix_m"],
        values["visibility_prefix"],
        values["motion_valid_prefix"],
        observations,
        case_id=str(protocol["case"]),
        fit_end_frame=int(protocol["fit_end_frame_exclusive"]),
        train_end_frame=int(protocol["train_end_frame_exclusive"]),
        num_surface_points=int(protocol["num_surface_points"]),
        source_revision=str(protocol["implementation_commit"]),
        source_artifact_sha256=str(inputs["cues"]["sha256"]),
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "prediction.npz"
    _save_prediction(prediction_path, prediction)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinRecursiveGaugeRbfPredictionV1",
        "created_at_utc": _utc_now(),
        "case": protocol["case"],
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
        },
        "implementation_commit": _git_revision(),
        "prediction": {
            "path": str(prediction_path),
            "sha256": _sha256(prediction_path),
        },
        "prefix_admitted": prediction.prefix_admitted,
        "prefix_baseline_cd_m": prediction.prefix_baseline_cd_m,
        "prefix_candidate_cd_m": prediction.prefix_candidate_cd_m,
        "diagnostics": prediction.diagnostics,
        "future_target_loaded": False,
        "manual_tracks_loaded": False,
        "held_v8_access": False,
    }
    manifest_path = output / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "prediction_manifest": str(manifest_path),
                "prediction_manifest_sha256": _sha256(manifest_path),
                "prediction_sha256": _sha256(prediction_path),
                "prefix_admitted": prediction.prefix_admitted,
                "prefix_baseline_cd_m": prediction.prefix_baseline_cd_m,
                "prefix_candidate_cd_m": prediction.prefix_candidate_cd_m,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_prediction(
    path: Path,
    manifest: dict[str, Any],
) -> PhysTwinRecursiveGaugeRbfPrediction:
    with np.load(path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return PhysTwinRecursiveGaugeRbfPrediction(
        dense_baseline=arrays["dense_baseline"],
        candidate=arrays["candidate"],
        correction_mean_m=arrays["correction_mean_m"],
        correction_covariance_m2=arrays[
            "correction_covariance_m2"
        ],
        center_ids=arrays["center_ids"],
        prefix_admitted=bool(manifest["prefix_admitted"]),
        prefix_baseline_cd_m=float(manifest["prefix_baseline_cd_m"]),
        prefix_candidate_cd_m=float(manifest["prefix_candidate_cd_m"]),
        diagnostics=dict(manifest["diagnostics"]),
    )


def _horizon_breakdown(
    frame_values: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    count = len(next(iter(frame_values.values())))
    boundaries = np.linspace(0, count, 4, dtype=int)
    result: dict[str, dict[str, float]] = {}
    for name, start, end in zip(
        ("early", "middle", "late"),
        boundaries[:-1],
        boundaries[1:],
    ):
        result[name] = {
            metric: float(np.mean(values[start:end]))
            for metric, values in frame_values.items()
        }
    return result


def _uncertainty_audit(
    prediction: PhysTwinRecursiveGaugeRbfPrediction,
    manual_tracks_m: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    if not prediction.prefix_admitted:
        return {
            "available": False,
            "reason": "exact-fallback carries no admitted correction posterior",
        }
    tracks = np.asarray(manual_tracks_m, dtype=np.float64)
    initial = np.all(np.isfinite(tracks[0]), axis=1)
    _, node_ids = _nearest_distances(
        prediction.dense_baseline[0],
        tracks[0, initial],
        p=2,
    )
    covered: list[np.ndarray] = []
    nees: list[float] = []
    z90 = 1.6448536269514722
    for frame in range(start_frame, end_frame):
        target = tracks[frame, initial]
        valid = np.all(np.isfinite(target), axis=1)
        if not np.any(valid):
            continue
        residual = (
            prediction.candidate[frame, node_ids][valid]
            - target[valid]
        )
        covariance = prediction.correction_covariance_m2[
            frame,
            node_ids,
        ][valid]
        standard_deviation = np.sqrt(
            np.maximum(
                np.diagonal(covariance, axis1=1, axis2=2),
                1e-12,
            )
        )
        covered.append(np.abs(residual) <= z90 * standard_deviation)
        for row, matrix in zip(residual, covariance):
            regularized = matrix + 1e-12 * np.eye(3)
            nees.append(float(row @ np.linalg.solve(regularized, row)))
    if not covered:
        return {"available": False, "reason": "no finite future manual track"}
    return {
        "available": True,
        "nominal_coordinate_coverage": 0.90,
        "coordinate_coverage": float(np.mean(np.concatenate(covered))),
        "mean_point_nees": float(np.mean(nees)),
        "median_point_nees": float(np.median(nees)),
        "point_count": len(nees),
    }


def score_prediction(args: argparse.Namespace) -> int:
    """Open only the authorized source future after prediction sealing."""

    protocol_path = Path(args.protocol)
    protocol = _load_protocol(protocol_path)
    prediction_manifest_path = Path(args.prediction_manifest)
    prediction_manifest = json.loads(
        prediction_manifest_path.read_text(encoding="utf-8")
    )
    if prediction_manifest["protocol"]["sha256"] != _sha256(
        protocol_path
    ):
        raise ValueError("prediction was not bound to this protocol")
    prediction_path = Path(prediction_manifest["prediction"]["path"])
    _verify_hash(
        prediction_path,
        prediction_manifest["prediction"]["sha256"],
        "prediction",
    )
    prediction = _load_prediction(
        prediction_path,
        prediction_manifest,
    )

    final_data_path = Path(args.case_dir) / "final_data.pkl"
    tracks_path = Path(args.case_dir) / "gt_track_3d.pkl"
    expected = protocol["score_inputs"]
    _verify_hash(
        final_data_path,
        expected["final_data_sha256"],
        "future final_data",
    )
    _verify_hash(
        tracks_path,
        expected["gt_track_sha256"],
        "future manual tracks",
    )
    data = _load_pickle(final_data_path)
    observed = np.asarray(data["object_points"], dtype=np.float64)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    tracks = np.asarray(_load_pickle(tracks_path), dtype=np.float64)
    start = int(protocol["train_end_frame_exclusive"])
    end = int(protocol["future_end_frame_exclusive"])
    score = score_recursive_gauge_rbf_prediction(
        prediction,
        observed,
        visible,
        tracks,
        num_surface_points=int(protocol["num_surface_points"]),
        start_frame=start,
        end_frame=end,
    )
    baseline_path = Path(protocol["inputs"]["baseline"]["path"])
    raw = np.asarray(_load_pickle(baseline_path))
    raw_metrics_by_frame = official_metrics_by_frame(
        raw,
        observed,
        visible,
        tracks,
        num_surface_points=int(protocol["num_surface_points"]),
        start_frame=start,
        end_frame=end,
    )
    raw_metrics = {
        name: float(np.mean(value))
        for name, value in raw_metrics_by_frame.items()
    }
    change = score["candidate_relative_change_fraction"]
    gates = {
        "prefix_admitted": prediction.prefix_admitted,
        "future_track_improvement_at_least_5_percent": (
            change["track_error_m"] <= -0.05
        ),
        "future_cd_regression_at_most_1_percent": (
            change["chamfer_distance_m"] <= 0.01
        ),
        "both_metrics_no_worse_than_raw_physical": all(
            score["candidate"][name] <= raw_metrics[name]
            for name in raw_metrics
        ),
    }
    gates["smoke_gate_passed"] = all(gates.values())
    result = {
        "schema_version": 1,
        "status": (
            "opened one-case source development smoke; "
            "not independent confirmation"
        ),
        "case": protocol["case"],
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
        },
        "prediction_manifest": {
            "path": str(prediction_manifest_path),
            "sha256": _sha256(prediction_manifest_path),
        },
        "raw_physical": raw_metrics,
        **score,
        "horizon_breakdown": {
            arm: _horizon_breakdown(values)
            for arm, values in score["frame_metrics"].items()
        },
        "uncertainty": _uncertainty_audit(
            prediction,
            tracks,
            start_frame=start,
            end_frame=end,
        ),
        "gates": gates,
        "recommendation": (
            "Design an object-disjoint source panel without opening fresh "
            "targets."
            if gates["smoke_gate_passed"]
            else "Stop this exact recursive camera-only arm without tuning "
            "against the opened future."
        ),
        "scored_at_utc": _utc_now(),
        "held_v8_access": False,
    }
    output = Path(args.output)
    _write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-prefix")
    prepare.add_argument("--case", required=True)
    prepare.add_argument("--case-dir", required=True)
    prepare.add_argument("--baseline", required=True)
    prepare.add_argument("--cues", required=True)
    prepare.add_argument("--fit-end", required=True, type=int)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(function=prepare_prefix)

    predict = subparsers.add_parser("seal-prediction")
    predict.add_argument("--protocol", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.set_defaults(function=seal_prediction)

    score = subparsers.add_parser("score")
    score.add_argument("--protocol", required=True)
    score.add_argument("--prediction-manifest", required=True)
    score.add_argument("--case-dir", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(function=score_prediction)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
