#!/usr/bin/env python3
"""Audit causal metric-gauge support without opening a source suffix or target."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

import bayesian_phystwin.deform360_covariance_source_producer_v1 as producer
from bayesian_phystwin.deform360_joint_sparse_public_inputs_v5 import (
    _load_metric_prefix,
    load_motioncrafter_prediction,
    robust_similarity_transform,
)

DEFAULT_CLUSTER_SIZES = (64, 48, 32, 24, 16, 12, 8, 4)
DEFAULT_REQUIRED_CLUSTERS = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--source-execution-lock", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--upstream-run-root", type=Path, required=True)
    parser.add_argument("--forbidden-confirmation-root", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_id(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_revision(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("implementation revision must be a lowercase Git SHA-1")
    return value


def _ordinary_directory(path: Path, *, label: str) -> Path:
    requested = path.absolute()
    if not requested.is_dir() or requested.is_symlink():
        raise ValueError(f"invalid {label}: {requested}")
    resolved = requested.resolve(strict=True)
    if resolved != requested:
        raise ValueError(f"{label} must be a canonical directory")
    return resolved


def _assert_disjoint(admitted: Sequence[Path], forbidden: Path) -> None:
    for path in admitted:
        if path == forbidden or forbidden in path.parents or path in forbidden.parents:
            raise ValueError(f"admitted path overlaps confirmation root: {path}")


def _request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    sizes = payload.get("cluster_sizes_pixels")
    if (
        payload.get("source_only") is not True
        or payload.get("confirmation_access_authorized") is not False
        or payload.get("prior_target_access") is not False
        or tuple(sizes if isinstance(sizes, list) else ()) != DEFAULT_CLUSTER_SIZES
        or payload.get("minimum_independent_clusters") != DEFAULT_REQUIRED_CLUSTERS
    ):
        raise ValueError("request differs from the source-only audit contract")
    return cast(dict[str, Any], payload)


def _first_fit(
    prediction: Any,
    metric: Any,
    *,
    prediction_index: int,
    metric_index: int,
    raw_frame: int,
    rows: np.ndarray,
    columns: np.ndarray,
    clusters: np.ndarray,
    required_clusters: int,
) -> dict[str, Any]:
    try:
        trim_fraction = max(0.8, min(1.0, 8.0 / len(rows)))
        transform = robust_similarity_transform(
            np.asarray(prediction.point_map[prediction_index, rows, columns]),
            np.asarray(metric.points_world_m[metric_index, rows, columns]),
            trim_fraction=trim_fraction,
            iterations=5,
        )
        inlier = np.asarray(transform["inlier_mask"], dtype=np.bool_)
        inlier_clusters = int(len(np.unique(clusters[inlier], axis=0)))
        return {
            "error": None,
            "raw_frame_index": raw_frame,
            "input_pair_count": int(transform["input_pair_count"]),
            "inlier_pair_count": int(transform["inlier_pair_count"]),
            "independent_cluster_count": int(len(np.unique(clusters, axis=0))),
            "inlier_independent_cluster_count": inlier_clusters,
            "inlier_rmse_m": float(transform["inlier_rmse_m"]),
            "contract_pass": inlier_clusters >= required_clusters,
        }
    except Exception as error:  # bounded diagnostic receipt
        return {
            "error": f"{type(error).__name__}: {error}",
            "raw_frame_index": raw_frame,
            "contract_pass": False,
        }


def _camera_record(
    unit: Any,
    camera_id: str,
    visual_path: Path,
    metric_path: Path,
    *,
    cluster_sizes: Sequence[int],
    required_clusters: int,
) -> dict[str, Any]:
    prediction = load_motioncrafter_prediction(visual_path)
    frames = prediction.frame_indices
    if frames is None:
        raise ValueError(f"missing frame indices: {unit.object_id}/{camera_id}")
    image_shape = (
        int(prediction.valid_mask.shape[1]),
        int(prediction.valid_mask.shape[2]),
    )
    metric = _load_metric_prefix(
        metric_path,
        raw_prefix_range_half_open=unit.raw_prefix_range_half_open,
        image_shape=image_shape,
    )
    start, stop = unit.raw_prefix_range_half_open
    stats: dict[int, dict[str, Any]] = {
        size: {
            "maximum_pair_count": 0,
            "maximum_independent_cluster_count": 0,
            "qualifying_frame_count": 0,
            "first": None,
        }
        for size in cluster_sizes
    }
    for prediction_index, raw_value in enumerate(np.asarray(frames).tolist()):
        raw_frame = int(raw_value)
        if raw_frame < start or raw_frame >= stop:
            continue
        metric_index = raw_frame - start
        active = np.asarray(
            prediction.valid_mask[prediction_index], dtype=np.bool_
        ) & np.asarray(metric.valid_mask[metric_index], dtype=np.bool_)
        rows, columns = np.nonzero(active)
        pair_count = int(len(rows))
        for size in cluster_sizes:
            entry = stats[size]
            entry["maximum_pair_count"] = max(
                int(entry["maximum_pair_count"]), pair_count
            )
            clusters = (
                np.empty((0, 2), dtype=np.int64)
                if pair_count == 0
                else np.column_stack((rows // size, columns // size))
            )
            cluster_count = int(len(np.unique(clusters, axis=0)))
            entry["maximum_independent_cluster_count"] = max(
                int(entry["maximum_independent_cluster_count"]), cluster_count
            )
            if cluster_count >= required_clusters:
                entry["qualifying_frame_count"] = int(entry["qualifying_frame_count"]) + 1
                if entry["first"] is None:
                    entry["first"] = (
                        prediction_index,
                        metric_index,
                        raw_frame,
                        rows.copy(),
                        columns.copy(),
                        clusters.copy(),
                    )

    scale_records: dict[str, Any] = {}
    for size in cluster_sizes:
        entry = stats[size]
        first = entry.pop("first")
        fit_record = None
        if first is not None:
            fit_record = _first_fit(
                prediction,
                metric,
                prediction_index=first[0],
                metric_index=first[1],
                raw_frame=first[2],
                rows=first[3],
                columns=first[4],
                clusters=first[5],
                required_clusters=required_clusters,
            )
        scale_records[str(size)] = {
            **entry,
            "prefit_support": int(entry["qualifying_frame_count"]) > 0,
            "first_frame_robust_fit": fit_record,
            "contract_pass": bool(
                fit_record is not None and fit_record.get("contract_pass") is True
            ),
        }
    return {
        "object_id": unit.object_id,
        "episode": unit.episode,
        "stratum": unit.stratum,
        "camera_id": camera_id,
        "image_shape": list(image_shape),
        "raw_prefix_range_half_open": list(unit.raw_prefix_range_half_open),
        "visual_sha256": producer._sha256_file(visual_path),
        "metric_sha256": producer._sha256_file(metric_path),
        "scales": scale_records,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    request_path = arguments.request.resolve(strict=True)
    request = _request(request_path)
    revision = _canonical_revision(arguments.implementation_revision)
    compact = _ordinary_directory(arguments.compact_root, label="compact source root")
    results_root = _ordinary_directory(arguments.results_root, label="results root")
    upstream_root = _ordinary_directory(
        arguments.upstream_run_root, label="upstream run root"
    )
    forbidden = _ordinary_directory(
        arguments.forbidden_confirmation_root, label="forbidden confirmation root"
    )
    _assert_disjoint((compact, results_root, upstream_root), forbidden)

    plan_path = compact / "source-plan.json"
    execution_receipt = compact / "execution-receipt.json"
    if producer._sha256_file(plan_path) != producer.UPSTREAM_SOURCE_PLAN_FILE_SHA256:
        raise ValueError("source plan identity changed")
    lock_path = arguments.source_execution_lock.resolve(strict=True)
    lock = producer.load_deform360_joint_sparse_source_execution_lock_v5(lock_path)
    plan = producer.validate_deform360_joint_sparse_source_prediction_plan_v5(
        producer.load_strict_json_object(plan_path, label="source plan"),
        lock=lock,
    )
    units = producer._resolve_source_inputs(
        source_plan=plan,
        input_root=producer._ordinary_root(results_root),
        upstream_run_root=producer._ordinary_directory(
            upstream_root, name="upstream run root"
        ),
        upstream_evidence_root=execution_receipt.parent,
        common_artifacts={"audit/source-plan.json": producer._file_record(plan_path)},
    )

    cluster_sizes = tuple(int(value) for value in request["cluster_sizes_pixels"])
    required = int(request["minimum_independent_clusters"])
    camera_records: list[dict[str, Any]] = []
    for unit in units:
        for camera_id, visual_path, metric_path in unit.visual_inputs:
            record = _camera_record(
                unit,
                camera_id,
                visual_path,
                metric_path,
                cluster_sizes=cluster_sizes,
                required_clusters=required,
            )
            camera_records.append(record)
            default = record["scales"]["32"]
            print(
                f"{unit.object_id} {camera_id} default32={default['contract_pass']} "
                f"max32={default['maximum_independent_cluster_count']}",
                flush=True,
            )

    unit_ids = tuple(
        sorted(
            {
                f"{row['object_id']}#ep{int(row['episode']):04d}"
                for row in camera_records
            }
        )
    )
    aggregate: dict[str, Any] = {}
    for size in cluster_sizes:
        key = str(size)
        camera_passes = sum(
            bool(row["scales"][key]["contract_pass"]) for row in camera_records
        )
        unit_passes = 0
        for unit_id in unit_ids:
            rows = [
                row
                for row in camera_records
                if f"{row['object_id']}#ep{int(row['episode']):04d}" == unit_id
            ]
            if rows and all(bool(row["scales"][key]["contract_pass"]) for row in rows):
                unit_passes += 1
        aggregate[key] = {
            "camera_pass_count": camera_passes,
            "camera_count": len(camera_records),
            "unit_full_panel_pass_count": unit_passes,
            "unit_count": len(unit_ids),
            "all_cameras_pass": camera_passes == len(camera_records),
            "all_units_pass": unit_passes == len(unit_ids),
        }
    common_scales = [
        size for size in cluster_sizes if aggregate[str(size)]["all_cameras_pass"]
    ]
    identity: dict[str, Any] = {
        "schema": "bayesian-phystwin.deform360-metric-gauge-support-audit-v1",
        "schema_version": 1,
        "implementation_revision": revision,
        "request_sha256": producer._sha256_file(request_path),
        "upstream_source_plan_id": plan["plan_id"],
        "upstream_source_plan_sha256": producer._sha256_file(plan_path),
        "prior_failure": {
            "run_id": request["prior_failed_run_id"],
            "receipt_id": request["prior_failure_receipt_id"],
            "diagnostic": "metric gauge lacks eight independent causal clusters",
            "target_access": False,
        },
        "audit_contract": {
            "algorithm": "exact-first-qualifying-causal-frame-per-cluster-scale",
            "cluster_sizes_pixels": list(cluster_sizes),
            "minimum_independent_clusters": required,
            "robust_similarity_iterations": 5,
            "registered_predictor_changed": False,
        },
        "aggregate": aggregate,
        "common_full_panel_passing_scales_pixels": common_scales,
        "camera_records": camera_records,
        "information_boundary": {
            "source_prefix_arrays_opened": True,
            "source_suffix_opened": False,
            "confirmation_root_entered": False,
            "confirmation_payload_opened": False,
            "target_outcome_opened": False,
            "future_object_observations_used": False,
            "registered_prediction_run": False,
            "model_or_hyperparameter_selected": False,
        },
        "claim_authorized": False,
    }
    payload = dict(identity)
    payload["audit_id"] = _canonical_id(identity)
    output = arguments.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_id": payload["audit_id"],
                "camera_count": len(camera_records),
                "unit_count": len(unit_ids),
                "default_32": aggregate["32"],
                "common_full_panel_passing_scales_pixels": common_scales,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return payload


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
