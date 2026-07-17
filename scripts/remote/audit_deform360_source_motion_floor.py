#!/usr/bin/env python3
"""Measure source-only Deform360 persistence and action-response floors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from causal4d_public.deform360_dense_reusable_panel import (
    authorize_dense_panel_episode,
    load_dense_reusable_panel_config,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _symmetric_chamfer_m(first: np.ndarray, second: np.ndarray) -> float:
    _require(len(first) > 0 and len(second) > 0, "Chamfer inputs are empty")
    first_to_second = cKDTree(second).query(first, workers=-1)[0]
    second_to_first = cKDTree(first).query(second, workers=-1)[0]
    return float(0.5 * (first_to_second.mean() + second_to_first.mean()))


def _mean_point_path_m(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        np.mean(np.linalg.norm(np.diff(points, axis=0), axis=-1), axis=1).sum()
    )


def _score_range(
    object_points: np.ndarray,
    valid: np.ndarray,
    controller_points: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, float | int]:
    persistence = object_points[0]
    coordinate_rmse = []
    euclidean_error = []
    chamfer = []
    mean_displacement = []
    maximum_displacement = []
    valid_counts = []
    for frame in range(start, stop):
        frame_valid = valid[0] & valid[frame]
        _require(np.any(frame_valid), f"frame {frame} has no persistent valid points")
        difference = object_points[frame, frame_valid] - persistence[frame_valid]
        norm = np.linalg.norm(difference, axis=1)
        coordinate_rmse.append(float(np.sqrt(np.mean(difference**2))))
        euclidean_error.append(float(np.mean(norm)))
        mean_displacement.append(float(np.mean(norm)))
        maximum_displacement.append(float(np.max(norm)))
        chamfer.append(
            _symmetric_chamfer_m(
                persistence[frame_valid], object_points[frame, frame_valid]
            )
        )
        valid_counts.append(int(np.count_nonzero(frame_valid)))

    selected_controller = controller_points[start:stop]
    controller_displacement = np.linalg.norm(
        selected_controller - controller_points[0], axis=-1
    )
    return {
        "frame_count": stop - start,
        "minimum_valid_point_count": min(valid_counts),
        "persistence_coordinate_rmse_m": float(np.mean(coordinate_rmse)),
        "persistence_mean_euclidean_track_error_m": float(np.mean(euclidean_error)),
        "persistence_symmetric_chamfer_m": float(np.mean(chamfer)),
        "object_mean_displacement_m": float(np.mean(mean_displacement)),
        "object_maximum_displacement_m": float(np.max(maximum_displacement)),
        "controller_mean_displacement_m": float(np.mean(controller_displacement)),
        "controller_maximum_displacement_m": float(np.max(controller_displacement)),
        "controller_mean_point_path_m": _mean_point_path_m(selected_controller),
    }


def _load_episode(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    _require(isinstance(payload, Mapping), f"{path} is not a mapping")
    return payload


def _parse_episode(value: str) -> tuple[str, int, Path]:
    fields = value.split("=", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("episode must be OBJECT_ID=EPISODE_ID=PATH")
    try:
        episode_id = int(fields[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("episode id must be an integer") from error
    return fields[0], episode_id, Path(fields[2])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--episode",
        type=_parse_episode,
        action="append",
        required=True,
        help="source episode as OBJECT_ID=EPISODE_ID=FINAL_DATA_PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_dense_reusable_panel_config(args.config)
    config = protocol["config"]
    horizons = config["frame_protocol"]["horizon_ranges_half_open"]
    cohort = {row["object_id"]: row for row in config["cohort"]}
    records = []

    for object_id, episode_id, data_path in args.episode:
        authorize_dense_panel_episode(
            protocol,
            object_id=object_id,
            episode_id=episode_id,
            phase="source",
            source_admission_passed=False,
        )
        _require(data_path.is_file(), f"missing source data {data_path}")
        data = _load_episode(data_path)
        object_points = np.asarray(data["object_points"], dtype=np.float64)
        visibility = np.asarray(data["object_visibilities"], dtype=bool)
        motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
        controller_points = np.asarray(data["controller_points"], dtype=np.float64)
        _require(
            object_points.ndim == 3
            and object_points.shape[2] == 3
            and visibility.shape == object_points.shape[:2]
            and motion_valid.shape == object_points.shape[:2],
            f"invalid object arrays in {data_path}",
        )
        _require(
            controller_points.ndim == 3
            and controller_points.shape[0] == len(object_points)
            and controller_points.shape[2] == 3,
            f"invalid controller array in {data_path}",
        )
        valid = visibility & motion_valid & np.isfinite(object_points).all(axis=2)
        full = _score_range(
            object_points, valid, controller_points, 1, len(object_points)
        )
        horizon_scores = {
            name: _score_range(
                object_points,
                valid,
                controller_points,
                int(bounds[0]),
                min(int(bounds[1]), len(object_points)),
            )
            for name, bounds in horizons.items()
        }
        records.append(
            {
                "object_id": object_id,
                "stratum": cohort[object_id]["stratum"],
                "episode_id": episode_id,
                "final_data_path": str(data_path.resolve()),
                "final_data_sha256": _sha256_file(data_path),
                "object_point_count": int(object_points.shape[1]),
                "controller_point_count": int(controller_points.shape[1]),
                "full_future": full,
                "horizons": horizon_scores,
            }
        )

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[str(record["stratum"])].append(record)
    aggregate = {}
    metric_names = (
        "persistence_coordinate_rmse_m",
        "persistence_mean_euclidean_track_error_m",
        "persistence_symmetric_chamfer_m",
        "object_mean_displacement_m",
        "controller_mean_displacement_m",
    )
    for stratum, selected in sorted(by_stratum.items()):
        aggregate[stratum] = {
            name: float(np.mean([row["full_future"][name] for row in selected]))
            for name in metric_names
        }
        aggregate[stratum]["episode_count"] = len(selected)

    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360SourceMotionFloorAudit",
        "protocol_id": config["protocol_id"],
        "config_sha256": protocol["config_sha256"],
        "evidence_scope": "source-only",
        "record_count": len(records),
        "records": records,
        "aggregate_by_stratum": aggregate,
        "published_particleformer_reference": config["target_panel"][
            "published_particleformer_multi_episode"
        ],
        "reference_comparison_is_confirmatory": False,
        "target_initial_frame_read": False,
        "target_post_initial_object_observations_read": False,
        "target_future_read": False,
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
