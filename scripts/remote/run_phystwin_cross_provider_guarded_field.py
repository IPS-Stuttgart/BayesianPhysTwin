#!/usr/bin/env python3
"""Run a sealed sparse-witness, dense-field PhysTwin source smoke."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bayesian_phystwin.phystwin_cross_provider_guarded_field import (  # noqa: E402
    CrossProviderGuardedFieldConfig,
    build_guarded_dense_field,
)
from bayesian_phystwin.phystwin_mvtracker_competence import (  # noqa: E402
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.phystwin_official_evaluation import (  # noqa: E402
    _nearest_distances,
)

PROTOCOL_ID = "phystwin-cross-provider-guarded-field-v1"
CASE_ID = "single_lift_cloth"
PREDICTION_FILENAME = "prediction.npz"
PREDICTION_REPORT_FILENAME = "prediction_report.json"
PREDICTION_SEAL_FILENAME = "prediction_seal.json"
SCORE_FILENAME = "score.json"
VALIDATION_FILENAME = "prefix_validation.npz"
FUTURE_FILENAME = "future_score.npz"
STAGING_REPORT_FILENAME = "staging_report.json"

BASELINE_SHA256 = (
    "5e41ce3bfea780add79c20841084422ad7cad5e6e2443f3c2d2fca9729b8dd72"
)
MANUAL_TRACKS_SHA256 = (
    "dca0398d8660cc17d58f12142e144b930c5db67e13892820f43cf89edaabdf1e"
)
FINAL_DATA_SHA256 = (
    "0ca33031250c5efa8d25500b27488973770c47191d7707fc3499924e462f464b"
)
PROVIDER_IDENTITY_IDS = np.array([3, 4, 6, 8], dtype=np.int64)
EXPECTED_VALIDATION_IDENTITY_IDS = np.array([1, 5], dtype=np.int64)
EXPECTED_FUTURE_IDENTITY_IDS = np.array([0, 2, 7], dtype=np.int64)
MAXIMUM_ASSOCIATION_DISTANCE_M = 0.005

_NUMPY_PICKLE_MODULE_ALIASES = {
    "numpy._core.multiarray": "numpy.core.multiarray",
    "numpy._core.numeric": "numpy.core.numeric",
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _load_pickle(path: str | Path) -> Any:
    class NumpyCompatibilityUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            compatible = _NUMPY_PICKLE_MODULE_ALIASES.get(module)
            if compatible is not None:
                try:
                    importlib.import_module(module)
                except ModuleNotFoundError:
                    module = compatible
            return super().find_class(module, name)

    with Path(path).open("rb") as handle:
        return NumpyCompatibilityUnpickler(handle).load()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id changed")
    _require(int(protocol.get("schema_version", -1)) == 1, "protocol schema changed")
    return protocol


def _verified_input(protocol: dict[str, Any], name: str) -> Path:
    record = protocol["inputs"][name]
    path = Path(record["path"]).resolve()
    _require(path.is_file(), f"missing input {name}: {path}")
    _require(file_sha256(path) == record["sha256"], f"input hash changed: {name}")
    return path


def _verify_implementation(protocol: dict[str, Any]) -> None:
    implementation = protocol["implementation"]
    runner = Path(__file__).resolve()
    module = (
        REPO_ROOT
        / "src"
        / "bayesian_phystwin"
        / "phystwin_cross_provider_guarded_field.py"
    )
    _require(file_sha256(runner) == implementation["runner_sha256"], "runner changed")
    _require(
        file_sha256(module) == implementation["module_sha256"],
        "guarded-field module changed",
    )


def _association(
    baseline_frame_zero_m: np.ndarray,
    tracks_frame_zero_m: np.ndarray,
    identity_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(baseline_frame_zero_m, dtype=np.float64)
    tracks = np.asarray(tracks_frame_zero_m, dtype=np.float64)
    identities = np.asarray(identity_ids, dtype=np.int64)
    query = tracks[identities]
    _require(np.all(np.isfinite(query)), "frame-zero identity is not finite")
    distance = np.linalg.norm(baseline[:, None] - query[None], axis=2)
    nodes = np.argmin(distance, axis=0).astype(np.int64)
    nearest = distance[nodes, np.arange(len(identities))]
    _require(len(np.unique(nodes)) == len(nodes), "identity nodes are not unique")
    _require(
        np.all(nearest <= MAXIMUM_ASSOCIATION_DISTANCE_M),
        "identity-to-node association exceeds 5 mm",
    )
    return nodes, nearest


def _stage(
    baseline_path: Path,
    manual_tracks_path: Path,
    final_data_path: Path,
    output: Path,
) -> None:
    expected = {
        baseline_path: BASELINE_SHA256,
        manual_tracks_path: MANUAL_TRACKS_SHA256,
        final_data_path: FINAL_DATA_SHA256,
    }
    for path, digest in expected.items():
        _require(file_sha256(path) == digest, f"staging input changed: {path}")
    _require(not output.exists(), "staging output already exists")
    output.mkdir(parents=True)

    baseline = np.asarray(_load_pickle(baseline_path))
    tracks = np.asarray(_load_pickle(manual_tracks_path), dtype=np.float64)
    data = _load_pickle(final_data_path)
    object_points = np.asarray(data["object_points"], dtype=np.float64)
    object_visible = np.asarray(data["object_visibilities"], dtype=bool)
    _require(
        baseline.shape == object_points.shape
        and object_visible.shape == object_points.shape[:2]
        and tracks.shape == (len(baseline), 9, 3),
        "source shapes changed",
    )

    cfg = CrossProviderGuardedFieldConfig()
    all_ids = np.arange(tracks.shape[1], dtype=np.int64)
    candidate_ids = np.setdiff1d(all_ids, PROVIDER_IDENTITY_IDS)
    prefix_available = np.sum(
        np.all(
            np.isfinite(
                tracks[
                    cfg.apply_frame_start : cfg.validation_frame_end_exclusive,
                    candidate_ids,
                ]
            ),
            axis=2,
        ),
        axis=0,
    )
    order = np.lexsort((candidate_ids, -prefix_available))
    validation_ids = np.sort(candidate_ids[order[:2]])
    future_ids = np.setdiff1d(candidate_ids, validation_ids)
    _require(
        np.array_equal(validation_ids, EXPECTED_VALIDATION_IDENTITY_IDS),
        "prefix-only validation identity selection changed",
    )
    _require(
        np.array_equal(future_ids, EXPECTED_FUTURE_IDENTITY_IDS),
        "future identity complement changed",
    )
    _require(
        np.all(prefix_available[order[:2]] > 0),
        "validation identities have no prefix support",
    )

    provider_nodes, provider_distance = _association(
        baseline[0],
        tracks[0],
        PROVIDER_IDENTITY_IDS,
    )
    validation_nodes, validation_distance = _association(
        baseline[0],
        tracks[0],
        validation_ids,
    )
    future_nodes, future_distance = _association(
        baseline[0],
        tracks[0],
        future_ids,
    )

    validation_path = output / VALIDATION_FILENAME
    _atomic_npz(
        validation_path,
        provider_identity_ids=PROVIDER_IDENTITY_IDS,
        provider_node_ids=provider_nodes,
        provider_association_distance_m=provider_distance,
        validation_identity_ids=validation_ids,
        validation_node_ids=validation_nodes,
        validation_association_distance_m=validation_distance,
        validation_tracks_world_m=tracks[
            cfg.apply_frame_start : cfg.validation_frame_end_exclusive,
            validation_ids,
        ],
        validation_frame_start=np.array(cfg.apply_frame_start, dtype=np.int64),
        validation_frame_end_exclusive=np.array(
            cfg.validation_frame_end_exclusive,
            dtype=np.int64,
        ),
    )
    future_start = cfg.validation_frame_end_exclusive
    future_end = len(baseline)
    future_path = output / FUTURE_FILENAME
    _atomic_npz(
        future_path,
        future_identity_ids=future_ids,
        future_node_ids=future_nodes,
        future_association_distance_m=future_distance,
        future_tracks_world_m=tracks[future_start:future_end, future_ids],
        object_points_world_m=object_points[future_start:future_end],
        object_visibility=object_visible[future_start:future_end],
        future_frame_start=np.array(future_start, dtype=np.int64),
        future_frame_end_exclusive=np.array(future_end, dtype=np.int64),
        num_surface_points=np.array(object_points.shape[1], dtype=np.int64),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinCrossProviderGuardedFieldStaging",
        "protocol_id": PROTOCOL_ID,
        "case_id": CASE_ID,
        "source_inputs": {
            "baseline_sha256": BASELINE_SHA256,
            "manual_tracks_sha256": MANUAL_TRACKS_SHA256,
            "final_data_sha256": FINAL_DATA_SHA256,
        },
        "identity_boundary": {
            "provider_identity_ids": PROVIDER_IDENTITY_IDS.tolist(),
            "validation_identity_ids": validation_ids.tolist(),
            "future_identity_ids": future_ids.tolist(),
            "sets_are_pairwise_disjoint": True,
            "validation_selection_rule": (
                "top two non-provider identities by finite availability on "
                "frames [88,121), ties by identity id; values and future "
                "availability are not used"
            ),
        },
        "association": {
            "reference_frame": 0,
            "maximum_distance_m": MAXIMUM_ASSOCIATION_DISTANCE_M,
            "provider_node_ids": provider_nodes.tolist(),
            "provider_distance_m": provider_distance.tolist(),
            "validation_node_ids": validation_nodes.tolist(),
            "validation_distance_m": validation_distance.tolist(),
            "future_node_ids": future_nodes.tolist(),
            "future_distance_m": future_distance.tolist(),
        },
        "outputs": {
            "prefix_validation": {
                "path": str(validation_path),
                "sha256": file_sha256(validation_path),
            },
            "future_score": {
                "path": str(future_path),
                "sha256": file_sha256(future_path),
            },
        },
        "information_boundary": {
            "future_outcomes_used_for_identity_selection": False,
            "future_score_artifact_is_for_post_seal_score_only": True,
            "held_v8_accessed": False,
        },
    }
    report["result_sha256"] = canonical_sha256(report)
    _atomic_json(output / STAGING_REPORT_FILENAME, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def _prediction_inputs(protocol: dict[str, Any]) -> dict[str, Path]:
    return {
        name: _verified_input(protocol, name)
        for name in (
            "baseline",
            "cotracker_cues",
            "complement_prediction",
            "complement_seal",
            "graph_basis",
            "prefix_validation",
        )
    }


def _predict(protocol_path: Path, output: Path) -> None:
    protocol = _load_protocol(protocol_path)
    _verify_implementation(protocol)
    paths = _prediction_inputs(protocol)
    _require(not output.exists(), "prediction output already exists")
    output.mkdir(parents=True)

    complement_seal = json.loads(
        paths["complement_seal"].read_text(encoding="utf-8")
    )
    _require(
        complement_seal.get("artifact_kind")
        == "PhysTwinTAPNextPPCoTrackerComplementPredictionSeal",
        "complement seal kind changed",
    )
    _require(
        complement_seal.get("prediction_archive_sha256")
        == protocol["inputs"]["complement_prediction"]["sha256"],
        "complement seal does not bind the provider archive",
    )

    baseline = np.asarray(_load_pickle(paths["baseline"]))
    with np.load(paths["graph_basis"], allow_pickle=False) as stored:
        graph_basis = np.asarray(stored["graph_basis"], dtype=np.float64)
    with np.load(paths["prefix_validation"], allow_pickle=False) as stored:
        provider_ids = np.asarray(stored["provider_identity_ids"], dtype=np.int64)
        provider_nodes = np.asarray(stored["provider_node_ids"], dtype=np.int64)
        validation_ids = np.asarray(
            stored["validation_identity_ids"],
            dtype=np.int64,
        )
        validation_nodes = np.asarray(stored["validation_node_ids"], dtype=np.int64)
        validation_tracks = np.asarray(
            stored["validation_tracks_world_m"],
            dtype=np.float64,
        )
        validation_start = int(stored["validation_frame_start"])
        validation_stop = int(stored["validation_frame_end_exclusive"])
    with np.load(paths["complement_prediction"], allow_pickle=False) as stored:
        complement_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        provider_trajectory = np.asarray(stored["trajectory_world_m"])
        provider_support = np.asarray(stored["accepted_support"], dtype=bool)
        provider_code = np.asarray(stored["provider_code"], dtype=np.int8)
        dense_local = np.asarray(stored["complement_trajectory_world_m"])
        dense_local_available = np.asarray(
            stored["complement_available"],
            dtype=bool,
        )
    cfg = CrossProviderGuardedFieldConfig(**protocol["method_config"])
    _require(np.array_equal(provider_ids, complement_ids), "provider ids changed")
    _require(
        (validation_start, validation_stop)
        == (cfg.apply_frame_start, cfg.validation_frame_end_exclusive),
        "validation interval changed",
    )
    with np.load(paths["cotracker_cues"], allow_pickle=False) as stored:
        source = slice(cfg.source_frame_start, cfg.source_frame_end_exclusive)
        dense_points = np.asarray(
            stored["multiview_points_world_m"][source],
            dtype=np.float64,
        )
        dense_valid = np.asarray(
            stored["multiview_point_valid"][source],
            dtype=bool,
        )
        dense_camera_count = np.asarray(
            stored["multiview_camera_count"][source],
        )
        dense_reprojection = np.asarray(
            stored["multiview_reprojection_error_px"][source],
            dtype=np.float64,
        )
        dense_quality = np.asarray(
            stored["cotracker_quality_probability"][source],
            dtype=np.float64,
        )

    result = build_guarded_dense_field(
        baseline,
        graph_basis,
        dense_points,
        dense_valid,
        dense_camera_count,
        dense_reprojection,
        dense_quality,
        provider_trajectory,
        provider_support,
        provider_code,
        provider_ids,
        provider_nodes,
        dense_local,
        dense_local_available,
        validation_tracks,
        validation_ids,
        validation_nodes,
        config=cfg,
    )
    prediction_path = output / PREDICTION_FILENAME
    _atomic_npz(
        prediction_path,
        baseline_trajectory_m=baseline,
        candidate_trajectory_m=result["candidate_trajectory_m"],
        raw_candidate_trajectory_m=result["raw_candidate_trajectory_m"],
        sparse_comparator_trajectory_m=result[
            "sparse_comparator_trajectory_m"
        ],
        dense_field_m=result["dense_field_m"],
        sparse_field_m=result["sparse_field_m"],
        dense_coefficients_m=result["dense_coefficients_m"],
        sparse_coefficients_m=result["sparse_coefficients_m"],
        provider_identity_ids=provider_ids,
        provider_node_ids=provider_nodes,
        validation_identity_ids=validation_ids,
        validation_node_ids=validation_nodes,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinCrossProviderGuardedFieldPrediction",
        "protocol_id": PROTOCOL_ID,
        "case_id": CASE_ID,
        "protocol_sha256": file_sha256(protocol_path),
        "accepted": bool(result["accepted"]),
        "selection_reason": str(result["reason"]),
        "diagnostics": result["diagnostics"],
        "output": {
            "prediction_path": str(prediction_path),
            "prediction_sha256": file_sha256(prediction_path),
        },
        "implementation": {
            "repository_commit": _git_commit(),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
        },
        "information_boundary": {
            "prefix_validation_read": True,
            "future_score_artifact_read": False,
            "future_object_observation_read": False,
            "future_manual_identity_read": False,
            "held_v8_accessed": False,
        },
        "outcomes": None,
        "claim_boundary": (
            "Post-open one-case source capacity smoke with manual disjoint "
            "prefix validation; not deployable, independent, confirmatory, "
            "or state-of-the-art evidence."
        ),
    }
    report["result_sha256"] = canonical_sha256(report)
    _atomic_json(output / PREDICTION_REPORT_FILENAME, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def _seal(protocol_path: Path, output: Path) -> None:
    protocol = _load_protocol(protocol_path)
    _verify_implementation(protocol)
    report_path = output / PREDICTION_REPORT_FILENAME
    prediction_path = output / PREDICTION_FILENAME
    seal_path = output / PREDICTION_SEAL_FILENAME
    _require(report_path.is_file() and prediction_path.is_file(), "prediction missing")
    _require(not seal_path.exists(), "prediction is already sealed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("result_sha256") == canonical_sha256(report),
        "prediction report self-hash changed",
    )
    _require(
        report.get("protocol_sha256") == file_sha256(protocol_path),
        "prediction protocol changed",
    )
    prediction_sha256 = file_sha256(prediction_path)
    _require(
        report.get("output", {}).get("prediction_sha256") == prediction_sha256,
        "prediction archive changed",
    )
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinCrossProviderGuardedFieldPredictionSeal",
        "protocol_id": PROTOCOL_ID,
        "case_id": CASE_ID,
        "protocol_sha256": file_sha256(protocol_path),
        "prediction_report_sha256": file_sha256(report_path),
        "prediction_archive_sha256": prediction_sha256,
        "prediction_result_sha256": report["result_sha256"],
        "information_boundary": {
            "future_score_artifact_read_before_seal": False,
            "future_score_authorized_after_seal": True,
            "held_v8_accessed": False,
        },
    }
    seal["result_sha256"] = canonical_sha256(seal)
    _atomic_json(seal_path, seal)
    print(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False))


def _metric_summary(values: np.ndarray) -> dict[str, Any]:
    metric = np.asarray(values, dtype=np.float64)
    boundaries = np.linspace(0, len(metric), 4, dtype=int)

    def summarize(section: np.ndarray) -> float | None:
        finite = section[np.isfinite(section)]
        return float(np.mean(finite)) if len(finite) else None

    return {
        "mean_m": summarize(metric),
        "early_mean_m": summarize(metric[boundaries[0] : boundaries[1]]),
        "middle_mean_m": summarize(metric[boundaries[1] : boundaries[2]]),
        "late_mean_m": summarize(metric[boundaries[2] : boundaries[3]]),
        "finite_frame_count": int(np.sum(np.isfinite(metric))),
        "by_frame_m": [
            float(value) if np.isfinite(value) else None for value in metric
        ],
    }


def _chamfer_by_frame(
    trajectory: np.ndarray,
    object_points: np.ndarray,
    object_visible: np.ndarray,
    *,
    frame_start: int,
    num_surface_points: int,
) -> np.ndarray:
    values = np.empty(len(object_points), dtype=np.float64)
    for offset in range(len(object_points)):
        observed = object_points[offset, object_visible[offset]]
        _require(len(observed) > 0, "future frame has no visible object points")
        distance, _ = _nearest_distances(
            trajectory[frame_start + offset, :num_surface_points],
            observed,
            p=1,
        )
        values[offset] = np.mean(distance)
    return values


def _track_error_by_frame(
    trajectory: np.ndarray,
    tracks: np.ndarray,
    node_ids: np.ndarray,
    *,
    frame_start: int,
) -> tuple[np.ndarray, int]:
    prediction = trajectory[
        frame_start : frame_start + len(tracks),
        node_ids,
    ]
    valid = np.all(np.isfinite(tracks), axis=2)
    residual = np.linalg.norm(prediction - tracks, axis=2)
    values = np.full(len(tracks), np.nan, dtype=np.float64)
    for frame in range(len(tracks)):
        if np.any(valid[frame]):
            values[frame] = np.mean(residual[frame, valid[frame]])
    return values, int(np.sum(valid))


def _score(protocol_path: Path, output: Path) -> None:
    protocol = _load_protocol(protocol_path)
    _verify_implementation(protocol)
    report_path = output / PREDICTION_REPORT_FILENAME
    prediction_path = output / PREDICTION_FILENAME
    seal_path = output / PREDICTION_SEAL_FILENAME
    score_path = output / SCORE_FILENAME
    _require(not score_path.exists(), "score already exists")
    _require(seal_path.is_file(), "prediction is not sealed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("artifact_kind")
        == "PhysTwinCrossProviderGuardedFieldPredictionSeal",
        "prediction seal kind changed",
    )
    _require(seal.get("result_sha256") == canonical_sha256(seal), "seal changed")
    _require(
        seal.get("protocol_sha256") == file_sha256(protocol_path)
        and seal.get("prediction_report_sha256") == file_sha256(report_path)
        and seal.get("prediction_archive_sha256") == file_sha256(prediction_path),
        "sealed prediction provenance changed",
    )

    future_path = _verified_input(protocol, "future_score")
    with np.load(future_path, allow_pickle=False) as stored:
        future_ids = np.asarray(stored["future_identity_ids"], dtype=np.int64)
        future_nodes = np.asarray(stored["future_node_ids"], dtype=np.int64)
        tracks = np.asarray(stored["future_tracks_world_m"], dtype=np.float64)
        object_points = np.asarray(stored["object_points_world_m"], dtype=np.float64)
        object_visible = np.asarray(stored["object_visibility"], dtype=bool)
        frame_start = int(stored["future_frame_start"])
        frame_end = int(stored["future_frame_end_exclusive"])
        num_surface_points = int(stored["num_surface_points"])
    _require(
        np.array_equal(future_ids, EXPECTED_FUTURE_IDENTITY_IDS),
        "future identity split changed",
    )
    _require(frame_end - frame_start == len(tracks), "future interval changed")
    with np.load(prediction_path, allow_pickle=False) as stored:
        trajectories = {
            name: np.asarray(stored[f"{name}_trajectory_m"])
            for name in (
                "baseline",
                "candidate",
                "raw_candidate",
                "sparse_comparator",
            )
        }

    scores: dict[str, Any] = {}
    for name, trajectory in trajectories.items():
        chamfer = _chamfer_by_frame(
            trajectory,
            object_points,
            object_visible,
            frame_start=frame_start,
            num_surface_points=num_surface_points,
        )
        track, track_point_frames = _track_error_by_frame(
            trajectory,
            tracks,
            future_nodes,
            frame_start=frame_start,
        )
        scores[name] = {
            "chamfer_distance_m": _metric_summary(chamfer),
            "hidden_track_error_m": _metric_summary(track),
            "hidden_track_point_frame_count": track_point_frames,
        }

    baseline_cd = scores["baseline"]["chamfer_distance_m"]["mean_m"]
    candidate_cd = scores["candidate"]["chamfer_distance_m"]["mean_m"]
    baseline_track = scores["baseline"]["hidden_track_error_m"]["mean_m"]
    candidate_track = scores["candidate"]["hidden_track_error_m"]["mean_m"]
    _require(
        None not in (baseline_cd, candidate_cd, baseline_track, candidate_track),
        "future metrics have no finite support",
    )
    gates = {
        "prefix_guard_accepted": bool(
            json.loads(report_path.read_text(encoding="utf-8"))["accepted"]
        ),
        "future_chamfer_improved": bool(candidate_cd < baseline_cd),
        "future_hidden_track_improved": bool(candidate_track < baseline_track),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinCrossProviderGuardedFieldSourceScore",
        "protocol_id": PROTOCOL_ID,
        "case_id": CASE_ID,
        "status": "opened-source-capacity-smoke",
        "prediction_seal_sha256": file_sha256(seal_path),
        "future_score_sha256": file_sha256(future_path),
        "scores": scores,
        "candidate_comparison": {
            "chamfer_improvement_fraction": float(
                1.0 - candidate_cd / baseline_cd
            ),
            "hidden_track_improvement_fraction": float(
                1.0 - candidate_track / baseline_track
            ),
        },
        "gates": gates,
        "all_gates_passed": bool(all(gates.values())),
        "information_boundary": {
            "future_score_read_only_after_prediction_seal": True,
            "provider_validation_and_future_identities_pairwise_disjoint": True,
            "future_object_observations_used_for_prediction": False,
            "held_v8_accessed": False,
        },
        "claim_boundary": (
            "Post-open one-case source capacity smoke. A pass only authorizes "
            "a separately locked source panel; it is not deployable, "
            "independent, confirmatory, or state-of-the-art evidence."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    _atomic_json(score_path, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage")
    stage.add_argument("--baseline", type=Path, required=True)
    stage.add_argument("--manual-tracks", type=Path, required=True)
    stage.add_argument("--final-data", type=Path, required=True)
    stage.add_argument("--output-dir", type=Path, required=True)

    for command in ("predict", "seal", "score"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--protocol", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "stage":
        _stage(
            args.baseline.resolve(),
            args.manual_tracks.resolve(),
            args.final_data.resolve(),
            args.output_dir.resolve(),
        )
    elif args.command == "predict":
        _predict(args.protocol.resolve(), args.output_dir.resolve())
    elif args.command == "seal":
        _seal(args.protocol.resolve(), args.output_dir.resolve())
    else:
        _score(args.protocol.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
