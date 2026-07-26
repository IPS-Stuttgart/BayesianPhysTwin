"""Explicit evaluator boundary for Deform360 state-of-the-art comparisons.

Deform360 now publishes processed episodes with ordered advected particles and
per-episode train/test windows.  The complete Table 4 split, exact metric
semantics, baseline predictions, and world-model evaluator remain unreleased.
This module consumes the newly released evidence while refusing to silently
turn an independent development score into a protocol-mismatched SOTA claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_reusable_sota_protocol import (
    EXPECTED_FIT_EPISODES,
    EXPECTED_HELD_EPISODES,
    validate_reusable_sota_config,
)
from .deform360_sota_processing import (
    DEVELOPMENT_OBSERVATIONS_KIND,
    PINNED_COTRACKER_CHECKPOINT_SHA256,
    PINNED_COTRACKER_REVISION,
    PINNED_DEFORM360_PROCESSING_REVISION,
)


DEFORM360_EVALUATOR_CONTRACT_SCHEMA_VERSION = 2
DEFORM360_EVALUATOR_CONTRACT_KIND = "Deform360EvaluatorContract"
DEFORM360_EPISODE_SCORE_KIND = "Deform360EpisodeScore"
DEFORM360_PANEL_SCORE_KIND = "Deform360PanelScore"
DEFORM360_RELEASED_PROCESSED_EPISODE_KIND = "Deform360ReleasedProcessedEpisodeManifest"
DEFORM360_RELEASED_PROCESSED_EPISODE_SCHEMA_VERSION = 1

_CONTRACT_STATUSES = {
    "unresolved-non-authorizing",
    "independent-protocol",
    "official-parity",
}
_CHAMFER_DEFINITIONS = {
    "symmetric_mean_euclidean_m",
    "symmetric_mean_squared_euclidean_m2",
}
_TRACK_DEFINITIONS = {
    "mean_euclidean_m",
    "root_mean_squared_euclidean_m",
    "mean_squared_euclidean_m2",
}
_VISIBILITY_POLICIES = {
    "all_finite_material_points",
    "visible_and_finite_material_points",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_git_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    canonical = dict(value)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_identity_sha256(points: np.ndarray) -> str:
    values = np.ascontiguousarray(points, dtype="<f4")
    digest = hashlib.sha256(b"deform360-ordered-material-identity-v1\0")
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def deform360_evaluator_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical checksum used by evaluator-contract artifacts."""

    return _canonical_sha256(payload)


def _episode_key(object_id: str, episode_id: int) -> str:
    return f"{object_id}/{int(episode_id)}"


def validate_deform360_evaluator_contract(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a contract without pretending unresolved fields are known."""

    _require(
        payload.get("schema_version") == DEFORM360_EVALUATOR_CONTRACT_SCHEMA_VERSION,
        "unsupported Deform360 evaluator-contract schema",
    )
    _require(
        payload.get("artifact_kind") == DEFORM360_EVALUATOR_CONTRACT_KIND,
        "unexpected Deform360 evaluator-contract kind",
    )
    _require(
        payload.get("result_sha256") == _canonical_sha256(payload),
        "Deform360 evaluator-contract checksum mismatch",
    )
    status = payload.get("status")
    _require(status in _CONTRACT_STATUSES, "invalid evaluator-contract status")
    unresolved = payload.get("unresolved_fields")
    _require(
        isinstance(unresolved, list)
        and all(isinstance(value, str) and value for value in unresolved),
        "unresolved_fields must be a list of nonempty paths",
    )
    dataset = payload.get("dataset")
    _require(isinstance(dataset, Mapping), "contract dataset is missing")
    _require(dataset.get("coordinate_unit") == "m", "coordinate unit must be metres")
    split = payload.get("split")
    _require(isinstance(split, Mapping), "contract split is missing")
    metrics = payload.get("metrics")
    _require(isinstance(metrics, Mapping), "contract metrics are missing")
    temporal = payload.get("temporal")
    _require(isinstance(temporal, Mapping), "contract temporal policy is missing")
    particles = payload.get("particles")
    _require(isinstance(particles, Mapping), "contract particle policy is missing")
    aggregation = payload.get("aggregation")
    _require(isinstance(aggregation, Mapping), "contract aggregation is missing")
    reference = payload.get("published_reference")
    _require(isinstance(reference, Mapping), "published reference is missing")
    _require(
        float(reference.get("future_chamfer_m", -1.0)) == 0.051
        and float(reference.get("future_track_error_m", -1.0)) == 0.079,
        "published Deform360 Table 4 reference changed",
    )
    if status == "official-parity":
        _validate_official_parity_contract(payload)
    return {
        "passed": True,
        "status": status,
        "official_table4_authorizing": status == "official-parity",
        "unresolved_field_count": len(unresolved),
        "result_sha256": payload["result_sha256"],
    }


def _validate_official_parity_contract(payload: Mapping[str, Any]) -> None:
    _require(not payload["unresolved_fields"], "official parity has unresolved fields")
    split = payload["split"]
    object_ids = split.get("object_ids")
    fit = split.get("fit_episode_ids_by_object")
    held = split.get("held_episode_ids_by_object")
    _require(
        isinstance(object_ids, list) and object_ids,
        "official parity requires the complete ordered object split",
    )
    _require(
        isinstance(fit, Mapping)
        and isinstance(held, Mapping)
        and set(fit) == set(held) == set(object_ids),
        "official split maps differ from object_ids",
    )
    _require(
        all(fit[object_id] and held[object_id] for object_id in object_ids),
        "official split contains an empty fit or held set",
    )
    expected_keys = {
        _episode_key(object_id, episode_id)
        for object_id in object_ids
        for episode_id in held[object_id]
    }
    temporal = payload["temporal"]
    global_temporal = all(
        isinstance(temporal.get(field), int)
        for field in (
            "evaluation_start_frame",
            "evaluation_stop_frame_exclusive",
            "frame_stride",
        )
    )
    per_episode_temporal = temporal.get("evaluation_frame_indices_by_episode")
    if global_temporal:
        _require(
            0
            <= temporal["evaluation_start_frame"]
            < temporal["evaluation_stop_frame_exclusive"]
            and temporal["frame_stride"] >= 1,
            "official temporal range is invalid",
        )
    else:
        _require(
            isinstance(per_episode_temporal, Mapping)
            and set(per_episode_temporal) == expected_keys
            and all(
                isinstance(indices, list)
                and indices
                and all(
                    isinstance(frame_index, int) and frame_index >= 0
                    for frame_index in indices
                )
                and indices == sorted(set(indices))
                for indices in per_episode_temporal.values()
            ),
            "official per-episode temporal ranges are unresolved",
        )
    metrics = payload["metrics"]
    _require(
        metrics.get("chamfer", {}).get("definition") in _CHAMFER_DEFINITIONS,
        "official Chamfer definition is unresolved",
    )
    _require(
        metrics.get("chamfer", {}).get("visibility_policy") in _VISIBILITY_POLICIES,
        "official Chamfer visibility policy is unresolved",
    )
    _require(
        metrics.get("track", {}).get("definition") in _TRACK_DEFINITIONS,
        "official track definition is unresolved",
    )
    _require(
        metrics.get("track", {}).get("visibility_policy") in _VISIBILITY_POLICIES,
        "official track visibility policy is unresolved",
    )
    _require(
        payload["aggregation"].get("panel")
        in {"object_balanced_mean", "episode_balanced_mean"},
        "official panel aggregation is unresolved",
    )
    identities = payload["particles"].get("identity_sha256_by_episode")
    _require(
        isinstance(identities, Mapping) and set(identities) == expected_keys,
        "official particle identities do not cover the complete held split",
    )
    _require(
        all(_valid_sha256(value) for value in identities.values()),
        "official particle identity checksum is invalid",
    )
    provenance = payload.get("evaluator_provenance")
    _require(
        isinstance(provenance, Mapping), "official evaluator provenance is missing"
    )
    _require(
        provenance.get("released_by_deform360_authors") is True,
        "evaluator is not author-released",
    )
    _require(
        _valid_sha256(provenance.get("source_revision_sha256"))
        and _valid_sha256(provenance.get("entrypoint_sha256")),
        "official evaluator provenance checksum is invalid",
    )
    reproduction = provenance.get("particleformer_table4_reproduction")
    _require(
        isinstance(reproduction, Mapping)
        and reproduction.get("passed") is True
        and float(reproduction.get("future_chamfer_m", -1.0)) == 0.051
        and float(reproduction.get("future_track_error_m", -1.0)) == 0.079,
        "official evaluator has not reproduced the published reference row",
    )


def load_deform360_evaluator_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "evaluator contract must contain an object")
    validate_deform360_evaluator_contract(payload)
    return payload


def write_deform360_evaluator_contract(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    validate_deform360_evaluator_contract(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _load_released_pcd_frame(
    path: Path,
    *,
    expected_camera_count: int | None = None,
) -> dict[str, np.ndarray]:
    _require(path.is_file(), f"released pcd frame is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "pts",
            "colors",
            "vels",
            "camera_indices",
            "visibility_matrix",
        }
        _require(
            required.issubset(archive.files),
            f"released pcd frame has missing arrays: {path}",
        )
        arrays = {name: np.asarray(archive[name]) for name in required}
    points = arrays["pts"]
    colors = arrays["colors"]
    velocities = arrays["vels"]
    camera_indices = arrays["camera_indices"]
    visibility = arrays["visibility_matrix"]
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and points.dtype == np.float32
        and colors.shape == points.shape
        and colors.dtype == np.float32
        and velocities.shape == points.shape
        and velocities.dtype == np.float32,
        f"released pcd geometry contract changed: {path}",
    )
    _require(
        camera_indices.shape == (len(points),)
        and camera_indices.dtype == np.int32
        and visibility.ndim == 2
        and visibility.shape[0] == len(points)
        and visibility.dtype == np.uint8,
        f"released pcd support contract changed: {path}",
    )
    if expected_camera_count is not None:
        _require(
            visibility.shape[1] == expected_camera_count,
            f"released pcd camera count changed: {path}",
        )
    _require(
        len(points) > 0
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(colors))
        and np.all(np.isfinite(velocities)),
        f"released pcd frame contains invalid values: {path}",
    )
    return arrays


def validate_deform360_released_processed_episode(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one author-released processed-episode manifest."""

    _require(
        payload.get("schema_version")
        == DEFORM360_RELEASED_PROCESSED_EPISODE_SCHEMA_VERSION,
        "unsupported released processed-episode schema",
    )
    _require(
        payload.get("artifact_kind") == DEFORM360_RELEASED_PROCESSED_EPISODE_KIND,
        "unexpected released processed-episode kind",
    )
    _require(
        payload.get("result_sha256") == _canonical_sha256(payload),
        "released processed-episode checksum mismatch",
    )
    _require(
        isinstance(payload.get("object_id"), str) and payload["object_id"],
        "released processed-episode object_id is missing",
    )
    _require(
        isinstance(payload.get("episode_id"), int) and payload["episode_id"] >= 0,
        "released processed-episode episode_id is invalid",
    )
    dataset = payload.get("dataset")
    _require(
        isinstance(dataset, Mapping)
        and dataset.get("repository") == "brownu/deform360"
        and _valid_git_sha1(dataset.get("revision"))
        and dataset.get("coordinate_unit") == "m",
        "released processed-episode dataset provenance changed",
    )
    split = payload.get("split")
    _require(isinstance(split, Mapping), "released episode split is missing")
    train = split.get("train_source_frames")
    test = split.get("test_source_frames")
    _require(
        isinstance(train, list)
        and len(train) == 2
        and all(isinstance(value, int) for value in train)
        and isinstance(test, list)
        and len(test) == 2
        and all(isinstance(value, int) for value in test)
        and 0 <= train[0] < train[1] == test[0] < test[1],
        "released episode split is invalid",
    )
    evaluation = payload.get("evaluation")
    _require(isinstance(evaluation, Mapping), "released evaluation window is missing")
    source_indices = evaluation.get("source_frame_indices")
    local_indices = evaluation.get("local_frame_indices")
    _require(
        isinstance(source_indices, list)
        and source_indices == list(range(test[0], test[1]))
        and local_indices == list(range(len(source_indices)))
        and evaluation.get("prediction_origin_source_frame") == test[0] - 1,
        "released evaluation frame mapping changed",
    )
    particles = payload.get("particles")
    _require(isinstance(particles, Mapping), "released particle contract is missing")
    trajectory_range = particles.get("released_trajectory_source_frames")
    _require(
        isinstance(particles.get("point_count"), int)
        and particles["point_count"] > 0
        and _valid_sha256(particles.get("identity_sha256"))
        and _valid_sha256(particles.get("target_sequence_sha256")),
        "released particle identity is invalid",
    )
    _require(
        isinstance(trajectory_range, list)
        and len(trajectory_range) == 2
        and all(isinstance(value, int) for value in trajectory_range)
        and trajectory_range[0] == 0
        and trajectory_range[1] >= test[1],
        "released particle trajectory range is invalid",
    )
    frame_hashes = particles.get("pcd_file_sha256_by_source_frame")
    required_frames = set(range(trajectory_range[0], trajectory_range[1]))
    _require(
        isinstance(frame_hashes, Mapping)
        and {int(key) for key in frame_hashes} == required_frames
        and all(_valid_sha256(value) for value in frame_hashes.values()),
        "released pcd frame checksum coverage changed",
    )
    advection = particles.get("ordered_advection_check")
    _require(
        isinstance(advection, Mapping)
        and advection.get("passed") is True
        and float(advection.get("maximum_absolute_residual_m", -1.0)) >= 0.0
        and float(advection.get("tolerance_m", -1.0))
        >= float(advection["maximum_absolute_residual_m"]),
        "released ordered-advection check failed",
    )
    inputs = payload.get("input_sha256")
    _require(
        isinstance(inputs, Mapping)
        and _valid_sha256(inputs.get("metadata_json"))
        and _valid_sha256(inputs.get("split_json")),
        "released episode input checksums are invalid",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("development_only") is True
        and boundary.get("public_processed_outcomes_read") is True
        and boundary.get("prediction_metric_computed") is False
        and boundary.get("confirmatory_object_opened") is False
        and boundary.get("held_v8_accessed") is False,
        "released episode information boundary changed",
    )
    return {
        "passed": True,
        "object_id": payload["object_id"],
        "episode_id": payload["episode_id"],
        "point_count": particles["point_count"],
        "test_frame_count": len(source_indices),
        "result_sha256": payload["result_sha256"],
    }


def inspect_deform360_released_processed_episode(
    episode_dir: str | Path,
    *,
    object_id: str,
    episode_id: int,
    dataset_revision: str,
    advection_tolerance_m: float = 5.0e-7,
) -> dict[str, Any]:
    """Bind the author-released split and ordered ``pcd_clean`` trajectory."""

    _require(isinstance(object_id, str) and object_id, "object_id is invalid")
    _require(isinstance(episode_id, int) and episode_id >= 0, "episode_id is invalid")
    _require(_valid_git_sha1(dataset_revision), "dataset revision is invalid")
    _require(
        np.isfinite(advection_tolerance_m) and advection_tolerance_m >= 0.0,
        "advection tolerance is invalid",
    )
    root = Path(episode_dir)
    metadata_path = root / "metadata.json"
    split_path = root / "split.json"
    _require(metadata_path.is_file(), "released metadata.json is missing")
    _require(split_path.is_file(), "released split.json is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_raw = json.loads(split_path.read_text(encoding="utf-8"))
    _require(
        isinstance(metadata, Mapping) and isinstance(split_raw, Mapping),
        "released metadata or split is not an object",
    )
    frame_rate_hz = float(metadata.get("fps", -1.0))
    start_frame = int(metadata.get("start_frame", -1))
    end_frame = int(metadata.get("end_frame", -1))
    frame_count = int(metadata.get("frame_num", -1))
    cameras = metadata.get("cameras")
    train = split_raw.get("train")
    test = split_raw.get("test")
    _require(
        frame_rate_hz > 0.0
        and isinstance(cameras, list)
        and cameras
        and len(set(cameras)) == len(cameras),
        "released episode timing or camera metadata is invalid",
    )
    _require(
        isinstance(train, list)
        and len(train) == 2
        and all(isinstance(value, int) for value in train)
        and isinstance(test, list)
        and len(test) == 2
        and all(isinstance(value, int) for value in test),
        "released split ranges are invalid",
    )
    _require(
        start_frame == train[0]
        and train[1] == test[0]
        and end_frame + 1 == test[1]
        and frame_count == end_frame - start_frame + 1
        and int(split_raw.get("frame_len", -1)) == frame_count
        and train[1] == start_frame + int(0.8 * frame_count),
        "released metadata and 80/20 split disagree",
    )
    source_indices = list(range(test[0], test[1]))
    prediction_origin = test[0] - 1
    _require(prediction_origin >= 0 and source_indices, "released test window is empty")
    pcd_dir = root / "pcd_clean"
    pcd_paths = sorted(pcd_dir.glob("*.npz"))
    _require(pcd_paths, "released pcd_clean trajectory is empty")
    _require(
        all(
            path.stem.isdigit()
            and len(path.stem) == 6
            and path.name == f"{int(path.stem):06d}.npz"
            for path in pcd_paths
        ),
        "released pcd frame naming changed",
    )
    released_source_indices = [int(path.stem) for path in pcd_paths]
    _require(
        released_source_indices == list(range(len(released_source_indices)))
        and source_indices[-1] < len(released_source_indices),
        "released pcd trajectory is not contiguous over the evaluation window",
    )
    frame_hashes: dict[str, str] = {}
    identity_sha256: str | None = None
    point_count: int | None = None
    previous: dict[str, np.ndarray] | None = None
    maximum_advection_residual = 0.0
    for frame_index, path in zip(released_source_indices, pcd_paths, strict=True):
        current = _load_released_pcd_frame(
            path,
            expected_camera_count=len(cameras),
        )
        frame_hashes[str(frame_index)] = _sha256_file(path)
        if point_count is None:
            point_count = len(current["pts"])
            identity_sha256 = _ordered_identity_sha256(current["pts"])
        _require(
            len(current["pts"]) == point_count,
            "released material point count changes inside the full trajectory",
        )
        if previous is not None:
            predicted = previous["pts"] + previous["vels"] / frame_rate_hz
            residual = float(np.max(np.abs(current["pts"] - predicted)))
            maximum_advection_residual = max(maximum_advection_residual, residual)
        previous = current
    assert identity_sha256 is not None and point_count is not None
    _require(
        maximum_advection_residual <= advection_tolerance_m,
        "released points do not preserve ordered velocity advection",
    )
    target_sequence_sha256 = hashlib.sha256(
        json.dumps(
            [
                [frame_index, frame_hashes[str(frame_index)]]
                for frame_index in source_indices
            ],
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_RELEASED_PROCESSED_EPISODE_SCHEMA_VERSION,
        "artifact_kind": DEFORM360_RELEASED_PROCESSED_EPISODE_KIND,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "dataset": {
            "repository": "brownu/deform360",
            "revision": dataset_revision,
            "coordinate_unit": "m",
            "episode_relative_path": f"processed/{object_id}/episode_{episode_id}",
        },
        "metadata": {
            "frame_rate_hz": frame_rate_hz,
            "camera_count": len(cameras),
            "frame_count": frame_count,
            "start_source_frame": start_frame,
            "end_source_frame_inclusive": end_frame,
        },
        "split": {
            "frame_count": frame_count,
            "train_source_frames": list(train),
            "test_source_frames": list(test),
            "train_fraction_rule": 0.8,
        },
        "evaluation": {
            "prediction_origin_source_frame": prediction_origin,
            "source_frame_indices": source_indices,
            "local_frame_indices": list(range(len(source_indices))),
        },
        "particles": {
            "identity_policy": (
                "ordered_frame_zero_seed_advected_by_author_released_velocities"
            ),
            "point_count": point_count,
            "identity_sha256": identity_sha256,
            "target_sequence_sha256": target_sequence_sha256,
            "released_trajectory_source_frames": [
                released_source_indices[0],
                released_source_indices[-1] + 1,
            ],
            "pcd_file_sha256_by_source_frame": frame_hashes,
            "ordered_advection_check": {
                "passed": True,
                "frame_rate_hz": frame_rate_hz,
                "first_transition_source_frame": released_source_indices[0],
                "last_transition_source_frame": released_source_indices[-1],
                "maximum_absolute_residual_m": maximum_advection_residual,
                "tolerance_m": float(advection_tolerance_m),
            },
        },
        "input_sha256": {
            "metadata_json": _sha256_file(metadata_path),
            "split_json": _sha256_file(split_path),
        },
        "information_boundary": {
            "development_only": True,
            "public_processed_outcomes_read": True,
            "prediction_metric_computed": False,
            "confirmatory_object_opened": False,
            "held_v8_accessed": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    validate_deform360_released_processed_episode(payload)
    return payload


def load_deform360_released_processed_evaluation(
    episode_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Load test targets and an exact-persistence prediction from a sealed manifest."""

    validate_deform360_released_processed_episode(manifest)
    root = Path(episode_dir)
    _require(
        _sha256_file(root / "metadata.json")
        == manifest["input_sha256"]["metadata_json"]
        and _sha256_file(root / "split.json") == manifest["input_sha256"]["split_json"],
        "released episode metadata changed after inspection",
    )
    frame_hashes = manifest["particles"]["pcd_file_sha256_by_source_frame"]
    source_indices = list(manifest["evaluation"]["source_frame_indices"])
    prediction_origin = int(manifest["evaluation"]["prediction_origin_source_frame"])
    required = sorted({0, prediction_origin, *source_indices})
    frames: dict[int, dict[str, np.ndarray]] = {}
    for frame_index in required:
        path = root / "pcd_clean" / f"{frame_index:06d}.npz"
        _require(
            _sha256_file(path) == frame_hashes[str(frame_index)],
            f"released pcd frame changed after inspection: {frame_index}",
        )
        frames[frame_index] = _load_released_pcd_frame(
            path,
            expected_camera_count=int(manifest["metadata"]["camera_count"]),
        )
    _require(
        _ordered_identity_sha256(frames[0]["pts"])
        == manifest["particles"]["identity_sha256"],
        "released material identity changed after inspection",
    )
    target = np.stack([frames[index]["pts"] for index in source_indices], axis=0)
    origin = frames[prediction_origin]["pts"]
    persistence = np.repeat(origin[None, :, :], len(source_indices), axis=0)
    visibility = np.stack(
        [
            np.any(frames[index]["visibility_matrix"].astype(bool), axis=1)
            for index in source_indices
        ],
        axis=0,
    )
    return {
        "target_m": target,
        "persistence_m": persistence,
        "visibility": visibility,
        "source_frame_indices": source_indices,
        "prediction_origin_source_frame": prediction_origin,
        "particle_identity_sha256": manifest["particles"]["identity_sha256"],
    }


def build_released_processed_evaluator_contract(
    episode_manifests: Sequence[Mapping[str, Any]],
    *,
    chamfer_definition: str = "symmetric_mean_euclidean_m",
    track_definition: str = "mean_euclidean_m",
    panel_aggregation: str = "object_balanced_mean",
) -> dict[str, Any]:
    """Build a non-authorizing evaluator from author-released processed episodes."""

    records = [dict(record) for record in episode_manifests]
    _require(records, "released processed evaluator requires episode manifests")
    _require(
        chamfer_definition in _CHAMFER_DEFINITIONS,
        "released evaluator Chamfer definition is invalid",
    )
    _require(
        track_definition in _TRACK_DEFINITIONS,
        "released evaluator track definition is invalid",
    )
    _require(
        panel_aggregation in {"object_balanced_mean", "episode_balanced_mean"},
        "released evaluator panel aggregation is invalid",
    )
    for record in records:
        validate_deform360_released_processed_episode(record)
    repository = records[0]["dataset"]["repository"]
    revision = records[0]["dataset"]["revision"]
    _require(
        all(
            record["dataset"]["repository"] == repository
            and record["dataset"]["revision"] == revision
            for record in records
        ),
        "released processed episodes use different dataset revisions",
    )
    by_object: dict[str, list[int]] = {}
    identities: dict[str, str] = {}
    manifest_sha256: dict[str, str] = {}
    local_indices: dict[str, list[int]] = {}
    source_indices: dict[str, list[int]] = {}
    prediction_origins: dict[str, int] = {}
    for record in records:
        object_id = str(record["object_id"])
        episode_id = int(record["episode_id"])
        key = _episode_key(object_id, episode_id)
        _require(key not in identities, "released processed episode is duplicated")
        by_object.setdefault(object_id, []).append(episode_id)
        identities[key] = str(record["particles"]["identity_sha256"])
        manifest_sha256[key] = str(record["result_sha256"])
        local_indices[key] = list(record["evaluation"]["local_frame_indices"])
        source_indices[key] = list(record["evaluation"]["source_frame_indices"])
        prediction_origins[key] = int(
            record["evaluation"]["prediction_origin_source_frame"]
        )
    object_ids = sorted(by_object)
    held = {object_id: sorted(by_object[object_id]) for object_id in object_ids}
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_EVALUATOR_CONTRACT_SCHEMA_VERSION,
        "artifact_kind": DEFORM360_EVALUATOR_CONTRACT_KIND,
        "contract_id": "deform360-author-released-processed-independent-v1",
        "status": "independent-protocol",
        "dataset": {
            "repository": repository,
            "revision": revision,
            "coordinate_unit": "m",
        },
        "annotation_pipeline": {
            "producer": "author-released processed pcd_clean artifacts",
            "author_released_processed_artifacts": True,
            "world_model_evaluator_released": False,
        },
        "split": {
            "object_ids": object_ids,
            "fit_episode_ids_by_object": {object_id: [] for object_id in object_ids},
            "held_episode_ids_by_object": held,
        },
        "input_boundary": {
            "prediction_origin": "last author-released train frame",
            "known_future_robot_action": True,
            "future_object_observation": False,
        },
        "temporal": {
            "source_frame_rate_hz": 30.0,
            "array_indexing": "evaluation_sequence_local_indices",
            "evaluation_frame_indices_by_episode": local_indices,
            "source_frame_indices_by_episode": source_indices,
            "prediction_origin_source_frame_by_episode": prediction_origins,
        },
        "particles": {
            "identity_policy": (
                "ordered_frame_zero_seed_advected_by_author_released_velocities"
            ),
            "identity_sha256_by_episode": identities,
            "released_processed_manifest_sha256_by_episode": manifest_sha256,
        },
        "metrics": {
            "chamfer": {
                "definition": chamfer_definition,
                "visibility_policy": "all_finite_material_points",
            },
            "track": {
                "definition": track_definition,
                "visibility_policy": "all_finite_material_points",
            },
        },
        "aggregation": {
            "frame": "mean",
            "episode": "mean",
            "object": "mean",
            "panel": panel_aggregation,
        },
        "published_reference": {
            "paper": "Deform360 arXiv:2607.05390v1",
            "table": 4,
            "setting": "multi-episode future prediction",
            "method": "ParticleFormer",
            "future_chamfer_m": 0.051,
            "future_track_error_m": 0.079,
        },
        "evaluator_provenance": {
            "released_by_deform360_authors": False,
            "source_revision_sha256": None,
            "entrypoint_sha256": None,
            "particleformer_table4_reproduction": {
                "passed": False,
                "future_chamfer_m": None,
                "future_track_error_m": None,
            },
        },
        "information_boundary": {
            "development_only": True,
            "public_processed_outcomes_read": True,
            "confirmatory_object_opened": False,
            "held_v8_accessed": False,
        },
        "unresolved_fields": [
            "official_parity.complete_object_and_episode_split",
            "official_parity.metric_definitions",
            "official_parity.panel_aggregation",
            "official_parity.evaluator",
            "official_parity.particleformer_table4_reproduction",
        ],
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    validate_deform360_evaluator_contract(payload)
    return payload


def build_development_evaluator_contract(
    protocol: Mapping[str, Any],
    observation_manifests: Sequence[Mapping[str, Any]],
    *,
    evaluation_start_frame: int,
    evaluation_stop_frame_exclusive: int,
    frame_stride: int = 1,
) -> dict[str, Any]:
    """Build an explicit non-authorizing contract for held development episodes."""

    validate_reusable_sota_config(protocol)
    _require(
        0 <= evaluation_start_frame < evaluation_stop_frame_exclusive
        and frame_stride >= 1,
        "independent evaluation horizon is invalid",
    )
    records = [dict(record) for record in observation_manifests]
    _require(records, "development evaluator requires observation manifests")
    development = protocol["config"]["development_objects"]
    development_ids = {
        str(object_id)
        for category in ("1d", "2d", "3d")
        for object_id in development[category]
    }
    protocol_id = str(protocol["config"]["protocol_id"])
    protocol_sha256 = str(protocol["config_sha256"])
    by_object: dict[str, list[int]] = {}
    identities: dict[str, str] = {}
    manifest_checksums: dict[str, str] = {}
    for record in records:
        _require(
            record.get("artifact_kind") == DEVELOPMENT_OBSERVATIONS_KIND
            and record.get("result_sha256") == _canonical_sha256(record),
            "development observation manifest is invalid",
        )
        authorization = record.get("authorization", {})
        implementation = record.get("implementation_revision", {})
        inputs = record.get("input_sha256", {})
        boundary = record.get("information_boundary", {})
        object_id = str(record.get("object_id"))
        episode_id = int(record.get("episode_id", -1))
        _require(
            object_id in development_ids
            and authorization.get("protocol_id") == protocol_id
            and authorization.get("protocol_config_sha256") == protocol_sha256
            and authorization.get("object_id") == object_id
            and int(authorization.get("episode_id", -1)) == episode_id
            and authorization.get("role") == "held-development"
            and authorization.get("development_only") is True
            and authorization.get("confirmatory_object_opened") is False
            and record.get("role") == "held-development"
            and episode_id in EXPECTED_HELD_EPISODES,
            "evaluator input is not a locked held-development episode",
        )
        _require(
            implementation.get("deform360_processing")
            == PINNED_DEFORM360_PROCESSING_REVISION
            and implementation.get("cotracker") == PINNED_COTRACKER_REVISION
            and inputs.get("cotracker_checkpoint")
            == PINNED_COTRACKER_CHECKPOINT_SHA256,
            "development observation implementation changed",
        )
        _require(
            boundary.get("development_only") is True
            and boundary.get("prediction_metric_computed") is False
            and boundary.get("confirmatory_object_opened") is False
            and boundary.get("pokeflex_target_opened") is False,
            "development observation information boundary changed",
        )
        _require(
            int(record.get("point_frame_count", -1)) >= evaluation_stop_frame_exclusive,
            f"evaluation horizon exceeds processed points for {object_id}/{episode_id}",
        )
        identity = record.get("material_identity_sha256")
        _require(_valid_sha256(identity), "material identity checksum is invalid")
        key = _episode_key(object_id, episode_id)
        _require(key not in identities, "development evaluator episode is duplicated")
        by_object.setdefault(object_id, []).append(episode_id)
        identities[key] = str(identity)
        manifest_checksums[key] = str(record["result_sha256"])

    object_ids = sorted(by_object)
    held = {object_id: sorted(by_object[object_id]) for object_id in object_ids}
    fit = {object_id: list(EXPECTED_FIT_EPISODES) for object_id in object_ids}
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_EVALUATOR_CONTRACT_SCHEMA_VERSION,
        "artifact_kind": DEFORM360_EVALUATOR_CONTRACT_KIND,
        "contract_id": "deform360-reusable-sota-development-independent-v1",
        "status": "independent-protocol",
        "dataset": {
            "repository": "brownu/deform360",
            "revision": protocol["config"]["dataset"]["revision"],
            "coordinate_unit": "m",
        },
        "annotation_pipeline": {
            "repository": "lhy0807/deform360",
            "revision": "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317",
            "producer": "public SAM2 development fallback plus released processing",
            "official_sam3_parity": False,
        },
        "split": {
            "object_ids": object_ids,
            "fit_episode_ids_by_object": fit,
            "held_episode_ids_by_object": held,
        },
        "input_boundary": {
            "initial_object_frame_count": 1,
            "known_future_robot_action": True,
            "future_object_observation": False,
        },
        "temporal": {
            "source_frame_rate_hz": 30.0,
            "evaluation_start_frame": int(evaluation_start_frame),
            "evaluation_stop_frame_exclusive": int(evaluation_stop_frame_exclusive),
            "frame_stride": int(frame_stride),
        },
        "particles": {
            "identity_policy": "ordered_frame_zero_seed_advected_by_released_pipeline",
            "identity_sha256_by_episode": identities,
            "observation_manifest_sha256_by_episode": manifest_checksums,
        },
        "metrics": {
            "chamfer": {
                "definition": "symmetric_mean_euclidean_m",
                "visibility_policy": "all_finite_material_points",
            },
            "track": {
                "definition": "mean_euclidean_m",
                "visibility_policy": "all_finite_material_points",
            },
        },
        "aggregation": {
            "frame": "mean",
            "episode": "mean",
            "object": "mean",
            "panel": "object_balanced_mean",
        },
        "published_reference": {
            "paper": "Deform360 arXiv:2607.05390v1",
            "table": 4,
            "setting": "multi-episode future prediction",
            "method": "ParticleFormer",
            "future_chamfer_m": 0.051,
            "future_track_error_m": 0.079,
        },
        "evaluator_provenance": {
            "released_by_deform360_authors": False,
            "source_revision_sha256": None,
            "entrypoint_sha256": None,
            "particleformer_table4_reproduction": {
                "passed": False,
                "future_chamfer_m": None,
                "future_track_error_m": None,
            },
        },
        "information_boundary": {
            "development_only": True,
            "confirmatory_object_opened": False,
            "pokeflex_target_opened": False,
        },
        "unresolved_fields": [
            "official_parity.split",
            "official_parity.temporal",
            "official_parity.metrics",
            "official_parity.evaluator",
            "official_parity.particleformer_table4_reproduction",
        ],
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    validate_deform360_evaluator_contract(payload)
    return payload


def _score_ready_contract(payload: Mapping[str, Any], episode_key: str) -> None:
    validate_deform360_evaluator_contract(payload)
    temporal = payload["temporal"]
    per_episode = temporal.get("evaluation_frame_indices_by_episode")
    _require(
        (
            all(
                isinstance(temporal.get(field), int)
                for field in (
                    "evaluation_start_frame",
                    "evaluation_stop_frame_exclusive",
                    "frame_stride",
                )
            )
            or (
                isinstance(per_episode, Mapping)
                and isinstance(per_episode.get(episode_key), list)
                and bool(per_episode[episode_key])
            )
        ),
        "temporal evaluator fields are unresolved",
    )
    metrics = payload["metrics"]
    _require(
        metrics.get("chamfer", {}).get("definition") in _CHAMFER_DEFINITIONS,
        "Chamfer definition is unresolved",
    )
    _require(
        metrics.get("chamfer", {}).get("visibility_policy") in _VISIBILITY_POLICIES,
        "Chamfer visibility policy is unresolved",
    )
    _require(
        metrics.get("track", {}).get("definition") in _TRACK_DEFINITIONS,
        "track definition is unresolved",
    )
    _require(
        metrics.get("track", {}).get("visibility_policy") in _VISIBILITY_POLICIES,
        "track visibility policy is unresolved",
    )
    identities = payload["particles"].get("identity_sha256_by_episode")
    _require(
        isinstance(identities, Mapping) and episode_key in identities,
        f"particle identity is unresolved for {episode_key}",
    )


def _evaluation_frame_indices(
    payload: Mapping[str, Any],
    episode_key: str,
) -> np.ndarray:
    temporal = payload["temporal"]
    per_episode = temporal.get("evaluation_frame_indices_by_episode")
    if isinstance(per_episode, Mapping) and episode_key in per_episode:
        indices = np.asarray(per_episode[episode_key], dtype=np.int64)
    else:
        indices = np.arange(
            temporal["evaluation_start_frame"],
            temporal["evaluation_stop_frame_exclusive"],
            temporal["frame_stride"],
            dtype=np.int64,
        )
    _require(
        len(indices) > 0 and np.all(indices >= 0) and np.all(np.diff(indices) > 0),
        f"evaluation frame indices are invalid for {episode_key}",
    )
    return indices


def _evaluated_source_frame_indices(
    payload: Mapping[str, Any],
    episode_key: str,
    local_indices: np.ndarray,
) -> list[int]:
    mapping = payload["temporal"].get("source_frame_indices_by_episode")
    if not isinstance(mapping, Mapping) or episode_key not in mapping:
        return local_indices.tolist()
    source = mapping[episode_key]
    _require(
        isinstance(source, list)
        and len(source) == len(local_indices)
        and all(isinstance(value, int) and value >= 0 for value in source),
        f"source frame mapping is invalid for {episode_key}",
    )
    return list(source)


def _chamfer(target: np.ndarray, prediction: np.ndarray, definition: str) -> float:
    def directed_minimum_squared(
        source: np.ndarray, target_set: np.ndarray
    ) -> np.ndarray:
        target_squared = np.sum(target_set * target_set, axis=1)
        minima: list[np.ndarray] = []
        for start in range(0, len(source), 1024):
            chunk = source[start : start + 1024]
            squared = (
                np.sum(chunk * chunk, axis=1)[:, None]
                + target_squared[None, :]
                - 2.0 * chunk @ target_set.T
            )
            minima.append(np.maximum(np.min(squared, axis=1), 0.0))
        return np.concatenate(minima)

    target_to_prediction = directed_minimum_squared(target, prediction)
    prediction_to_target = directed_minimum_squared(prediction, target)
    if definition == "symmetric_mean_euclidean_m":
        target_to_prediction = np.sqrt(target_to_prediction)
        prediction_to_target = np.sqrt(prediction_to_target)
    elif definition != "symmetric_mean_squared_euclidean_m2":
        raise ValueError(f"unsupported Chamfer definition: {definition}")
    return 0.5 * (
        float(np.mean(target_to_prediction)) + float(np.mean(prediction_to_target))
    )


def _track(displacement: np.ndarray, definition: str) -> float:
    squared = np.sum(displacement * displacement, axis=1)
    if definition == "mean_euclidean_m":
        return float(np.mean(np.sqrt(squared)))
    if definition == "root_mean_squared_euclidean_m":
        return float(np.sqrt(np.mean(squared)))
    if definition == "mean_squared_euclidean_m2":
        return float(np.mean(squared))
    raise ValueError(f"unsupported track definition: {definition}")


def score_deform360_episode(
    contract: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
    particle_identity_sha256: str,
    target_m: np.ndarray,
    prediction_m: np.ndarray,
    visibility: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score one episode only under the explicitly named metric contract."""

    episode_key = _episode_key(object_id, episode_id)
    _score_ready_contract(contract, episode_key)
    expected_identity = contract["particles"]["identity_sha256_by_episode"][episode_key]
    _require(
        particle_identity_sha256 == expected_identity,
        "particle identity differs from the evaluator contract",
    )
    target = np.asarray(target_m, dtype=np.float64)
    prediction = np.asarray(prediction_m, dtype=np.float64)
    _require(
        target.shape == prediction.shape and target.ndim == 3 and target.shape[2] == 3,
        "target and prediction must share shape (T,N,3)",
    )
    indices = _evaluation_frame_indices(contract, episode_key)
    _require(
        len(indices) and int(indices[-1]) < len(target),
        "evaluation horizon exceeds data",
    )
    source_indices = _evaluated_source_frame_indices(
        contract,
        episode_key,
        indices,
    )
    chamfer_visibility_policy = contract["metrics"]["chamfer"]["visibility_policy"]
    track_visibility_policy = contract["metrics"]["track"]["visibility_policy"]
    if "visible_and_finite_material_points" in {
        chamfer_visibility_policy,
        track_visibility_policy,
    }:
        _require(visibility is not None, "visibility mask is required by the contract")
        visible = np.asarray(visibility, dtype=bool)
        _require(visible.shape == target.shape[:2], "visibility shape differs")
    else:
        visible = np.ones(target.shape[:2], dtype=bool)
    chamfer_definition = contract["metrics"]["chamfer"]["definition"]
    track_definition = contract["metrics"]["track"]["definition"]
    frame_chamfer: list[float] = []
    frame_track: list[float] = []
    frame_chamfer_counts: list[int] = []
    frame_track_counts: list[int] = []
    for frame_index in indices:
        finite = np.all(np.isfinite(target[frame_index]), axis=1) & np.all(
            np.isfinite(prediction[frame_index]), axis=1
        )
        chamfer_selected = finite.copy()
        if chamfer_visibility_policy == "visible_and_finite_material_points":
            chamfer_selected &= visible[frame_index]
        track_selected = finite.copy()
        if track_visibility_policy == "visible_and_finite_material_points":
            track_selected &= visible[frame_index]
        _require(
            np.any(chamfer_selected),
            f"no valid Chamfer particles at frame {frame_index}",
        )
        _require(
            np.any(track_selected),
            f"no valid track particles at frame {frame_index}",
        )
        frame_chamfer.append(
            _chamfer(
                target[frame_index, chamfer_selected],
                prediction[frame_index, chamfer_selected],
                chamfer_definition,
            )
        )
        frame_track.append(
            _track(
                prediction[frame_index, track_selected]
                - target[frame_index, track_selected],
                track_definition,
            )
        )
        frame_chamfer_counts.append(int(np.count_nonzero(chamfer_selected)))
        frame_track_counts.append(int(np.count_nonzero(track_selected)))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": DEFORM360_EPISODE_SCORE_KIND,
        "contract_result_sha256": contract["result_sha256"],
        "object_id": object_id,
        "episode_id": int(episode_id),
        "episode_key": episode_key,
        "particle_identity_sha256": particle_identity_sha256,
        "evaluated_frame_indices": indices.tolist(),
        "evaluated_source_frame_indices": source_indices,
        "valid_chamfer_particle_count_by_frame": frame_chamfer_counts,
        "valid_track_particle_count_by_frame": frame_track_counts,
        "metrics": {
            "future_chamfer": float(np.mean(frame_chamfer)),
            "future_track_error": float(np.mean(frame_track)),
            "chamfer_definition": chamfer_definition,
            "chamfer_visibility_policy": chamfer_visibility_policy,
            "track_definition": track_definition,
            "track_visibility_policy": track_visibility_policy,
            "per_frame_chamfer": frame_chamfer,
            "per_frame_track_error": frame_track,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def score_deform360_released_processed_persistence(
    contract: Mapping[str, Any],
    *,
    episode_dir: str | Path,
    episode_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Score exact endpoint persistence on one released processed episode."""

    validate_deform360_released_processed_episode(episode_manifest)
    object_id = str(episode_manifest["object_id"])
    episode_id = int(episode_manifest["episode_id"])
    episode_key = _episode_key(object_id, episode_id)
    expected_manifests = contract["particles"].get(
        "released_processed_manifest_sha256_by_episode"
    )
    _require(
        isinstance(expected_manifests, Mapping)
        and expected_manifests.get(episode_key) == episode_manifest["result_sha256"],
        "released processed manifest differs from the evaluator contract",
    )
    arrays = load_deform360_released_processed_evaluation(
        episode_dir,
        episode_manifest,
    )
    result = score_deform360_episode(
        contract,
        object_id=object_id,
        episode_id=episode_id,
        particle_identity_sha256=arrays["particle_identity_sha256"],
        target_m=arrays["target_m"],
        prediction_m=arrays["persistence_m"],
        visibility=arrays["visibility"],
    )
    result["prediction_kind"] = "exact_last_train_frame_persistence"
    result["prediction_origin_source_frame"] = arrays["prediction_origin_source_frame"]
    result["released_processed_manifest_sha256"] = episode_manifest["result_sha256"]
    result["result_sha256"] = _canonical_sha256(result)
    return result


def aggregate_deform360_panel(
    contract: Mapping[str, Any], episode_scores: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Aggregate episodes without silently changing the declared replicate unit."""

    validate_deform360_evaluator_contract(contract)
    rows = tuple(episode_scores)
    _require(rows, "panel score requires at least one episode")
    observed_keys = {str(row.get("episode_key")) for row in rows}
    _require(len(observed_keys) == len(rows), "panel contains duplicate episodes")
    for row in rows:
        _require(
            row.get("artifact_kind") == DEFORM360_EPISODE_SCORE_KIND
            and row.get("result_sha256") == _canonical_sha256(row)
            and row.get("contract_result_sha256") == contract["result_sha256"],
            "panel contains an invalid or mismatched episode score",
        )
    held = contract["split"].get("held_episode_ids_by_object")
    if held:
        expected = {
            _episode_key(object_id, episode_id)
            for object_id, episode_ids in held.items()
            for episode_id in episode_ids
        }
        _require(
            observed_keys == expected, "panel differs from the declared held split"
        )
    by_object: dict[str, dict[str, float | int]] = {}
    for object_id in sorted({str(row["object_id"]) for row in rows}):
        selected = [row for row in rows if row["object_id"] == object_id]
        by_object[object_id] = {
            "episode_count": len(selected),
            "future_chamfer": float(
                np.mean([row["metrics"]["future_chamfer"] for row in selected])
            ),
            "future_track_error": float(
                np.mean([row["metrics"]["future_track_error"] for row in selected])
            ),
        }
    panel_rule = contract["aggregation"].get("panel")
    if panel_rule == "object_balanced_mean":
        chamfer = float(np.mean([row["future_chamfer"] for row in by_object.values()]))
        track = float(
            np.mean([row["future_track_error"] for row in by_object.values()])
        )
    elif panel_rule == "episode_balanced_mean":
        chamfer = float(np.mean([row["metrics"]["future_chamfer"] for row in rows]))
        track = float(np.mean([row["metrics"]["future_track_error"] for row in rows]))
    else:
        raise ValueError("panel aggregation is unresolved")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": DEFORM360_PANEL_SCORE_KIND,
        "contract_result_sha256": contract["result_sha256"],
        "episode_count": len(rows),
        "object_count": len(by_object),
        "aggregation": panel_rule,
        "metrics": {
            "future_chamfer": chamfer,
            "future_track_error": track,
            "by_object": by_object,
        },
        "episode_result_sha256": {
            str(row["episode_key"]): str(row["result_sha256"])
            for row in sorted(rows, key=lambda value: str(value["episode_key"]))
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def authorize_deform360_table4_claim(
    contract: Mapping[str, Any], panel_score: Mapping[str, Any]
) -> dict[str, Any]:
    """Authorize a direct Table 4 claim only after exact evaluator parity."""

    validation = validate_deform360_evaluator_contract(contract)
    _require(
        validation["official_table4_authorizing"],
        "direct Table 4 comparison refused: evaluator parity is not established",
    )
    _require(
        panel_score.get("artifact_kind") == DEFORM360_PANEL_SCORE_KIND
        and panel_score.get("result_sha256") == _canonical_sha256(panel_score)
        and panel_score.get("contract_result_sha256") == contract["result_sha256"],
        "panel score does not belong to the official evaluator contract",
    )
    reference = contract["published_reference"]
    chamfer = float(panel_score["metrics"]["future_chamfer"])
    track = float(panel_score["metrics"]["future_track_error"])
    gates = {
        "future_chamfer_below_particleformer": chamfer
        < float(reference["future_chamfer_m"]),
        "future_track_below_particleformer": track
        < float(reference["future_track_error_m"]),
    }
    return {
        "authorized": all(gates.values()),
        "protocol_parity_established": True,
        "gates": gates,
        "candidate": {
            "future_chamfer": chamfer,
            "future_track_error": track,
        },
        "reference": {
            "future_chamfer_m": float(reference["future_chamfer_m"]),
            "future_track_error_m": float(reference["future_track_error_m"]),
        },
        "contract_result_sha256": contract["result_sha256"],
        "panel_result_sha256": panel_score["result_sha256"],
    }


__all__ = [
    "DEFORM360_EVALUATOR_CONTRACT_KIND",
    "DEFORM360_EVALUATOR_CONTRACT_SCHEMA_VERSION",
    "DEFORM360_RELEASED_PROCESSED_EPISODE_KIND",
    "DEFORM360_RELEASED_PROCESSED_EPISODE_SCHEMA_VERSION",
    "aggregate_deform360_panel",
    "authorize_deform360_table4_claim",
    "build_development_evaluator_contract",
    "build_released_processed_evaluator_contract",
    "deform360_evaluator_contract_sha256",
    "inspect_deform360_released_processed_episode",
    "load_deform360_released_processed_evaluation",
    "load_deform360_evaluator_contract",
    "score_deform360_episode",
    "score_deform360_released_processed_persistence",
    "validate_deform360_evaluator_contract",
    "validate_deform360_released_processed_episode",
    "write_deform360_evaluator_contract",
]
