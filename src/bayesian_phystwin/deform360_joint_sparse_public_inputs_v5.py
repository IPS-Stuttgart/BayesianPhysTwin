"""Public Deform360 prefix inputs for the joint-sparse v5 predictor.

This module is deliberately prediction-side only.  It aligns a released
decoded-uniform Prob4D prefix to the Deform360 world frame using sparse robot
geometry from the same causal prefix, then emits the residual-independent
visual rows consumed by the v5 materializer.  It has no endpoint or outcome
interface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from ._portable_contracts import content_id, nonempty_string, source_artifact_mapping
from .deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseContactRowsV5,
    Deform360JointSparseExtractionConfigV5,
    Deform360JointSparsePrefixFitV5,
    Deform360JointSparseVisualWindowRowsV5,
    extract_deform360_joint_sparse_visual_rows_v5,
)
from .deform360_public_contact_prefix import (
    validate_deform360_public_contact_prefix,
)
from .phystwin_motioncrafter_association import (
    MotionCrafterPrediction,
    align_motioncrafter_prediction,
    load_motioncrafter_prediction,
    robust_similarity_transform,
)

METRIC_PREFIX_MEMBERS: Final = frozenset(
    {"frame_indices", "points_world_m", "valid_mask"}
)
METRIC_GAUGE_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-metric-gauge-fit"
)
METRIC_GAUGE_VERSION: Final = 5
CONTACT_LOCALIZATION_FLOOR_M: Final = 0.005
CONTACT_ASSOCIATION_SCALE_M: Final = 0.010
CONTACT_ASSOCIATION_CANDIDATES: Final = 4


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _immutable_float(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind in "iuf", f"{name} must be real")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    _require(result.shape == shape, f"{name} shape changed")
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class Deform360JointSparseMetricGaugeFitV5:
    """One causal local-to-world Sim(3) fit and its complete lineage."""

    camera_id: str
    raw_frame_index: int
    linear: np.ndarray
    translation: np.ndarray
    input_pair_count: int
    inlier_pair_count: int
    independent_cluster_count: int
    inlier_independent_cluster_count: int
    inlier_rmse_m: float
    source_artifact_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        camera = nonempty_string(self.camera_id, name="camera_id")
        _require(
            type(self.raw_frame_index) is int and self.raw_frame_index >= 0,
            "raw_frame_index must be a nonnegative integer",
        )
        linear = _immutable_float(self.linear, name="linear", shape=(3, 3))
        translation = _immutable_float(
            self.translation,
            name="translation",
            shape=(3,),
        )
        singular = np.linalg.svd(linear, compute_uv=False)
        _require(
            np.all(singular > 0.0)
            and np.max(singular) / np.min(singular) <= 1.0 + 1e-6
            and np.linalg.det(linear) > 0.0,
            "linear transform must be a proper scaled rotation",
        )
        for name, minimum in (
            ("input_pair_count", 8),
            ("inlier_pair_count", 4),
            ("independent_cluster_count", 8),
            ("inlier_independent_cluster_count", 8),
        ):
            value = getattr(self, name)
            _require(type(value) is int and value >= minimum, f"invalid {name}")
        _require(
            self.inlier_pair_count <= self.input_pair_count,
            "inlier count exceeds input count",
        )
        _require(
            self.inlier_independent_cluster_count
            <= self.independent_cluster_count,
            "inlier cluster count exceeds input cluster count",
        )
        _require(
            not isinstance(self.inlier_rmse_m, (bool, np.bool_))
            and np.isfinite(self.inlier_rmse_m)
            and self.inlier_rmse_m >= 0.0,
            "inlier_rmse_m must be finite and nonnegative",
        )
        sources = source_artifact_mapping(
            self.source_artifact_ids,
            name="source_artifact_ids",
        )
        object.__setattr__(self, "camera_id", camera)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "source_artifact_ids", sources)

    @property
    def artifact_id(self) -> str:
        return content_id(
            {
                "schema": METRIC_GAUGE_SCHEMA,
                "schema_version": METRIC_GAUGE_VERSION,
                "camera_id": self.camera_id,
                "raw_frame_index": self.raw_frame_index,
                "linear_sha256": _array_sha256(self.linear),
                "translation_sha256": _array_sha256(self.translation),
                "input_pair_count": self.input_pair_count,
                "inlier_pair_count": self.inlier_pair_count,
                "independent_cluster_count": self.independent_cluster_count,
                "inlier_independent_cluster_count": (
                    self.inlier_independent_cluster_count
                ),
                "inlier_rmse_m": self.inlier_rmse_m,
                "source_artifact_ids": dict(self.source_artifact_ids),
                "information_boundary": {
                    "released_prefix_measurements_used": True,
                    "future_object_observations_used": False,
                    "physical_state_residual_used": False,
                    "human_approval_used": False,
                    "new_measurements_collected": False,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class _MetricPrefixV5:
    frame_indices: np.ndarray
    points_world_m: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class Deform360JointSparseContactConfigV5:
    """Frozen conversion from public tactile patches to graph observations."""

    localization_floor_m: float = CONTACT_LOCALIZATION_FLOOR_M
    association_scale_m: float = CONTACT_ASSOCIATION_SCALE_M
    association_candidate_count: int = CONTACT_ASSOCIATION_CANDIDATES

    def __post_init__(self) -> None:
        for name in ("localization_floor_m", "association_scale_m"):
            value = getattr(self, name)
            _require(
                not isinstance(value, (bool, np.bool_))
                and np.isfinite(value)
                and value > 0.0,
                f"{name} must be finite and positive",
            )
            object.__setattr__(self, name, float(value))
        _require(
            type(self.association_candidate_count) is int
            and self.association_candidate_count >= 1,
            "association_candidate_count must be a positive integer",
        )


def _load_metric_prefix(
    path: Path,
    *,
    raw_prefix_range_half_open: tuple[int, int],
    image_shape: tuple[int, int],
) -> _MetricPrefixV5:
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == METRIC_PREFIX_MEMBERS,
                "metric-prefix member roster changed",
            )
            frames = np.asarray(archive["frame_indices"])
            points = np.asarray(archive["points_world_m"], dtype=np.float64)
            valid = np.asarray(archive["valid_mask"])
    except (OSError, ValueError) as error:
        raise ValueError("cannot load metric-prefix archive") from error
    start, stop = raw_prefix_range_half_open
    _require(
        type(start) is int and type(stop) is int and 0 <= start < stop,
        "raw prefix range is invalid",
    )
    expected: np.ndarray = np.arange(start, stop, dtype=np.int64)
    _require(
        frames.dtype.kind in "iu" and np.array_equal(frames, expected),
        "metric prefix does not contain the complete causal range",
    )
    expected_shape = (len(expected), *image_shape)
    _require(
        points.shape == (*expected_shape, 3),
        "metric point grid shape changed",
    )
    _require(
        valid.dtype.kind == "b" and valid.shape == expected_shape,
        "metric valid-mask shape changed",
    )
    _require(np.all(np.isfinite(points[valid])), "valid metric points are non-finite")
    return _MetricPrefixV5(
        frame_indices=np.asarray(frames, dtype=np.int64),
        points_world_m=points,
        valid_mask=np.asarray(valid, dtype=np.bool_),
    )


def _fit_metric_gauge(
    prediction: MotionCrafterPrediction,
    metric: _MetricPrefixV5,
    *,
    camera_id: str,
    raw_prefix_range_half_open: tuple[int, int],
    source_artifact_ids: Mapping[str, str],
    cluster_size_pixels: int,
) -> Deform360JointSparseMetricGaugeFitV5:
    frames = prediction.frame_indices
    _require(frames is not None, "decoded-uniform input must carry frame_indices")
    _require(
        type(cluster_size_pixels) is int and cluster_size_pixels >= 1,
        "cluster_size_pixels must be a positive integer",
    )
    prefix_start, prefix_stop = raw_prefix_range_half_open
    selected: tuple[int, int, np.ndarray, np.ndarray, np.ndarray] | None = None
    for prediction_index, raw_frame in enumerate(frames):
        frame = int(raw_frame)
        if frame < prefix_start or frame >= prefix_stop:
            continue
        metric_index = frame - prefix_start
        active = np.asarray(prediction.valid_mask[prediction_index]) & np.asarray(
            metric.valid_mask[metric_index]
        )
        rows, columns = np.nonzero(active)
        if len(rows) < 8:
            continue
        clusters = np.column_stack(
            (rows // cluster_size_pixels, columns // cluster_size_pixels)
        )
        if len(np.unique(clusters, axis=0)) >= 8:
            selected = prediction_index, metric_index, rows, columns, clusters
            break
    if selected is None:
        raise ValueError("metric gauge lacks eight independent causal clusters")
    prediction_index, metric_index, rows, columns, clusters = selected
    trim_fraction = max(0.8, min(1.0, 8.0 / len(rows)))
    transform = robust_similarity_transform(
        np.asarray(prediction.point_map[prediction_index, rows, columns]),
        metric.points_world_m[metric_index, rows, columns],
        trim_fraction=trim_fraction,
        iterations=5,
    )
    inlier = np.asarray(transform["inlier_mask"], dtype=np.bool_)
    _require(inlier.shape == (len(rows),), "metric-gauge inlier mask changed")
    inlier_clusters = len(np.unique(clusters[inlier], axis=0))
    _require(
        inlier_clusters >= 8,
        "metric gauge has fewer than eight independent inlier clusters",
    )
    return Deform360JointSparseMetricGaugeFitV5(
        camera_id=camera_id,
        raw_frame_index=int(metric.frame_indices[metric_index]),
        linear=np.asarray(transform["linear"]),
        translation=np.asarray(transform["translation"]),
        input_pair_count=int(transform["input_pair_count"]),
        inlier_pair_count=int(transform["inlier_pair_count"]),
        independent_cluster_count=len(np.unique(clusters, axis=0)),
        inlier_independent_cluster_count=inlier_clusters,
        inlier_rmse_m=float(transform["inlier_rmse_m"]),
        source_artifact_ids=source_artifact_ids,
    )


def prepare_deform360_joint_sparse_visual_window_v5(
    *,
    camera_id: str,
    decoded_uniform_path: str | Path,
    metric_prefix_path: str | Path,
    raw_prefix_range_half_open: tuple[int, int],
    fit: Deform360JointSparsePrefixFitV5,
    source_artifact_ids: Mapping[str, str],
    extraction_config: Deform360JointSparseExtractionConfigV5 | None = None,
    metric_cluster_size_pixels: int = 32,
) -> tuple[
    Deform360JointSparseVisualWindowRowsV5,
    Deform360JointSparseMetricGaugeFitV5,
]:
    """Prepare one causal public visual stream without opening an endpoint.

    The PhysTwin state and its residual are intentionally absent from this
    interface.  Contributor count is retained for dependence accounting but
    never converted into higher prior confidence.
    """

    if not isinstance(fit, Deform360JointSparsePrefixFitV5):
        raise TypeError("fit must be a Deform360JointSparsePrefixFitV5")
    camera = nonempty_string(camera_id, name="camera_id")
    decoded_path = Path(decoded_uniform_path).resolve(strict=True)
    metric_path = Path(metric_prefix_path).resolve(strict=True)
    supplied_sources = source_artifact_mapping(
        source_artifact_ids,
        name="source_artifact_ids",
        allow_empty=True,
    )
    automatic_sources = {
        f"prob4d-decoded-uniform/{camera}.npz": _sha256_file(decoded_path),
        f"robot-metric-prefix/{camera}.npz": _sha256_file(metric_path),
    }
    overlap = set(supplied_sources) & set(automatic_sources)
    _require(
        all(supplied_sources[key] == automatic_sources[key] for key in overlap),
        "source artifact digest conflicts with the opened public input",
    )
    sources = MappingProxyType(
        dict(sorted({**supplied_sources, **automatic_sources}.items()))
    )
    prediction = load_motioncrafter_prediction(decoded_path)
    frames = prediction.frame_indices
    _require(frames is not None, "decoded-uniform input lacks frame_indices")
    prefix_start, prefix_stop = raw_prefix_range_half_open
    causal = (frames >= prefix_start) & (frames < prefix_stop)
    _require(np.any(causal), "decoded-uniform input has no causal frame")
    image_shape = (
        int(prediction.valid_mask.shape[1]),
        int(prediction.valid_mask.shape[2]),
    )
    metric = _load_metric_prefix(
        metric_path,
        raw_prefix_range_half_open=raw_prefix_range_half_open,
        image_shape=image_shape,
    )
    gauge = _fit_metric_gauge(
        prediction,
        metric,
        camera_id=camera,
        raw_prefix_range_half_open=raw_prefix_range_half_open,
        source_artifact_ids=sources,
        cluster_size_pixels=metric_cluster_size_pixels,
    )
    aligned = align_motioncrafter_prediction(
        prediction,
        {"linear": gauge.linear, "translation": gauge.translation},
    )
    selected = np.flatnonzero(causal)
    local_frames = np.asarray(frames[selected] - prefix_start, dtype=np.int64)
    contributors = (
        np.ones_like(aligned.valid_mask[selected], dtype=np.int64)
        if aligned.contributors is None
        else np.asarray(aligned.contributors[selected], dtype=np.int64)
    )
    source_confidence = (
        np.ones_like(aligned.valid_mask[selected], dtype=np.float64)
        if aligned.source_confidence is None
        else np.asarray(aligned.source_confidence[selected], dtype=np.float64)
    )
    _require(
        np.all(np.isfinite(source_confidence[aligned.valid_mask[selected]]))
        and np.all(
            (source_confidence[aligned.valid_mask[selected]] >= 0.0)
            & (source_confidence[aligned.valid_mask[selected]] <= 1.0)
        ),
        "source confidence must lie in [0,1] on valid pixels",
    )
    covariance = (
        None
        if aligned.point_covariance_m2 is None
        else np.asarray(aligned.point_covariance_m2[selected], dtype=np.float64)
    )
    rows = extract_deform360_joint_sparse_visual_rows_v5(
        camera_id=camera,
        window_id=f"prob4d-decoded-uniform:{camera}",
        frame_indices=local_frames,
        point_map_world_m=np.asarray(aligned.point_map[selected], dtype=np.float64),
        valid_mask=np.asarray(aligned.valid_mask[selected], dtype=np.bool_),
        object_mask=np.asarray(aligned.deform_mask[selected], dtype=np.bool_),
        causal_frame_stop=prefix_stop - prefix_start,
        fit=fit,
        source_artifact_ids={
            **dict(sources),
            f"metric-gauge/{camera}.json": gauge.artifact_id,
        },
        point_covariance_m2=covariance,
        source_confidence=source_confidence,
        overlap_disagreement_m=np.zeros_like(source_confidence),
        contributor_count=contributors,
        config=extraction_config,
    )
    return rows, gauge


def _contact_patch_moments(
    tactile_response: np.ndarray,
    taxel_world_positions_m: np.ndarray,
    *,
    localization_floor_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(tactile_response)
    centroids: np.ndarray = np.empty((count, 3), dtype=np.float64)
    covariance: np.ndarray = np.empty((count, 3, 3), dtype=np.float64)
    floor = localization_floor_m**2 * np.eye(3, dtype=np.float64)
    for index in range(count):
        positive = np.maximum(tactile_response[index], 0.0)
        active = positive > 0.0
        active_positions = taxel_world_positions_m[index, active]
        _require(
            len(np.unique(active_positions, axis=0)) >= 2,
            "contact row has fewer than two unique active taxels",
        )
        weights = positive[active]
        weights /= np.sum(weights)
        centroid = np.einsum(
            "k,kc->c",
            weights,
            active_positions,
            optimize=True,
        )
        centered = active_positions - centroid
        scatter = np.einsum(
            "k,ki,kj->ij",
            weights,
            centered,
            centered,
            optimize=True,
        )
        centroids[index] = centroid
        covariance[index] = 0.5 * (scatter + scatter.T) + floor
    return centroids, covariance


def _contact_graph_association(
    physical_prediction_m: np.ndarray,
    frame_indices: np.ndarray,
    observed_point_world_m: np.ndarray,
    *,
    candidate_count: int,
    association_scale_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_count = physical_prediction_m.shape[1]
    count = min(candidate_count, node_count)
    _require(count >= 1, "physical prediction has no graph nodes")
    candidates = physical_prediction_m[frame_indices]
    squared = np.sum(
        np.square(candidates - observed_point_world_m[:, None]),
        axis=2,
    )
    indices = np.argpartition(squared, kth=count - 1, axis=1)[:, :count]
    selected_squared = np.take_along_axis(squared, indices, axis=1)
    order = np.argsort(selected_squared, axis=1, kind="mergesort")
    indices = np.take_along_axis(indices, order, axis=1)
    selected_squared = np.take_along_axis(selected_squared, order, axis=1)
    logits = -0.5 * selected_squared / association_scale_m**2
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.clip(logits, -700.0, 0.0))
    weights /= np.sum(weights, axis=1, keepdims=True)
    selected = np.take_along_axis(candidates, indices[..., None], axis=1)
    mean = np.sum(weights[..., None] * selected, axis=1)
    centered = selected - mean[:, None]
    spread = np.einsum(
        "ak,aki,akj->aij",
        weights,
        centered,
        centered,
        optimize=True,
    )
    return indices, weights, 0.5 * (spread + np.swapaxes(spread, -1, -2))


def _load_contact_prefix_arrays(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    try:
        frames = np.asarray(
            np.load(root / "frame-ids.npy", allow_pickle=False),
            dtype=np.int64,
        )
        response = np.asarray(
            np.load(root / "tactile-response.npy", allow_pickle=False),
            dtype=np.float64,
        )
        positions = np.asarray(
            np.load(root / "taxel-world-positions-m.npy", allow_pickle=False),
            dtype=np.float64,
        )
        reliability = np.asarray(
            np.load(root / "source-reliability.npy", allow_pickle=False),
            dtype=np.float64,
        )
        sensors_value = json.loads(
            (root / "sensor-names.json").read_text(encoding="utf-8")
        )
        episodes_value = json.loads(
            (root / "contact-episode-ids.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("cannot load validated contact-prefix arrays") from error
    row_count = manifest["row_count"]
    _require(type(row_count) is int and row_count > 0, "contact row count changed")
    _require(
        isinstance(sensors_value, list)
        and isinstance(episodes_value, list)
        and all(type(value) is str and value for value in sensors_value)
        and all(type(value) is str and value for value in episodes_value),
        "contact identities changed",
    )
    sensors = tuple(sensors_value)
    episodes = tuple(episodes_value)
    _require(
        frames.shape == (row_count,)
        and response.shape[0] == row_count
        and positions.shape == (row_count, response.shape[1], 3)
        and reliability.shape == (row_count,)
        and len(sensors) == len(episodes) == row_count,
        "contact-prefix array dimensions changed",
    )
    return frames, sensors, episodes, response, positions, reliability


def prepare_deform360_joint_sparse_contact_rows_v5(
    *,
    contact_prefix_directory: str | Path,
    object_id: str,
    episode_id: int,
    raw_prefix_range_half_open: tuple[int, int],
    physical_prediction_m: object,
    source_artifact_ids: Mapping[str, str] | None = None,
    config: Deform360JointSparseContactConfigV5 | None = None,
) -> Deform360JointSparseContactRowsV5 | None:
    """Convert one released contact prefix without using the state residual.

    Candidate physical geometry determines only the graph-assignment mixture.
    The prior reliability is copied from the independently materialized public
    sensor artifact, and the assignment-mixture spread is added to covariance.
    A validated support-negative prefix returns ``None`` for exact fallback.
    """

    cfg = config or Deform360JointSparseContactConfigV5()
    if not isinstance(cfg, Deform360JointSparseContactConfigV5):
        raise TypeError("config must be Deform360JointSparseContactConfigV5")
    identifier = nonempty_string(object_id, name="object_id")
    _require(type(episode_id) is int and episode_id >= 0, "invalid episode_id")
    prefix_start, prefix_stop = raw_prefix_range_half_open
    _require(
        type(prefix_start) is int
        and type(prefix_stop) is int
        and 0 <= prefix_start < prefix_stop,
        "raw prefix range is invalid",
    )
    requested_root = Path(contact_prefix_directory).absolute()
    manifest = validate_deform360_public_contact_prefix(requested_root)
    root = requested_root.resolve(strict=True)
    _require(manifest["object_id"] == identifier, "contact object_id changed")
    _require(manifest["episode_id"] == episode_id, "contact episode_id changed")
    _require(
        manifest["prefix_raw_frame_range_half_open"]
        == [prefix_start, prefix_stop],
        "contact prefix range changed",
    )
    if manifest["status"] == "support-negative":
        return None
    _require(manifest["status"] == "materialized", "unknown contact status")

    physical = np.asarray(physical_prediction_m, dtype=np.float64)
    _require(
        physical.ndim == 3
        and physical.shape[0] >= prefix_stop - prefix_start
        and physical.shape[1] >= 1
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical)),
        "physical prediction must have shape (T,N,3) and be finite",
    )
    frames, sensors, episodes, response, positions, reliability = (
        _load_contact_prefix_arrays(root, manifest)
    )
    # Revalidate after opening every member so a concurrent mutation cannot
    # silently change the sensor evidence used by this adapter.
    _require(
        validate_deform360_public_contact_prefix(root)["materialization_id"]
        == manifest["materialization_id"],
        "contact prefix changed while it was opened",
    )
    local_frames: np.ndarray = frames - prefix_start
    _require(
        np.all((local_frames >= 0) & (local_frames < prefix_stop - prefix_start)),
        "contact rows leave the causal prefix",
    )
    observed, covariance = _contact_patch_moments(
        response,
        positions,
        localization_floor_m=cfg.localization_floor_m,
    )
    indices, weights, mixture_spread = _contact_graph_association(
        physical,
        local_frames,
        observed,
        candidate_count=cfg.association_candidate_count,
        association_scale_m=cfg.association_scale_m,
    )
    covariance += mixture_spread

    supplied_sources = source_artifact_mapping(
        {} if source_artifact_ids is None else source_artifact_ids,
        name="contact source_artifact_ids",
        allow_empty=True,
    )
    automatic_sources = {
        f"public-contact-prefix/{name}": _sha256_file(root / name)
        for name in (
            "contact-prefix.json",
            "contact-episode-ids.json",
            "frame-ids.npy",
            "sensor-names.json",
            "source-reliability.npy",
            "tactile-response.npy",
            "taxel-world-positions-m.npy",
        )
    }
    overlap = set(supplied_sources) & set(automatic_sources)
    _require(
        all(supplied_sources[key] == automatic_sources[key] for key in overlap),
        "contact source artifact digest conflicts with public input",
    )
    sources = dict(sorted({**supplied_sources, **automatic_sources}.items()))
    return Deform360JointSparseContactRowsV5(
        frame_indices=local_frames,
        observed_point_world_m=observed,
        graph_node_indices=indices,
        graph_node_weights=weights,
        covariance_m2=covariance,
        prior_reliability=reliability,
        correlation_group_ids=tuple(
            f"deform360-contact:{sensor}:{contact_episode}"
            for sensor, contact_episode in zip(sensors, episodes, strict=True)
        ),
        source_artifact_ids=sources,
    )


__all__ = [
    "Deform360JointSparseContactConfigV5",
    "Deform360JointSparseMetricGaugeFitV5",
    "prepare_deform360_joint_sparse_contact_rows_v5",
    "prepare_deform360_joint_sparse_visual_window_v5",
]
