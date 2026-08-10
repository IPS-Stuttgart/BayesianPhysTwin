"""Prefix-only Deform360 materialization for the joint-sparse v5 study.

The routines in this module turn released, metric, causal-prefix measurements
into the numerical prediction contract.  They deliberately do not load or
score future geometry.  Perception reliability is built only from provider and
mask cues; physical innovation enters the robust Bayesian likelihood later.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from ._canonical_contracts import plain_json
from ._gauge_aware_contracts import GaugeAwareObservationBatch
from ._portable_contracts import (
    exact_revision,
    nonempty_string,
    sha256_digest,
    source_artifact_mapping,
)
from .deform360_joint_sparse_geometric_batch_v4 import _mode_matrices
from .deform360_joint_sparse_prediction_v5 import (
    Deform360JointSparsePredictionInputV5,
)
from .mask_distance import interior_mask_distance
from .sparse_prior_aware_gauge_belief import TreeSparseGaugeDesignV1

Stratum = Literal["sheet", "volumetric"]

VISUAL_ROWS_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-visual-window-rows"
)
PREFIX_FIT_SCHEMA: Final = "bayesian-phystwin.deform360-joint-sparse-prefix-fit"
ADMISSION_SCHEMA: Final = "bayesian-phystwin.deform360-joint-sparse-admission"
MATERIALIZATION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-materialization"
)
MATERIALIZATION_VERSION: Final = 5


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _readonly_float(
    value: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind in "iuf", f"{name} must be real")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    _require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    result.setflags(write=False)
    return result


def _readonly_integer(
    value: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(value)
    _require(
        raw.dtype.kind in "iu" and raw.dtype.kind != "b",
        f"{name} must be integer",
    )
    result = np.array(raw, dtype=np.int64, order="C", copy=True)
    _require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    result.setflags(write=False)
    return result


def _readonly_boolean(
    value: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind == "b", f"{name} must be Boolean")
    result = np.array(raw, dtype=np.bool_, order="C", copy=True)
    _require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    result.setflags(write=False)
    return result


def _positive(value: object, *, name: str, allow_zero: bool = False) -> float:
    _require(not isinstance(value, (bool, np.bool_)), f"{name} must be numeric")
    raw = np.asarray(value)
    _require(raw.shape == () and raw.dtype.kind in "iuf", f"{name} must be scalar")
    result = float(raw.item())
    _require(np.isfinite(result), f"{name} must be finite")
    _require(result >= 0.0 if allow_zero else result > 0.0, f"invalid {name}")
    return result


def _probability(value: object, *, name: str, open_interval: bool = False) -> float:
    result = _positive(value, name=name, allow_zero=not open_interval)
    _require(result <= 1.0, f"{name} exceeds one")
    if open_interval:
        _require(result < 1.0, f"{name} must be below one")
    return result


def _positive_definite(covariance: np.ndarray, *, name: str) -> np.ndarray:
    matrix = 0.5 * (np.asarray(covariance) + np.swapaxes(covariance, -1, -2))
    _require(matrix.shape[-2:] == (3, 3), f"{name} must end in (3,3)")
    eigenvalues = np.linalg.eigvalsh(matrix)
    _require(np.all(eigenvalues > 0.0), f"{name} must be positive definite")
    return matrix


def _merge_sources(*values: Mapping[str, str]) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for value in values:
        normalized = source_artifact_mapping(
            value,
            name="source artifacts",
            allow_empty=True,
        )
        overlap = set(result) & set(normalized)
        _require(
            all(result[key] == normalized[key] for key in overlap),
            "source artifact key has conflicting digests",
        )
        result.update(normalized)
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True, slots=True)
class Deform360JointSparsePrefixFitV5:
    """Prefix-only fold fit used to materialize one held-out prediction."""

    fit_object_ids: tuple[str, ...]
    source_artifact_ids: Mapping[str, str]
    fallback_point_std_m: float = 0.010
    observation_variance_floor_m2: float = 4e-6
    root_gauge_prior_std_m: float = 0.020
    camera_gauge_innovation_std_m: float = 0.010
    window_gauge_innovation_std_m: float = 0.005
    shared_bias_prior_std_m: float = 0.020
    view_bias_prior_std_m: float = 0.010
    state_prior_std_m: float = 0.020
    contact_anchor_bias_std_m: float = 0.005
    association_scale_m: float = 0.010
    maximum_association_distance_m: float = 0.040
    association_entropy_strength: float = 0.5
    overlap_disagreement_scale_m: float = 0.015
    boundary_reliability_scale_pixels: float = 8.0
    boundary_reliability_floor: float = 0.25
    nominal_inlier_probability: float = 0.90
    suffix_outcomes_used: bool = False

    def __post_init__(self) -> None:
        _require(
            type(self.fit_object_ids) is tuple and bool(self.fit_object_ids),
            "fit_object_ids must be a nonempty tuple",
        )
        identifiers = tuple(
            nonempty_string(value, name="fit object ID")
            for value in self.fit_object_ids
        )
        _require(len(set(identifiers)) == len(identifiers), "fit object IDs repeat")
        _require(
            type(self.suffix_outcomes_used) is bool
            and not self.suffix_outcomes_used,
            "prefix fit must not use suffix outcomes",
        )
        for name in (
            "fallback_point_std_m",
            "observation_variance_floor_m2",
            "root_gauge_prior_std_m",
            "camera_gauge_innovation_std_m",
            "window_gauge_innovation_std_m",
            "shared_bias_prior_std_m",
            "view_bias_prior_std_m",
            "state_prior_std_m",
            "contact_anchor_bias_std_m",
            "association_scale_m",
            "maximum_association_distance_m",
            "overlap_disagreement_scale_m",
            "boundary_reliability_scale_pixels",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "association_entropy_strength",
            _positive(
                self.association_entropy_strength,
                name="association_entropy_strength",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "boundary_reliability_floor",
            _probability(
                self.boundary_reliability_floor,
                name="boundary_reliability_floor",
            ),
        )
        object.__setattr__(
            self,
            "nominal_inlier_probability",
            _probability(
                self.nominal_inlier_probability,
                name="nominal_inlier_probability",
                open_interval=True,
            ),
        )
        object.__setattr__(self, "fit_object_ids", identifiers)
        object.__setattr__(
            self,
            "source_artifact_ids",
            source_artifact_mapping(
                self.source_artifact_ids,
                name="fit source_artifact_ids",
            ),
        )

    @property
    def fit_artifact_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": PREFIX_FIT_SCHEMA,
                "schema_version": MATERIALIZATION_VERSION,
                "fit_object_ids": list(self.fit_object_ids),
                "source_artifact_ids": dict(self.source_artifact_ids),
                "fallback_point_std_m": self.fallback_point_std_m,
                "observation_variance_floor_m2": (
                    self.observation_variance_floor_m2
                ),
                "root_gauge_prior_std_m": self.root_gauge_prior_std_m,
                "camera_gauge_innovation_std_m": (
                    self.camera_gauge_innovation_std_m
                ),
                "window_gauge_innovation_std_m": (
                    self.window_gauge_innovation_std_m
                ),
                "shared_bias_prior_std_m": self.shared_bias_prior_std_m,
                "view_bias_prior_std_m": self.view_bias_prior_std_m,
                "state_prior_std_m": self.state_prior_std_m,
                "contact_anchor_bias_std_m": self.contact_anchor_bias_std_m,
                "association_scale_m": self.association_scale_m,
                "maximum_association_distance_m": (
                    self.maximum_association_distance_m
                ),
                "association_entropy_strength": self.association_entropy_strength,
                "overlap_disagreement_scale_m": (
                    self.overlap_disagreement_scale_m
                ),
                "boundary_reliability_scale_pixels": (
                    self.boundary_reliability_scale_pixels
                ),
                "boundary_reliability_floor": self.boundary_reliability_floor,
                "nominal_inlier_probability": self.nominal_inlier_probability,
                "suffix_outcomes_used": self.suffix_outcomes_used,
                "information_boundary": {
                    "development_prefix_only": True,
                    "development_suffix_outcomes_used": False,
                    "confirmation_payloads_opened": False,
                    "target_outcomes_used": False,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class Deform360JointSparseExtractionConfigV5:
    """Outcome-blind deterministic pixel extraction settings."""

    measurement_stride_pixels: int = 4
    maximum_rows_per_window: int = 512

    def __post_init__(self) -> None:
        for name in ("measurement_stride_pixels", "maximum_rows_per_window"):
            value = getattr(self, name)
            _require(type(value) is int and value >= 1, f"invalid {name}")


@dataclass(frozen=True, slots=True)
class Deform360JointSparseVisualWindowRowsV5:
    """Sparse metric rows from one released causal MotionCrafter window."""

    camera_id: str
    window_id: str
    frame_indices: np.ndarray
    pixel_yx: np.ndarray
    point_world_m: np.ndarray
    point_covariance_m2: np.ndarray
    source_confidence: np.ndarray
    mask_distance_pixels: np.ndarray
    overlap_disagreement_m: np.ndarray
    contributor_count: np.ndarray
    source_artifact_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        camera = nonempty_string(self.camera_id, name="camera_id")
        window = nonempty_string(self.window_id, name="window_id")
        frames = _readonly_integer(self.frame_indices, name="frame_indices", ndim=1)
        pixels = _readonly_integer(self.pixel_yx, name="pixel_yx", ndim=2)
        points = _readonly_float(self.point_world_m, name="point_world_m", ndim=2)
        covariance = _readonly_float(
            self.point_covariance_m2,
            name="point_covariance_m2",
            ndim=3,
        )
        confidence = _readonly_float(
            self.source_confidence,
            name="source_confidence",
            ndim=1,
        )
        distance = _readonly_float(
            self.mask_distance_pixels,
            name="mask_distance_pixels",
            ndim=1,
        )
        disagreement = _readonly_float(
            self.overlap_disagreement_m,
            name="overlap_disagreement_m",
            ndim=1,
        )
        contributors = _readonly_integer(
            self.contributor_count,
            name="contributor_count",
            ndim=1,
        )
        count = len(frames)
        _require(count > 0, "visual window has no rows")
        _require(pixels.shape == (count, 2), "pixel_yx shape changed")
        _require(points.shape == (count, 3), "point_world_m shape changed")
        _require(covariance.shape == (count, 3, 3), "point covariance shape changed")
        _require(confidence.shape == (count,), "source confidence shape changed")
        _require(distance.shape == (count,), "mask distance shape changed")
        _require(disagreement.shape == (count,), "overlap disagreement shape changed")
        _require(contributors.shape == (count,), "contributor count shape changed")
        _require(np.all(frames >= 0), "frame indices must be nonnegative")
        _require(np.all(pixels >= 0), "pixel indices must be nonnegative")
        _require(
            np.all((confidence >= 0.0) & (confidence <= 1.0)),
            "source confidence must lie in [0,1]",
        )
        _require(np.all(distance >= 0.0), "mask distance must be nonnegative")
        _require(
            np.all(disagreement >= 0.0),
            "overlap disagreement must be nonnegative",
        )
        _require(np.all(contributors >= 1), "contributors must be positive")
        covariance = _positive_definite(covariance, name="point covariance")
        covariance.setflags(write=False)
        object.__setattr__(self, "camera_id", camera)
        object.__setattr__(self, "window_id", window)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "pixel_yx", pixels)
        object.__setattr__(self, "point_world_m", points)
        object.__setattr__(self, "point_covariance_m2", covariance)
        object.__setattr__(self, "source_confidence", confidence)
        object.__setattr__(self, "mask_distance_pixels", distance)
        object.__setattr__(self, "overlap_disagreement_m", disagreement)
        object.__setattr__(self, "contributor_count", contributors)
        object.__setattr__(
            self,
            "source_artifact_ids",
            source_artifact_mapping(
                self.source_artifact_ids,
                name="visual source_artifact_ids",
            ),
        )

    @property
    def artifact_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": VISUAL_ROWS_SCHEMA,
                "schema_version": MATERIALIZATION_VERSION,
                "camera_id": self.camera_id,
                "window_id": self.window_id,
                "array_sha256": {
                    "frame_indices": _array_sha256(self.frame_indices),
                    "pixel_yx": _array_sha256(self.pixel_yx),
                    "point_world_m": _array_sha256(self.point_world_m),
                    "point_covariance_m2": _array_sha256(
                        self.point_covariance_m2
                    ),
                    "source_confidence": _array_sha256(self.source_confidence),
                    "mask_distance_pixels": _array_sha256(
                        self.mask_distance_pixels
                    ),
                    "overlap_disagreement_m": _array_sha256(
                        self.overlap_disagreement_m
                    ),
                    "contributor_count": _array_sha256(self.contributor_count),
                },
                "source_artifact_ids": dict(self.source_artifact_ids),
                "information_boundary": {
                    "released_prefix_measurements_used": True,
                    "future_object_observations_used": False,
                    "confirmation_payloads_opened": False,
                },
            }
        )


def _ranked_indices(
    frame: np.ndarray,
    pixel_yx: np.ndarray,
    *,
    maximum: int,
    seed: str,
) -> np.ndarray:
    if len(frame) <= maximum:
        return np.arange(len(frame), dtype=np.int64)
    order = sorted(
        range(len(frame)),
        key=lambda index: hashlib.sha256(
            (
                f"{seed}\0{int(frame[index])}\0{int(pixel_yx[index, 0])}"
                f"\0{int(pixel_yx[index, 1])}"
            ).encode()
        ).digest(),
    )
    return np.asarray(sorted(order[:maximum]), dtype=np.int64)


def extract_deform360_joint_sparse_visual_rows_v5(
    *,
    camera_id: str,
    window_id: str,
    frame_indices: np.ndarray,
    point_map_world_m: np.ndarray,
    valid_mask: np.ndarray,
    object_mask: np.ndarray,
    causal_frame_stop: int,
    fit: Deform360JointSparsePrefixFitV5,
    source_artifact_ids: Mapping[str, str],
    point_covariance_m2: np.ndarray | None = None,
    source_confidence: np.ndarray | None = None,
    overlap_disagreement_m: np.ndarray | None = None,
    contributor_count: np.ndarray | None = None,
    config: Deform360JointSparseExtractionConfigV5 | None = None,
) -> Deform360JointSparseVisualWindowRowsV5:
    """Extract deterministic sparse rows without consulting physical residuals."""

    if not isinstance(fit, Deform360JointSparsePrefixFitV5):
        raise TypeError("fit must be a Deform360JointSparsePrefixFitV5")
    cfg = config or Deform360JointSparseExtractionConfigV5()
    frames = _readonly_integer(frame_indices, name="frame_indices", ndim=1)
    raw_points = np.asarray(point_map_world_m)
    _require(raw_points.dtype.kind in "iuf", "point_map_world_m must be real")
    points = np.array(raw_points, dtype=np.float64, order="C", copy=True)
    _require(points.ndim == 4, "point_map_world_m must have 4 dimensions")
    points.setflags(write=False)
    valid = _readonly_boolean(valid_mask, name="valid_mask", ndim=3)
    mask = _readonly_boolean(object_mask, name="object_mask", ndim=3)
    _require(
        points.shape[:-1] == valid.shape == mask.shape
        and len(frames) == len(points),
        "visual window arrays disagree",
    )
    _require(
        type(causal_frame_stop) is int
        and causal_frame_stop >= 1
        and np.all(frames < causal_frame_stop),
        "visual window leaves the causal prefix",
    )
    grid_y, grid_x = np.indices(valid.shape[1:])
    stride = (grid_y % cfg.measurement_stride_pixels == 0) & (
        grid_x % cfg.measurement_stride_pixels == 0
    )
    active = valid & mask & stride[None] & np.all(np.isfinite(points), axis=3)
    local_frame, row, column = np.nonzero(active)
    _require(len(local_frame) > 0, "visual window has no causal object rows")
    absolute_frame = frames[local_frame]
    pixel = np.column_stack((row, column)).astype(np.int64)
    selected = _ranked_indices(
        absolute_frame,
        pixel,
        maximum=cfg.maximum_rows_per_window,
        seed=f"{camera_id}\0{window_id}\0v5-visual-row-v1",
    )
    local_frame = local_frame[selected]
    absolute_frame = absolute_frame[selected]
    row = row[selected]
    column = column[selected]
    pixel = pixel[selected]

    if point_covariance_m2 is None:
        covariance = np.broadcast_to(
            fit.fallback_point_std_m**2 * np.eye(3),
            points.shape[:-1] + (3, 3),
        )
    else:
        covariance = np.asarray(point_covariance_m2, dtype=np.float64)
        _require(
            covariance.shape == points.shape[:-1] + (3, 3),
            "point covariance grid shape changed",
        )
    if source_confidence is None:
        confidence = np.ones(valid.shape, dtype=np.float64)
    else:
        confidence = np.asarray(source_confidence, dtype=np.float64)
        _require(confidence.shape == valid.shape, "source confidence shape changed")
    if overlap_disagreement_m is None:
        disagreement = np.zeros(valid.shape, dtype=np.float64)
    else:
        disagreement = np.asarray(overlap_disagreement_m, dtype=np.float64)
        _require(disagreement.shape == valid.shape, "overlap disagreement shape changed")
    if contributor_count is None:
        contributors = np.ones(valid.shape, dtype=np.int64)
    else:
        raw_contributors = np.asarray(contributor_count)
        _require(
            raw_contributors.dtype.kind in "iu"
            and raw_contributors.shape == valid.shape,
            "contributor count shape or dtype changed",
        )
        contributors = np.asarray(raw_contributors, dtype=np.int64)

    distance: np.ndarray = np.empty(len(selected), dtype=np.float64)
    for output_index, frame_index in enumerate(local_frame):
        distance[output_index] = interior_mask_distance(mask[frame_index])[
            row[output_index], column[output_index]
        ]
    return Deform360JointSparseVisualWindowRowsV5(
        camera_id=camera_id,
        window_id=window_id,
        frame_indices=absolute_frame,
        pixel_yx=pixel,
        point_world_m=points[local_frame, row, column],
        point_covariance_m2=covariance[local_frame, row, column],
        source_confidence=confidence[local_frame, row, column],
        mask_distance_pixels=distance,
        overlap_disagreement_m=disagreement[local_frame, row, column],
        contributor_count=contributors[local_frame, row, column],
        source_artifact_ids=source_artifact_ids,
    )


@dataclass(frozen=True, slots=True)
class Deform360JointSparseContactRowsV5:
    """Released causal tactile/robot anchors mapped to physical graph patches."""

    frame_indices: np.ndarray
    observed_point_world_m: np.ndarray
    graph_node_indices: np.ndarray
    graph_node_weights: np.ndarray
    covariance_m2: np.ndarray
    prior_reliability: np.ndarray
    correlation_group_ids: tuple[str, ...]
    source_artifact_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        frames = _readonly_integer(self.frame_indices, name="contact frames", ndim=1)
        observed = _readonly_float(
            self.observed_point_world_m,
            name="contact observed points",
            ndim=2,
        )
        indices = _readonly_integer(
            self.graph_node_indices,
            name="contact graph indices",
            ndim=2,
        )
        weights = _readonly_float(
            self.graph_node_weights,
            name="contact graph weights",
            ndim=2,
        )
        covariance = _readonly_float(
            self.covariance_m2,
            name="contact covariance",
            ndim=3,
        )
        reliability = _readonly_float(
            self.prior_reliability,
            name="contact prior reliability",
            ndim=1,
        )
        count = len(frames)
        _require(count > 0, "contact anchor set is empty")
        _require(observed.shape == (count, 3), "contact point shape changed")
        _require(
            indices.ndim == 2
            and indices.shape == weights.shape
            and indices.shape[0] == count
            and indices.shape[1] >= 1,
            "contact patch shape changed",
        )
        _require(covariance.shape == (count, 3, 3), "contact covariance shape changed")
        _require(reliability.shape == (count,), "contact reliability shape changed")
        _require(np.all(indices >= 0), "contact graph indices must be nonnegative")
        _require(
            np.all(weights >= 0.0)
            and np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-12),
            "contact graph weights must be row probabilities",
        )
        _require(
            np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "contact reliability must lie in [0,1]",
        )
        _require(
            type(self.correlation_group_ids) is tuple
            and len(self.correlation_group_ids) == count,
            "contact correlation group IDs changed",
        )
        groups = tuple(
            nonempty_string(value, name="contact correlation group")
            for value in self.correlation_group_ids
        )
        covariance = _positive_definite(covariance, name="contact covariance")
        covariance.setflags(write=False)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "observed_point_world_m", observed)
        object.__setattr__(self, "graph_node_indices", indices)
        object.__setattr__(self, "graph_node_weights", weights)
        object.__setattr__(self, "covariance_m2", covariance)
        object.__setattr__(self, "prior_reliability", reliability)
        object.__setattr__(self, "correlation_group_ids", groups)
        object.__setattr__(
            self,
            "source_artifact_ids",
            source_artifact_mapping(
                self.source_artifact_ids,
                name="contact source_artifact_ids",
            ),
        )


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = map(float, value)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _nearest_neighbors(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    count: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    _require(1 <= count <= len(reference), "invalid association candidate count")
    distances: np.ndarray = np.empty((len(query), count), dtype=np.float64)
    indices: np.ndarray = np.empty((len(query), count), dtype=np.int64)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        squared = np.sum(
            np.square(query[start:stop, None] - reference[None]), axis=2
        )
        local = np.argpartition(squared, kth=count - 1, axis=1)[:, :count]
        local_squared = np.take_along_axis(squared, local, axis=1)
        order = np.argsort(local_squared, axis=1, kind="mergesort")
        local = np.take_along_axis(local, order, axis=1)
        indices[start:stop] = local
        distances[start:stop] = np.sqrt(
            np.take_along_axis(squared, local, axis=1)
        )
    return distances, indices


def _association(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    candidate_count: int,
    scale_m: float,
    maximum_distance_m: float,
    entropy_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance, indices = _nearest_neighbors(
        reference,
        query,
        count=min(candidate_count, len(reference)),
    )
    shifted = np.square(distance / scale_m)
    shifted -= shifted[:, :1]
    weights = np.exp(np.clip(-0.5 * shifted, -700.0, 0.0))
    weights /= np.sum(weights, axis=1, keepdims=True)
    positive = weights > 0.0
    entropy = -np.sum(
        np.where(positive, weights * np.log(np.maximum(weights, 1e-300)), 0.0),
        axis=1,
    )
    if weights.shape[1] > 1:
        entropy /= math.log(weights.shape[1])
    distance_probability = np.exp(
        -0.5 * np.square(distance[:, 0] / maximum_distance_m)
    )
    distance_probability[distance[:, 0] > maximum_distance_m] = 0.0
    probability = distance_probability * np.exp(-entropy_strength * entropy)
    return indices, weights, entropy, np.clip(probability, 0.0, 1.0)


def _state_basis(
    physical: np.ndarray,
    *,
    causal_frame_stop: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    reference = physical[causal_frame_stop - 1]
    centroid = np.mean(reference, axis=0)
    rms = float(
        np.sqrt(np.mean(np.sum(np.square(reference - centroid), axis=1)))
    )
    rms = max(rms, 1e-6)
    frame_centroid = np.mean(physical, axis=1, keepdims=True)
    normalized = (physical - frame_centroid) / rms
    modes = _mode_matrices()
    basis = np.einsum("sij,tnj->tnis", modes, normalized, optimize=True)
    return basis, centroid, rms


def _tree_covariance(
    *,
    local_gauge: np.ndarray,
    gauge_indices: np.ndarray,
    parent_indices: np.ndarray,
    transition_matrices: np.ndarray,
    innovation_scale_tril: np.ndarray,
    gauge_ids: tuple[str, ...],
    prior_id: str,
) -> tuple[np.ndarray, TreeSparseGaugeDesignV1]:
    design = TreeSparseGaugeDesignV1(
        local_gauge_jacobian=local_gauge,
        gauge_indices=gauge_indices,
        parent_indices=parent_indices,
        transition_matrices=transition_matrices,
        innovation_scale_tril=innovation_scale_tril,
        gauge_ids=gauge_ids,
        prior_id=prior_id,
    )
    information = design.prior_information_matrix()
    covariance = np.linalg.solve(information, np.eye(len(information)))
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, design


def _group_composite_weight(
    groups: Sequence[str],
    *,
    cap: float,
) -> np.ndarray:
    counts = Counter(groups)
    return np.asarray(
        [1.0 / min(cap, float(counts[group])) for group in groups],
        dtype=np.float64,
    )


def _effective_row_weight(
    groups: tuple[str, ...],
    reliability: np.ndarray,
    association: np.ndarray,
    composite: np.ndarray,
    *,
    cap: float,
) -> np.ndarray:
    raw = reliability * association
    result: np.ndarray = np.zeros(len(raw), dtype=np.float64)
    labels = np.asarray(groups, dtype=object)
    for group in dict.fromkeys(groups):
        selected = np.flatnonzero(labels == group)
        active = selected[raw[selected] > 0.0]
        if len(active):
            consumer_scale = min(cap, float(len(active))) / len(active)
            result[active] = (
                raw[active] * float(composite[selected[0]]) * consumer_scale
            )
    return result


def _marginal_information(
    *,
    covariance: np.ndarray,
    state: np.ndarray,
    nuisance: np.ndarray,
    nuisance_prior_covariance: np.ndarray,
    row_weight: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    state_count = state.shape[2]
    nuisance_count = nuisance.shape[2]
    state_information = np.zeros((state_count, state_count), dtype=np.float64)
    cross = np.zeros((state_count, nuisance_count), dtype=np.float64)
    nuisance_information = np.linalg.solve(
        nuisance_prior_covariance,
        np.eye(nuisance_count),
    )
    for index in np.flatnonzero(mask & (row_weight > 0.0)):
        whitener = np.linalg.solve(
            np.linalg.cholesky(covariance[index]),
            np.eye(3),
        )
        state_row = whitener @ state[index]
        nuisance_row = whitener @ nuisance[index]
        weight = float(row_weight[index])
        state_information += weight * state_row.T @ state_row
        cross += weight * state_row.T @ nuisance_row
        nuisance_information += weight * nuisance_row.T @ nuisance_row
    marginal = state_information - cross @ np.linalg.solve(
        nuisance_information,
        cross.T,
    )
    return 0.5 * (marginal + marginal.T)


def _rank(value: np.ndarray) -> tuple[int, np.ndarray, float | None]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (value + value.T))
    tolerance = max(1e-12, 1e-9 * max(float(eigenvalues[-1]), 0.0))
    _require(np.all(eigenvalues >= -tolerance), "state information is indefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    positive = eigenvalues > tolerance
    rank = int(np.count_nonzero(positive))
    condition = (
        None
        if rank < len(eigenvalues)
        else float(eigenvalues[-1] / eigenvalues[0])
    )
    return rank, eigenvalues, condition


@dataclass(frozen=True, slots=True)
class Deform360JointSparseAdmissionResultV5:
    """Prefix-only object/query admission with explicit dependence handling."""

    gate_passed: bool
    checks: Mapping[str, bool]
    factor_count: int
    excluded_factor_count: int
    distinct_camera_count: int
    distinct_window_count: int
    distinct_spatial_cluster_count: int
    distinct_correlation_group_count: int
    query_rank: int
    query_precision_eigenvalues: tuple[float, ...]
    query_condition_number: float | None
    maximum_single_camera_information_fraction: float
    minimum_leave_one_camera_rank_fraction: float
    minimum_leave_one_window_rank_fraction: float
    effective_row_weight_sum: float
    input_id: str

    def __post_init__(self) -> None:
        _require(type(self.gate_passed) is bool, "gate_passed must be Boolean")
        checks = dict(self.checks)
        _require(
            bool(checks)
            and all(type(key) is str and type(value) is bool for key, value in checks.items()),
            "admission checks changed",
        )
        _require(self.gate_passed == all(checks.values()), "gate decision changed")
        for name in (
            "factor_count",
            "excluded_factor_count",
            "distinct_camera_count",
            "distinct_window_count",
            "distinct_spatial_cluster_count",
            "distinct_correlation_group_count",
            "query_rank",
        ):
            value = getattr(self, name)
            _require(type(value) is int and value >= 0, f"invalid {name}")
        eigenvalues = tuple(
            _positive(value, name="query eigenvalue", allow_zero=True)
            for value in self.query_precision_eigenvalues
        )
        _require(
            eigenvalues == tuple(sorted(eigenvalues)) and bool(eigenvalues),
            "query eigenvalues changed",
        )
        condition = self.query_condition_number
        if condition is not None:
            condition = _positive(condition, name="query condition number")
        for name in (
            "maximum_single_camera_information_fraction",
            "minimum_leave_one_camera_rank_fraction",
            "minimum_leave_one_window_rank_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "effective_row_weight_sum",
            _positive(
                self.effective_row_weight_sum,
                name="effective_row_weight_sum",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "input_id",
            sha256_digest(self.input_id, name="admission input_id"),
        )
        object.__setattr__(self, "checks", MappingProxyType(dict(sorted(checks.items()))))
        object.__setattr__(self, "query_precision_eigenvalues", eigenvalues)
        object.__setattr__(self, "query_condition_number", condition)

    @property
    def result_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": ADMISSION_SCHEMA,
                "schema_version": MATERIALIZATION_VERSION,
                "input_id": self.input_id,
                "gate_passed": self.gate_passed,
                "checks": dict(self.checks),
                "factor_count": self.factor_count,
                "excluded_factor_count": self.excluded_factor_count,
                "distinct_camera_count": self.distinct_camera_count,
                "distinct_window_count": self.distinct_window_count,
                "distinct_spatial_cluster_count": (
                    self.distinct_spatial_cluster_count
                ),
                "distinct_correlation_group_count": (
                    self.distinct_correlation_group_count
                ),
                "query_rank": self.query_rank,
                "query_precision_eigenvalues": list(
                    self.query_precision_eigenvalues
                ),
                "query_condition_number": self.query_condition_number,
                "maximum_single_camera_information_fraction": (
                    self.maximum_single_camera_information_fraction
                ),
                "minimum_leave_one_camera_rank_fraction": (
                    self.minimum_leave_one_camera_rank_fraction
                ),
                "minimum_leave_one_window_rank_fraction": (
                    self.minimum_leave_one_window_rank_fraction
                ),
                "effective_row_weight_sum": self.effective_row_weight_sum,
                "information_boundary": {
                    "released_prefix_point_values_used": True,
                    "future_object_observations_used": False,
                    "development_suffix_outcomes_used": False,
                    "confirmation_payloads_opened": False,
                    "target_outcomes_used": False,
                    "human_approval_required": False,
                    "new_measurements_required": False,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class Deform360JointSparseMaterializationResultV5:
    """Numerical problem plus the prefix-only admission that controls fallback."""

    problem: Deform360JointSparsePredictionInputV5
    admission: Deform360JointSparseAdmissionResultV5
    fit_artifact_id: str
    visual_row_count: int
    contact_row_count: int
    source_artifact_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.problem, Deform360JointSparsePredictionInputV5):
            raise TypeError("problem has the wrong type")
        if not isinstance(self.admission, Deform360JointSparseAdmissionResultV5):
            raise TypeError("admission has the wrong type")
        _require(
            self.problem.factor_admitted == self.admission.gate_passed,
            "problem/admission decision differs",
        )
        object.__setattr__(
            self,
            "fit_artifact_id",
            sha256_digest(self.fit_artifact_id, name="fit_artifact_id"),
        )
        for name in ("visual_row_count", "contact_row_count"):
            value = getattr(self, name)
            _require(type(value) is int and value >= 0, f"invalid {name}")
        object.__setattr__(
            self,
            "source_artifact_ids",
            source_artifact_mapping(
                self.source_artifact_ids,
                name="materialization source_artifact_ids",
            ),
        )

    @property
    def materialization_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": MATERIALIZATION_SCHEMA,
                "schema_version": MATERIALIZATION_VERSION,
                "prediction_input_id": self.problem.input_id,
                "admission_result_id": self.admission.result_id,
                "fit_artifact_id": self.fit_artifact_id,
                "visual_row_count": self.visual_row_count,
                "contact_row_count": self.contact_row_count,
                "source_artifact_ids": dict(self.source_artifact_ids),
            }
        )


def materialize_deform360_joint_sparse_prediction_v5(
    *,
    object_id: str,
    episode_id: int,
    stratum: Stratum,
    physical_prediction_m: np.ndarray,
    persistence_m: np.ndarray,
    last_causal_residual_m: np.ndarray,
    physical_mode: str,
    causal_frame_stop: int,
    evaluation_frame_range_half_open: tuple[int, int],
    visual_windows: Sequence[Deform360JointSparseVisualWindowRowsV5],
    contact_rows: Deform360JointSparseContactRowsV5 | None,
    fit: Deform360JointSparsePrefixFitV5,
    implementation_revision: str,
    source_artifact_ids: Mapping[str, str],
    association_candidate_count: int = 4,
    spatial_cluster_size_m: float = 0.020,
    effective_samples_per_correlation_group: float = 64.0,
) -> Deform360JointSparseMaterializationResultV5:
    """Build one causal v5 prediction problem from released prefix measurements."""

    if not isinstance(fit, Deform360JointSparsePrefixFitV5):
        raise TypeError("fit must be a Deform360JointSparsePrefixFitV5")
    identifier = nonempty_string(object_id, name="object_id")
    _require(type(episode_id) is int and episode_id >= 0, "invalid episode_id")
    _require(stratum in {"sheet", "volumetric"}, "invalid stratum")
    revision = exact_revision(implementation_revision, name="implementation_revision")
    _require(
        type(causal_frame_stop) is int and causal_frame_stop >= 1,
        "invalid causal_frame_stop",
    )
    _require(
        type(association_candidate_count) is int
        and association_candidate_count >= 1,
        "invalid association_candidate_count",
    )
    cluster_size = _positive(
        spatial_cluster_size_m,
        name="spatial_cluster_size_m",
    )
    effective_cap = _positive(
        effective_samples_per_correlation_group,
        name="effective_samples_per_correlation_group",
    )
    physical = np.asarray(physical_prediction_m)
    persistence = np.asarray(persistence_m)
    _require(
        physical.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        and persistence.dtype == physical.dtype
        and physical.shape == persistence.shape
        and physical.ndim == 3
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical))
        and np.all(np.isfinite(persistence)),
        "physical and persistence trajectories changed",
    )
    _require(
        causal_frame_stop <= len(physical),
        "causal cutoff leaves the physical trajectory",
    )
    _require(bool(visual_windows), "at least one visual window is required")
    windows = tuple(visual_windows)
    _require(
        all(isinstance(value, Deform360JointSparseVisualWindowRowsV5) for value in windows),
        "visual window type changed",
    )
    key_order = tuple((value.camera_id, value.window_id) for value in windows)
    _require(len(set(key_order)) == len(key_order), "visual camera/window repeats")
    windows = tuple(sorted(windows, key=lambda value: (value.camera_id, int(np.min(value.frame_indices)), value.window_id)))
    _require(
        all(np.all(value.frame_indices < causal_frame_stop) for value in windows),
        "visual rows leave the causal prefix",
    )

    state_basis, object_centroid, object_rms = _state_basis(
        np.asarray(physical, dtype=np.float64),
        causal_frame_stop=causal_frame_stop,
    )
    state_count = state_basis.shape[-1]
    node_count = physical.shape[1]

    gauge_ids_list = ["global-similarity-root-v5"]
    parent_list = [-1]
    gauge_scale_list = [fit.root_gauge_prior_std_m]
    window_gauge_index: dict[tuple[str, str], int] = {}
    previous_by_camera: dict[str, int] = {}
    for window in windows:
        key = (window.camera_id, window.window_id)
        gauge_ids_list.append(
            f"camera-window-similarity-v5:{window.camera_id}:{window.window_id}"
        )
        parent_list.append(previous_by_camera.get(window.camera_id, 0))
        gauge_scale_list.append(
            fit.camera_gauge_innovation_std_m
            if window.camera_id not in previous_by_camera
            else fit.window_gauge_innovation_std_m
        )
        gauge_index = len(gauge_ids_list) - 1
        window_gauge_index[key] = gauge_index
        previous_by_camera[window.camera_id] = gauge_index
    gauge_ids = tuple(gauge_ids_list)
    gauge_count = len(gauge_ids)
    parent_indices = np.asarray(parent_list, dtype=np.int64)
    transitions: np.ndarray = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    transitions[1:] = np.eye(7)
    scales = np.asarray(
        [np.eye(7) * value for value in gauge_scale_list],
        dtype=np.float64,
    )

    points = np.concatenate([value.point_world_m for value in windows], axis=0)
    frames = np.concatenate([value.frame_indices for value in windows], axis=0)
    covariance_source = np.concatenate(
        [value.point_covariance_m2 for value in windows], axis=0
    )
    confidence = np.concatenate([value.source_confidence for value in windows])
    mask_distance = np.concatenate(
        [value.mask_distance_pixels for value in windows]
    )
    disagreement = np.concatenate(
        [value.overlap_disagreement_m for value in windows]
    )
    contributors = np.concatenate([value.contributor_count for value in windows])
    camera_ids = tuple(
        camera for value in windows for camera in [value.camera_id] * len(value.frame_indices)
    )
    window_ids = tuple(
        window for value in windows for window in [value.window_id] * len(value.frame_indices)
    )
    row_count = len(points)
    gauge_indices = np.asarray(
        [window_gauge_index[(camera, window)] for camera, window in zip(camera_ids, window_ids, strict=True)],
        dtype=np.int64,
    )

    candidate_indices = np.empty(
        (row_count, min(association_candidate_count, node_count)),
        dtype=np.int64,
    )
    candidate_weights = np.empty_like(candidate_indices, dtype=np.float64)
    association_entropy: np.ndarray = np.empty(row_count, dtype=np.float64)
    association_probability: np.ndarray = np.empty(row_count, dtype=np.float64)
    for frame in np.unique(frames):
        selected = np.flatnonzero(frames == frame)
        indices, weights, entropy, probability = _association(
            np.asarray(physical[int(frame)], dtype=np.float64),
            points[selected],
            candidate_count=association_candidate_count,
            scale_m=fit.association_scale_m,
            maximum_distance_m=fit.maximum_association_distance_m,
            entropy_strength=fit.association_entropy_strength,
        )
        candidate_indices[selected] = indices
        candidate_weights[selected] = weights
        association_entropy[selected] = entropy
        association_probability[selected] = probability

    candidate_points = np.asarray(physical, dtype=np.float64)[
        frames[:, None], candidate_indices
    ]
    predicted = np.sum(candidate_weights[..., None] * candidate_points, axis=1)
    offset = candidate_points - predicted[:, None]
    assignment_covariance = np.einsum(
        "mk,mki,mkj->mij",
        candidate_weights,
        offset,
        offset,
        optimize=True,
    )
    covariance = covariance_source + assignment_covariance
    covariance += fit.observation_variance_floor_m2 * np.eye(3)[None]
    covariance = _positive_definite(covariance, name="visual covariance")
    innovation = points - predicted

    state = np.sum(
        candidate_weights[:, :, None, None]
        * state_basis[frames[:, None], candidate_indices],
        axis=1,
    )
    local_gauge: np.ndarray = np.zeros((row_count, 3, 7), dtype=np.float64)
    normalized_predicted = (predicted - object_centroid) / object_rms
    local_gauge[:, :, :3] = np.eye(3)
    for index, normalized in enumerate(normalized_predicted):
        local_gauge[index, :, 3:6] = -_skew(normalized)
        local_gauge[index, :, 6] = normalized

    full_gauge: np.ndarray = np.zeros(
        (row_count, 3, gauge_count * 7), dtype=np.float64
    )
    for index, gauge_index in enumerate(gauge_indices):
        block = slice(int(gauge_index) * 7, (int(gauge_index) + 1) * 7)
        full_gauge[index, :, block] = local_gauge[index]
    prior_id = _canonical_sha256(
        {
            "schema": "bayesian-phystwin.deform360-joint-sparse-gauge-tree-v5",
            "schema_version": MATERIALIZATION_VERSION,
            "gauge_ids": list(gauge_ids),
            "parent_indices_sha256": _array_sha256(parent_indices),
            "transition_matrices_sha256": _array_sha256(transitions),
            "innovation_scale_tril_sha256": _array_sha256(scales),
            "fit_artifact_id": fit.fit_artifact_id,
        }
    )
    gauge_prior_covariance, _ = _tree_covariance(
        local_gauge=local_gauge,
        gauge_indices=gauge_indices,
        parent_indices=parent_indices,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        gauge_ids=gauge_ids,
        prior_id=prior_id,
    )

    boundary_reliability = fit.boundary_reliability_floor + (
        1.0 - fit.boundary_reliability_floor
    ) * (
        1.0
        - np.exp(-mask_distance / fit.boundary_reliability_scale_pixels)
    )
    overlap_reliability = np.exp(
        -0.5 * np.square(disagreement / fit.overlap_disagreement_scale_m)
    )
    prior_reliability = np.clip(
        confidence * boundary_reliability * overlap_reliability,
        0.0,
        1.0,
    )

    voxel = np.floor(predicted / cluster_size).astype(np.int64)
    spatial_clusters = tuple(
        hashlib.sha256(
            f"physical-world-voxel-v5:{x}:{y}:{z}".encode("ascii")
        ).hexdigest()
        for x, y, z in voxel
    )
    correlation_groups = tuple(
        hashlib.sha256(
            f"physical-frame-voxel-v5:{identifier}:{episode_id}:{int(frame)}:{cluster}".encode(
                "ascii"
            )
        ).hexdigest()
        for frame, cluster in zip(frames, spatial_clusters, strict=True)
    )
    composite = _group_composite_weight(
        correlation_groups,
        cap=effective_cap,
    )
    factor_ids = tuple(
        _canonical_sha256(
            {
                "schema": "bayesian-phystwin.deform360-joint-sparse-factor-v5",
                "schema_version": MATERIALIZATION_VERSION,
                "object_id": identifier,
                "episode_id": episode_id,
                "camera_id": camera,
                "window_id": window,
                "frame": int(frame),
                "point_world_m_sha256": _array_sha256(points[index]),
                "covariance_m2_sha256": _array_sha256(covariance[index]),
                "source_window_artifact_id": next(
                    value.artifact_id
                    for value in windows
                    if value.camera_id == camera and value.window_id == window
                ),
            }
        )
        for index, (camera, window, frame) in enumerate(
            zip(camera_ids, window_ids, frames, strict=True)
        )
    )

    cameras = tuple(sorted(set(camera_ids)))
    camera_index = {camera: index for index, camera in enumerate(cameras)}
    shared_bias = np.broadcast_to(np.eye(3), (row_count, 3, 3)).copy()
    view_bias: np.ndarray = np.zeros(
        (row_count, 3, 3 * len(cameras)), dtype=np.float64
    )
    for index, camera in enumerate(camera_ids):
        block = slice(3 * camera_index[camera], 3 * (camera_index[camera] + 1))
        view_bias[index, :, block] = np.eye(3)

    future_state_jacobian = np.zeros_like(state_basis)
    evaluation_start, evaluation_stop = evaluation_frame_range_half_open
    future_state_jacobian[evaluation_start:evaluation_stop] = state_basis[
        evaluation_start:evaluation_stop
    ]
    query = future_state_jacobian[evaluation_start:evaluation_stop].reshape(
        -1, 3, state_count
    )

    anchor_innovation: np.ndarray | None = None
    anchor_covariance: np.ndarray | None = None
    anchor_state: np.ndarray | None = None
    anchor_groups: tuple[str, ...] | None = None
    anchor_reliability: np.ndarray | None = None
    anchor_composite: np.ndarray | None = None
    anchor_bias: np.ndarray | None = None
    anchor_bias_prior: np.ndarray | None = None
    contact_count = 0
    contact_sources: Mapping[str, str] = {}
    if contact_rows is not None:
        if not isinstance(contact_rows, Deform360JointSparseContactRowsV5):
            raise TypeError("contact_rows has the wrong type")
        _require(
            np.all(contact_rows.frame_indices < causal_frame_stop),
            "contact rows leave the causal prefix",
        )
        _require(
            np.all(contact_rows.graph_node_indices < node_count),
            "contact row references an unknown graph node",
        )
        contact_count = len(contact_rows.frame_indices)
        contact_candidate = np.asarray(physical, dtype=np.float64)[
            contact_rows.frame_indices[:, None],
            contact_rows.graph_node_indices,
        ]
        contact_predicted = np.sum(
            contact_rows.graph_node_weights[..., None] * contact_candidate,
            axis=1,
        )
        anchor_innovation = contact_rows.observed_point_world_m - contact_predicted
        anchor_covariance = (
            contact_rows.covariance_m2
            + fit.observation_variance_floor_m2 * np.eye(3)[None]
        )
        anchor_state = np.sum(
            contact_rows.graph_node_weights[:, :, None, None]
            * state_basis[
                contact_rows.frame_indices[:, None],
                contact_rows.graph_node_indices,
            ],
            axis=1,
        )
        anchor_groups = contact_rows.correlation_group_ids
        anchor_reliability = contact_rows.prior_reliability
        anchor_composite = _group_composite_weight(anchor_groups, cap=1.0)
        anchor_bias = np.broadcast_to(
            np.eye(3),
            (contact_count, 3, 3),
        ).copy()
        anchor_bias_prior = (
            fit.contact_anchor_bias_std_m**2 * np.eye(3)
        )
        contact_sources = contact_rows.source_artifact_ids

    all_sources = _merge_sources(
        source_artifact_ids,
        fit.source_artifact_ids,
        *(value.source_artifact_ids for value in windows),
        contact_sources,
    )
    metadata = {
        "schema": "bayesian-phystwin.deform360-joint-sparse-observation-metadata",
        "schema_version": MATERIALIZATION_VERSION,
        "fit_artifact_id": fit.fit_artifact_id,
        "implementation_revision": revision,
        "factor_ids": list(factor_ids),
        "camera_ids": list(camera_ids),
        "window_ids": list(window_ids),
        "spatial_cluster_ids": list(spatial_clusters),
        "gauge_ids": list(gauge_ids),
        "gauge_prior_id": prior_id,
        "association_entropy_sha256": _array_sha256(association_entropy),
        "contributor_count_sha256": _array_sha256(contributors),
        "prior_reliability_definition": (
            "source-confidence-times-mask-distance-times-overlap-consistency-v1"
        ),
        "prior_reliability_uses_physical_innovation": False,
        "association_probability_definition": (
            "candidate-geometry-distance-and-entropy-generalized-Bayes-power-v1"
        ),
        "state_innovation_robust_processing_count": 1,
        "assignment_mixture_spread_in_covariance": True,
        "unknown_cross_view_correlation_treatment": (
            "camera-independent-physical-frame-voxel-group-power-cap-v1"
        ),
        "dense_pixel_independence_assumed": False,
        "metric_observation_covariance_units": "m^2",
        "future_object_observations_used": False,
        "development_suffix_outcomes_used": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "released_real_world_measurements_used": True,
        "human_approval_required": False,
        "new_measurements_required": False,
    }
    batch = GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=covariance,
        state_jacobian=state,
        gauge_jacobian=full_gauge,
        shared_bias_jacobian=shared_bias,
        view_bias_jacobian=view_bias,
        query_state_jacobian=query,
        gauge_prior_covariance=gauge_prior_covariance,
        correlation_group_ids=correlation_groups,
        prior_reliability=prior_reliability,
        prior_nominal_probability=np.full(
            row_count,
            fit.nominal_inlier_probability,
        ),
        composite_weight=composite,
        association_probability=association_probability,
        physical_response_scale_m=max(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            np.square(
                                physical[evaluation_start:evaluation_stop]
                                - persistence[evaluation_start:evaluation_stop]
                            ),
                            axis=2,
                        )
                    )
                )
            ),
            1e-9,
        ),
        state_prior_covariance_m2=fit.state_prior_std_m**2 * np.eye(state_count),
        anchor_innovation_m=anchor_innovation,
        anchor_covariance_m2=anchor_covariance,
        anchor_state_jacobian=anchor_state,
        anchor_correlation_group_ids=anchor_groups,
        anchor_prior_reliability=anchor_reliability,
        anchor_prior_nominal_probability=(
            None
            if contact_count == 0
            else np.full(contact_count, fit.nominal_inlier_probability)
        ),
        anchor_composite_weight=anchor_composite,
        anchor_bias_jacobian=anchor_bias,
        anchor_bias_prior_covariance=anchor_bias_prior,
        metadata=metadata,
    )

    nuisance = np.concatenate((full_gauge, shared_bias, view_bias), axis=2)
    nuisance_prior = np.zeros((nuisance.shape[2], nuisance.shape[2]), dtype=np.float64)
    gauge_stop = gauge_prior_covariance.shape[0]
    shared_stop = gauge_stop + 3
    nuisance_prior[:gauge_stop, :gauge_stop] = gauge_prior_covariance
    nuisance_prior[gauge_stop:shared_stop, gauge_stop:shared_stop] = (
        fit.shared_bias_prior_std_m**2 * np.eye(3)
    )
    nuisance_prior[shared_stop:, shared_stop:] = (
        fit.view_bias_prior_std_m**2 * np.eye(3 * len(cameras))
    )
    row_weight = _effective_row_weight(
        correlation_groups,
        prior_reliability,
        association_probability,
        composite,
        cap=effective_cap,
    )
    all_mask: np.ndarray = np.ones(row_count, dtype=np.bool_)

    def information(mask: np.ndarray) -> np.ndarray:
        return _marginal_information(
            covariance=covariance,
            state=state,
            nuisance=nuisance,
            nuisance_prior_covariance=nuisance_prior,
            row_weight=row_weight,
            mask=mask,
        )

    full_information = information(all_mask)
    query_rank, eigenvalues, condition = _rank(full_information)
    full_trace = float(np.trace(full_information))
    camera_array = np.asarray(camera_ids, dtype=object)
    window_array = np.asarray(window_ids, dtype=object)
    single_camera_fraction: list[float] = []
    leave_camera_fraction: list[float] = []
    for camera in cameras:
        selected = camera_array == camera
        only_rank, _, _ = _rank(information(selected))
        del only_rank
        only_trace = float(np.trace(information(selected)))
        single_camera_fraction.append(
            0.0 if full_trace <= 0.0 else min(1.0, only_trace / full_trace)
        )
        leave_rank, _, _ = _rank(information(~selected))
        leave_camera_fraction.append(leave_rank / state_count)
    leave_window_fraction: list[float] = []
    for window_name in sorted(set(window_ids)):
        leave_rank, _, _ = _rank(information(window_array != window_name))
        leave_window_fraction.append(leave_rank / state_count)
    maximum_camera = max(single_camera_fraction)
    minimum_leave_camera = min(leave_camera_fraction)
    minimum_leave_window = min(leave_window_fraction)
    checks = {
        "minimum_distinct_cameras": len(cameras) >= 2,
        "minimum_distinct_windows": len(set(window_ids)) >= 2,
        "minimum_distinct_spatial_clusters": len(set(spatial_clusters)) >= 8,
        "query_rank": query_rank == state_count,
        "minimum_query_precision_eigenvalue": float(eigenvalues[0]) >= 1e-9,
        "maximum_query_condition_number": condition is not None
        and condition <= 1e10,
        "maximum_single_camera_information_fraction": maximum_camera <= 0.85,
        "minimum_leave_one_camera_rank_fraction": minimum_leave_camera >= 0.75,
        "minimum_leave_one_window_rank_fraction": minimum_leave_window >= 0.75,
    }
    admission_input_id = _canonical_sha256(
        {
            "schema": "bayesian-phystwin.deform360-joint-sparse-admission-input-v5",
            "schema_version": MATERIALIZATION_VERSION,
            "object_id": identifier,
            "episode_id": episode_id,
            "fit_artifact_id": fit.fit_artifact_id,
            "factor_ids": list(factor_ids),
            "covariance_sha256": _array_sha256(covariance),
            "state_jacobian_sha256": _array_sha256(state),
            "nuisance_jacobian_sha256": _array_sha256(nuisance),
            "nuisance_prior_covariance_sha256": _array_sha256(nuisance_prior),
            "row_weight_sha256": _array_sha256(row_weight),
            "implementation_revision": revision,
            "future_object_observations_used": False,
            "development_suffix_outcomes_used": False,
        }
    )
    admission = Deform360JointSparseAdmissionResultV5(
        gate_passed=all(checks.values()),
        checks=checks,
        factor_count=row_count,
        excluded_factor_count=int(np.count_nonzero(association_probability == 0.0)),
        distinct_camera_count=len(cameras),
        distinct_window_count=len(set(window_ids)),
        distinct_spatial_cluster_count=len(set(spatial_clusters)),
        distinct_correlation_group_count=len(set(correlation_groups)),
        query_rank=query_rank,
        query_precision_eigenvalues=tuple(map(float, eigenvalues)),
        query_condition_number=condition,
        maximum_single_camera_information_fraction=maximum_camera,
        minimum_leave_one_camera_rank_fraction=minimum_leave_camera,
        minimum_leave_one_window_rank_fraction=minimum_leave_window,
        effective_row_weight_sum=float(np.sum(row_weight)),
        input_id=admission_input_id,
    )
    problem_sources = _merge_sources(
        all_sources,
        {
            "fit/prefix-fit-v5.json": fit.fit_artifact_id,
            "admission/prefix-only-v5.json": admission.result_id,
        },
    )
    problem = Deform360JointSparsePredictionInputV5(
        object_id=identifier,
        episode_id=episode_id,
        stratum=stratum,
        physical_prediction_m=physical,
        persistence_m=persistence,
        last_causal_residual_m=last_causal_residual_m,
        future_state_jacobian_m=future_state_jacobian,
        observation_batch=batch,
        causal_frame_stop=causal_frame_stop,
        evaluation_frame_range_half_open=evaluation_frame_range_half_open,
        factor_admitted=admission.gate_passed,
        physical_mode=physical_mode,
        source_artifact_ids=problem_sources,
    )
    return Deform360JointSparseMaterializationResultV5(
        problem=problem,
        admission=admission,
        fit_artifact_id=fit.fit_artifact_id,
        visual_row_count=row_count,
        contact_row_count=contact_count,
        source_artifact_ids=problem_sources,
    )


__all__ = [
    "Deform360JointSparseAdmissionResultV5",
    "Deform360JointSparseContactRowsV5",
    "Deform360JointSparseExtractionConfigV5",
    "Deform360JointSparseMaterializationResultV5",
    "Deform360JointSparsePrefixFitV5",
    "Deform360JointSparseVisualWindowRowsV5",
    "extract_deform360_joint_sparse_visual_rows_v5",
    "materialize_deform360_joint_sparse_prediction_v5",
]
