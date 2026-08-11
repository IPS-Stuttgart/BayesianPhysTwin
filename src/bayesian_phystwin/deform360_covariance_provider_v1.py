"""Source-frozen covariance-only provider contracts for fresh Deform360 data.

The adapter in this module is intentionally separate from the v5 last-residual
point estimator.  The latter returns one spatially completed endpoint.  The
endpoint covariance model instead needs an identity-aligned causal history with
explicit missingness, and must not interpret a prior-only track as empirical
donor evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    nonempty_string,
    source_artifact_mapping,
)
from .deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseVisualWindowRowsV5,
)
from .endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

DEFORM360_COVARIANCE_PROVIDER_VERSION: Final = 1
DEFORM360_COVARIANCE_PROVIDER_SCHEMA: Final = (
    "bayesian-phystwin/deform360-covariance-only-provider-v1"
)
HORIZON_COVARIANCE_MULTIPLIER_V1: Final = {
    "early": 8.0,
    "middle": 16.0,
    "late": 16.0,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _readonly(value: object, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _identity_ids(values: Sequence[str], *, expected_count: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("material_identity_ids must be a sequence of strings")
    result = tuple(values)
    _require(
        len(result) == expected_count
        and len(set(result)) == len(result)
        and all(type(value) is str and value and value.strip() == value for value in result),
        "material identity order must contain unique nonempty strings",
    )
    return result


def _validate_covariance(value: np.ndarray, *, name: str) -> None:
    _require(
        value.ndim == 4 and value.shape[-2:] == (3, 3),
        f"{name} must have shape (H,N,3,3)",
    )
    _require(np.all(np.isfinite(value)), f"{name} must be finite")
    _require(
        np.allclose(value, value.swapaxes(-1, -2), atol=1e-12, rtol=1e-12),
        f"{name} must be symmetric",
    )
    eigenvalues = np.linalg.eigvalsh(value)
    _require(
        np.min(eigenvalues, initial=0.0) >= -1e-12,
        f"{name} must be positive semidefinite",
    )


@dataclass(frozen=True, slots=True)
class Deform360CausalResidualHistoryV1:
    """Identity-aligned causal residuals with missing frames left unobserved."""

    frame_indices: np.ndarray
    material_identity_ids: tuple[str, ...]
    residual_world_m: np.ndarray
    valid_mask: np.ndarray
    prefix_range_half_open: tuple[int, int]
    provider_camera_ids: tuple[str, ...]
    source_artifact_ids: Mapping[str, str]
    coordinate_frame: str = "deform360_world"
    position_units: str = "m"
    covariance_units: str = "m^2"

    def __post_init__(self) -> None:
        raw_frames = np.asarray(self.frame_indices)
        _require(raw_frames.dtype.kind in "iu", "frame_indices must be integers")
        frames = _readonly(raw_frames, dtype=np.int64)
        residual = _readonly(self.residual_world_m, dtype=np.float64)
        raw_valid = np.asarray(self.valid_mask)
        _require(raw_valid.dtype.kind == "b", "valid_mask must be boolean")
        valid = _readonly(raw_valid, dtype=np.bool_)
        _require(
            residual.ndim == 3 and residual.shape[2] == 3,
            "residual_world_m must have shape (T,N,3)",
        )
        _require(
            valid.shape == residual.shape[:2],
            "valid_mask must match the residual frame and identity dimensions",
        )
        _require(
            frames.shape == (len(residual),),
            "frame_indices must match the residual history length",
        )
        _require(np.all(np.isfinite(residual)), "residual history must be finite")
        _require(
            np.all(residual[~valid] == 0.0),
            "invalid residual entries must be explicit zero, never filled observations",
        )
        start, stop = self.prefix_range_half_open
        _require(
            type(start) is int and type(stop) is int and 0 <= start < stop,
            "prefix_range_half_open is invalid",
        )
        _require(
            np.array_equal(frames, np.arange(start, stop, dtype=np.int64)),
            "residual history must bind every causal prefix frame exactly once",
        )
        identities = _identity_ids(
            self.material_identity_ids,
            expected_count=residual.shape[1],
        )
        cameras = canonical_sorted_strings(
            self.provider_camera_ids,
            name="provider_camera_ids",
        )
        _require(
            self.coordinate_frame == "deform360_world",
            "coordinate_frame must be deform360_world",
        )
        _require(self.position_units == "m", "position_units must be metres")
        _require(
            self.covariance_units == "m^2",
            "covariance_units must be square metres",
        )
        sources = source_artifact_mapping(
            self.source_artifact_ids,
            name="residual-history source_artifact_ids",
        )
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "material_identity_ids", identities)
        object.__setattr__(self, "residual_world_m", residual)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "provider_camera_ids", cameras)
        object.__setattr__(self, "source_artifact_ids", sources)

    @property
    def artifact_id(self) -> str:
        return content_id(
            {
                "schema": DEFORM360_COVARIANCE_PROVIDER_SCHEMA,
                "schema_version": DEFORM360_COVARIANCE_PROVIDER_VERSION,
                "kind": "causal-residual-history",
                "frame_indices_sha256": _array_sha256(self.frame_indices),
                "material_identity_ids": list(self.material_identity_ids),
                "residual_world_m_sha256": _array_sha256(self.residual_world_m),
                "valid_mask_sha256": _array_sha256(self.valid_mask),
                "prefix_range_half_open": list(self.prefix_range_half_open),
                "provider_camera_ids": list(self.provider_camera_ids),
                "source_artifact_ids": dict(self.source_artifact_ids),
                "coordinate_frame": self.coordinate_frame,
                "position_units": self.position_units,
                "covariance_units": self.covariance_units,
                "missing_frame_policy": "explicit-invalid-zero-never-nearest-filled",
            }
        )


@dataclass(frozen=True, slots=True)
class Deform360CovarianceDonorSupportConfigV1:
    """Prospective case and identity support gate for empirical covariance."""

    minimum_observed_frame_count: int = 2
    minimum_updates_per_identity: int = 2
    minimum_empirical_identity_fraction: float = 0.5

    def __post_init__(self) -> None:
        for name in ("minimum_observed_frame_count", "minimum_updates_per_identity"):
            value = getattr(self, name)
            _require(type(value) is int and value >= 1, f"{name} must be positive")
        fraction = self.minimum_empirical_identity_fraction
        _require(
            not isinstance(fraction, (bool, np.bool_))
            and np.isfinite(fraction)
            and 0.0 < fraction <= 1.0,
            "minimum_empirical_identity_fraction must lie in (0,1]",
        )
        object.__setattr__(self, "minimum_empirical_identity_fraction", float(fraction))


@dataclass(frozen=True, slots=True)
class Deform360ObservationSplitV1:
    """Disjoint provider/scoring views and independently built reconstructions."""

    provider_camera_ids: tuple[str, ...]
    scoring_camera_ids: tuple[str, ...]
    provider_reconstruction_artifact_id: str
    scoring_reconstruction_artifact_id: str

    def __post_init__(self) -> None:
        provider = canonical_sorted_strings(
            self.provider_camera_ids,
            name="provider_camera_ids",
        )
        scoring = canonical_sorted_strings(
            self.scoring_camera_ids,
            name="scoring_camera_ids",
        )
        _require(
            set(provider).isdisjoint(scoring),
            "provider and scoring cameras must be disjoint",
        )
        provider_artifact = nonempty_string(
            self.provider_reconstruction_artifact_id,
            name="provider_reconstruction_artifact_id",
        )
        scoring_artifact = nonempty_string(
            self.scoring_reconstruction_artifact_id,
            name="scoring_reconstruction_artifact_id",
        )
        _require(
            provider_artifact != scoring_artifact,
            "provider and scoring reconstructions must be distinct artifacts",
        )
        object.__setattr__(self, "provider_camera_ids", provider)
        object.__setattr__(self, "scoring_camera_ids", scoring)
        object.__setattr__(self, "provider_reconstruction_artifact_id", provider_artifact)
        object.__setattr__(self, "scoring_reconstruction_artifact_id", scoring_artifact)

    @property
    def artifact_id(self) -> str:
        return content_id(
            {
                "schema": DEFORM360_COVARIANCE_PROVIDER_SCHEMA,
                "schema_version": DEFORM360_COVARIANCE_PROVIDER_VERSION,
                "kind": "observation-split",
                "provider_camera_ids": list(self.provider_camera_ids),
                "scoring_camera_ids": list(self.scoring_camera_ids),
                "provider_reconstruction_artifact_id": (
                    self.provider_reconstruction_artifact_id
                ),
                "scoring_reconstruction_artifact_id": (
                    self.scoring_reconstruction_artifact_id
                ),
                "shared_reconstruction_artifact": False,
            }
        )


@dataclass(frozen=True, slots=True)
class Deform360CovarianceOnlyForecastV1:
    """A last-residual mean with selectively admitted empirical covariance."""

    mean_world_m: np.ndarray
    covariance_world_m2: np.ndarray
    fallback_covariance_world_m2: np.ndarray
    future_frame_indices: np.ndarray
    horizon_labels: tuple[str, ...]
    update_count: np.ndarray
    empirical_donor_mask: np.ndarray
    prior_only_mask: np.ndarray
    case_donor_admitted: bool
    fallback_reason: str
    history_artifact_id: str
    reference_mean_sha256: str

    def __post_init__(self) -> None:
        mean = _readonly(self.mean_world_m)
        covariance = _readonly(self.covariance_world_m2)
        fallback = _readonly(self.fallback_covariance_world_m2)
        frames = _readonly(self.future_frame_indices, dtype=np.int64)
        count = _readonly(self.update_count, dtype=np.int64)
        empirical = _readonly(self.empirical_donor_mask, dtype=np.bool_)
        prior_only = _readonly(self.prior_only_mask, dtype=np.bool_)
        _require(
            mean.ndim == 3 and mean.shape[2] == 3 and np.all(np.isfinite(mean)),
            "mean_world_m must have finite shape (H,N,3)",
        )
        _validate_covariance(covariance, name="covariance_world_m2")
        _validate_covariance(fallback, name="fallback_covariance_world_m2")
        _require(
            covariance.shape[:2] == mean.shape[:2]
            and fallback.shape == covariance.shape,
            "forecast mean and covariance dimensions changed",
        )
        _require(
            frames.shape == (mean.shape[0],)
            and np.all(np.diff(frames) > 0),
            "future_frame_indices must be strictly increasing",
        )
        _require(
            len(self.horizon_labels) == mean.shape[0]
            and all(label in HORIZON_COVARIANCE_MULTIPLIER_V1 for label in self.horizon_labels),
            "horizon labels changed",
        )
        _require(
            count.shape == empirical.shape == prior_only.shape == (mean.shape[1],),
            "identity support vectors changed shape",
        )
        _require(
            np.all(count >= 0) and np.array_equal(prior_only, ~empirical),
            "prior-only and empirical donor identities must partition the tracks",
        )
        _require(type(self.case_donor_admitted) is bool, "case_donor_admitted must be bool")
        reason = nonempty_string(self.fallback_reason, name="fallback_reason")
        history_id = nonempty_string(self.history_artifact_id, name="history_artifact_id")
        expected_mean = nonempty_string(
            self.reference_mean_sha256,
            name="reference_mean_sha256",
        )
        _require(
            _array_sha256(mean) == expected_mean,
            "candidate mean is not byte-identical to the registered reference mean",
        )
        if not self.case_donor_admitted:
            _require(
                np.array_equal(covariance, fallback),
                "failed case support must return exact fallback covariance",
            )
            _require(not np.any(empirical), "failed case support cannot admit donor tracks")
        object.__setattr__(self, "mean_world_m", mean)
        object.__setattr__(self, "covariance_world_m2", covariance)
        object.__setattr__(self, "fallback_covariance_world_m2", fallback)
        object.__setattr__(self, "future_frame_indices", frames)
        object.__setattr__(self, "horizon_labels", tuple(self.horizon_labels))
        object.__setattr__(self, "update_count", count)
        object.__setattr__(self, "empirical_donor_mask", empirical)
        object.__setattr__(self, "prior_only_mask", prior_only)
        object.__setattr__(self, "fallback_reason", reason)
        object.__setattr__(self, "history_artifact_id", history_id)

    @property
    def artifact_id(self) -> str:
        return content_id(
            {
                "schema": DEFORM360_COVARIANCE_PROVIDER_SCHEMA,
                "schema_version": DEFORM360_COVARIANCE_PROVIDER_VERSION,
                "kind": "covariance-only-forecast",
                "mean_world_m_sha256": _array_sha256(self.mean_world_m),
                "covariance_world_m2_sha256": _array_sha256(
                    self.covariance_world_m2
                ),
                "fallback_covariance_world_m2_sha256": _array_sha256(
                    self.fallback_covariance_world_m2
                ),
                "future_frame_indices_sha256": _array_sha256(
                    self.future_frame_indices
                ),
                "horizon_labels": list(self.horizon_labels),
                "update_count_sha256": _array_sha256(self.update_count),
                "empirical_donor_mask_sha256": _array_sha256(
                    self.empirical_donor_mask
                ),
                "case_donor_admitted": self.case_donor_admitted,
                "fallback_reason": self.fallback_reason,
                "history_artifact_id": self.history_artifact_id,
                "coordinate_frame": "deform360_world",
                "position_units": "m",
                "covariance_units": "m^2",
            }
        )


def estimate_deform360_causal_residual_history_v1(
    *,
    visual_windows: Sequence[Deform360JointSparseVisualWindowRowsV5],
    physical_prediction_world_m: object,
    frame_indices: object,
    material_identity_ids: Sequence[str],
    source_artifact_ids: Mapping[str, str],
    association_candidate_count: int = 4,
    association_scale_m: float = 0.010,
    maximum_residual_m: float = 0.030,
) -> Deform360CausalResidualHistoryV1:
    """Build a complete causal residual series without filling missing frames."""

    windows = tuple(visual_windows)
    _require(
        bool(windows)
        and all(
            isinstance(window, Deform360JointSparseVisualWindowRowsV5)
            for window in windows
        ),
        "visual_windows must contain validated v5 rows",
    )
    raw_frames = np.asarray(frame_indices)
    _require(
        raw_frames.dtype.kind in "iu" and raw_frames.ndim == 1 and len(raw_frames) >= 1,
        "frame_indices must be a nonempty integer vector",
    )
    frames = np.asarray(raw_frames, dtype=np.int64)
    _require(
        np.array_equal(frames, np.arange(frames[0], frames[-1] + 1)),
        "frame_indices must be one complete causal range",
    )
    physical = np.asarray(physical_prediction_world_m, dtype=np.float64)
    _require(
        physical.ndim == 3
        and physical.shape[0] == len(frames)
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical)),
        "physical_prediction_world_m must have finite shape (T,N,3)",
    )
    identities = _identity_ids(material_identity_ids, expected_count=physical.shape[1])
    _require(
        type(association_candidate_count) is int and association_candidate_count >= 1,
        "association_candidate_count must be positive",
    )
    for name, value in (
        ("association_scale_m", association_scale_m),
        ("maximum_residual_m", maximum_residual_m),
    ):
        _require(
            not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and value > 0.0,
            f"{name} must be finite and positive",
        )
    frame_to_local = {int(frame): index for index, frame in enumerate(frames)}
    _require(
        all(
            all(int(frame) in frame_to_local for frame in window.frame_indices)
            for window in windows
        ),
        "visual row leaves the registered causal prefix",
    )
    residual = np.zeros_like(physical, dtype=np.float64)
    valid = np.zeros(physical.shape[:2], dtype=np.bool_)
    candidate_count = min(association_candidate_count, physical.shape[1])
    for frame, local_index in frame_to_local.items():
        selected_points = [
            np.asarray(window.point_world_m[window.frame_indices == frame])
            for window in windows
            if np.any(window.frame_indices == frame)
        ]
        if not selected_points:
            continue
        points = np.concatenate(selected_points, axis=0)
        reference = physical[local_index]
        squared = np.sum(np.square(points[:, None] - reference[None]), axis=2)
        indices = np.argpartition(
            squared,
            kth=candidate_count - 1,
            axis=1,
        )[:, :candidate_count]
        selected_squared = np.take_along_axis(squared, indices, axis=1)
        order = np.argsort(selected_squared, axis=1, kind="mergesort")
        indices = np.take_along_axis(indices, order, axis=1)
        selected_squared = np.take_along_axis(selected_squared, order, axis=1)
        logits = -0.5 * selected_squared / association_scale_m**2
        logits -= np.max(logits, axis=1, keepdims=True)
        assignment = np.exp(np.clip(logits, -700.0, 0.0))
        assignment /= np.sum(assignment, axis=1, keepdims=True)
        predicted = np.sum(assignment[..., None] * reference[indices], axis=1)
        row_residual = points - predicted
        numerator = np.zeros_like(reference)
        denominator = np.zeros(len(reference), dtype=np.float64)
        for candidate in range(candidate_count):
            np.add.at(
                numerator,
                indices[:, candidate],
                assignment[:, candidate, None] * row_residual,
            )
            np.add.at(denominator, indices[:, candidate], assignment[:, candidate])
        direct = denominator > 0.0
        residual[local_index, direct] = (
            numerator[direct] / denominator[direct, None]
        )
        norms = np.linalg.norm(residual[local_index, direct], axis=1)
        clipped = norms > maximum_residual_m
        if np.any(clipped):
            selected = np.flatnonzero(direct)[clipped]
            residual[local_index, selected] *= (
                maximum_residual_m / norms[clipped]
            )[:, None]
        valid[local_index] = direct
    sources = dict(source_artifact_ids)
    for index, window in enumerate(windows):
        key = f"visual-window/{index:04d}"
        _require(key not in sources, "visual-window source key conflicts")
        sources[key] = window.artifact_id
    return Deform360CausalResidualHistoryV1(
        frame_indices=frames,
        material_identity_ids=identities,
        residual_world_m=residual,
        valid_mask=valid,
        prefix_range_half_open=(int(frames[0]), int(frames[-1]) + 1),
        provider_camera_ids=tuple(sorted({window.camera_id for window in windows})),
        source_artifact_ids=sources,
    )


def build_deform360_covariance_only_forecast_v1(
    *,
    reference_mean_world_m: object,
    fallback_covariance_world_m2: object,
    future_frame_indices: object,
    horizon_labels: Sequence[str],
    history: Deform360CausalResidualHistoryV1,
    support_config: Deform360CovarianceDonorSupportConfigV1 | None = None,
    endpoint_config: ModelAveragedEndpointConfigV1 | None = None,
) -> Deform360CovarianceOnlyForecastV1:
    """Apply empirical covariance only where the frozen support gate permits it."""

    if not isinstance(history, Deform360CausalResidualHistoryV1):
        raise TypeError("history must be a Deform360CausalResidualHistoryV1")
    support = support_config or Deform360CovarianceDonorSupportConfigV1()
    if not isinstance(support, Deform360CovarianceDonorSupportConfigV1):
        raise TypeError("support_config must be a donor support config")
    raw_mean = np.asarray(reference_mean_world_m)
    _require(
        raw_mean.dtype.kind in "f"
        and raw_mean.ndim == 3
        and raw_mean.shape[1:] == (len(history.material_identity_ids), 3)
        and np.all(np.isfinite(raw_mean)),
        "reference_mean_world_m must have finite floating shape (H,N,3)",
    )
    reference_mean_sha256 = _array_sha256(raw_mean)
    mean = np.array(raw_mean, copy=True, order="C")
    raw_fallback = np.asarray(fallback_covariance_world_m2)
    _require(raw_fallback.dtype.kind == "f", "fallback covariance must be floating")
    fallback = np.array(raw_fallback, copy=True, order="C")
    _validate_covariance(fallback, name="fallback_covariance_world_m2")
    _require(
        fallback.shape[:2] == mean.shape[:2],
        "fallback covariance must match the reference mean",
    )
    raw_future = np.asarray(future_frame_indices)
    _require(raw_future.dtype.kind in "iu", "future_frame_indices must be integers")
    future = np.asarray(raw_future, dtype=np.int64)
    _require(
        future.shape == (len(mean),)
        and np.all(np.diff(future) > 0)
        and np.all(future >= history.prefix_range_half_open[1]),
        "future frames must be strictly increasing after the causal prefix",
    )
    labels = tuple(horizon_labels)
    _require(
        len(labels) == len(mean)
        and all(label in HORIZON_COVARIANCE_MULTIPLIER_V1 for label in labels),
        "horizon labels must be registered early/middle/late values",
    )
    posterior = infer_model_averaged_endpoint(
        history.residual_world_m,
        history.valid_mask,
        end_frame=len(history.frame_indices),
        config=endpoint_config,
    )
    empirical = posterior.update_count >= support.minimum_updates_per_identity
    observed_frame_count = int(np.sum(np.any(history.valid_mask, axis=1)))
    identity_fraction = float(np.mean(empirical))
    admitted = bool(
        observed_frame_count >= support.minimum_observed_frame_count
        and identity_fraction >= support.minimum_empirical_identity_fraction
    )
    covariance = np.array(fallback, copy=True, order="C")
    fallback_reason = "none"
    if admitted:
        prefix_last = history.prefix_range_half_open[1] - 1
        for index, (frame, label) in enumerate(zip(future, labels, strict=True)):
            prediction = predict_model_averaged_endpoint(
                posterior,
                horizon_steps=int(frame) - prefix_last,
            )
            covariance[index, empirical] = (
                HORIZON_COVARIANCE_MULTIPLIER_V1[label]
                * prediction.covariance_m2[empirical]
            )
    else:
        empirical = np.zeros_like(empirical)
        reasons: list[str] = []
        if observed_frame_count < support.minimum_observed_frame_count:
            reasons.append("insufficient-observed-frames")
        if identity_fraction < support.minimum_empirical_identity_fraction:
            reasons.append("insufficient-empirical-identities")
        fallback_reason = "+".join(reasons)
    _validate_covariance(covariance, name="candidate covariance")
    return Deform360CovarianceOnlyForecastV1(
        mean_world_m=mean,
        covariance_world_m2=covariance,
        fallback_covariance_world_m2=fallback,
        future_frame_indices=future,
        horizon_labels=labels,
        update_count=posterior.update_count,
        empirical_donor_mask=empirical,
        prior_only_mask=~empirical,
        case_donor_admitted=admitted,
        fallback_reason=fallback_reason,
        history_artifact_id=history.artifact_id,
        reference_mean_sha256=reference_mean_sha256,
    )


__all__ = [
    "DEFORM360_COVARIANCE_PROVIDER_SCHEMA",
    "DEFORM360_COVARIANCE_PROVIDER_VERSION",
    "HORIZON_COVARIANCE_MULTIPLIER_V1",
    "Deform360CausalResidualHistoryV1",
    "Deform360CovarianceDonorSupportConfigV1",
    "Deform360CovarianceOnlyForecastV1",
    "Deform360ObservationSplitV1",
    "build_deform360_covariance_only_forecast_v1",
    "estimate_deform360_causal_residual_history_v1",
]
