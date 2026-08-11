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
    sha256_digest,
    source_artifact_mapping,
)
from .contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from .deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseVisualWindowRowsV5,
    associate_deform360_joint_sparse_geometry_v5,
)
from .endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    ModelAveragedEndpointConfigV1,
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
CAMERA_PARTITION_NAMESPACE_V1: Final = "deform360-covariance-camera-partition-v1"
ASSOCIATION_CANDIDATE_COUNT_V1: Final = 4
ASSOCIATION_SCALE_M_V1: Final = 0.010
MAXIMUM_ASSOCIATION_DISTANCE_M_V1: Final = 0.040
ASSOCIATION_ENTROPY_STRENGTH_V1: Final = 0.5
MINIMUM_EFFECTIVE_ROW_SUPPORT_V1: Final = 0.05
BOUNDARY_RELIABILITY_SCALE_PIXELS_V1: Final = 8.0
BOUNDARY_RELIABILITY_FLOOR_V1: Final = 0.25
OVERLAP_DISAGREEMENT_SCALE_M_V1: Final = 0.015
OBSERVATION_VARIANCE_FLOOR_M2_V1: Final = 4e-6


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
    observation_covariance_world_m2: np.ndarray
    prior_reliability: np.ndarray
    valid_mask: np.ndarray
    prefix_range_half_open: tuple[int, int]
    provider_camera_ids: tuple[str, ...]
    observation_split_artifact_id: str
    source_artifact_ids: Mapping[str, str]
    coordinate_frame: str = "deform360_world"
    position_units: str = "m"
    covariance_units: str = "m^2"

    def __post_init__(self) -> None:
        raw_frames = np.asarray(self.frame_indices)
        _require(raw_frames.dtype.kind in "iu", "frame_indices must be integers")
        frames = _readonly(raw_frames, dtype=np.int64)
        residual = _readonly(self.residual_world_m, dtype=np.float64)
        covariance = _readonly(
            self.observation_covariance_world_m2,
            dtype=np.float64,
        )
        reliability = _readonly(self.prior_reliability, dtype=np.float64)
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
        _validate_covariance(
            covariance,
            name="observation_covariance_world_m2",
        )
        _require(
            covariance.shape[:2] == residual.shape[:2],
            "observation covariance must match the residual history",
        )
        _require(
            reliability.shape == residual.shape[:2]
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior_reliability must match the history and lie in [0,1]",
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
        _require(
            np.all(covariance[~valid] == 0.0)
            and np.all(reliability[~valid] == 0.0),
            "invalid history entries cannot carry covariance or reliability evidence",
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
        object.__setattr__(self, "observation_covariance_world_m2", covariance)
        object.__setattr__(self, "prior_reliability", reliability)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "provider_camera_ids", cameras)
        object.__setattr__(
            self,
            "observation_split_artifact_id",
            sha256_digest(
                self.observation_split_artifact_id,
                name="observation_split_artifact_id",
            ),
        )
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
                "observation_covariance_world_m2_sha256": _array_sha256(
                    self.observation_covariance_world_m2
                ),
                "prior_reliability_sha256": _array_sha256(
                    self.prior_reliability
                ),
                "valid_mask_sha256": _array_sha256(self.valid_mask),
                "prefix_range_half_open": list(self.prefix_range_half_open),
                "provider_camera_ids": list(self.provider_camera_ids),
                "observation_split_artifact_id": (
                    self.observation_split_artifact_id
                ),
                "source_artifact_ids": dict(self.source_artifact_ids),
                "coordinate_frame": self.coordinate_frame,
                "position_units": self.position_units,
                "covariance_units": self.covariance_units,
                "missing_frame_policy": "explicit-invalid-zero-never-nearest-filled",
                "endpoint_noise_policy": (
                    "source-frozen-model-average-noise-grid-plus-metric-row-covariance;"
                    "cue-only-reliability-scales-row-covariance-once"
                ),
                "state_residual_used_for_prior_reliability": False,
                "association_probability_used_for_prior_reliability": False,
                "association_probability_used_for_assignment_and_admission": True,
                "row_covariance_used_by_endpoint_filter": True,
                "prior_reliability_used_by_endpoint_filter": True,
                "prior_reliability_application_count": 1,
                "innovation_robustification_count": 1,
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
        provider_artifact = sha256_digest(
            self.provider_reconstruction_artifact_id,
            name="provider_reconstruction_artifact_id",
        )
        scoring_artifact = sha256_digest(
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


def plan_deform360_camera_partition_v1(
    *,
    camera_ids: Sequence[str],
    object_session_hash: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition names only into deterministic disjoint provider/scoring views."""

    cameras = canonical_sorted_strings(camera_ids, name="camera_ids")
    _require(len(cameras) >= 4, "camera partition requires at least four cameras")
    session_hash = sha256_digest(object_session_hash, name="object_session_hash")
    ranked = sorted(
        cameras,
        key=lambda camera: hashlib.sha256(
            (
                f"{CAMERA_PARTITION_NAMESPACE_V1}\0{session_hash}\0{camera}"
            ).encode()
        ).digest(),
    )
    provider = tuple(sorted(ranked[::2]))
    scoring = tuple(sorted(ranked[1::2]))
    _require(
        len(provider) >= 2
        and len(scoring) >= 2
        and set(provider).isdisjoint(scoring),
        "camera partition did not produce two disjoint multiview panels",
    )
    return provider, scoring


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
    observation_split_artifact_id: str
    scoring_reconstruction_artifact_id: str
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
        history_id = sha256_digest(self.history_artifact_id, name="history_artifact_id")
        split_id = sha256_digest(
            self.observation_split_artifact_id,
            name="observation_split_artifact_id",
        )
        scoring_id = sha256_digest(
            self.scoring_reconstruction_artifact_id,
            name="scoring_reconstruction_artifact_id",
        )
        expected_mean = sha256_digest(
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
        object.__setattr__(self, "observation_split_artifact_id", split_id)
        object.__setattr__(self, "scoring_reconstruction_artifact_id", scoring_id)
        object.__setattr__(self, "reference_mean_sha256", expected_mean)

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
                "observation_split_artifact_id": (
                    self.observation_split_artifact_id
                ),
                "scoring_reconstruction_artifact_id": (
                    self.scoring_reconstruction_artifact_id
                ),
                "reference_mean_sha256": self.reference_mean_sha256,
                "coordinate_frame": "deform360_world",
                "position_units": "m",
                "covariance_units": "m^2",
            }
        )


@dataclass(frozen=True, slots=True)
class _HeteroscedasticEndpointPosteriorV1:
    mean_m: np.ndarray
    component_mean_m: np.ndarray
    component_covariance_m2: np.ndarray
    component_weights: np.ndarray
    component_process_variance_m2: np.ndarray
    update_count: np.ndarray


def _project_psd(value: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (value + value.swapaxes(-1, -2))
    eigenvalue, eigenvector = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalue, 0.0)
    return np.einsum(
        "...ik,...k,...jk->...ij",
        eigenvector,
        clipped,
        eigenvector,
        optimize=True,
    )


def _gaussian_log_density(
    innovation: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    sign, log_determinant = np.linalg.slogdet(covariance)
    _require(np.all(sign > 0.0), "endpoint innovation covariance is not positive")
    solved = np.linalg.solve(covariance, innovation[..., None])[..., 0]
    mahalanobis = np.einsum("ni,ni->n", innovation, solved, optimize=True)
    return -0.5 * (
        3.0 * np.log(2.0 * np.pi) + log_determinant + mahalanobis
    )


def _kalman_update(
    mean: np.ndarray,
    predicted_covariance: np.ndarray,
    innovation: np.ndarray,
    observation_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    innovation_covariance = predicted_covariance + observation_covariance
    gain = np.linalg.solve(
        innovation_covariance,
        predicted_covariance,
    ).swapaxes(-1, -2)
    updated_mean = mean + np.einsum(
        "nij,nj->ni",
        gain,
        innovation,
        optimize=True,
    )
    identity_minus_gain = np.eye(3)[None] - gain
    updated_covariance = (
        identity_minus_gain
        @ predicted_covariance
        @ identity_minus_gain.swapaxes(-1, -2)
        + gain @ observation_covariance @ gain.swapaxes(-1, -2)
    )
    return updated_mean, _project_psd(updated_covariance)


def _filter_heteroscedastic_component(
    residual_m: np.ndarray,
    valid: np.ndarray,
    observation_covariance_m2: np.ndarray,
    prior_reliability: np.ndarray,
    *,
    component: FixedBayesianAnchorConfigV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    track_count = residual_m.shape[1]
    identity = np.eye(3, dtype=np.float64)
    process_variance = float(component.process_std_m) ** 2
    observation_variance = float(component.observation_std_m) ** 2
    initial_variance = float(component.initial_std_m) ** 2
    mean = np.zeros((track_count, 3), dtype=np.float64)
    covariance = np.broadcast_to(
        initial_variance * identity,
        (track_count, 3, 3),
    ).copy()
    update_count = np.zeros(track_count, dtype=np.int64)
    log_evidence = np.zeros(track_count, dtype=np.float64)
    log_inlier_prior = np.log(float(component.inlier_prior))
    log_outlier_prior = np.log1p(-float(component.inlier_prior))
    outlier_multiplier = float(component.outlier_variance_multiplier)
    for frame in range(len(residual_m)):
        predicted_covariance = covariance + process_variance * identity[None]
        covariance = predicted_covariance
        mask = valid[frame] & (prior_reliability[frame] > 0.0)
        if not np.any(mask):
            continue
        reliability = prior_reliability[frame, mask]
        row_covariance = (
            observation_covariance_m2[frame, mask]
            + observation_variance * identity[None]
        ) / reliability[:, None, None]
        predicted = predicted_covariance[mask]
        innovation = residual_m[frame, mask] - mean[mask]
        inlier_innovation_covariance = predicted + row_covariance
        outlier_row_covariance = outlier_multiplier * row_covariance
        outlier_innovation_covariance = predicted + outlier_row_covariance
        log_inlier = log_inlier_prior + _gaussian_log_density(
            innovation,
            inlier_innovation_covariance,
        )
        log_outlier = log_outlier_prior + _gaussian_log_density(
            innovation,
            outlier_innovation_covariance,
        )
        log_mixture = np.logaddexp(log_inlier, log_outlier)
        inlier_probability = np.exp(log_inlier - log_mixture)
        inlier_mean, inlier_covariance = _kalman_update(
            mean[mask],
            predicted,
            innovation,
            row_covariance,
        )
        outlier_mean, outlier_covariance = _kalman_update(
            mean[mask],
            predicted,
            innovation,
            outlier_row_covariance,
        )
        updated_mean = (
            inlier_probability[:, None] * inlier_mean
            + (1.0 - inlier_probability)[:, None] * outlier_mean
        )
        inlier_offset = inlier_mean - updated_mean
        outlier_offset = outlier_mean - updated_mean
        updated_covariance = (
            inlier_probability[:, None, None]
            * (
                inlier_covariance
                + inlier_offset[..., :, None] * inlier_offset[..., None, :]
            )
            + (1.0 - inlier_probability)[:, None, None]
            * (
                outlier_covariance
                + outlier_offset[..., :, None] * outlier_offset[..., None, :]
            )
        )
        mean[mask] = updated_mean
        covariance[mask] = _project_psd(updated_covariance)
        update_count[mask] += 1
        log_evidence[mask] += log_mixture
    return mean, covariance, update_count, log_evidence


def _infer_heteroscedastic_endpoint(
    history: Deform360CausalResidualHistoryV1,
    *,
    config: ModelAveragedEndpointConfigV1 | None,
) -> _HeteroscedasticEndpointPosteriorV1:
    settings = (
        DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1 if config is None else config
    )
    if not isinstance(settings, ModelAveragedEndpointConfigV1):
        raise TypeError("config must be a ModelAveragedEndpointConfigV1")
    component_count = len(settings.components)
    track_count = len(history.material_identity_ids)
    component_mean = np.empty((component_count, track_count, 3))
    component_covariance = np.empty((component_count, track_count, 3, 3))
    component_evidence = np.empty((track_count, component_count))
    component_process_variance = np.empty(component_count)
    common_update_count: np.ndarray | None = None
    for index, component in enumerate(settings.components):
        (
            component_mean[index],
            component_covariance[index],
            update_count,
            component_evidence[:, index],
        ) = _filter_heteroscedastic_component(
            history.residual_world_m,
            history.valid_mask,
            history.observation_covariance_world_m2,
            history.prior_reliability,
            component=component,
        )
        component_process_variance[index] = component.process_std_m**2
        if common_update_count is None:
            common_update_count = update_count
        elif not np.array_equal(common_update_count, update_count):
            raise AssertionError("endpoint components used different observations")
    assert common_update_count is not None
    log_prior = np.log(
        np.asarray(settings.component_prior_probability, dtype=np.float64)
    )
    unnormalized = component_evidence + log_prior[None]
    normalizer = np.logaddexp.reduce(unnormalized, axis=1)
    weights = np.exp(unnormalized - normalizer[:, None])
    mean = np.einsum("nk,knc->nc", weights, component_mean, optimize=True)
    return _HeteroscedasticEndpointPosteriorV1(
        mean_m=mean,
        component_mean_m=component_mean,
        component_covariance_m2=component_covariance,
        component_weights=weights,
        component_process_variance_m2=component_process_variance,
        update_count=common_update_count,
    )


def _predict_heteroscedastic_covariance(
    posterior: _HeteroscedasticEndpointPosteriorV1,
    *,
    horizon_steps: int,
) -> np.ndarray:
    component_covariance = (
        posterior.component_covariance_m2
        + horizon_steps
        * posterior.component_process_variance_m2[:, None, None, None]
        * np.eye(3)[None, None]
    )
    centered = posterior.component_mean_m - posterior.mean_m[None]
    outer = centered[..., :, None] * centered[..., None, :]
    covariance = np.einsum(
        "nk,knij->nij",
        posterior.component_weights,
        component_covariance + outer,
        optimize=True,
    )
    return _project_psd(covariance)


def estimate_deform360_causal_residual_history_v1(
    *,
    visual_windows: Sequence[Deform360JointSparseVisualWindowRowsV5],
    physical_prediction_world_m: object,
    frame_indices: object,
    material_identity_ids: Sequence[str],
    observation_split: Deform360ObservationSplitV1,
    source_artifact_ids: Mapping[str, str],
    association_candidate_count: int = ASSOCIATION_CANDIDATE_COUNT_V1,
    association_scale_m: float = ASSOCIATION_SCALE_M_V1,
    maximum_association_distance_m: float = MAXIMUM_ASSOCIATION_DISTANCE_M_V1,
    association_entropy_strength: float = ASSOCIATION_ENTROPY_STRENGTH_V1,
    minimum_effective_row_support: float = MINIMUM_EFFECTIVE_ROW_SUPPORT_V1,
) -> Deform360CausalResidualHistoryV1:
    """Build an admissible cue-weighted history without clipping innovations."""

    if not isinstance(observation_split, Deform360ObservationSplitV1):
        raise TypeError("observation_split must be a Deform360ObservationSplitV1")
    windows = tuple(visual_windows)
    _require(
        bool(windows)
        and all(
            isinstance(window, Deform360JointSparseVisualWindowRowsV5)
            for window in windows
        ),
        "visual_windows must contain validated v5 rows",
    )
    keys = tuple((window.camera_id, window.window_id) for window in windows)
    _require(len(set(keys)) == len(keys), "visual camera/window repeats")
    windows = tuple(
        sorted(
            windows,
            key=lambda window: (
                window.camera_id,
                int(np.min(window.frame_indices)),
                window.window_id,
            ),
        )
    )
    provider_cameras = tuple(sorted({window.camera_id for window in windows}))
    _require(
        provider_cameras == observation_split.provider_camera_ids,
        "visual windows do not match the registered provider camera panel",
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
        ("maximum_association_distance_m", maximum_association_distance_m),
        ("minimum_effective_row_support", minimum_effective_row_support),
    ):
        _require(
            not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and value > 0.0,
            f"{name} must be finite and positive",
        )
    _require(
        not isinstance(association_entropy_strength, (bool, np.bool_))
        and np.isfinite(association_entropy_strength)
        and association_entropy_strength >= 0.0,
        "association_entropy_strength must be finite and nonnegative",
    )
    _require(
        minimum_effective_row_support <= 1.0,
        "minimum_effective_row_support must not exceed one",
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
    covariance = np.zeros((*physical.shape[:2], 3, 3), dtype=np.float64)
    reliability = np.zeros(physical.shape[:2], dtype=np.float64)
    valid = np.zeros(physical.shape[:2], dtype=np.bool_)
    for frame, local_index in frame_to_local.items():
        selected_windows = [
            (window, np.flatnonzero(window.frame_indices == frame))
            for window in windows
            if np.any(window.frame_indices == frame)
        ]
        if not selected_windows:
            continue
        points = np.concatenate(
            [window.point_world_m[selected] for window, selected in selected_windows],
            axis=0,
        )
        source_covariance = np.concatenate(
            [
                window.point_covariance_m2[selected]
                for window, selected in selected_windows
            ],
            axis=0,
        )
        source_confidence = np.concatenate(
            [window.source_confidence[selected] for window, selected in selected_windows]
        )
        mask_distance = np.concatenate(
            [window.mask_distance_pixels[selected] for window, selected in selected_windows]
        )
        disagreement = np.concatenate(
            [
                window.overlap_disagreement_m[selected]
                for window, selected in selected_windows
            ]
        )
        contributors = np.concatenate(
            [window.contributor_count[selected] for window, selected in selected_windows]
        )
        reference = physical[local_index]
        indices, assignment, _, association_probability = (
            associate_deform360_joint_sparse_geometry_v5(
                reference,
                points,
                candidate_count=association_candidate_count,
                scale_m=association_scale_m,
                maximum_distance_m=maximum_association_distance_m,
                entropy_strength=association_entropy_strength,
            )
        )
        candidate_count = indices.shape[1]
        candidate_points = reference[indices]
        predicted = np.sum(assignment[..., None] * candidate_points, axis=1)
        offset = candidate_points - predicted[:, None]
        assignment_covariance = np.einsum(
            "mk,mki,mkj->mij",
            assignment,
            offset,
            offset,
            optimize=True,
        )
        row_covariance = (
            source_covariance
            + assignment_covariance
            + OBSERVATION_VARIANCE_FLOOR_M2_V1 * np.eye(3)[None]
        )
        boundary_reliability = BOUNDARY_RELIABILITY_FLOOR_V1 + (
            1.0 - BOUNDARY_RELIABILITY_FLOOR_V1
        ) * (
            1.0
            - np.exp(-mask_distance / BOUNDARY_RELIABILITY_SCALE_PIXELS_V1)
        )
        overlap_reliability = np.exp(
            -0.5
            * np.square(disagreement / OVERLAP_DISAGREEMENT_SCALE_M_V1)
        )
        # Contributor count never increases confidence when overlap correlation
        # is unknown. The penalty is residual-independent and duplicate-safe.
        correlation_penalty = 1.0 / np.sqrt(contributors.astype(np.float64))
        cue_reliability = np.clip(
            source_confidence
            * boundary_reliability
            * overlap_reliability
            * correlation_penalty,
            0.0,
            1.0,
        )
        association_contribution = (
            assignment * association_probability[:, None]
        )
        effective_support = association_contribution * cue_reliability[:, None]
        candidate_admitted = effective_support >= minimum_effective_row_support
        contribution = np.where(
            candidate_admitted,
            association_contribution,
            0.0,
        )
        candidate_residual = points[:, None, :] - candidate_points
        candidate_second_moment = (
            row_covariance[:, None]
            + candidate_residual[..., :, None]
            * candidate_residual[..., None, :]
        )
        numerator = np.zeros_like(reference)
        denominator: np.ndarray = np.zeros(len(reference), dtype=np.float64)
        second_moment: np.ndarray = np.zeros(
            (len(reference), 3, 3),
            dtype=np.float64,
        )
        maximum_effective_support: np.ndarray = np.zeros(
            len(reference),
            dtype=np.float64,
        )
        maximum_prior_reliability: np.ndarray = np.zeros(
            len(reference),
            dtype=np.float64,
        )
        for candidate in range(candidate_count):
            np.add.at(
                numerator,
                indices[:, candidate],
                contribution[:, candidate, None]
                * candidate_residual[:, candidate],
            )
            np.add.at(
                denominator,
                indices[:, candidate],
                contribution[:, candidate],
            )
            np.add.at(
                second_moment,
                indices[:, candidate],
                contribution[:, candidate, None, None]
                * candidate_second_moment[:, candidate],
            )
            np.maximum.at(
                maximum_effective_support,
                indices[:, candidate],
                effective_support[:, candidate],
            )
            np.maximum.at(
                maximum_prior_reliability,
                indices[:, candidate],
                np.where(
                    candidate_admitted[:, candidate],
                    cue_reliability,
                    0.0,
                ),
            )
        admitted = (
            (denominator > 0.0)
            & (maximum_effective_support >= minimum_effective_row_support)
        )
        residual[local_index, admitted] = (
            numerator[admitted] / denominator[admitted, None]
        )
        covariance[local_index, admitted] = (
            second_moment[admitted] / denominator[admitted, None, None]
            - residual[local_index, admitted, :, None]
            * residual[local_index, admitted, None, :]
        )
        covariance[local_index, admitted] = 0.5 * (
            covariance[local_index, admitted]
            + covariance[local_index, admitted].swapaxes(-1, -2)
        )
        minimum_eigenvalue = np.linalg.eigvalsh(
            covariance[local_index, admitted]
        )[:, 0]
        adjustment = np.maximum(-minimum_eigenvalue, 0.0) + 1e-15
        covariance[local_index, admitted] += adjustment[:, None, None] * np.eye(3)
        # Geometry determines association and admission only. Once admitted, the
        # stored perception reliability is made exclusively from source cues.
        reliability[local_index, admitted] = maximum_prior_reliability[admitted]
        valid[local_index] = admitted
    sources = dict(source_artifact_ids)
    for key, digest in (
        ("observation-split/v1", observation_split.artifact_id),
        (
            "reconstruction/provider",
            observation_split.provider_reconstruction_artifact_id,
        ),
    ):
        _require(key not in sources, f"{key} source key conflicts")
        sources[key] = digest
    for index, window in enumerate(windows):
        key = f"visual-window/{index:04d}"
        _require(key not in sources, "visual-window source key conflicts")
        sources[key] = window.artifact_id
    return Deform360CausalResidualHistoryV1(
        frame_indices=frames,
        material_identity_ids=identities,
        residual_world_m=residual,
        observation_covariance_world_m2=covariance,
        prior_reliability=reliability,
        valid_mask=valid,
        prefix_range_half_open=(int(frames[0]), int(frames[-1]) + 1),
        provider_camera_ids=provider_cameras,
        observation_split_artifact_id=observation_split.artifact_id,
        source_artifact_ids=sources,
    )


def build_deform360_covariance_only_forecast_v1(
    *,
    reference_mean_world_m: object,
    fallback_covariance_world_m2: object,
    future_frame_indices: object,
    horizon_labels: Sequence[str],
    history: Deform360CausalResidualHistoryV1,
    observation_split: Deform360ObservationSplitV1,
    registered_reference_mean_sha256: str,
    support_config: Deform360CovarianceDonorSupportConfigV1 | None = None,
    endpoint_config: ModelAveragedEndpointConfigV1 | None = None,
) -> Deform360CovarianceOnlyForecastV1:
    """Apply empirical covariance only where the frozen support gate permits it."""

    if not isinstance(history, Deform360CausalResidualHistoryV1):
        raise TypeError("history must be a Deform360CausalResidualHistoryV1")
    if not isinstance(observation_split, Deform360ObservationSplitV1):
        raise TypeError("observation_split must be a Deform360ObservationSplitV1")
    _require(
        history.observation_split_artifact_id == observation_split.artifact_id,
        "history is not bound to the registered observation split",
    )
    _require(
        history.provider_camera_ids == observation_split.provider_camera_ids,
        "history provider cameras changed after split registration",
    )
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
    reference_mean_sha256 = sha256_digest(
        registered_reference_mean_sha256,
        name="registered_reference_mean_sha256",
    )
    _require(
        _array_sha256(raw_mean) == reference_mean_sha256,
        "reference mean bytes do not match the registered baseline digest",
    )
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
    posterior = _infer_heteroscedastic_endpoint(
        history,
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
            prediction_covariance = _predict_heteroscedastic_covariance(
                posterior,
                horizon_steps=int(frame) - prefix_last,
            )
            covariance[index, empirical] = (
                HORIZON_COVARIANCE_MULTIPLIER_V1[label]
                * prediction_covariance[empirical]
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
        observation_split_artifact_id=observation_split.artifact_id,
        scoring_reconstruction_artifact_id=(
            observation_split.scoring_reconstruction_artifact_id
        ),
        reference_mean_sha256=reference_mean_sha256,
    )


__all__ = [
    "ASSOCIATION_CANDIDATE_COUNT_V1",
    "ASSOCIATION_ENTROPY_STRENGTH_V1",
    "ASSOCIATION_SCALE_M_V1",
    "BOUNDARY_RELIABILITY_FLOOR_V1",
    "BOUNDARY_RELIABILITY_SCALE_PIXELS_V1",
    "DEFORM360_COVARIANCE_PROVIDER_SCHEMA",
    "DEFORM360_COVARIANCE_PROVIDER_VERSION",
    "HORIZON_COVARIANCE_MULTIPLIER_V1",
    "MAXIMUM_ASSOCIATION_DISTANCE_M_V1",
    "MINIMUM_EFFECTIVE_ROW_SUPPORT_V1",
    "OBSERVATION_VARIANCE_FLOOR_M2_V1",
    "OVERLAP_DISAGREEMENT_SCALE_M_V1",
    "Deform360CausalResidualHistoryV1",
    "Deform360CovarianceDonorSupportConfigV1",
    "Deform360CovarianceOnlyForecastV1",
    "Deform360ObservationSplitV1",
    "build_deform360_covariance_only_forecast_v1",
    "estimate_deform360_causal_residual_history_v1",
    "plan_deform360_camera_partition_v1",
]
