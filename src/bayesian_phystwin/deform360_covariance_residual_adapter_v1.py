"""Source-only Deform360 residual-history adapter for covariance-only forecasts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from .covariance_only_hybrid import compose_covariance_only_hybrid
from .endpoint_model_average import (
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

DEFORM360_RESIDUAL_ADAPTER_SCHEMA: Final = (
    "bayesian-phystwin/deform360-covariance-residual-adapter-v1"
)
DEFORM360_RESIDUAL_ADAPTER_VERSION: Final = 1
DEFAULT_HORIZON_SCALES: Final = (
    ("early", 8.0),
    ("middle", 16.0),
    ("late", 16.0),
)

CovarianceProvider = Callable[
    [np.ndarray, np.ndarray, tuple[int, ...]],
    np.ndarray,
]


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be one canonical line")
    return value


def _canonical_text_set(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty sequence")
    items = tuple(
        _canonical_text(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not items or len(set(items)) != len(items):
        raise ValueError(f"{name} must be nonempty and unique")
    return tuple(sorted(items))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _readonly(value: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class Deform360ResidualHistoryIdentityV1:
    """Identity and independence boundary for one source-only adapter call."""

    object_id: str
    session_id: str
    material_id: str
    coordinate_frame: str
    provider_camera_ids: Sequence[str]
    scoring_camera_ids: Sequence[str]
    provider_artifact_ids: Sequence[str]
    scoring_artifact_ids: Sequence[str]

    def __post_init__(self) -> None:
        for field_name in (
            "object_id",
            "session_id",
            "material_id",
            "coordinate_frame",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_text(getattr(self, field_name), name=field_name),
            )
        for field_name in (
            "provider_camera_ids",
            "scoring_camera_ids",
            "provider_artifact_ids",
            "scoring_artifact_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_text_set(getattr(self, field_name), name=field_name),
            )
        if set(self.provider_camera_ids) & set(self.scoring_camera_ids):
            raise ValueError("provider and scoring cameras must be disjoint")
        if set(self.provider_artifact_ids) & set(self.scoring_artifact_ids):
            raise ValueError("provider and scoring artifacts must be disjoint")

    def descriptor(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "session_id": self.session_id,
            "material_id": self.material_id,
            "coordinate_frame": self.coordinate_frame,
            "provider_camera_ids": list(self.provider_camera_ids),
            "scoring_camera_ids": list(self.scoring_camera_ids),
            "provider_artifact_ids": list(self.provider_artifact_ids),
            "scoring_artifact_ids": list(self.scoring_artifact_ids),
        }


@dataclass(frozen=True, slots=True)
class Deform360ResidualHistoryAdapterConfigV1:
    """Frozen source-only support and covariance scaling configuration."""

    minimum_valid_observations_per_track: int = 3
    horizon_scales: tuple[tuple[str, float], ...] = DEFAULT_HORIZON_SCALES
    covariance_psd_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        minimum = self.minimum_valid_observations_per_track
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("minimum_valid_observations_per_track must be positive")
        scales: list[tuple[str, float]] = []
        for index, pair in enumerate(self.horizon_scales):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"horizon_scales[{index}] must be a pair")
            label = _canonical_text(pair[0], name=f"horizon_scales[{index}].label")
            scale = float(pair[1])
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "horizon covariance scales must be finite and positive"
                )
            scales.append((label, scale))
        if tuple(label for label, _ in scales) != ("early", "middle", "late"):
            raise ValueError("horizon scales must retain early, middle, late order")
        tolerance = float(self.covariance_psd_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("covariance_psd_tolerance must be finite and positive")
        object.__setattr__(self, "horizon_scales", tuple(scales))
        object.__setattr__(self, "covariance_psd_tolerance", tolerance)

    @property
    def scale_by_label(self) -> Mapping[str, float]:
        return dict(self.horizon_scales)


@dataclass(frozen=True, slots=True)
class Deform360ResidualAdapterRecordV1:
    """Content-addressed proof of the adapter and its fallback boundary."""

    identity: Deform360ResidualHistoryIdentityV1
    residual_history_shape: tuple[int, int, int]
    future_shape: tuple[int, int, int]
    minimum_valid_observations_per_track: int
    valid_observation_count_by_track: tuple[int, ...]
    supported_track_count: int
    unsupported_track_count: int
    validity_sha256: str
    canonical_residual_sha256: str
    physical_future_sha256: str
    reference_mean_sha256: str
    output_covariance_sha256: str
    provider_status: str
    provider_error_type: str | None
    provider_error_sha256: str | None
    covariance_hybrid_artifact_id: str | None
    nearest_fill_used: bool
    validity_preserved: bool
    mean_object_identity_preserved: bool
    unsupported_tracks_use_physical_fallback: bool
    target_payload_opened: bool
    target_outcomes_opened: bool
    claim_authorized: bool
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        expected = _canonical_sha256(self.descriptor())
        if self.artifact_id is None:
            object.__setattr__(self, "artifact_id", expected)
        elif self.artifact_id != expected:
            raise ValueError("artifact_id does not match the adapter record")

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_RESIDUAL_ADAPTER_SCHEMA,
            "schema_version": DEFORM360_RESIDUAL_ADAPTER_VERSION,
            "identity": self.identity.descriptor(),
            "residual_history_shape": list(self.residual_history_shape),
            "future_shape": list(self.future_shape),
            "minimum_valid_observations_per_track": (
                self.minimum_valid_observations_per_track
            ),
            "valid_observation_count_by_track": list(
                self.valid_observation_count_by_track
            ),
            "supported_track_count": self.supported_track_count,
            "unsupported_track_count": self.unsupported_track_count,
            "validity_sha256": self.validity_sha256,
            "canonical_residual_sha256": self.canonical_residual_sha256,
            "physical_future_sha256": self.physical_future_sha256,
            "reference_mean_sha256": self.reference_mean_sha256,
            "output_covariance_sha256": self.output_covariance_sha256,
            "provider_status": self.provider_status,
            "provider_error_type": self.provider_error_type,
            "provider_error_sha256": self.provider_error_sha256,
            "covariance_hybrid_artifact_id": self.covariance_hybrid_artifact_id,
            "nearest_fill_used": self.nearest_fill_used,
            "validity_preserved": self.validity_preserved,
            "mean_object_identity_preserved": self.mean_object_identity_preserved,
            "unsupported_tracks_use_physical_fallback": (
                self.unsupported_tracks_use_physical_fallback
            ),
            "information_boundary": {
                "residual_unit": "m",
                "covariance_unit": "m2",
                "source_only": True,
                "target_payload_opened": self.target_payload_opened,
                "target_outcomes_opened": self.target_outcomes_opened,
                "claim_authorized": self.claim_authorized,
            },
        }


@dataclass(frozen=True, slots=True)
class Deform360ResidualAdapterPredictionV1:
    """Covariance-only prediction plus exact support/fallback evidence."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    supported_track_mask: np.ndarray
    record: Deform360ResidualAdapterRecordV1


def _validate_inputs(
    residual_history_m: object,
    valid: object,
    physical_future_m: np.ndarray,
    horizon_labels: Sequence[str],
    horizon_steps: Sequence[int],
    config: Deform360ResidualHistoryAdapterConfigV1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[int, ...],
]:
    raw_residual = np.asarray(residual_history_m)
    raw_valid = np.asarray(valid)
    if raw_residual.dtype.kind not in "iuf":
        raise ValueError("residual_history_m must contain real numeric values")
    if raw_residual.ndim != 3 or raw_residual.shape[-1] != 3:
        raise ValueError("residual_history_m must have shape (T, N, 3)")
    if raw_residual.shape[0] < 1 or raw_residual.shape[1] < 1:
        raise ValueError("residual_history_m must contain frames and tracks")
    if raw_valid.dtype.kind != "b" or raw_valid.shape != raw_residual.shape[:2]:
        raise ValueError("valid must be Boolean with shape (T, N)")
    validity = np.array(raw_valid, dtype=bool, copy=True, order="C")
    selected = np.asarray(raw_residual, dtype=np.float64)[validity]
    if not np.all(np.isfinite(selected)):
        raise ValueError("valid residual entries must be finite")
    residual = np.zeros(raw_residual.shape, dtype=np.float64)
    residual[validity] = selected

    if not isinstance(physical_future_m, np.ndarray):
        raise TypeError("physical_future_m must be a NumPy array")
    if physical_future_m.dtype != np.dtype(np.float64):
        raise ValueError("physical_future_m must have dtype float64")
    if not physical_future_m.flags.c_contiguous:
        raise ValueError("physical_future_m must be C-contiguous")
    if (
        physical_future_m.ndim != 3
        or physical_future_m.shape[1:] != raw_residual.shape[1:]
        or physical_future_m.shape[0] < 1
    ):
        raise ValueError("physical_future_m must have shape (H, N, 3)")
    if not np.all(np.isfinite(physical_future_m)):
        raise ValueError("physical_future_m must be finite")

    labels = tuple(
        _canonical_text(label, name=f"horizon_labels[{index}]")
        for index, label in enumerate(horizon_labels)
    )
    if len(labels) != len(physical_future_m):
        raise ValueError("horizon_labels must match the future frame count")
    scale_labels = tuple(label for label, _ in config.horizon_scales)
    if any(label not in scale_labels for label in labels):
        raise ValueError("horizon_labels contain an unregistered horizon")
    positions = tuple(scale_labels.index(label) for label in labels)
    if positions != tuple(sorted(positions)):
        raise ValueError("horizon_labels must be ordered early, middle, late")

    steps: list[int] = []
    for index, value in enumerate(horizon_steps):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"horizon_steps[{index}] must be an integer")
        step = int(value)
        if step < 1:
            raise ValueError("horizon_steps must be positive")
        steps.append(step)
    if len(steps) != len(physical_future_m):
        raise ValueError("horizon_steps must match the future frame count")
    if any(second <= first for first, second in zip(steps, steps[1:], strict=False)):
        raise ValueError("horizon_steps must be strictly increasing")
    return residual, validity, physical_future_m, labels, tuple(steps)


def _reference_mean(
    residual: np.ndarray,
    validity: np.ndarray,
    physical_future: np.ndarray,
    *,
    minimum_valid_observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.sum(validity, axis=0, dtype=np.int64)
    supported = counts >= minimum_valid_observations
    endpoint = np.zeros((residual.shape[1], 3), dtype=np.float64)
    for track in np.flatnonzero(supported):
        frames = np.flatnonzero(validity[:, track])
        endpoint[track] = residual[frames[-1], track]
    if not np.any(supported):
        return physical_future, supported, counts
    reference = np.array(physical_future, dtype=np.float64, copy=True, order="C")
    reference[:, supported, :] += endpoint[supported][None, :, :]
    return reference, supported, counts


def _endpoint_covariance_provider(
    residual: np.ndarray,
    validity: np.ndarray,
    horizon_steps: tuple[int, ...],
) -> np.ndarray:
    posterior = infer_model_averaged_endpoint(
        residual,
        validity,
        end_frame=len(residual),
    )
    return np.stack(
        [
            predict_model_averaged_endpoint(
                posterior,
                horizon_steps=step,
            ).covariance_m2
            for step in horizon_steps
        ],
        axis=0,
    )


def _validate_donor_covariance(
    value: object,
    *,
    expected_shape: tuple[int, int, int, int],
    tolerance: float,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("covariance provider returned nonnumeric values")
    covariance = np.asarray(raw, dtype=np.float64)
    if covariance.shape != expected_shape or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance provider returned an invalid shape or value")
    transpose = np.swapaxes(covariance, -1, -2)
    if not np.allclose(covariance, transpose, atol=tolerance, rtol=tolerance):
        raise ValueError("covariance provider returned a nonsymmetric covariance")
    symmetric = 0.5 * (covariance + transpose)
    if float(np.min(np.linalg.eigvalsh(symmetric), initial=0.0)) < -tolerance:
        raise ValueError("covariance provider returned a non-PSD covariance")
    return np.array(symmetric, dtype=np.float64, copy=True, order="C")


def adapt_deform360_covariance_residual_history_v1(
    residual_history_m: object,
    valid: object,
    physical_future_m: np.ndarray,
    *,
    horizon_labels: Sequence[str],
    horizon_steps: Sequence[int],
    identity: Deform360ResidualHistoryIdentityV1,
    config: Deform360ResidualHistoryAdapterConfigV1 | None = None,
    covariance_provider: CovarianceProvider | None = None,
) -> Deform360ResidualAdapterPredictionV1:
    """Adapt causal residual history without opening or scoring target outcomes."""

    if not isinstance(identity, Deform360ResidualHistoryIdentityV1):
        raise TypeError("identity must be a Deform360ResidualHistoryIdentityV1")
    settings = (
        Deform360ResidualHistoryAdapterConfigV1() if config is None else config
    )
    if not isinstance(settings, Deform360ResidualHistoryAdapterConfigV1):
        raise TypeError("config must be a Deform360ResidualHistoryAdapterConfigV1")
    residual, validity, physical_future, labels, steps = _validate_inputs(
        residual_history_m,
        valid,
        physical_future_m,
        horizon_labels,
        horizon_steps,
        settings,
    )
    reference, supported, counts = _reference_mean(
        residual,
        validity,
        physical_future,
        minimum_valid_observations=settings.minimum_valid_observations_per_track,
    )
    expected_covariance_shape = (
        int(reference.shape[0]),
        int(reference.shape[1]),
        int(reference.shape[2]),
        3,
    )
    scale_by_label = settings.scale_by_label
    schedule = np.asarray(
        [scale_by_label[label] for label in labels],
        dtype=np.float64,
    )[:, None]
    provider = (
        _endpoint_covariance_provider
        if covariance_provider is None
        else covariance_provider
    )
    provider_status = "success"
    provider_error_type: str | None = None
    provider_error_sha256: str | None = None
    hybrid_artifact_id: str | None = None
    mean = reference

    if not np.any(supported):
        provider_status = "fallback-no-supported-tracks"
        covariance = _readonly(
            np.zeros(expected_covariance_shape, dtype=np.float64),
            dtype=np.float64,
        )
    else:
        try:
            provider_validity = validity & supported[None, :]
            provider_residual = np.zeros_like(residual)
            provider_residual[provider_validity] = residual[provider_validity]
            donor = _validate_donor_covariance(
                provider(provider_residual, provider_validity, steps),
                expected_shape=expected_covariance_shape,
                tolerance=settings.covariance_psd_tolerance,
            )
            donor[:, ~supported, :, :] = 0.0
            hybrid = compose_covariance_only_hybrid(
                reference,
                donor,
                reference_predictor_id="last_residual",
                covariance_donor_id="independent_endpoint_v1",
                covariance_scale=schedule,
                covariance_psd_tolerance=settings.covariance_psd_tolerance,
                metadata={
                    "coordinate_frame": identity.coordinate_frame,
                    "material_id": identity.material_id,
                    "horizon_labels": list(labels),
                    "horizon_steps": list(steps),
                    "provider_camera_ids": list(identity.provider_camera_ids),
                    "scoring_camera_ids": list(identity.scoring_camera_ids),
                    "source_only": True,
                },
            )
            mean = hybrid.mean_m
            covariance = hybrid.covariance_m2
            hybrid_artifact_id = hybrid.record.artifact_id
        except Exception as error:
            provider_status = "fallback-provider-failure"
            provider_error_type = type(error).__name__
            provider_error_sha256 = hashlib.sha256(
                f"{provider_error_type}:{error}".encode()
            ).hexdigest()
            covariance = _readonly(
                np.zeros(expected_covariance_shape, dtype=np.float64),
                dtype=np.float64,
            )

    supported_mask = _readonly(supported, dtype=bool)
    validity_readonly = _readonly(validity, dtype=bool)
    residual_readonly = _readonly(residual, dtype=np.float64)
    validity_preserved = np.array_equal(validity_readonly, validity)
    unsupported_exact = np.array_equal(
        mean[:, ~supported, :],
        physical_future[:, ~supported, :],
    )
    if mean is not reference:
        raise AssertionError("adapter changed the last_residual mean object")
    if not unsupported_exact:
        raise AssertionError("unsupported tracks changed the physical fallback")
    record = Deform360ResidualAdapterRecordV1(
        identity=identity,
        residual_history_shape=(
            int(residual.shape[0]),
            int(residual.shape[1]),
            int(residual.shape[2]),
        ),
        future_shape=(
            int(reference.shape[0]),
            int(reference.shape[1]),
            int(reference.shape[2]),
        ),
        minimum_valid_observations_per_track=(
            settings.minimum_valid_observations_per_track
        ),
        valid_observation_count_by_track=tuple(int(value) for value in counts),
        supported_track_count=int(np.sum(supported)),
        unsupported_track_count=int(np.sum(~supported)),
        validity_sha256=_array_sha256(validity_readonly),
        canonical_residual_sha256=_array_sha256(residual_readonly),
        physical_future_sha256=_array_sha256(physical_future),
        reference_mean_sha256=_array_sha256(mean),
        output_covariance_sha256=_array_sha256(covariance),
        provider_status=provider_status,
        provider_error_type=provider_error_type,
        provider_error_sha256=provider_error_sha256,
        covariance_hybrid_artifact_id=hybrid_artifact_id,
        nearest_fill_used=False,
        validity_preserved=validity_preserved,
        mean_object_identity_preserved=True,
        unsupported_tracks_use_physical_fallback=unsupported_exact,
        target_payload_opened=False,
        target_outcomes_opened=False,
        claim_authorized=False,
    )
    return Deform360ResidualAdapterPredictionV1(
        mean_m=mean,
        covariance_m2=covariance,
        supported_track_mask=supported_mask,
        record=record,
    )


__all__ = [
    "DEFAULT_HORIZON_SCALES",
    "DEFORM360_RESIDUAL_ADAPTER_SCHEMA",
    "DEFORM360_RESIDUAL_ADAPTER_VERSION",
    "CovarianceProvider",
    "Deform360ResidualAdapterPredictionV1",
    "Deform360ResidualAdapterRecordV1",
    "Deform360ResidualHistoryAdapterConfigV1",
    "Deform360ResidualHistoryIdentityV1",
    "adapt_deform360_covariance_residual_history_v1",
]
