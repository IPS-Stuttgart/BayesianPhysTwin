"""One registered source-only covariance path for Deform360 studies.

The caller supplies the already registered ``last_residual`` future mean and the
exact physical fallback arrays. This module independently verifies the mean
from causal source residuals, constructs ``independent_endpoint_v1`` internally
at consecutive horizons, applies the frozen ``[8, 16, 16]`` scale schedule, and
otherwise returns the exact physical fallback objects.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .covariance_only_hybrid import (
    CovarianceOnlyHybridPredictionV1,
    compose_covariance_only_hybrid,
)
from .endpoint_model_average import (
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

REGISTERED_COVARIANCE_SOURCE_SCHEMA: Final = (
    "bayesian-phystwin-deform360-registered-covariance-source-v1"
)
REGISTERED_COVARIANCE_SOURCE_VERSION: Final = 1
REFERENCE_PREDICTOR_ID: Final = "last_residual"
COVARIANCE_DONOR_ID: Final = "independent_endpoint_v1"
HORIZON_LABELS: Final = ("early", "middle", "late")
COVARIANCE_SCALES: Final = (8.0, 16.0, 16.0)
MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL: Final = 2
CLAIM_BOUNDARY: Final = (
    "source-only implementation evidence; no target roster, payload, outcome, "
    "provider-competence, physical-benefit, calibration, deployment, or "
    "state-of-the-art claim"
)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _owned_mean(
    value: object,
    *,
    name: str,
    shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array to preserve identity")
    if value.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must have dtype float64")
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, N, 3)")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} shape changed")
    if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite and C-contiguous")
    return value


def _owned_covariance(
    value: object,
    *,
    mean_shape: tuple[int, int, int],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(
            "physical_fallback_covariance_m2 must be a NumPy array to preserve identity"
        )
    expected = mean_shape + (3,)
    if value.dtype != np.dtype(np.float64) or value.shape != expected:
        raise ValueError(
            f"physical_fallback_covariance_m2 must be float64 with shape {expected}"
        )
    if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
        raise ValueError(
            "physical_fallback_covariance_m2 must be finite and C-contiguous"
        )
    if not np.allclose(value, np.swapaxes(value, -1, -2), atol=1e-12, rtol=1e-12):
        raise ValueError("physical_fallback_covariance_m2 must be symmetric")
    if float(np.min(np.linalg.eigvalsh(value), initial=0.0)) < -1e-12:
        raise ValueError(
            "physical_fallback_covariance_m2 must be positive semidefinite"
        )
    return value


def _source_inputs(
    residual_history_m: object,
    validity: object,
    *,
    end_frame: object,
) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        raw = np.asarray(residual_history_m)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "residual_history_m must contain real numeric values"
        ) from error
    if raw.dtype.kind not in "iuf":
        raise ValueError("residual_history_m must contain real numeric values")
    residual = np.asarray(raw, dtype=np.float64)
    if residual.ndim != 3 or residual.shape[-1] != 3 or residual.shape[1] < 1:
        raise ValueError("residual_history_m must have shape (T, N>=1, 3)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_history_m must be finite")
    raw_validity = np.asarray(validity)
    if raw_validity.dtype.kind != "b" or raw_validity.shape != residual.shape[:2]:
        raise ValueError("validity must be a matching Boolean matrix")
    valid = np.array(raw_validity, dtype=bool, copy=True, order="C")
    frame_stop = genuine_integer(end_frame, name="end_frame", minimum=1)
    if frame_stop > len(residual):
        raise ValueError("end_frame lies outside residual_history_m")
    if np.any(residual[~valid] != 0.0):
        raise ValueError("invalid residual entries must be stored as exact zero")
    return residual, valid, frame_stop


def _bins(value: object, *, count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.shape != (count,):
        raise ValueError(
            "future_horizon_bins must be an integer vector with one "
            "entry per future frame"
        )
    result = np.asarray(raw, dtype=np.int64)
    if np.any((result < 0) | (result >= len(HORIZON_LABELS))):
        raise ValueError("future_horizon_bins must use indices 0, 1, 2")
    return np.array(result, copy=True, order="C")


def _last_residual_and_support(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.sum(valid[:end_frame], axis=0, dtype=np.int64)
    result = np.zeros((residual.shape[1], 3), dtype=np.float64)
    for material in range(residual.shape[1]):
        support = np.flatnonzero(valid[:end_frame, material])
        if len(support):
            result[material] = residual[support[-1], material]
    return result, counts


def _donor_covariance(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    future_count: int,
) -> np.ndarray:
    posterior = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=end_frame,
    )
    return np.stack(
        [
            np.asarray(
                predict_model_averaged_endpoint(
                    posterior,
                    horizon_steps=horizon,
                ).covariance_m2,
                dtype=np.float64,
            )
            for horizon in range(1, future_count + 1)
        ]
    )


@dataclass(frozen=True, slots=True)
class RegisteredCovarianceSourceRecordV1:
    """Content-addressed admission, fallback, and source provenance."""

    source_unit_id: str
    source_residual_artifact_id: str
    registered_reference_artifact_id: str
    physical_fallback_belief_id: str
    end_frame: int
    future_horizon_bins: Sequence[int]
    residual_history_sha256: str
    validity_sha256: str
    support_count_sha256: str
    registered_mean_sha256: str
    expected_mean_sha256: str
    physical_fallback_mean_sha256: str
    physical_fallback_covariance_sha256: str
    deployed_covariance_sha256: str
    accepted: bool
    reason: str
    registered_mean_identity_preserved: bool
    exact_fallback_identity_preserved: bool
    hybrid_artifact_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    record_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_unit_id",
            _canonical_string(self.source_unit_id, name="source_unit_id"),
        )
        for name in (
            "source_residual_artifact_id",
            "registered_reference_artifact_id",
            "physical_fallback_belief_id",
            "residual_history_sha256",
            "validity_sha256",
            "support_count_sha256",
            "registered_mean_sha256",
            "expected_mean_sha256",
            "physical_fallback_mean_sha256",
            "physical_fallback_covariance_sha256",
            "deployed_covariance_sha256",
        ):
            object.__setattr__(
                self,
                name,
                literal_lower_hex(
                    getattr(self, name),
                    name=name,
                    lengths={64},
                ),
            )
        hybrid_id = self.hybrid_artifact_id
        if hybrid_id is not None:
            hybrid_id = literal_lower_hex(
                hybrid_id,
                name="hybrid_artifact_id",
                lengths={64},
            )
        bins = tuple(
            genuine_integer(value, name="future_horizon_bins", minimum=0)
            for value in self.future_horizon_bins
        )
        if not bins or any(value >= len(HORIZON_LABELS) for value in bins):
            raise ValueError("future_horizon_bins must use indices 0, 1, 2")
        end_frame = genuine_integer(self.end_frame, name="end_frame", minimum=1)
        reason = _canonical_string(self.reason, name="reason")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a Boolean")
        if type(self.registered_mean_identity_preserved) is not bool:
            raise ValueError("registered_mean_identity_preserved must be a Boolean")
        if type(self.exact_fallback_identity_preserved) is not bool:
            raise ValueError("exact_fallback_identity_preserved must be a Boolean")
        if self.accepted:
            if (
                reason != "accepted"
                or hybrid_id is None
                or self.registered_mean_sha256 != self.expected_mean_sha256
                or not self.registered_mean_identity_preserved
                or self.exact_fallback_identity_preserved
            ):
                raise ValueError("accepted source record is inconsistent")
        elif (
            reason == "accepted"
            or hybrid_id is not None
            or self.registered_mean_identity_preserved
            or not self.exact_fallback_identity_preserved
            or (
                self.deployed_covariance_sha256
                != self.physical_fallback_covariance_sha256
            )
        ):
            raise ValueError("fallback source record is inconsistent")
        object.__setattr__(self, "hybrid_artifact_id", hybrid_id)
        object.__setattr__(self, "future_horizon_bins", bins)
        object.__setattr__(self, "end_frame", end_frame)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )
        expected = content_id(self.descriptor())
        if self.record_id is None:
            object.__setattr__(self, "record_id", expected)
        elif (
            literal_lower_hex(
                self.record_id,
                name="record_id",
                lengths={64},
            )
            != expected
        ):
            raise ValueError("record_id does not match the source record")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": REGISTERED_COVARIANCE_SOURCE_SCHEMA,
            "schema_version": REGISTERED_COVARIANCE_SOURCE_VERSION,
            "source_unit_id": self.source_unit_id,
            "source_residual_artifact_id": self.source_residual_artifact_id,
            "registered_reference_artifact_id": (self.registered_reference_artifact_id),
            "physical_fallback_belief_id": self.physical_fallback_belief_id,
            "reference_predictor_id": REFERENCE_PREDICTOR_ID,
            "covariance_donor_id": COVARIANCE_DONOR_ID,
            "covariance_scales": list(COVARIANCE_SCALES),
            "minimum_valid_observations_per_material": (
                MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL
            ),
            "end_frame": self.end_frame,
            "future_horizon_bins": list(self.future_horizon_bins),
            "residual_history_sha256": self.residual_history_sha256,
            "validity_sha256": self.validity_sha256,
            "support_count_sha256": self.support_count_sha256,
            "registered_mean_sha256": self.registered_mean_sha256,
            "expected_mean_sha256": self.expected_mean_sha256,
            "physical_fallback_mean_sha256": (self.physical_fallback_mean_sha256),
            "physical_fallback_covariance_sha256": (
                self.physical_fallback_covariance_sha256
            ),
            "deployed_covariance_sha256": self.deployed_covariance_sha256,
            "accepted": self.accepted,
            "reason": self.reason,
            "registered_mean_identity_preserved": (
                self.registered_mean_identity_preserved
            ),
            "exact_fallback_identity_preserved": (
                self.exact_fallback_identity_preserved
            ),
            "hybrid_artifact_id": self.hybrid_artifact_id,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class RegisteredCovarianceSourceResultV1:
    """One accepted registered forecast or exact physical fallback."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    record: RegisteredCovarianceSourceRecordV1
    hybrid: CovarianceOnlyHybridPredictionV1 | None

    @property
    def accepted(self) -> bool:
        return self.record.accepted


def _result(
    *,
    accepted: bool,
    reason: str,
    mean: np.ndarray,
    covariance: np.ndarray,
    registered_mean: np.ndarray,
    expected_mean: np.ndarray,
    physical_mean: np.ndarray,
    physical_covariance: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray,
    support_count: np.ndarray,
    bins: np.ndarray,
    end_frame: int,
    source_unit_id: str,
    source_residual_artifact_id: str,
    registered_reference_artifact_id: str,
    physical_fallback_belief_id: str,
    hybrid: CovarianceOnlyHybridPredictionV1 | None,
    metadata: Mapping[str, Any] | None,
) -> RegisteredCovarianceSourceResultV1:
    record = RegisteredCovarianceSourceRecordV1(
        source_unit_id=source_unit_id,
        source_residual_artifact_id=source_residual_artifact_id,
        registered_reference_artifact_id=registered_reference_artifact_id,
        physical_fallback_belief_id=physical_fallback_belief_id,
        end_frame=end_frame,
        future_horizon_bins=tuple(int(value) for value in bins),
        residual_history_sha256=_sha256(residual),
        validity_sha256=_sha256(valid),
        support_count_sha256=_sha256(support_count),
        registered_mean_sha256=_sha256(registered_mean),
        expected_mean_sha256=_sha256(expected_mean),
        physical_fallback_mean_sha256=_sha256(physical_mean),
        physical_fallback_covariance_sha256=_sha256(physical_covariance),
        deployed_covariance_sha256=_sha256(covariance),
        accepted=accepted,
        reason=reason,
        registered_mean_identity_preserved=accepted,
        exact_fallback_identity_preserved=not accepted,
        hybrid_artifact_id=None if hybrid is None else hybrid.record.artifact_id,
        metadata={} if metadata is None else metadata,
    )
    result = RegisteredCovarianceSourceResultV1(
        mean_m=mean,
        covariance_m2=covariance,
        record=record,
        hybrid=hybrid,
    )
    expected_identity = registered_mean if accepted else physical_mean
    if result.mean_m is not expected_identity:
        raise AssertionError("deployed mean object identity changed")
    if not accepted and result.covariance_m2 is not physical_covariance:
        raise AssertionError("exact physical covariance identity changed")
    return result


def run_registered_deform360_covariance_source_v1(
    residual_history_m: object,
    validity: object,
    registered_last_residual_future_mean_m: np.ndarray,
    physical_fallback_future_mean_m: np.ndarray,
    physical_fallback_covariance_m2: np.ndarray,
    *,
    end_frame: int,
    future_horizon_bins: object,
    source_unit_id: str,
    source_residual_artifact_id: str,
    registered_reference_artifact_id: str,
    physical_fallback_belief_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> RegisteredCovarianceSourceResultV1:
    """Verify and deploy the frozen covariance-only source candidate.

    Structural contract violations raise. Insufficient per-material support,
    a registered-mean mismatch, or an internal covariance failure returns the
    exact caller-owned physical mean and covariance arrays.
    """

    residual, valid, frame_stop = _source_inputs(
        residual_history_m,
        validity,
        end_frame=end_frame,
    )
    physical_mean = _owned_mean(
        physical_fallback_future_mean_m,
        name="physical_fallback_future_mean_m",
    )
    registered_mean = _owned_mean(
        registered_last_residual_future_mean_m,
        name="registered_last_residual_future_mean_m",
        shape=physical_mean.shape,
    )
    if residual.shape[1] != physical_mean.shape[1]:
        raise ValueError("residual material count must match future means")
    physical_covariance = _owned_covariance(
        physical_fallback_covariance_m2,
        mean_shape=physical_mean.shape,
    )
    bins = _bins(future_horizon_bins, count=physical_mean.shape[0])
    source_unit = _canonical_string(source_unit_id, name="source_unit_id")
    source_id = literal_lower_hex(
        source_residual_artifact_id,
        name="source_residual_artifact_id",
        lengths={64},
    )
    reference_id = literal_lower_hex(
        registered_reference_artifact_id,
        name="registered_reference_artifact_id",
        lengths={64},
    )
    fallback_id = literal_lower_hex(
        physical_fallback_belief_id,
        name="physical_fallback_belief_id",
        lengths={64},
    )
    last_residual, support_count = _last_residual_and_support(
        residual,
        valid,
        end_frame=frame_stop,
    )
    expected_mean = np.array(
        physical_mean + last_residual[None, ...],
        dtype=np.float64,
        copy=True,
        order="C",
    )
    common = {
        "registered_mean": registered_mean,
        "expected_mean": expected_mean,
        "physical_mean": physical_mean,
        "physical_covariance": physical_covariance,
        "residual": residual,
        "valid": valid,
        "support_count": support_count,
        "bins": bins,
        "end_frame": frame_stop,
        "source_unit_id": source_unit,
        "source_residual_artifact_id": source_id,
        "registered_reference_artifact_id": reference_id,
        "physical_fallback_belief_id": fallback_id,
        "metadata": metadata,
    }
    if np.any(support_count < MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL):
        return _result(
            accepted=False,
            reason="insufficient-material-support",
            mean=physical_mean,
            covariance=physical_covariance,
            hybrid=None,
            **common,
        )
    if _sha256(registered_mean) != _sha256(expected_mean):
        return _result(
            accepted=False,
            reason="registered-reference-mean-mismatch",
            mean=physical_mean,
            covariance=physical_covariance,
            hybrid=None,
            **common,
        )
    try:
        donor = _donor_covariance(
            residual,
            valid,
            end_frame=frame_stop,
            future_count=physical_mean.shape[0],
        )
        scale = np.asarray(COVARIANCE_SCALES, dtype=np.float64)[bins]
        hybrid = compose_covariance_only_hybrid(
            registered_mean,
            donor,
            reference_predictor_id=REFERENCE_PREDICTOR_ID,
            covariance_donor_id=COVARIANCE_DONOR_ID,
            covariance_scale=scale[:, None],
            metadata={
                "source_unit_id": source_unit,
                "source_residual_artifact_id": source_id,
                "registered_reference_artifact_id": reference_id,
                "physical_fallback_belief_id": fallback_id,
                "end_frame": frame_stop,
            },
        )
    except (
        ArithmeticError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        fallback_metadata: dict[str, Any] = {
            "covariance_rejection_type": type(error).__name__,
            "covariance_rejection_message": str(error),
        }
        if metadata is not None:
            fallback_metadata["source_metadata"] = plain_json(metadata)
        return _result(
            accepted=False,
            reason="covariance-contract-rejection",
            mean=physical_mean,
            covariance=physical_covariance,
            hybrid=None,
            **{**common, "metadata": fallback_metadata},
        )
    return _result(
        accepted=True,
        reason="accepted",
        mean=hybrid.mean_m,
        covariance=hybrid.covariance_m2,
        hybrid=hybrid,
        **common,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "COVARIANCE_DONOR_ID",
    "COVARIANCE_SCALES",
    "HORIZON_LABELS",
    "MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL",
    "REFERENCE_PREDICTOR_ID",
    "REGISTERED_COVARIANCE_SOURCE_SCHEMA",
    "REGISTERED_COVARIANCE_SOURCE_VERSION",
    "RegisteredCovarianceSourceRecordV1",
    "RegisteredCovarianceSourceResultV1",
    "run_registered_deform360_covariance_source_v1",
]
