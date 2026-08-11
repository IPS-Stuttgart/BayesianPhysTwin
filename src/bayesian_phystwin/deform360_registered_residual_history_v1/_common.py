"""Shared frozen identities and strict canonicalization helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Final

import numpy as np

from .._canonical_contracts import literal_lower_hex
from .._portable_contracts import content_id
from ..contracts.fixed_anchor import FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION
from ..endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
    ModelAveragedEndpointPredictionV1,
)

REGISTERED_REFERENCE_PREDICTOR_ID: Final = "last_residual"
REGISTERED_COVARIANCE_DONOR_ID: Final = "independent_endpoint_v1"
REGISTERED_COVARIANCE_SCALES: Final = (8.0, 16.0, 16.0)
REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK: Final = 2
REGISTERED_SOURCE_PROVENANCE_SCHEMA: Final = (
    "bayesian-phystwin.deform360-registered-residual-history-source-provenance-v1"
)
REGISTERED_DECISION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-registered-residual-history-decision-v1"
)
REGISTERED_SCHEMA_VERSION: Final = 1
CLAIM_BOUNDARY: Final = (
    "Source-only implementation evidence for one frozen exact-mean covariance "
    "candidate. It does not authorize target access, scoring, promotion, "
    "deployment, physical-state identification, Causal4D benefit, or a claim."
)
_ALLOWED_FALLBACK_REASONS: Final = frozenset(
    {
        "covariance-contract-rejection",
        "insufficient-per-track-support",
        "registered-mean-mismatch",
    }
)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _sha256(value: object, *, name: str, length: int = 64) -> str:
    return literal_lower_hex(value, name=name, lengths={length})


def _optional_sha256(value: object, *, name: str) -> str | None:
    return None if value is None else _sha256(value, name=name)


def _canonical_string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    result = tuple(
        _canonical_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _digest_tuple(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    result = tuple(
        _sha256(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )
    if not result and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _float64_array(
    value: object,
    *,
    name: str,
    ndim: int,
    require_finite: bool,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must have dtype float64")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if require_finite and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _boolean_array(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have Boolean dtype")
    if value.ndim != 2:
        raise ValueError(f"{name} must have two dimensions")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _canonical_horizon_bins(future_count: int) -> np.ndarray:
    count = _integer(future_count, name="future_count", minimum=3)
    bins = np.empty(count, dtype=np.int64)
    positions = np.arange(count, dtype=np.int64)
    for label, indices in enumerate(np.array_split(positions, 3)):
        bins[indices] = label
    bins.setflags(write=False)
    return bins


def _future_horizon_steps(future_count: int) -> np.ndarray:
    steps = np.arange(1, future_count + 1, dtype=np.int64)
    steps.setflags(write=False)
    return steps


def _scale_schedule(*, bins: np.ndarray, track_count: int) -> np.ndarray:
    scale_values = np.asarray(REGISTERED_COVARIANCE_SCALES, dtype=np.float64)
    schedule = np.broadcast_to(
        scale_values[bins, None],
        (len(bins), track_count),
    ).copy(order="C")
    schedule.setflags(write=False)
    return schedule


def _endpoint_config_descriptor(
    config: ModelAveragedEndpointConfigV1,
) -> dict[str, Any]:
    if not isinstance(config, ModelAveragedEndpointConfigV1):
        raise TypeError("config must be a ModelAveragedEndpointConfigV1")
    return {
        "schema": "bayesian-phystwin.independent-endpoint-config-lineage-v1",
        "endpoint_contract_version": MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
        "fixed_anchor_contract_version": FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION,
        "components": [
            {
                "process_std_m": component.process_std_m,
                "observation_std_m": component.observation_std_m,
                "initial_std_m": component.initial_std_m,
                "inlier_prior": component.inlier_prior,
                "outlier_variance_multiplier": (component.outlier_variance_multiplier),
            }
            for component in config.components
        ],
        "component_prior_probability": list(config.component_prior_probability or ()),
    }


def _frozen_endpoint_config_id() -> str:
    return content_id(
        _endpoint_config_descriptor(DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1)
    )


def _endpoint_posterior_descriptor(
    posterior: ModelAveragedEndpointPosteriorV1,
    *,
    residual_history_sha256: str,
    validity_sha256: str,
    config_id: str,
) -> dict[str, Any]:
    if not isinstance(posterior, ModelAveragedEndpointPosteriorV1):
        raise TypeError("posterior must be ModelAveragedEndpointPosteriorV1")
    return {
        "schema": "bayesian-phystwin.independent-endpoint-posterior-lineage-v1",
        "endpoint_contract_version": MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
        "residual_history_sha256": residual_history_sha256,
        "validity_sha256": validity_sha256,
        "config_id": config_id,
        "end_frame": posterior.end_frame,
        "mean_sha256": _array_sha256(posterior.mean_m),
        "covariance_sha256": _array_sha256(posterior.covariance_m2),
        "final_nominal_probability_sha256": _array_sha256(
            posterior.final_nominal_probability
        ),
        "update_count_sha256": _array_sha256(posterior.update_count),
        "component_weights_sha256": _array_sha256(posterior.component_weights),
        "component_log_evidence_sha256": _array_sha256(
            posterior.component_log_evidence
        ),
        "component_mean_sha256": _array_sha256(posterior.component_mean_m),
        "component_variance_sha256": _array_sha256(posterior.component_variance_m2),
        "component_process_variance_sha256": _array_sha256(
            posterior.component_process_variance_m2
        ),
    }


def _endpoint_prediction_descriptor(
    prediction: ModelAveragedEndpointPredictionV1,
    *,
    posterior_id: str,
) -> dict[str, Any]:
    if not isinstance(prediction, ModelAveragedEndpointPredictionV1):
        raise TypeError("prediction must be ModelAveragedEndpointPredictionV1")
    return {
        "schema": "bayesian-phystwin.independent-endpoint-prediction-lineage-v1",
        "posterior_id": posterior_id,
        "horizon_steps": prediction.horizon_steps,
        "mean_sha256": _array_sha256(prediction.mean_m),
        "covariance_sha256": _array_sha256(prediction.covariance_m2),
        "component_weights_sha256": _array_sha256(prediction.component_weights),
    }
