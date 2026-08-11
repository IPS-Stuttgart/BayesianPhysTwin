"""Sole registered execution and exact fallback implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._canonical_contracts import plain_json
from .._portable_contracts import content_id
from ..contracts.fixed_anchor import FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION
from ..covariance_only_hybrid import compose_covariance_only_hybrid
from ..endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)
from ._common import (
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    _array_sha256,
    _boolean_array,
    _canonical_horizon_bins,
    _canonical_string,
    _endpoint_posterior_descriptor,
    _endpoint_prediction_descriptor,
    _float64_array,
    _frozen_endpoint_config_id,
    _future_horizon_steps,
    _scale_schedule,
    _sha256,
)
from ._decision import (
    RegisteredResidualHistoryDecisionV1,
    RegisteredResidualHistoryPredictionV1,
)
from ._provenance import ResidualHistorySourceProvenanceV1


@dataclass(frozen=True, slots=True)
class _DonorArtifacts:
    covariance_m2: np.ndarray
    config_id: str
    posterior_id: str
    prediction_ids: tuple[str, ...]


def _validate_execution_arrays(
    physical_prefix_m: object,
    provider_observation_prefix_m: object,
    observed_validity: object,
    physical_future_m: object,
    registered_last_residual_mean_m: object,
    reference_covariance_m2: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    physical_prefix = _float64_array(
        physical_prefix_m,
        name="physical_prefix_m",
        ndim=3,
        require_finite=True,
    )
    observation = _float64_array(
        provider_observation_prefix_m,
        name="provider_observation_prefix_m",
        ndim=3,
        require_finite=False,
    )
    validity = _boolean_array(observed_validity, name="observed_validity")
    physical_future = _float64_array(
        physical_future_m,
        name="physical_future_m",
        ndim=3,
        require_finite=True,
    )
    registered_mean = _float64_array(
        registered_last_residual_mean_m,
        name="registered_last_residual_mean_m",
        ndim=3,
        require_finite=True,
    )
    reference_covariance = _float64_array(
        reference_covariance_m2,
        name="reference_covariance_m2",
        ndim=4,
        require_finite=True,
    )
    if (
        physical_prefix.shape != observation.shape
        or physical_prefix.shape[:2] != validity.shape
        or physical_prefix.shape[2:] != (3,)
        or physical_prefix.shape[0] < 1
        or physical_prefix.shape[1] < 1
    ):
        raise ValueError("prefix arrays must have matching shape (T>=1, N>=1, 3)")
    future_count, track_count = physical_future.shape[:2]
    if physical_future.shape[2:] != (3,) or future_count < 3:
        raise ValueError("physical_future_m must have shape (H>=3, N, 3)")
    if track_count != physical_prefix.shape[1]:
        raise ValueError("prefix and future track rosters differ")
    if registered_mean.shape != physical_future.shape:
        raise ValueError("registered_last_residual_mean_m shape changed")
    if reference_covariance.shape != physical_future.shape + (3,):
        raise ValueError("reference_covariance_m2 shape changed")
    if np.any(reference_covariance != 0.0):
        raise ValueError("reference_covariance_m2 must be the exact zero covariance")
    if not np.all(np.isfinite(observation[validity])):
        raise ValueError("valid provider observations must be finite")
    invalid_rows = observation[~validity]
    if len(invalid_rows) and not np.all(np.isnan(invalid_rows)):
        raise ValueError("invalid provider observations must be explicit NaN rows")
    return (
        physical_prefix,
        observation,
        validity,
        physical_future,
        registered_mean,
        reference_covariance,
    )


def _residual_history(
    physical_prefix: np.ndarray,
    observation: np.ndarray,
    validity: np.ndarray,
) -> np.ndarray:
    residual = np.zeros_like(physical_prefix)
    residual[validity] = observation[validity] - physical_prefix[validity]
    residual.setflags(write=False)
    return residual


def _reconstructed_reference_mean(
    physical_future: np.ndarray,
    residual: np.ndarray,
    validity: np.ndarray,
) -> np.ndarray:
    last = np.zeros((residual.shape[1], 3), dtype=np.float64)
    for track in range(residual.shape[1]):
        support = np.flatnonzero(validity[:, track])
        if len(support):
            last[track] = residual[int(support[-1]), track]
    return np.asarray(physical_future + last[None, :, :], dtype=np.float64, order="C")


def _build_donor(
    residual: np.ndarray,
    validity: np.ndarray,
    *,
    counts: tuple[int, ...],
    steps: np.ndarray,
) -> _DonorArtifacts:
    config = DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1
    config_id = _frozen_endpoint_config_id()
    posterior = infer_model_averaged_endpoint(
        residual,
        validity,
        end_frame=len(residual),
        config=config,
    )
    if not np.array_equal(
        posterior.update_count,
        np.asarray(counts, dtype=np.int64),
    ):
        raise AssertionError("endpoint donor used different causal observations")
    posterior_id = content_id(
        _endpoint_posterior_descriptor(
            posterior,
            residual_history_sha256=_array_sha256(residual),
            validity_sha256=_array_sha256(validity),
            config_id=config_id,
        )
    )
    predictions = tuple(
        predict_model_averaged_endpoint(
            posterior,
            horizon_steps=int(step),
        )
        for step in steps
    )
    covariance = np.stack(
        [prediction.covariance_m2 for prediction in predictions],
        axis=0,
    )
    covariance.setflags(write=False)
    return _DonorArtifacts(
        covariance_m2=covariance,
        config_id=config_id,
        posterior_id=posterior_id,
        prediction_ids=tuple(
            content_id(
                _endpoint_prediction_descriptor(
                    prediction,
                    posterior_id=posterior_id,
                )
            )
            for prediction in predictions
        ),
    )


def _fallback_result(
    *,
    source_unit_id: str,
    provenance: ResidualHistorySourceProvenanceV1,
    residual: np.ndarray,
    validity: np.ndarray,
    registered_mean: np.ndarray,
    reconstructed_mean: np.ndarray,
    reference_covariance: np.ndarray,
    counts: tuple[int, ...],
    bins: np.ndarray,
    steps: np.ndarray,
    schedule: np.ndarray,
    reasons: tuple[str, ...],
    donor: _DonorArtifacts | None,
    metadata: Mapping[str, Any] | None,
) -> RegisteredResidualHistoryPredictionV1:
    decision = RegisteredResidualHistoryDecisionV1(
        source_unit_id=source_unit_id,
        provenance_id=_sha256(provenance.provenance_id, name="provenance_id"),
        residual_history_sha256=_array_sha256(residual),
        validity_sha256=_array_sha256(validity),
        registered_mean_sha256=_array_sha256(registered_mean),
        reconstructed_reference_mean_sha256=_array_sha256(reconstructed_mean),
        reference_covariance_sha256=_array_sha256(reference_covariance),
        valid_observation_count_by_track=counts,
        future_horizon_count=len(registered_mean),
        future_horizon_bins=tuple(int(value) for value in bins),
        future_horizon_steps=tuple(int(value) for value in steps),
        scale_schedule_sha256=_array_sha256(schedule),
        endpoint_contract_version=MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
        fixed_anchor_contract_version=FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION,
        endpoint_config_id=_frozen_endpoint_config_id(),
        accepted=False,
        fallback_reasons=reasons,
        endpoint_posterior_id=None if donor is None else donor.posterior_id,
        endpoint_prediction_ids=() if donor is None else donor.prediction_ids,
        donor_covariance_sha256=(
            None if donor is None else _array_sha256(donor.covariance_m2)
        ),
        output_covariance_sha256=_array_sha256(reference_covariance),
        hybrid_artifact_id=None,
        registered_mean_identity_preserved=True,
        reference_covariance_identity_preserved=True,
        metadata={} if metadata is None else metadata,
    )
    result = RegisteredResidualHistoryPredictionV1(
        mean_m=registered_mean,
        covariance_m2=reference_covariance,
        provenance=provenance,
        decision=decision,
        hybrid=None,
    )
    if result.mean_m is not registered_mean:
        raise AssertionError("fallback copied the registered mean")
    if result.covariance_m2 is not reference_covariance:
        raise AssertionError("fallback copied the reference covariance")
    return result


def _rejection_metadata(
    error: BaseException,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "covariance_rejection_type": type(error).__name__,
        "covariance_rejection_message": str(error),
    }
    if metadata is not None:
        result["source_metadata"] = plain_json(metadata)
    return result


def run_registered_residual_history_v1(
    physical_prefix_m: np.ndarray,
    provider_observation_prefix_m: np.ndarray,
    observed_validity: np.ndarray,
    physical_future_m: np.ndarray,
    registered_last_residual_mean_m: np.ndarray,
    reference_covariance_m2: np.ndarray,
    *,
    source_unit_id: str,
    provenance: ResidualHistorySourceProvenanceV1,
    metadata: Mapping[str, Any] | None = None,
) -> RegisteredResidualHistoryPredictionV1:
    """Execute the sole frozen source-side residual-history covariance path."""

    source_unit = _canonical_string(source_unit_id, name="source_unit_id")
    if not isinstance(provenance, ResidualHistorySourceProvenanceV1):
        raise TypeError("provenance must be ResidualHistorySourceProvenanceV1")
    (
        physical_prefix,
        observation,
        validity,
        physical_future,
        registered_mean,
        reference_covariance,
    ) = _validate_execution_arrays(
        physical_prefix_m,
        provider_observation_prefix_m,
        observed_validity,
        physical_future_m,
        registered_last_residual_mean_m,
        reference_covariance_m2,
    )
    residual = _residual_history(physical_prefix, observation, validity)
    counts = tuple(int(value) for value in np.sum(validity, axis=0, dtype=np.int64))
    reconstructed_mean = _reconstructed_reference_mean(
        physical_future,
        residual,
        validity,
    )
    bins = _canonical_horizon_bins(len(physical_future))
    steps = _future_horizon_steps(len(physical_future))
    schedule = _scale_schedule(bins=bins, track_count=physical_future.shape[1])
    reasons: list[str] = []
    if any(count < REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK for count in counts):
        reasons.append("insufficient-per-track-support")
    if registered_mean.tobytes(order="C") != reconstructed_mean.tobytes(order="C"):
        reasons.append("registered-mean-mismatch")
    if reasons:
        return _fallback_result(
            source_unit_id=source_unit,
            provenance=provenance,
            residual=residual,
            validity=validity,
            registered_mean=registered_mean,
            reconstructed_mean=reconstructed_mean,
            reference_covariance=reference_covariance,
            counts=counts,
            bins=bins,
            steps=steps,
            schedule=schedule,
            reasons=tuple(sorted(reasons)),
            donor=None,
            metadata=metadata,
        )

    donor: _DonorArtifacts | None = None
    try:
        donor = _build_donor(
            residual,
            validity,
            counts=counts,
            steps=steps,
        )
    except (
        ArithmeticError,
        AssertionError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return _fallback_result(
            source_unit_id=source_unit,
            provenance=provenance,
            residual=residual,
            validity=validity,
            registered_mean=registered_mean,
            reconstructed_mean=reconstructed_mean,
            reference_covariance=reference_covariance,
            counts=counts,
            bins=bins,
            steps=steps,
            schedule=schedule,
            reasons=("covariance-contract-rejection",),
            donor=None,
            metadata=_rejection_metadata(error, metadata),
        )

    try:
        hybrid = compose_covariance_only_hybrid(
            registered_mean,
            donor.covariance_m2,
            reference_predictor_id=REGISTERED_REFERENCE_PREDICTOR_ID,
            covariance_donor_id=REGISTERED_COVARIANCE_DONOR_ID,
            covariance_scale=schedule,
            metadata={
                "source_unit_id": source_unit,
                "source_provenance_id": provenance.provenance_id,
                "endpoint_config_id": donor.config_id,
                "endpoint_posterior_id": donor.posterior_id,
                "endpoint_prediction_ids": list(donor.prediction_ids),
                "execution": "registered-source-only-single-path-v1",
            },
        )
        if hybrid.mean_m is not registered_mean:
            raise AssertionError("registered covariance path copied the reference mean")
    except (
        ArithmeticError,
        AssertionError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return _fallback_result(
            source_unit_id=source_unit,
            provenance=provenance,
            residual=residual,
            validity=validity,
            registered_mean=registered_mean,
            reconstructed_mean=reconstructed_mean,
            reference_covariance=reference_covariance,
            counts=counts,
            bins=bins,
            steps=steps,
            schedule=schedule,
            reasons=("covariance-contract-rejection",),
            donor=donor,
            metadata=_rejection_metadata(error, metadata),
        )

    decision = RegisteredResidualHistoryDecisionV1(
        source_unit_id=source_unit,
        provenance_id=_sha256(provenance.provenance_id, name="provenance_id"),
        residual_history_sha256=_array_sha256(residual),
        validity_sha256=_array_sha256(validity),
        registered_mean_sha256=_array_sha256(registered_mean),
        reconstructed_reference_mean_sha256=_array_sha256(reconstructed_mean),
        reference_covariance_sha256=_array_sha256(reference_covariance),
        valid_observation_count_by_track=counts,
        future_horizon_count=len(registered_mean),
        future_horizon_bins=tuple(int(value) for value in bins),
        future_horizon_steps=tuple(int(value) for value in steps),
        scale_schedule_sha256=_array_sha256(schedule),
        endpoint_contract_version=MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
        fixed_anchor_contract_version=FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION,
        endpoint_config_id=donor.config_id,
        accepted=True,
        fallback_reasons=(),
        endpoint_posterior_id=donor.posterior_id,
        endpoint_prediction_ids=donor.prediction_ids,
        donor_covariance_sha256=_array_sha256(donor.covariance_m2),
        output_covariance_sha256=_array_sha256(hybrid.covariance_m2),
        hybrid_artifact_id=_sha256(
            hybrid.record.artifact_id,
            name="hybrid.record.artifact_id",
        ),
        registered_mean_identity_preserved=True,
        reference_covariance_identity_preserved=False,
        metadata={} if metadata is None else metadata,
    )
    return RegisteredResidualHistoryPredictionV1(
        mean_m=registered_mean,
        covariance_m2=hybrid.covariance_m2,
        provenance=provenance,
        decision=decision,
        hybrid=hybrid,
    )
