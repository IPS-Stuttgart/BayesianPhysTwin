"""Registered donor construction for the Deform360 covariance-only source path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._canonical_contracts import genuine_integer, plain_json
from ._deform360_covariance_residual_history_adapter_v1 import (
    build_residual_history_adapter,
)
from ._deform360_covariance_residual_history_common_v1 import (
    CLAIM_BOUNDARY,
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_COVARIANCE_SCALES,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    CameraRecorderFamilyMapV1,
    ReconstructionManifestV1,
    ResidualHistoryDryRunPolicyV1,
    _array_sha256,
    _integer_vector,
    _readonly_float_array,
    _required_sha256,
    _validate_covariance,
)
from ._deform360_covariance_residual_history_decision_v1 import (
    ResidualHistoryDryRunResultV1,
    _horizon_bins,
    _physical_future_mean,
)
from ._deform360_covariance_residual_history_last_valid_v1 import (
    run_source_only_residual_history_dry_run,
)
from ._portable_contracts import content_id
from .contracts.fixed_anchor import FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION
from .endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

REGISTERED_DONOR_RECORD_SCHEMA = (
    "bayesian-phystwin.deform360-independent-endpoint-covariance-donor-v1"
)
REGISTERED_EXECUTION_SCHEMA = (
    "bayesian-phystwin.deform360-registered-covariance-residual-history-execution-v1"
)
REGISTERED_EXECUTION_VERSION = 1


def _canonical_digest_tuple(
    value: object,
    *,
    name: str,
    expected_length: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of SHA-256 identities")
    result = tuple(
        _required_sha256(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != expected_length:
        raise ValueError(f"{name} must contain one identity per future horizon")
    return result


def _config_descriptor(config: ModelAveragedEndpointConfigV1) -> dict[str, Any]:
    if not isinstance(config, ModelAveragedEndpointConfigV1):
        raise TypeError("config must be ModelAveragedEndpointConfigV1")
    return {
        "model_averaged_endpoint_contract_version": (
            MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION
        ),
        "fixed_bayesian_anchor_contract_version": (
            FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION
        ),
        "components": [
            {
                "process_std_m": component.process_std_m,
                "observation_std_m": component.observation_std_m,
                "initial_std_m": component.initial_std_m,
                "inlier_prior": component.inlier_prior,
                "outlier_variance_multiplier": (
                    component.outlier_variance_multiplier
                ),
            }
            for component in config.components
        ],
        "component_prior_probability": list(
            config.component_prior_probability or ()
        ),
    }


def _posterior_descriptor(
    posterior: ModelAveragedEndpointPosteriorV1,
    *,
    adapter_id: str,
    config_id: str,
) -> dict[str, Any]:
    if not isinstance(posterior, ModelAveragedEndpointPosteriorV1):
        raise TypeError("posterior must be ModelAveragedEndpointPosteriorV1")
    return {
        "schema": "bayesian-phystwin.independent-endpoint-posterior-lineage-v1",
        "adapter_id": _required_sha256(adapter_id, name="adapter_id"),
        "config_id": _required_sha256(config_id, name="config_id"),
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
        "component_variance_sha256": _array_sha256(
            posterior.component_variance_m2
        ),
        "component_process_variance_sha256": _array_sha256(
            posterior.component_process_variance_m2
        ),
    }


def _prediction_descriptor(
    *,
    posterior_id: str,
    horizon_steps: int,
    mean_m: np.ndarray,
    covariance_m2: np.ndarray,
    component_weights: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema": "bayesian-phystwin.independent-endpoint-prediction-lineage-v1",
        "posterior_id": _required_sha256(posterior_id, name="posterior_id"),
        "horizon_steps": genuine_integer(
            horizon_steps,
            name="horizon_steps",
            minimum=1,
        ),
        "mean_sha256": _array_sha256(mean_m),
        "covariance_sha256": _array_sha256(covariance_m2),
        "component_weights_sha256": _array_sha256(component_weights),
    }


@dataclass(frozen=True, slots=True)
class IndependentEndpointCovarianceDonorV1:
    """Content-addressed covariance produced by the frozen endpoint model."""

    adapter_id: str
    config_id: str
    posterior_id: str
    prediction_ids: tuple[str, ...]
    future_horizon_steps: np.ndarray
    covariance_m2: np.ndarray
    donor_id: str | None = None

    def __post_init__(self) -> None:
        adapter_id = _required_sha256(self.adapter_id, name="adapter_id")
        config_id = _required_sha256(self.config_id, name="config_id")
        posterior_id = _required_sha256(self.posterior_id, name="posterior_id")
        steps = _integer_vector(
            self.future_horizon_steps,
            name="future_horizon_steps",
        )
        if np.any(steps < 1) or not np.all(np.diff(steps) > 0):
            raise ValueError("future_horizon_steps must be positive and increasing")
        prediction_ids = _canonical_digest_tuple(
            self.prediction_ids,
            name="prediction_ids",
            expected_length=len(steps),
        )
        raw_covariance = _readonly_float_array(
            self.covariance_m2,
            name="covariance_m2",
            ndim=4,
        )
        if raw_covariance.shape[0] != len(steps):
            raise ValueError("covariance_m2 must contain one future covariance per step")
        if raw_covariance.shape[1] < 1 or raw_covariance.shape[2:] != (3, 3):
            raise ValueError("covariance_m2 must have shape (H, N>=1, 3, 3)")
        covariance = _validate_covariance(
            raw_covariance,
            name="covariance_m2",
            expected_shape=raw_covariance.shape,
        )
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "config_id", config_id)
        object.__setattr__(self, "posterior_id", posterior_id)
        object.__setattr__(self, "prediction_ids", prediction_ids)
        object.__setattr__(self, "future_horizon_steps", steps)
        object.__setattr__(self, "covariance_m2", covariance)
        expected = content_id(self.descriptor())
        if self.donor_id is None:
            object.__setattr__(self, "donor_id", expected)
        elif _required_sha256(self.donor_id, name="donor_id") != expected:
            raise ValueError("donor_id does not match the registered donor record")

    @property
    def covariance_sha256(self) -> str:
        return _array_sha256(self.covariance_m2)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": REGISTERED_DONOR_RECORD_SCHEMA,
            "schema_version": REGISTERED_EXECUTION_VERSION,
            "covariance_donor_id": REGISTERED_COVARIANCE_DONOR_ID,
            "adapter_id": self.adapter_id,
            "config_id": self.config_id,
            "posterior_id": self.posterior_id,
            "prediction_ids": list(self.prediction_ids),
            "future_horizon_steps": self.future_horizon_steps.tolist(),
            "future_horizon_steps_sha256": _array_sha256(
                self.future_horizon_steps
            ),
            "covariance_shape": list(self.covariance_m2.shape),
            "covariance_sha256": self.covariance_sha256,
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class RegisteredResidualHistoryExecutionV1:
    """Strict registered execution binding donor production and deployment."""

    result: ResidualHistoryDryRunResultV1
    donor: IndependentEndpointCovarianceDonorV1
    future_horizon_bins: np.ndarray
    execution_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, ResidualHistoryDryRunResultV1):
            raise TypeError("result must be ResidualHistoryDryRunResultV1")
        if not isinstance(self.donor, IndependentEndpointCovarianceDonorV1):
            raise TypeError("donor must be IndependentEndpointCovarianceDonorV1")
        adapter_id = _required_sha256(
            self.result.adapter.adapter_id,
            name="result.adapter.adapter_id",
        )
        if self.donor.adapter_id != adapter_id:
            raise ValueError("donor and deployed result use different residual histories")
        bins = _horizon_bins(
            self.future_horizon_bins,
            future_count=len(self.donor.future_horizon_steps),
        )
        if self.result.accepted:
            scales = np.asarray(REGISTERED_COVARIANCE_SCALES, dtype=np.float64)[bins]
            expected = self.donor.covariance_m2 * scales[:, None, None, None]
            if self.result.covariance_m2.tobytes(order="C") != expected.tobytes(
                order="C"
            ):
                raise ValueError(
                    "accepted covariance differs from the registered donor and scales"
                )
            if self.result.hybrid is None:
                raise ValueError("accepted execution is missing the covariance hybrid")
        elif self.result.hybrid is not None:
            raise ValueError("fallback execution must not retain a covariance hybrid")
        object.__setattr__(self, "future_horizon_bins", bins)
        expected_id = content_id(self.descriptor())
        if self.execution_id is None:
            object.__setattr__(self, "execution_id", expected_id)
        elif _required_sha256(self.execution_id, name="execution_id") != expected_id:
            raise ValueError("execution_id does not match the registered execution")

    @property
    def accepted(self) -> bool:
        return self.result.accepted

    @property
    def mean_m(self) -> np.ndarray:
        return self.result.mean_m

    @property
    def covariance_m2(self) -> np.ndarray:
        return self.result.covariance_m2

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": REGISTERED_EXECUTION_SCHEMA,
            "schema_version": REGISTERED_EXECUTION_VERSION,
            "reference_predictor_id": REGISTERED_REFERENCE_PREDICTOR_ID,
            "covariance_donor_id": REGISTERED_COVARIANCE_DONOR_ID,
            "covariance_scales": list(REGISTERED_COVARIANCE_SCALES),
            "adapter_id": _required_sha256(
                self.result.adapter.adapter_id,
                name="result.adapter.adapter_id",
            ),
            "donor_id": _required_sha256(
                self.donor.donor_id,
                name="donor.donor_id",
            ),
            "decision_id": _required_sha256(
                self.result.decision.decision_id,
                name="result.decision.decision_id",
            ),
            "accepted": self.result.accepted,
            "future_horizon_bins": self.future_horizon_bins.tolist(),
            "future_horizon_bins_sha256": _array_sha256(
                self.future_horizon_bins
            ),
            "deployed_mean_sha256": _array_sha256(self.result.mean_m),
            "deployed_covariance_sha256": _array_sha256(
                self.result.covariance_m2
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }


def _build_registered_donor(
    *,
    adapter_id: str,
    residual_history_m: np.ndarray,
    observed_validity: np.ndarray,
    prefix_frame_count: int,
    valid_observation_count_by_material: tuple[int, ...],
    future_horizon_steps: object,
    future_count: int,
) -> IndependentEndpointCovarianceDonorV1:
    steps = _integer_vector(
        future_horizon_steps,
        name="future_horizon_steps",
    )
    if steps.shape != (future_count,):
        raise ValueError("future_horizon_steps must have one entry per future frame")
    if np.any(steps < 1) or not np.all(np.diff(steps) > 0):
        raise ValueError("future_horizon_steps must be positive and increasing")
    config = DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1
    config_id = content_id(_config_descriptor(config))
    posterior = infer_model_averaged_endpoint(
        residual_history_m,
        observed_validity,
        end_frame=prefix_frame_count,
        config=config,
    )
    expected_counts = np.asarray(
        valid_observation_count_by_material,
        dtype=np.int64,
    )
    if not np.array_equal(posterior.update_count, expected_counts):
        raise AssertionError("endpoint donor used different causal observations")
    posterior_id = content_id(
        _posterior_descriptor(
            posterior,
            adapter_id=adapter_id,
            config_id=config_id,
        )
    )
    covariances: list[np.ndarray] = []
    prediction_ids: list[str] = []
    for raw_step in steps:
        step = int(raw_step)
        prediction = predict_model_averaged_endpoint(
            posterior,
            horizon_steps=step,
        )
        covariances.append(prediction.covariance_m2)
        prediction_ids.append(
            content_id(
                _prediction_descriptor(
                    posterior_id=posterior_id,
                    horizon_steps=step,
                    mean_m=prediction.mean_m,
                    covariance_m2=prediction.covariance_m2,
                    component_weights=prediction.component_weights,
                )
            )
        )
    covariance = np.stack(covariances, axis=0)
    return IndependentEndpointCovarianceDonorV1(
        adapter_id=adapter_id,
        config_id=config_id,
        posterior_id=posterior_id,
        prediction_ids=tuple(prediction_ids),
        future_horizon_steps=steps,
        covariance_m2=covariance,
    )


def run_registered_source_only_residual_history(
    physical_prefix_m: object,
    provider_observation_prefix_m: object,
    observed_validity: object,
    physical_future_m: np.ndarray,
    physical_fallback_covariance_m2: np.ndarray,
    registered_last_residual_mean_m: np.ndarray,
    *,
    frame_indices: object,
    material_ids: object,
    future_horizon_steps: object,
    future_horizon_bins: object,
    camera_recorder_family_map: CameraRecorderFamilyMapV1,
    provider_reconstruction_manifest: ReconstructionManifestV1,
    scoring_reconstruction_manifest: ReconstructionManifestV1,
    source_unit_id: str,
    policy: ResidualHistoryDryRunPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> RegisteredResidualHistoryExecutionV1:
    """Run the hard-bound registered donor with no covariance injection surface."""

    adapter = build_residual_history_adapter(
        physical_prefix_m,
        provider_observation_prefix_m,
        observed_validity,
        frame_indices=frame_indices,
        material_ids=material_ids,
        camera_recorder_family_map=camera_recorder_family_map,
        provider_reconstruction_manifest=provider_reconstruction_manifest,
        scoring_reconstruction_manifest=scoring_reconstruction_manifest,
        source_unit_id=source_unit_id,
        policy=policy,
        metadata=metadata,
    )
    physical_future = _physical_future_mean(
        physical_future_m,
        material_count=adapter.material_count,
    )
    bins = _horizon_bins(
        future_horizon_bins,
        future_count=len(physical_future),
    )
    donor = _build_registered_donor(
        adapter_id=_required_sha256(adapter.adapter_id, name="adapter.adapter_id"),
        residual_history_m=adapter.residual_history_m,
        observed_validity=adapter.observed_validity,
        prefix_frame_count=adapter.prefix_frame_count,
        valid_observation_count_by_material=(
            adapter.valid_observation_count_by_material
        ),
        future_horizon_steps=future_horizon_steps,
        future_count=len(physical_future),
    )
    result = run_source_only_residual_history_dry_run(
        physical_prefix_m,
        provider_observation_prefix_m,
        observed_validity,
        physical_future_m,
        physical_fallback_covariance_m2,
        registered_last_residual_mean_m,
        donor.covariance_m2,
        frame_indices=frame_indices,
        material_ids=material_ids,
        future_horizon_bins=bins,
        camera_recorder_family_map=camera_recorder_family_map,
        provider_reconstruction_manifest=provider_reconstruction_manifest,
        scoring_reconstruction_manifest=scoring_reconstruction_manifest,
        source_unit_id=source_unit_id,
        policy=policy,
        metadata=metadata,
    )
    if result.adapter.adapter_id != adapter.adapter_id:
        raise AssertionError("registered and deployed residual histories differ")
    execution = RegisteredResidualHistoryExecutionV1(
        result=result,
        donor=donor,
        future_horizon_bins=bins,
    )
    if result.accepted and execution.mean_m is not registered_last_residual_mean_m:
        raise AssertionError("registered execution copied the caller-owned mean")
    if metadata is not None and plain_json(metadata) != plain_json(
        result.adapter.metadata
    ):
        raise AssertionError("registered execution changed source metadata")
    return execution


__all__ = [
    "IndependentEndpointCovarianceDonorV1",
    "REGISTERED_DONOR_RECORD_SCHEMA",
    "REGISTERED_EXECUTION_SCHEMA",
    "REGISTERED_EXECUTION_VERSION",
    "RegisteredResidualHistoryExecutionV1",
    "run_registered_source_only_residual_history",
]
