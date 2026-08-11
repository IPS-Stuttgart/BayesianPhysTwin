"""Single-path source-only execution for the frozen Deform360 covariance candidate.

This module deliberately exposes one execution function. It reconstructs the
registered ``independent_endpoint_v1`` covariance internally at consecutive
horizons, attaches the frozen ``[8, 16, 16]`` scale schedule to the exact
caller-owned ``last_residual`` mean, and otherwise returns the exact registered
reference objects. It contains no target roster, target access, scoring, or claim
promotion path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .covariance_only_hybrid import (
    CovarianceOnlyHybridPredictionV1,
    compose_covariance_only_hybrid,
)
from .endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
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
    {"insufficient-per-track-support", "registered-mean-mismatch"}
)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _sha256(value: object, *, name: str, length: int = 64) -> str:
    return literal_lower_hex(value, name=name, lengths={length})


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


def _digest_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    result = tuple(
        _sha256(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
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
    preserve_identity: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        qualifier = " to preserve identity" if preserve_identity else ""
        raise TypeError(f"{name} must be a NumPy array{qualifier}")
    if value.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must have dtype float64")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _boolean_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have Boolean dtype")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _canonical_horizon_bins(future_count: int) -> np.ndarray:
    if isinstance(future_count, bool) or future_count < 3:
        raise ValueError("registered execution requires at least three future frames")
    bins = np.empty(future_count, dtype=np.int64)
    indices = np.arange(future_count, dtype=np.int64)
    for label_index, chunk in enumerate(np.array_split(indices, 3)):
        if not len(chunk):
            raise AssertionError("canonical horizon partition contains an empty bin")
        bins[chunk] = label_index
    bins.setflags(write=False)
    return bins


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


def _endpoint_posterior_descriptor(
    posterior: ModelAveragedEndpointPosteriorV1,
    *,
    residual_history_sha256: str,
    validity_sha256: str,
    config_id: str,
) -> dict[str, Any]:
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
        "component_variance_sha256": _array_sha256(
            posterior.component_variance_m2
        ),
        "component_process_variance_sha256": _array_sha256(
            posterior.component_process_variance_m2
        ),
    }


def _endpoint_prediction_descriptor(
    *,
    posterior_id: str,
    horizon_steps: int,
    mean_m: np.ndarray,
    covariance_m2: np.ndarray,
    component_weights: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema": "bayesian-phystwin.independent-endpoint-prediction-lineage-v1",
        "posterior_id": posterior_id,
        "horizon_steps": horizon_steps,
        "mean_sha256": _array_sha256(mean_m),
        "covariance_sha256": _array_sha256(covariance_m2),
        "component_weights_sha256": _array_sha256(component_weights),
    }


@dataclass(frozen=True, slots=True)
class ResidualHistorySourceProvenanceV1:
    """Disjoint source reconstruction identity for one opened source unit."""

    source_inventory_id: str
    provider_reconstruction_id: str
    scoring_reconstruction_id: str
    provider_implementation_revision: str
    scoring_implementation_revision: str
    provider_configuration_id: str
    scoring_configuration_id: str
    provider_camera_family_ids: tuple[str, ...]
    scoring_camera_family_ids: tuple[str, ...]
    provider_input_artifact_ids: tuple[str, ...]
    scoring_input_artifact_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance_id: str | None = None

    def __post_init__(self) -> None:
        source_inventory_id = _sha256(
            self.source_inventory_id,
            name="source_inventory_id",
        )
        provider_reconstruction_id = _sha256(
            self.provider_reconstruction_id,
            name="provider_reconstruction_id",
        )
        scoring_reconstruction_id = _sha256(
            self.scoring_reconstruction_id,
            name="scoring_reconstruction_id",
        )
        if provider_reconstruction_id == scoring_reconstruction_id:
            raise ValueError("provider and scoring reconstructions must differ")
        provider_revision = _sha256(
            self.provider_implementation_revision,
            name="provider_implementation_revision",
            length=40,
        )
        scoring_revision = _sha256(
            self.scoring_implementation_revision,
            name="scoring_implementation_revision",
            length=40,
        )
        provider_configuration_id = _sha256(
            self.provider_configuration_id,
            name="provider_configuration_id",
        )
        scoring_configuration_id = _sha256(
            self.scoring_configuration_id,
            name="scoring_configuration_id",
        )
        provider_families = _canonical_string_tuple(
            self.provider_camera_family_ids,
            name="provider_camera_family_ids",
        )
        scoring_families = _canonical_string_tuple(
            self.scoring_camera_family_ids,
            name="scoring_camera_family_ids",
        )
        if set(provider_families) & set(scoring_families):
            raise ValueError("provider and scoring camera families must be disjoint")
        provider_inputs = _digest_tuple(
            self.provider_input_artifact_ids,
            name="provider_input_artifact_ids",
        )
        scoring_inputs = _digest_tuple(
            self.scoring_input_artifact_ids,
            name="scoring_input_artifact_ids",
        )
        if set(provider_inputs) & set(scoring_inputs):
            raise ValueError("provider and scoring input artifacts must be disjoint")
        metadata = frozen_finite_json_mapping(self.metadata, name="metadata")
        object.__setattr__(self, "source_inventory_id", source_inventory_id)
        object.__setattr__(
            self,
            "provider_reconstruction_id",
            provider_reconstruction_id,
        )
        object.__setattr__(
            self,
            "scoring_reconstruction_id",
            scoring_reconstruction_id,
        )
        object.__setattr__(
            self,
            "provider_implementation_revision",
            provider_revision,
        )
        object.__setattr__(
            self,
            "scoring_implementation_revision",
            scoring_revision,
        )
        object.__setattr__(
            self,
            "provider_configuration_id",
            provider_configuration_id,
        )
        object.__setattr__(
            self,
            "scoring_configuration_id",
            scoring_configuration_id,
        )
        object.__setattr__(self, "provider_camera_family_ids", provider_families)
        object.__setattr__(self, "scoring_camera_family_ids", scoring_families)
        object.__setattr__(self, "provider_input_artifact_ids", provider_inputs)
        object.__setattr__(self, "scoring_input_artifact_ids", scoring_inputs)
        object.__setattr__(self, "metadata", metadata)
        expected = content_id(self.descriptor())
        if self.provenance_id is None:
            object.__setattr__(self, "provenance_id", expected)
        elif _sha256(self.provenance_id, name="provenance_id") != expected:
            raise ValueError("provenance_id does not match source provenance")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": REGISTERED_SOURCE_PROVENANCE_SCHEMA,
            "schema_version": REGISTERED_SCHEMA_VERSION,
            "source_inventory_id": self.source_inventory_id,
            "provider_reconstruction_id": self.provider_reconstruction_id,
            "scoring_reconstruction_id": self.scoring_reconstruction_id,
            "provider_implementation_revision": (
                self.provider_implementation_revision
            ),
            "scoring_implementation_revision": (
                self.scoring_implementation_revision
            ),
            "provider_configuration_id": self.provider_configuration_id,
            "scoring_configuration_id": self.scoring_configuration_id,
            "provider_camera_family_ids": list(self.provider_camera_family_ids),
            "scoring_camera_family_ids": list(self.scoring_camera_family_ids),
            "provider_input_artifact_ids": list(self.provider_input_artifact_ids),
            "scoring_input_artifact_ids": list(self.scoring_input_artifact_ids),
            "metadata": plain_json(self.metadata),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class RegisteredResidualHistoryDecisionV1:
    """Content-addressed acceptance or exact registered-reference fallback."""

    source_unit_id: str
    provenance_id: str
    residual_history_sha256: str
    validity_sha256: str
    registered_mean_sha256: str
    reconstructed_reference_mean_sha256: str
    reference_covariance_sha256: str
    valid_observation_count_by_track: tuple[int, ...]
    future_horizon_count: int
    future_horizon_bins: tuple[int, ...]
    scale_schedule_sha256: str
    accepted: bool
    fallback_reasons: tuple[str, ...]
    endpoint_config_id: str | None
    endpoint_posterior_id: str | None
    endpoint_prediction_ids: tuple[str, ...]
    donor_covariance_sha256: str | None
    output_covariance_sha256: str
    hybrid_artifact_id: str | None
    registered_mean_identity_preserved: bool
    reference_covariance_identity_preserved: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    decision_id: str | None = None

    def __post_init__(self) -> None:
        source_unit_id = _canonical_string(self.source_unit_id, name="source_unit_id")
        provenance_id = _sha256(self.provenance_id, name="provenance_id")
        digests = {
            name: _sha256(getattr(self, name), name=name)
            for name in (
                "residual_history_sha256",
                "validity_sha256",
                "registered_mean_sha256",
                "reconstructed_reference_mean_sha256",
                "reference_covariance_sha256",
                "scale_schedule_sha256",
                "output_covariance_sha256",
            )
        }
        if type(self.valid_observation_count_by_track) is not tuple:
            raise ValueError(
                "valid_observation_count_by_track must be a canonical tuple"
            )
        counts: list[int] = []
        for index, value in enumerate(self.valid_observation_count_by_track):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"valid_observation_count_by_track[{index}] must be an integer"
                )
            count = int(value)
            if count < 0:
                raise ValueError("valid observation counts must be nonnegative")
            counts.append(count)
        count_tuple = tuple(counts)
        if not count_tuple:
            raise ValueError("valid observation counts must be nonempty")
        if isinstance(self.future_horizon_count, bool) or not isinstance(
            self.future_horizon_count,
            (int, np.integer),
        ):
            raise ValueError("future_horizon_count must be an integer")
        future_count = int(self.future_horizon_count)
        if type(self.future_horizon_bins) is not tuple:
            raise ValueError("future_horizon_bins must be a canonical tuple")
        expected_bins = tuple(
            int(value) for value in _canonical_horizon_bins(future_count)
        )
        if self.future_horizon_bins != expected_bins:
            raise ValueError("future_horizon_bins differ from the canonical partition")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a Boolean")
        if type(self.fallback_reasons) is not tuple:
            raise ValueError("fallback_reasons must be a canonical tuple")
        reasons = tuple(
            _canonical_string(value, name="fallback_reasons")
            for value in self.fallback_reasons
        )
        if reasons != tuple(sorted(set(reasons))):
            raise ValueError("fallback_reasons must be sorted and unique")
        if not set(reasons) <= _ALLOWED_FALLBACK_REASONS:
            raise ValueError("fallback_reasons contain an unsupported reason")
        insufficient = any(
            count < REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK
            for count in count_tuple
        )
        if ("insufficient-per-track-support" in reasons) != insufficient:
            raise ValueError("support fallback reason differs from source support")
        mean_mismatch = (
            digests["registered_mean_sha256"]
            != digests["reconstructed_reference_mean_sha256"]
        )
        if ("registered-mean-mismatch" in reasons) != mean_mismatch:
            raise ValueError("mean mismatch fallback reason differs from content")
        endpoint_config_id = self.endpoint_config_id
        endpoint_posterior_id = self.endpoint_posterior_id
        donor_covariance_sha256 = self.donor_covariance_sha256
        hybrid_artifact_id = self.hybrid_artifact_id
        if endpoint_config_id is not None:
            endpoint_config_id = _sha256(
                endpoint_config_id,
                name="endpoint_config_id",
            )
        if endpoint_posterior_id is not None:
            endpoint_posterior_id = _sha256(
                endpoint_posterior_id,
                name="endpoint_posterior_id",
            )
        if donor_covariance_sha256 is not None:
            donor_covariance_sha256 = _sha256(
                donor_covariance_sha256,
                name="donor_covariance_sha256",
            )
        if hybrid_artifact_id is not None:
            hybrid_artifact_id = _sha256(
                hybrid_artifact_id,
                name="hybrid_artifact_id",
            )
        if type(self.endpoint_prediction_ids) is not tuple:
            raise ValueError("endpoint_prediction_ids must be a canonical tuple")
        prediction_ids = tuple(
            _sha256(value, name=f"endpoint_prediction_ids[{index}]")
            for index, value in enumerate(self.endpoint_prediction_ids)
        )
        if len(set(prediction_ids)) != len(prediction_ids):
            raise ValueError("endpoint_prediction_ids must be unique")
        identity_values = (
            self.registered_mean_identity_preserved,
            self.reference_covariance_identity_preserved,
        )
        if any(type(value) is not bool for value in identity_values):
            raise ValueError("identity-preservation fields must be Booleans")
        if not self.registered_mean_identity_preserved:
            raise ValueError("registered mean identity must always be preserved")
        if self.accepted:
            if reasons:
                raise ValueError("accepted decisions cannot contain fallback reasons")
            if insufficient or mean_mismatch:
                raise ValueError("accepted decision violates registered admission")
            if (
                endpoint_config_id is None
                or endpoint_posterior_id is None
                or donor_covariance_sha256 is None
                or hybrid_artifact_id is None
                or len(prediction_ids) != future_count
            ):
                raise ValueError("accepted decision lacks endpoint donor lineage")
            if self.reference_covariance_identity_preserved:
                raise ValueError(
                    "accepted decision did not deploy reference covariance"
                )
        else:
            if not reasons:
                raise ValueError("fallback decisions require at least one reason")
            if any(
                value is not None
                for value in (
                    endpoint_config_id,
                    endpoint_posterior_id,
                    donor_covariance_sha256,
                    hybrid_artifact_id,
                )
            ) or prediction_ids:
                raise ValueError("fallback decision must not retain donor execution")
            if not self.reference_covariance_identity_preserved:
                raise ValueError("fallback must preserve reference covariance identity")
            if (
                digests["output_covariance_sha256"]
                != digests["reference_covariance_sha256"]
            ):
                raise ValueError("fallback covariance differs from the reference")
        metadata = frozen_finite_json_mapping(self.metadata, name="metadata")
        object.__setattr__(self, "source_unit_id", source_unit_id)
        object.__setattr__(self, "provenance_id", provenance_id)
        for name, value in digests.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "valid_observation_count_by_track",
            count_tuple,
        )
        object.__setattr__(self, "future_horizon_count", future_count)
        object.__setattr__(self, "future_horizon_bins", expected_bins)
        object.__setattr__(self, "fallback_reasons", reasons)
        object.__setattr__(self, "endpoint_config_id", endpoint_config_id)
        object.__setattr__(self, "endpoint_posterior_id", endpoint_posterior_id)
        object.__setattr__(self, "endpoint_prediction_ids", prediction_ids)
        object.__setattr__(
            self,
            "donor_covariance_sha256",
            donor_covariance_sha256,
        )
        object.__setattr__(self, "hybrid_artifact_id", hybrid_artifact_id)
        object.__setattr__(self, "metadata", metadata)
        expected = content_id(self.descriptor())
        if self.decision_id is None:
            object.__setattr__(self, "decision_id", expected)
        elif _sha256(self.decision_id, name="decision_id") != expected:
            raise ValueError("decision_id does not match the registered decision")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": REGISTERED_DECISION_SCHEMA,
            "schema_version": REGISTERED_SCHEMA_VERSION,
            "source_unit_id": self.source_unit_id,
            "provenance_id": self.provenance_id,
            "residual_history_sha256": self.residual_history_sha256,
            "validity_sha256": self.validity_sha256,
            "registered_mean_sha256": self.registered_mean_sha256,
            "reconstructed_reference_mean_sha256": (
                self.reconstructed_reference_mean_sha256
            ),
            "reference_covariance_sha256": self.reference_covariance_sha256,
            "valid_observation_count_by_track": list(
                self.valid_observation_count_by_track
            ),
            "minimum_valid_observations_per_track": (
                REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK
            ),
            "future_horizon_count": self.future_horizon_count,
            "future_horizon_bins": list(self.future_horizon_bins),
            "registered_covariance_scales": list(REGISTERED_COVARIANCE_SCALES),
            "scale_schedule_sha256": self.scale_schedule_sha256,
            "accepted": self.accepted,
            "fallback_reasons": list(self.fallback_reasons),
            "endpoint_contract_version": MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
            "endpoint_config_id": self.endpoint_config_id,
            "endpoint_posterior_id": self.endpoint_posterior_id,
            "endpoint_prediction_ids": list(self.endpoint_prediction_ids),
            "donor_covariance_sha256": self.donor_covariance_sha256,
            "output_covariance_sha256": self.output_covariance_sha256,
            "hybrid_artifact_id": self.hybrid_artifact_id,
            "registered_mean_identity_preserved": (
                self.registered_mean_identity_preserved
            ),
            "reference_covariance_identity_preserved": (
                self.reference_covariance_identity_preserved
            ),
            "reference_predictor_id": REGISTERED_REFERENCE_PREDICTOR_ID,
            "covariance_donor_id": REGISTERED_COVARIANCE_DONOR_ID,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class RegisteredResidualHistoryPredictionV1:
    """One source-only candidate or exact registered-reference fallback."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    provenance: ResidualHistorySourceProvenanceV1
    decision: RegisteredResidualHistoryDecisionV1
    hybrid: CovarianceOnlyHybridPredictionV1 | None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ResidualHistorySourceProvenanceV1):
            raise TypeError("provenance must be ResidualHistorySourceProvenanceV1")
        if not isinstance(self.decision, RegisteredResidualHistoryDecisionV1):
            raise TypeError("decision must be RegisteredResidualHistoryDecisionV1")
        if self.decision.provenance_id != self.provenance.provenance_id:
            raise ValueError("decision and source provenance differ")
        expected_mean_shape = (
            self.decision.future_horizon_count,
            len(self.decision.valid_observation_count_by_track),
            3,
        )
        if self.mean_m.shape != expected_mean_shape:
            raise ValueError("result mean shape differs from the registered decision")
        if self.covariance_m2.shape != expected_mean_shape + (3,):
            raise ValueError(
                "result covariance shape differs from the registered decision"
            )
        if _array_sha256(self.mean_m) != self.decision.registered_mean_sha256:
            raise ValueError("result mean content differs from the registered mean")
        if _array_sha256(self.covariance_m2) != self.decision.output_covariance_sha256:
            raise ValueError("result covariance content differs from the decision")
        if self.decision.accepted:
            if self.hybrid is None:
                raise ValueError("accepted result is missing the covariance hybrid")
            if (
                self.hybrid.mean_m is not self.mean_m
                or self.hybrid.covariance_m2 is not self.covariance_m2
            ):
                raise ValueError("accepted result does not retain hybrid objects")
        elif self.hybrid is not None:
            raise ValueError("fallback result must not retain a covariance hybrid")

    @property
    def accepted(self) -> bool:
        return self.decision.accepted


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
    )
    observation = _float64_array(
        provider_observation_prefix_m,
        name="provider_observation_prefix_m",
        ndim=3,
    )
    validity = _boolean_array(
        observed_validity,
        name="observed_validity",
        ndim=2,
    )
    physical_future = _float64_array(
        physical_future_m,
        name="physical_future_m",
        ndim=3,
    )
    registered_mean = _float64_array(
        registered_last_residual_mean_m,
        name="registered_last_residual_mean_m",
        ndim=3,
        preserve_identity=True,
    )
    reference_covariance = _float64_array(
        reference_covariance_m2,
        name="reference_covariance_m2",
        ndim=4,
        preserve_identity=True,
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
    if not np.all(np.isfinite(physical_prefix)):
        raise ValueError("physical_prefix_m must be finite")
    if not np.all(np.isfinite(physical_future)):
        raise ValueError("physical_future_m must be finite")
    if not np.all(np.isfinite(registered_mean)):
        raise ValueError("registered_last_residual_mean_m must be finite")
    if not np.all(np.isfinite(reference_covariance)):
        raise ValueError("reference_covariance_m2 must be finite")
    if np.any(reference_covariance != 0.0):
        raise ValueError("reference_covariance_m2 must be the exact zero covariance")
    valid_values = observation[validity]
    if not np.all(np.isfinite(valid_values)):
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
    schedule: np.ndarray,
    reasons: tuple[str, ...],
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
        scale_schedule_sha256=_array_sha256(schedule),
        accepted=False,
        fallback_reasons=reasons,
        endpoint_config_id=None,
        endpoint_posterior_id=None,
        endpoint_prediction_ids=(),
        donor_covariance_sha256=None,
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
    """Execute the sole frozen source-side residual-history covariance path.

    The caller cannot inject a covariance donor, horizon list, horizon partition,
    or scale schedule. Any source-support or registered-mean failure returns the
    exact caller-owned ``last_residual`` mean and zero-covariance reference.
    """

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
    schedule = _scale_schedule(bins=bins, track_count=physical_future.shape[1])
    reasons: list[str] = []
    if any(
        count < REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK for count in counts
    ):
        reasons.append("insufficient-per-track-support")
    if _array_sha256(registered_mean) != _array_sha256(reconstructed_mean):
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
            schedule=schedule,
            reasons=tuple(sorted(reasons)),
            metadata=metadata,
        )

    config = DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1
    config_id = content_id(_endpoint_config_descriptor(config))
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
    residual_digest = _array_sha256(residual)
    validity_digest = _array_sha256(validity)
    posterior_id = content_id(
        _endpoint_posterior_descriptor(
            posterior,
            residual_history_sha256=residual_digest,
            validity_sha256=validity_digest,
            config_id=config_id,
        )
    )
    covariances: list[np.ndarray] = []
    prediction_ids: list[str] = []
    for horizon_steps in range(1, len(physical_future) + 1):
        prediction = predict_model_averaged_endpoint(
            posterior,
            horizon_steps=horizon_steps,
        )
        covariances.append(prediction.covariance_m2)
        prediction_ids.append(
            content_id(
                _endpoint_prediction_descriptor(
                    posterior_id=posterior_id,
                    horizon_steps=horizon_steps,
                    mean_m=prediction.mean_m,
                    covariance_m2=prediction.covariance_m2,
                    component_weights=prediction.component_weights,
                )
            )
        )
    donor_covariance = np.stack(covariances, axis=0)
    hybrid = compose_covariance_only_hybrid(
        registered_mean,
        donor_covariance,
        reference_predictor_id=REGISTERED_REFERENCE_PREDICTOR_ID,
        covariance_donor_id=REGISTERED_COVARIANCE_DONOR_ID,
        covariance_scale=schedule,
        metadata={
            "source_unit_id": source_unit,
            "source_provenance_id": provenance.provenance_id,
            "endpoint_config_id": config_id,
            "endpoint_posterior_id": posterior_id,
            "execution": "registered-source-only-single-path-v1",
        },
    )
    if hybrid.mean_m is not registered_mean:
        raise AssertionError("registered covariance path copied the reference mean")
    decision = RegisteredResidualHistoryDecisionV1(
        source_unit_id=source_unit,
        provenance_id=_sha256(provenance.provenance_id, name="provenance_id"),
        residual_history_sha256=residual_digest,
        validity_sha256=validity_digest,
        registered_mean_sha256=_array_sha256(registered_mean),
        reconstructed_reference_mean_sha256=_array_sha256(reconstructed_mean),
        reference_covariance_sha256=_array_sha256(reference_covariance),
        valid_observation_count_by_track=counts,
        future_horizon_count=len(registered_mean),
        future_horizon_bins=tuple(int(value) for value in bins),
        scale_schedule_sha256=_array_sha256(schedule),
        accepted=True,
        fallback_reasons=(),
        endpoint_config_id=config_id,
        endpoint_posterior_id=posterior_id,
        endpoint_prediction_ids=tuple(prediction_ids),
        donor_covariance_sha256=_array_sha256(donor_covariance),
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


__all__ = [
    "CLAIM_BOUNDARY",
    "REGISTERED_COVARIANCE_DONOR_ID",
    "REGISTERED_COVARIANCE_SCALES",
    "REGISTERED_DECISION_SCHEMA",
    "REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK",
    "REGISTERED_REFERENCE_PREDICTOR_ID",
    "REGISTERED_SCHEMA_VERSION",
    "REGISTERED_SOURCE_PROVENANCE_SCHEMA",
    "RegisteredResidualHistoryDecisionV1",
    "RegisteredResidualHistoryPredictionV1",
    "ResidualHistorySourceProvenanceV1",
    "run_registered_residual_history_v1",
]
