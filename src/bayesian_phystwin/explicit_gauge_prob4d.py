"""Strict Bayesian-PhysTwin admission for explicit-gauge Prob4D factors.

The producer-owned ``prob4d.provider_v2_factors`` surface validates and loads the
neutral schema-v4 factor bundle, its claim-bearing envelope, and a sparse
``3 x 7`` gauge design. This module independently checks the fields required by
Bayesian-PhysTwin before an innovation is formed, then creates a bounded dense
compatibility batch for the existing prior-aware solver.

The compatibility bridge is deliberately fail-closed: it consumes conditional
point covariance and the complete joint gauge prior exactly once, keeps source
reliability and nominal-component probability separate, treats association
probability as an explicit generalized-Bayes row power, and refuses a dense
``M x 3 x 7K`` expansion above a declared byte limit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    integer_array,
    plain_json,
)
from ._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,
)
from .physical_linearization import PhysicalLinearizationV1
from .prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)
from .prospective_prob4d_update import ClaimBearingProb4DUpdateV1

EXPLICIT_GAUGE_FACTOR_BRIDGE_VERSION = 1
PROB4D_FACTOR_API_VERSION = 2
PROB4D_FACTOR_BUNDLE_SCHEMA_VERSION = 4
PROB4D_FROZEN_FACTOR_REPOSITORY = "FlorianPfaff/Prob4D"
DEFAULT_MAXIMUM_DENSE_GAUGE_DESIGN_BYTES = 256 * 1024 * 1024

_CALIBRATION_FIELDS = frozenset({"gauge_artifact_id", "point_artifact_id"})


class _GaugeDescriptor(Protocol):
    window_id: str


class _BundleProtocol(Protocol):
    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    causal_frame_stop: int
    factors: Sequence[object]
    gauges: Sequence[_GaugeDescriptor]


class _EnvelopeProtocol(Protocol):
    artifact_id: str | None
    bundle_schema_version: int
    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    causal_frame_stop: int
    factor_count: int
    observation_count: int
    gauge_ids: tuple[str, ...]
    gauge_covariance_semantics: str
    cross_window_gauge_covariance_preserved: bool
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    runtime_revision_independently_verified: bool
    provider_attestation: Mapping[str, Any]


class _ValidatedBundleProtocol(Protocol):
    bundle: _BundleProtocol
    envelope: _EnvelopeProtocol

    @property
    def artifact_id(self) -> str: ...


class _SparseStackProtocol(Protocol):
    world_mean_m: np.ndarray
    conditional_world_covariance_m2: np.ndarray
    marginal_world_covariance_m2: np.ndarray
    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    gauge_prior_covariance: np.ndarray
    association_probability: np.ndarray
    prior_reliability: np.ndarray
    prior_nominal_probability: np.ndarray
    composite_weight: np.ndarray
    point_ids: np.ndarray
    frame_indices: np.ndarray
    view_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    correlation_group_ids: tuple[str, ...]
    gauge_ids: tuple[str, ...]
    causal_frame_stop: int


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_revision(value: object, *, name: str) -> str:
    revision = _require_string(value, name=name)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase Git commit")
    return revision


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a genuine integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _calibration_ids(value: object) -> Mapping[str, str]:
    mapping = _require_mapping(value, name="calibration_artifact_ids")
    if set(mapping) != _CALIBRATION_FIELDS:
        raise ValueError(
            "calibration_artifact_ids must contain exactly "
            "gauge_artifact_id and point_artifact_id"
        )
    normalized = {
        name: _require_sha256(
            mapping[name],
            name=f"calibration artifact {name}",
        )
        for name in sorted(_CALIBRATION_FIELDS)
    }
    return MappingProxyType(normalized)


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"{name} must be a nonempty list or tuple")
    return tuple(_require_string(item, name=f"{name} item") for item in value)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_view_indices(view_ids: tuple[str, ...]) -> np.ndarray:
    names = tuple(sorted(set(view_ids)))
    positions = {name: index for index, name in enumerate(names)}
    return np.asarray([positions[name] for name in view_ids], dtype=np.int64)


def _validate_provider_attestation(
    envelope: _EnvelopeProtocol,
    *,
    provider_manifest_id: str,
    calibration_ids: Mapping[str, str],
    runtime_source: str,
    source_revision: str,
) -> None:
    attestation = _require_mapping(
        envelope.provider_attestation,
        name="provider_attestation",
    )
    if attestation.get("claim_bearing") is not True:
        raise ValueError("provider attestation is not claim-bearing")
    if attestation.get("export_mode") != "calibrated":
        raise ValueError("provider attestation is not calibrated")
    if attestation.get("provider_revision") != source_revision:
        raise ValueError(
            "provider attestation revision differs from the factor envelope"
        )
    if attestation.get("provider_manifest_id") != provider_manifest_id:
        raise ValueError(
            "provider attestation manifest differs from the factor envelope"
        )
    attested_calibration = _calibration_ids(attestation.get("calibration_artifact_ids"))
    if dict(attested_calibration) != dict(calibration_ids):
        raise ValueError(
            "provider attestation calibration IDs differ from the factor envelope"
        )
    runtime = _require_mapping(
        attestation.get("runtime_revision"),
        name="provider runtime revision",
    )
    if runtime.get("source") != runtime_source:
        raise ValueError("provider runtime source differs from the factor envelope")
    if runtime.get("independently_verified") is not True:
        raise ValueError("provider runtime revision is not independently verified")


def _validate_envelope_and_bundle(
    validated_bundle: _ValidatedBundleProtocol,
) -> tuple[
    str,
    str,
    Mapping[str, str],
    str,
    tuple[str, ...],
    int,
]:
    envelope = validated_bundle.envelope
    bundle = validated_bundle.bundle
    artifact_id = _require_sha256(
        envelope.artifact_id,
        name="factor envelope artifact_id",
    )
    if (
        _require_sha256(
            validated_bundle.artifact_id,
            name="validated factor artifact_id",
        )
        != artifact_id
    ):
        raise ValueError("validated factor artifact ID differs from its envelope")
    if (
        _require_integer(
            envelope.bundle_schema_version,
            name="bundle_schema_version",
            minimum=1,
        )
        != PROB4D_FACTOR_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("claim-bearing factors require schema version 4")
    repository = _require_string(
        envelope.source_repository,
        name="source_repository",
    )
    if repository != PROB4D_FROZEN_FACTOR_REPOSITORY:
        raise ValueError(
            "factor envelope does not use the frozen Prob4D producer identity"
        )
    source_revision = _require_revision(
        envelope.source_revision,
        name="source_revision",
    )
    causal_frame_stop = _require_integer(
        envelope.causal_frame_stop,
        name="causal_frame_stop",
        minimum=1,
    )
    factor_count = _require_integer(
        envelope.factor_count,
        name="factor_count",
        minimum=1,
    )
    observation_count = _require_integer(
        envelope.observation_count,
        name="observation_count",
        minimum=1,
    )
    gauge_ids = _string_tuple(envelope.gauge_ids, name="gauge_ids")
    if len(set(gauge_ids)) != len(gauge_ids):
        raise ValueError("gauge_ids must be unique")
    if envelope.gauge_covariance_semantics != "joint-cross-window":
        raise ValueError(
            "claim-bearing explicit gauges require joint cross-window covariance"
        )
    if envelope.cross_window_gauge_covariance_preserved is not True:
        raise ValueError("claim-bearing explicit gauges lost cross-window covariance")
    provider_manifest_id = _require_sha256(
        envelope.provider_manifest_id,
        name="provider_manifest_id",
    )
    calibration_ids = _calibration_ids(envelope.calibration_artifact_ids)
    runtime_source = _require_string(
        envelope.runtime_revision_source,
        name="runtime_revision_source",
    )
    if envelope.runtime_revision_independently_verified is not True:
        raise ValueError(
            "runtime_revision_independently_verified must be literally True"
        )
    _validate_provider_attestation(
        envelope,
        provider_manifest_id=provider_manifest_id,
        calibration_ids=calibration_ids,
        runtime_source=runtime_source,
        source_revision=source_revision,
    )

    expected_strings = {
        "sequence_id": _require_string(
            envelope.sequence_id,
            name="sequence_id",
        ),
        "case_id": _require_string(envelope.case_id, name="case_id"),
        "stream_id": _require_string(
            envelope.stream_id,
            name="stream_id",
        ),
        "source_repository": repository,
        "source_revision": source_revision,
    }
    for name, expected in expected_strings.items():
        if getattr(bundle, name) != expected:
            raise ValueError(f"factor bundle differs from envelope field {name}")
    if (
        _require_integer(
            bundle.causal_frame_stop,
            name="bundle causal_frame_stop",
            minimum=1,
        )
        != causal_frame_stop
    ):
        raise ValueError("factor bundle differs from envelope causal_frame_stop")
    if len(bundle.factors) != factor_count:
        raise ValueError("factor bundle differs from envelope factor_count")
    bundle_gauge_ids = tuple(
        _require_string(gauge.window_id, name="bundle gauge window_id")
        for gauge in bundle.gauges
    )
    if bundle_gauge_ids != gauge_ids:
        raise ValueError("factor bundle gauges differ from the envelope")

    return (
        artifact_id,
        provider_manifest_id,
        calibration_ids,
        runtime_source,
        gauge_ids,
        observation_count,
    )


def _validate_stack(
    stack: _SparseStackProtocol,
    *,
    gauge_ids: tuple[str, ...],
    causal_frame_stop: int,
    observation_count: int,
) -> dict[str, Any]:
    mean = np.asarray(stack.world_mean_m, dtype=np.float64)
    conditional = np.asarray(
        stack.conditional_world_covariance_m2,
        dtype=np.float64,
    )
    marginal = np.asarray(
        stack.marginal_world_covariance_m2,
        dtype=np.float64,
    )
    local_gauge = np.asarray(
        stack.local_gauge_jacobian,
        dtype=np.float64,
    )
    gauge_indices = integer_array(
        stack.gauge_indices,
        name="gauge_indices",
    )
    gauge_prior = np.asarray(
        stack.gauge_prior_covariance,
        dtype=np.float64,
    )
    association = np.asarray(
        stack.association_probability,
        dtype=np.float64,
    )
    reliability = np.asarray(stack.prior_reliability, dtype=np.float64)
    nominal = np.asarray(
        stack.prior_nominal_probability,
        dtype=np.float64,
    )
    composite = np.asarray(stack.composite_weight, dtype=np.float64)
    point_ids = integer_array(stack.point_ids, name="point_ids")
    frame_indices = integer_array(
        stack.frame_indices,
        name="frame_indices",
    )
    view_ids = _string_tuple(stack.view_ids, name="view_ids")
    factor_ids = _string_tuple(stack.factor_ids, name="factor_ids")
    groups = _string_tuple(
        stack.correlation_group_ids,
        name="correlation_group_ids",
    )
    stack_gauge_ids = _string_tuple(stack.gauge_ids, name="stack gauge_ids")
    count = len(mean)
    gauge_count = len(gauge_ids)
    gauge_dimension = 7 * gauge_count

    if count != observation_count:
        raise ValueError("sparse factor stack differs from envelope observation_count")
    if mean.shape != (count, 3):
        raise ValueError("world_mean_m must have shape (M, 3)")
    if conditional.shape != (count, 3, 3):
        raise ValueError("conditional_world_covariance_m2 must have shape (M, 3, 3)")
    if marginal.shape != (count, 3, 3):
        raise ValueError("marginal_world_covariance_m2 must have shape (M, 3, 3)")
    if local_gauge.shape != (count, 3, 7):
        raise ValueError("local_gauge_jacobian must have shape (M, 3, 7)")
    if gauge_indices.shape != (count,):
        raise ValueError("gauge_indices must have shape (M,)")
    if stack_gauge_ids != gauge_ids:
        raise ValueError("sparse factor gauges differ from the envelope")
    if np.any(gauge_indices < 0) or np.any(gauge_indices >= gauge_count):
        raise ValueError("gauge_indices reference an unknown gauge")
    if gauge_prior.shape != (gauge_dimension, gauge_dimension):
        raise ValueError("joint gauge prior has changed shape")
    if not np.all(np.isfinite(gauge_prior)):
        raise ValueError("joint gauge prior must be finite")
    if not np.allclose(gauge_prior, gauge_prior.T, atol=1e-12, rtol=1e-10):
        raise ValueError("joint gauge prior must be symmetric")
    if (
        np.min(
            np.linalg.eigvalsh(0.5 * (gauge_prior + gauge_prior.T)),
            initial=0.0,
        )
        < -1e-12
    ):
        raise ValueError("joint gauge prior must be positive semidefinite")

    for name, values, strictly_positive in (
        ("association_probability", association, True),
        ("prior_reliability", reliability, True),
        ("prior_nominal_probability", nominal, False),
        ("composite_weight", composite, True),
    ):
        if values.shape != (count,):
            raise ValueError(f"{name} must have shape (M,)")
        lower = values > 0.0 if strictly_positive else values >= 0.0
        if not np.all(np.isfinite(values)) or not np.all(lower) or np.any(values > 1.0):
            interval = "(0, 1]" if strictly_positive else "[0, 1]"
            raise ValueError(f"{name} must lie in {interval}")

    if point_ids.shape != (count,) or frame_indices.shape != (count,):
        raise ValueError("factor row identity arrays must have shape (M,)")
    if np.any(point_ids < 0):
        raise ValueError("point_ids must be nonnegative")
    if np.any(frame_indices < 0) or np.any(frame_indices >= causal_frame_stop):
        raise ValueError("factor rows cross the exclusive causal frame stop")
    if not (len(view_ids) == len(factor_ids) == len(groups) == count):
        raise ValueError("factor string identities must contain one value per row")
    if (
        _require_integer(
            stack.causal_frame_stop,
            name="stack causal_frame_stop",
            minimum=1,
        )
        != causal_frame_stop
    ):
        raise ValueError("sparse factor stack differs from envelope causal_frame_stop")
    active_arrays = (
        mean,
        conditional,
        marginal,
        local_gauge,
    )
    if any(not np.all(np.isfinite(values)) for values in active_arrays):
        raise ValueError("sparse factor stack contains non-finite values")
    if not np.allclose(
        conditional,
        conditional.swapaxes(1, 2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("conditional point covariances must be symmetric")
    if np.any(np.linalg.eigvalsh(conditional) <= 0.0):
        raise ValueError("conditional point covariances must be positive definite")
    if not np.allclose(
        marginal,
        marginal.swapaxes(1, 2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("marginal point covariances must be symmetric")
    if np.any(np.linalg.eigvalsh(marginal) < -1e-12):
        raise ValueError("marginal point covariances must be positive semidefinite")

    return {
        "mean": mean,
        "conditional": conditional,
        "marginal": marginal,
        "local_gauge": local_gauge,
        "gauge_indices": np.asarray(gauge_indices, dtype=np.int64),
        "gauge_prior": gauge_prior,
        "association": association,
        "reliability": reliability,
        "nominal": nominal,
        "composite": composite,
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "view_ids": view_ids,
        "factor_ids": factor_ids,
        "groups": groups,
    }


def _validate_linearization(
    linearization: PhysicalLinearizationV1,
    *,
    observation_artifact_id: str,
    frame_indices: np.ndarray,
    point_ids: np.ndarray,
    view_ids: tuple[str, ...],
    gauge_indices: np.ndarray,
) -> None:
    if not isinstance(linearization, PhysicalLinearizationV1):
        raise TypeError("linearization must be a PhysicalLinearizationV1")
    if linearization.observation_artifact_id != observation_artifact_id:
        raise ValueError(
            "physical linearization does not identify this factor envelope"
        )
    expected = {
        "frame_ids": frame_indices,
        "entity_ids": point_ids,
        "view_indices": _canonical_view_indices(view_ids),
        "window_indices": gauge_indices,
    }
    for name, values in expected.items():
        observed = integer_array(
            getattr(linearization, name),
            name=f"linearization {name}",
        )
        if not np.array_equal(observed, values):
            raise ValueError(f"factor rows and physical linearization {name} differ")


def _dense_gauge_design(
    local_gauge_jacobian: np.ndarray,
    gauge_indices: np.ndarray,
    *,
    gauge_count: int,
    maximum_bytes: int,
) -> tuple[np.ndarray, int]:
    count = len(local_gauge_jacobian)
    gauge_dimension = 7 * gauge_count
    required_bytes = count * 3 * gauge_dimension * np.dtype(np.float64).itemsize
    if required_bytes > maximum_bytes:
        raise MemoryError(
            "explicit-gauge dense compatibility design requires "
            f"{required_bytes} bytes, exceeding the declared "
            f"{maximum_bytes}-byte limit"
        )
    design: np.ndarray = np.zeros(
        (count, 3, gauge_dimension),
        dtype=np.float64,
    )
    for gauge_index in range(gauge_count):
        selected = gauge_indices == gauge_index
        start = 7 * gauge_index
        design[selected, :, start : start + 7] = local_gauge_jacobian[selected]
    return design, required_bytes


@dataclass(frozen=True, slots=True)
class ExplicitGaugeFactorAdapterResult:
    """A strict factor envelope adapted to the existing prior-aware solver."""

    batch: GaugeAwareObservationBatch
    observation_artifact_id: str
    linearization_artifact_id: str
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    dense_gauge_design_bytes: int
    dense_gauge_design_limit_bytes: int
    gauge_ids: tuple[str, ...]
    view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.batch, GaugeAwareObservationBatch):
            raise TypeError("batch must be a GaugeAwareObservationBatch")
        object.__setattr__(
            self,
            "observation_artifact_id",
            _require_sha256(
                self.observation_artifact_id,
                name="observation_artifact_id",
            ),
        )
        object.__setattr__(
            self,
            "linearization_artifact_id",
            _require_sha256(
                self.linearization_artifact_id,
                name="linearization_artifact_id",
            ),
        )
        object.__setattr__(
            self,
            "provider_manifest_id",
            _require_sha256(
                self.provider_manifest_id,
                name="provider_manifest_id",
            ),
        )
        object.__setattr__(
            self,
            "calibration_artifact_ids",
            _calibration_ids(self.calibration_artifact_ids),
        )
        object.__setattr__(
            self,
            "runtime_revision_source",
            _require_string(
                self.runtime_revision_source,
                name="runtime_revision_source",
            ),
        )
        object.__setattr__(
            self,
            "dense_gauge_design_bytes",
            _require_integer(
                self.dense_gauge_design_bytes,
                name="dense_gauge_design_bytes",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "dense_gauge_design_limit_bytes",
            _require_integer(
                self.dense_gauge_design_limit_bytes,
                name="dense_gauge_design_limit_bytes",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "gauge_ids",
            _string_tuple(self.gauge_ids, name="gauge_ids"),
        )
        object.__setattr__(
            self,
            "view_ids",
            _string_tuple(self.view_ids, name="view_ids"),
        )


def build_claim_bearing_explicit_gauge_batch(
    validated_bundle: _ValidatedBundleProtocol,
    sparse_stack: _SparseStackProtocol,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    maximum_dense_gauge_design_bytes: int = (DEFAULT_MAXIMUM_DENSE_GAUGE_DESIGN_BYTES),
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    anchor_correlation_group_ids: tuple[str, ...] | None = None,
    anchor_prior_reliability: np.ndarray | None = None,
    anchor_prior_nominal_probability: np.ndarray | None = None,
    anchor_composite_weight: np.ndarray | None = None,
    anchor_bias_jacobian: np.ndarray | None = None,
    anchor_bias_prior_covariance: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExplicitGaugeFactorAdapterResult:
    """Validate and adapt a strict explicit-gauge factor bundle.

    The adapter is a bounded compatibility bridge for the existing dense
    prior-aware solver. It performs every provenance and memory check before
    allocating the dense zero-block gauge design.
    """

    (
        artifact_id,
        provider_manifest_id,
        calibration_ids,
        runtime_source,
        gauge_ids,
        observation_count,
    ) = _validate_envelope_and_bundle(validated_bundle)
    causal_frame_stop = _require_integer(
        validated_bundle.envelope.causal_frame_stop,
        name="causal_frame_stop",
        minimum=1,
    )
    stack = _validate_stack(
        sparse_stack,
        gauge_ids=gauge_ids,
        causal_frame_stop=causal_frame_stop,
        observation_count=observation_count,
    )
    _validate_linearization(
        linearization,
        observation_artifact_id=artifact_id,
        frame_indices=stack["frame_indices"],
        point_ids=stack["point_ids"],
        view_ids=stack["view_ids"],
        gauge_indices=stack["gauge_indices"],
    )
    maximum_bytes = _require_integer(
        maximum_dense_gauge_design_bytes,
        name="maximum_dense_gauge_design_bytes",
        minimum=1,
    )
    gauge_design, design_bytes = _dense_gauge_design(
        stack["local_gauge"],
        stack["gauge_indices"],
        gauge_count=len(gauge_ids),
        maximum_bytes=maximum_bytes,
    )

    physical_prediction = np.asarray(
        physical_prediction_xyz_m,
        dtype=np.float64,
    )
    if physical_prediction.shape != (observation_count, 3):
        raise ValueError("physical_prediction_xyz_m must have shape (M, 3)")
    if not np.all(np.isfinite(physical_prediction)):
        raise ValueError("physical_prediction_xyz_m must be finite")
    shared = (
        np.zeros((observation_count, 3, 0), dtype=np.float64)
        if shared_bias_jacobian is None
        else np.asarray(shared_bias_jacobian, dtype=np.float64)
    )
    view = (
        np.zeros((observation_count, 3, 0), dtype=np.float64)
        if view_bias_jacobian is None
        else np.asarray(view_bias_jacobian, dtype=np.float64)
    )
    row_power = stack["association"] * stack["composite"]
    if np.any(row_power <= 0.0):
        raise ValueError("association-weighted composite power must be positive")

    extra_metadata = frozen_finite_json_mapping(metadata)
    reserved_metadata: dict[str, Any] = {
        "observation_artifact_id": artifact_id,
        "linearization_artifact_id": linearization.artifact_id,
        "baseline_belief_id": linearization.baseline_belief_id,
        "action_prefix_id": linearization.action_prefix_id,
        "simulator_revision": linearization.simulator_revision,
        "row_alignment_verified": True,
        "prob4d_claim_bearing_provider_v2_validated": True,
        "prob4d_claim_bearing_factor_api_version": (PROB4D_FACTOR_API_VERSION),
        "prob4d_claim_bearing_factor_bundle_schema_version": (
            PROB4D_FACTOR_BUNDLE_SCHEMA_VERSION
        ),
        "prob4d_claim_bearing_factor_bundle_envelope_artifact_id": (artifact_id),
        "prob4d_claim_bearing_provider_manifest_id": (provider_manifest_id),
        "prob4d_claim_bearing_calibration_artifact_ids": dict(calibration_ids),
        "prob4d_claim_bearing_runtime_revision_source": runtime_source,
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
        "prob4d_explicit_gauge_covariance_semantics": (
            "conditional-point-plus-explicit-joint-gauge-prior-v1"
        ),
        "prob4d_marginal_point_covariance_consumed": False,
        "prob4d_association_probability_semantics": (
            "generalized-Bayes-row-power-not-source-reliability-v1"
        ),
        "prob4d_association_probability_sha256": _array_sha256(stack["association"]),
        "prob4d_source_reliability_sha256": _array_sha256(stack["reliability"]),
        "prob4d_prior_nominal_probability_sha256": _array_sha256(stack["nominal"]),
        "prob4d_provider_composite_weight_sha256": _array_sha256(stack["composite"]),
        "prob4d_dense_compatibility_bridge": True,
        "prob4d_dense_gauge_design_bytes": design_bytes,
        "prob4d_dense_gauge_design_limit_bytes": maximum_bytes,
        "prob4d_gauge_ids": list(gauge_ids),
        "prob4d_view_ids_canonical_order": sorted(set(stack["view_ids"])),
        "prob4d_factor_ids_sha256": hashlib.sha256(
            json.dumps(
                list(stack["factor_ids"]),
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "physical_response_scale_source": (
            "PhysicalLinearizationV1.physical_response_m"
        ),
    }
    collisions = set(extra_metadata) & set(reserved_metadata)
    if collisions:
        raise ValueError(
            f"metadata overrides reserved explicit-gauge fields: {sorted(collisions)}"
        )
    extra_plain = plain_json(extra_metadata)
    if not isinstance(extra_plain, dict):
        raise TypeError("validated metadata lost its mapping type")
    batch_metadata = {
        **extra_plain,
        **reserved_metadata,
    }

    batch = GaugeAwareObservationBatch(
        innovation_m=stack["mean"] - physical_prediction,
        observation_covariance_m2=stack["conditional"],
        state_jacobian=linearization.state_jacobian,
        gauge_jacobian=gauge_design,
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        query_state_jacobian=linearization.query_state_jacobian,
        gauge_prior_covariance=stack["gauge_prior"],
        correlation_group_ids=stack["groups"],
        prior_reliability=stack["reliability"],
        prior_nominal_probability=stack["nominal"],
        composite_weight=row_power,
        physical_response_scale_m=(linearization.physical_response_scale_m),
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        anchor_correlation_group_ids=anchor_correlation_group_ids,
        anchor_prior_reliability=anchor_prior_reliability,
        anchor_prior_nominal_probability=(anchor_prior_nominal_probability),
        anchor_composite_weight=anchor_composite_weight,
        anchor_bias_jacobian=anchor_bias_jacobian,
        anchor_bias_prior_covariance=anchor_bias_prior_covariance,
        metadata=batch_metadata,
        composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    )
    return ExplicitGaugeFactorAdapterResult(
        batch=batch,
        observation_artifact_id=artifact_id,
        linearization_artifact_id=linearization.artifact_id,
        provider_manifest_id=provider_manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source=runtime_source,
        dense_gauge_design_bytes=design_bytes,
        dense_gauge_design_limit_bytes=maximum_bytes,
        gauge_ids=gauge_ids,
        view_ids=stack["view_ids"],
    )


def update_claim_bearing_explicit_gauge_from_artifacts(
    validated_bundle: _ValidatedBundleProtocol,
    sparse_stack: _SparseStackProtocol,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    maximum_dense_gauge_design_bytes: int = (DEFAULT_MAXIMUM_DENSE_GAUGE_DESIGN_BYTES),
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    anchor_correlation_group_ids: tuple[str, ...] | None = None,
    anchor_prior_reliability: np.ndarray | None = None,
    anchor_prior_nominal_probability: np.ndarray | None = None,
    anchor_composite_weight: np.ndarray | None = None,
    anchor_bias_jacobian: np.ndarray | None = None,
    anchor_bias_prior_covariance: np.ndarray | None = None,
    config: PriorAwareGaugeConfigV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ClaimBearingProb4DUpdateV1:
    """Run one strict explicit-gauge update or exact solver fallback."""

    adapted = build_claim_bearing_explicit_gauge_batch(
        validated_bundle,
        sparse_stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        maximum_dense_gauge_design_bytes=(maximum_dense_gauge_design_bytes),
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        anchor_correlation_group_ids=anchor_correlation_group_ids,
        anchor_prior_reliability=anchor_prior_reliability,
        anchor_prior_nominal_probability=(anchor_prior_nominal_probability),
        anchor_composite_weight=anchor_composite_weight,
        anchor_bias_jacobian=anchor_bias_jacobian,
        anchor_bias_prior_covariance=anchor_bias_prior_covariance,
        metadata=metadata,
    )
    result = update_prior_aware_gauge_belief(
        adapted.batch,
        config=config,
    )
    return ClaimBearingProb4DUpdateV1(
        result=result,
        observation_artifact_id=adapted.observation_artifact_id,
        linearization_artifact_id=adapted.linearization_artifact_id,
        provider_manifest_id=adapted.provider_manifest_id,
        calibration_artifact_ids=adapted.calibration_artifact_ids,
        runtime_revision_source=adapted.runtime_revision_source,
        runtime_revision_independently_verified=True,
    )


__all__ = [
    "DEFAULT_MAXIMUM_DENSE_GAUGE_DESIGN_BYTES",
    "EXPLICIT_GAUGE_FACTOR_BRIDGE_VERSION",
    "ExplicitGaugeFactorAdapterResult",
    "PROB4D_FACTOR_API_VERSION",
    "build_claim_bearing_explicit_gauge_batch",
    "update_claim_bearing_explicit_gauge_from_artifacts",
]
