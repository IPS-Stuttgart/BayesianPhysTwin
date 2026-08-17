"""Strict BayesianPhysTwin admission for claim-bearing tree-sparse Prob4D rows.

The producer-owned loader verifies the portable artifact.  This module then
independently checks the evidence fields needed by BayesianPhysTwin and adapts
the causal transition/innovation gauge prior to the native precision-form
solver without constructing a dense gauge covariance.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,
)
from .explicit_gauge_prob4d import (
    PROB4D_FACTOR_API_VERSION,
    PROB4D_FROZEN_FACTOR_REPOSITORY,
    _array_sha256,
    _calibration_ids,
    _require_integer,
    _require_mapping,
    _require_revision,
    _require_sha256,
    _require_string,
    _string_tuple,
    _validate_linearization,
    _validate_provider_attestation,
)
from .physical_linearization import PhysicalLinearizationV1
from .prior_aware_gauge_belief import PriorAwareGaugeConfigV1
from .prior_aware_gauge_belief_v2 import (
    update_sparse_prior_aware_gauge_belief_v2 as update_sparse_prior_aware_gauge_belief,
)
from .prospective_prob4d_update import ClaimBearingProb4DUpdateV1
from .sparse_prior_aware_gauge_belief import (
    SPARSE_PRIOR_AWARE_GAUGE_SOLVER_VERSION,
    TreeSparseGaugeDesignV1,
)

TREE_SPARSE_EXPLICIT_GAUGE_BRIDGE_VERSION = 1
PROB4D_TREE_SPARSE_OBSERVATION_SCHEMA_VERSION = 1
PROB4D_TREE_SPARSE_ENVELOPE_SCHEMA_VERSION = 1
PROB4D_TREE_PRIOR_SEMANTICS = (
    "zero-mean-linearized-causal-tree-independent-innovations-v1"
)
_REQUIRED_TREE_SPARSE_CAPABILITIES = frozenset(
    {
        "content_addressed_tree_sparse_observation_artifacts",
        "strict_claim_bearing_tree_sparse_observation_loading",
    }
)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return cast(Sequence[object], value)


def _integer_vector(
    value: object,
    *,
    name: str,
    count: int,
    minimum: int | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.shape != (count,)
        or not np.issubdtype(raw.dtype, np.integer)
        or raw.dtype.kind == "b"
    ):
        raise TypeError(f"{name} must be an integer vector of shape ({count},)")
    result = np.asarray(raw, dtype=np.int64)
    if minimum is not None and np.any(result < minimum):
        raise ValueError(f"{name} values must be at least {minimum}")
    return result


def _float_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _design_array(
    value: object,
    *,
    name: str,
    count: int,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or result.shape[:2] != (count, 3):
        raise ValueError(f"{name} must have shape ({count}, 3, D)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _probability_vector(
    value: object,
    *,
    name: str,
    count: int,
    strictly_positive: bool,
) -> np.ndarray:
    result = _float_array(value, name=name, shape=(count,))
    lower = result > 0.0 if strictly_positive else result >= 0.0
    if not np.all(lower) or np.any(result > 1.0):
        interval = "(0, 1]" if strictly_positive else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _validate_extended_provider_manifest(
    envelope: Any,
    *,
    provider_manifest_id: str,
) -> None:
    attestation = _require_mapping(
        envelope.provider_attestation,
        name="provider_attestation",
    )
    manifest = _require_mapping(
        attestation.get("provider_manifest"),
        name="provider_manifest",
    )
    if manifest.get("manifest_id") != provider_manifest_id:
        raise ValueError("provider manifest identity differs from the envelope")
    api_version = _require_integer(
        manifest.get("provider_api_version"),
        name="provider_api_version",
        minimum=1,
    )
    if api_version != PROB4D_FACTOR_API_VERSION:
        raise ValueError("tree-sparse observations require provider API version 2")
    capabilities = _sequence(
        manifest.get("capabilities"),
        name="provider capabilities",
    )
    capability_names = {
        _require_string(value, name="provider capability") for value in capabilities
    }
    if not _REQUIRED_TREE_SPARSE_CAPABILITIES.issubset(capability_names):
        raise ValueError("provider manifest lacks tree-sparse claim capabilities")
    versions = _require_mapping(
        manifest.get("artifact_schema_versions"),
        name="provider artifact schema versions",
    )
    observation_version = _require_integer(
        versions.get("TreeSparseObservationArtifactV1"),
        name="TreeSparseObservationArtifactV1 version",
        minimum=1,
    )
    envelope_version = _require_integer(
        versions.get("ClaimBearingTreeSparseObservationEnvelopeV1"),
        name="ClaimBearingTreeSparseObservationEnvelopeV1 version",
        minimum=1,
    )
    if observation_version != PROB4D_TREE_SPARSE_OBSERVATION_SCHEMA_VERSION:
        raise ValueError("provider manifest changed the tree-sparse artifact version")
    if envelope_version != PROB4D_TREE_SPARSE_ENVELOPE_SCHEMA_VERSION:
        raise ValueError("provider manifest changed the tree-sparse envelope version")


def _lineage_bounds(
    value: object,
    *,
    gauge_ids: tuple[str, ...],
    causal_frame_stop: int,
) -> dict[str, tuple[int, int]]:
    lineage = _require_mapping(value, name="causal_source_lineage")
    lineage_version = _require_integer(
        lineage.get("schema_version"),
        name="causal source lineage schema_version",
        minimum=1,
    )
    if lineage_version != 1:
        raise ValueError("causal source lineage changed schema version")
    if lineage.get("producer") != "Prob4D":
        raise ValueError("causal source lineage changed producer")
    stop = _require_integer(
        lineage.get("causal_frame_stop_exclusive"),
        name="causal lineage frame stop",
        minimum=1,
    )
    if stop != causal_frame_stop:
        raise ValueError("causal source lineage differs from the envelope cutoff")
    future_opened = _require_integer(
        lineage.get("future_prediction_payloads_opened"),
        name="future_prediction_payloads_opened",
        minimum=0,
    )
    if future_opened != 0:
        raise ValueError("causal source lineage opened future prediction payloads")
    windows = _sequence(
        lineage.get("selected_windows"),
        name="selected_windows",
    )
    bounds: dict[str, tuple[int, int]] = {}
    for raw_window in windows:
        window = _require_mapping(raw_window, name="selected window")
        window_id = _require_string(window.get("window_id"), name="window_id")
        if window_id in bounds:
            raise ValueError("causal source lineage contains duplicate windows")
        start = _require_integer(
            window.get("source_frame_start"),
            name=f"{window_id} source_frame_start",
            minimum=0,
        )
        window_stop = _require_integer(
            window.get("source_frame_stop_exclusive"),
            name=f"{window_id} source_frame_stop_exclusive",
            minimum=1,
        )
        if not start < window_stop <= causal_frame_stop:
            raise ValueError("causal source lineage contains an invalid window")
        bounds[window_id] = (start, window_stop)
    if tuple(bounds) != gauge_ids and set(bounds) != set(gauge_ids):
        raise ValueError("causal source lineage gauges differ from the envelope")
    return bounds


def _validate_tree_prior(
    prior: Any,
    *,
    gauge_ids: tuple[str, ...],
    prior_id: str,
) -> dict[str, np.ndarray]:
    prior_gauge_ids = _string_tuple(prior.gauge_ids, name="tree prior gauge_ids")
    if prior_gauge_ids != gauge_ids:
        raise ValueError("tree prior gauge order differs from the envelope")
    if _require_sha256(prior.prior_id, name="tree prior prior_id") != prior_id:
        raise ValueError("tree prior identity differs from the envelope")
    if prior.representation_semantics != PROB4D_TREE_PRIOR_SEMANTICS:
        raise ValueError("tree prior representation semantics changed")
    count = len(gauge_ids)
    parents = _integer_vector(
        prior.parent_indices,
        name="tree prior parent_indices",
        count=count,
    )
    if parents[0] != -1:
        raise ValueError("the first tree gauge must be the unique root")
    if count > 1 and np.any(parents[1:] < 0):
        raise ValueError("only the first tree gauge may be a root")
    for index in range(1, count):
        if int(parents[index]) >= index:
            raise ValueError("every tree parent must precede its child")
    transitions = _float_array(
        prior.transition_matrices,
        name="tree prior transition_matrices",
        shape=(count, 7, 7),
    )
    scales = _float_array(
        prior.innovation_scale_tril,
        name="tree prior innovation_scale_tril",
        shape=(count, 7, 7),
    )
    if not np.allclose(transitions[0], 0.0, atol=1e-14, rtol=0.0):
        raise ValueError("the root transition matrix must be zero")
    if not np.allclose(scales, np.tril(scales), atol=1e-14, rtol=0.0):
        raise ValueError("tree innovation scales must be lower triangular")
    if not np.all(np.diagonal(scales, axis1=1, axis2=2) > 0.0):
        raise ValueError("tree innovation scales must have positive diagonal")
    return {
        "parent_indices": parents,
        "transition_matrices": transitions,
        "innovation_scale_tril": scales,
    }


def _validate_rows(
    factors: Any,
    *,
    observation_count: int,
    gauge_ids: tuple[str, ...],
    causal_frame_stop: int,
    lineage_bounds: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    count = observation_count
    mean = _float_array(
        factors.world_mean_m,
        name="world_mean_m",
        shape=(count, 3),
    )
    conditional = _float_array(
        factors.conditional_world_covariance_m2,
        name="conditional_world_covariance_m2",
        shape=(count, 3, 3),
    )
    if not np.allclose(
        conditional,
        conditional.swapaxes(1, 2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("conditional point covariances must be symmetric")
    if np.any(np.linalg.eigvalsh(conditional) <= 0.0):
        raise ValueError("conditional point covariances must be positive definite")
    local_gauge = _float_array(
        factors.local_gauge_jacobian,
        name="local_gauge_jacobian",
        shape=(count, 3, 7),
    )
    gauge_indices = _integer_vector(
        factors.gauge_indices,
        name="gauge_indices",
        count=count,
        minimum=0,
    )
    if np.any(gauge_indices >= len(gauge_ids)):
        raise ValueError("gauge_indices reference an unknown gauge")
    association = _probability_vector(
        factors.association_probability,
        name="association_probability",
        count=count,
        strictly_positive=True,
    )
    reliability = _probability_vector(
        factors.prior_reliability,
        name="prior_reliability",
        count=count,
        strictly_positive=True,
    )
    nominal = _probability_vector(
        factors.prior_nominal_probability,
        name="prior_nominal_probability",
        count=count,
        strictly_positive=False,
    )
    composite = _probability_vector(
        factors.composite_weight,
        name="composite_weight",
        count=count,
        strictly_positive=True,
    )
    point_ids = _integer_vector(
        factors.point_ids,
        name="point_ids",
        count=count,
        minimum=0,
    )
    frame_indices = _integer_vector(
        factors.frame_indices,
        name="frame_indices",
        count=count,
        minimum=0,
    )
    if np.any(frame_indices >= causal_frame_stop):
        raise ValueError("tree-sparse rows cross the exclusive causal frame stop")
    view_ids = _string_tuple(factors.view_ids, name="view_ids")
    factor_ids = _string_tuple(factors.factor_ids, name="factor_ids")
    groups = _string_tuple(
        factors.correlation_group_ids,
        name="correlation_group_ids",
    )
    if not (len(view_ids) == len(factor_ids) == len(groups) == count):
        raise ValueError("tree-sparse string identities must contain one value per row")
    factor_gauge_ids = _string_tuple(factors.gauge_ids, name="factor gauge_ids")
    if factor_gauge_ids != gauge_ids:
        raise ValueError("tree-sparse row gauges differ from the envelope")
    if (
        _require_integer(
            factors.causal_frame_stop,
            name="factor causal_frame_stop",
            minimum=1,
        )
        != causal_frame_stop
    ):
        raise ValueError("tree-sparse rows differ from the envelope cutoff")
    for index in range(count):
        gauge_id = gauge_ids[int(gauge_indices[index])]
        start, stop = lineage_bounds[gauge_id]
        frame = int(frame_indices[index])
        if not start <= frame < stop:
            raise ValueError("tree-sparse row lies outside its causal source window")
    grouped: dict[str, tuple[float, float]] = {}
    for index, group in enumerate(groups):
        settings = (float(nominal[index]), float(composite[index]))
        if group in grouped and grouped[group] != settings:
            raise ValueError(
                "rows in one correlation group changed nominal probability "
                "or composite weight"
            )
        grouped[group] = settings
    return {
        "mean": mean,
        "conditional": conditional,
        "local_gauge": local_gauge,
        "gauge_indices": gauge_indices,
        "association": association,
        "reliability": reliability,
        "nominal": nominal,
        "composite": composite,
        "point_ids": point_ids,
        "frame_indices": frame_indices,
        "view_ids": view_ids,
        "factor_ids": factor_ids,
        "groups": groups,
    }


def _tree_stack_sha256(
    stack: Mapping[str, Any],
    tree: Mapping[str, np.ndarray],
    *,
    gauge_ids: tuple[str, ...],
    prior_id: str,
    causal_frame_stop: int,
) -> str:
    arrays = {
        name: _array_sha256(stack[name])
        for name in (
            "mean",
            "conditional",
            "local_gauge",
            "gauge_indices",
            "association",
            "reliability",
            "nominal",
            "composite",
            "point_ids",
            "frame_indices",
        )
    }
    arrays.update(
        {
            "parent_indices": _array_sha256(tree["parent_indices"]),
            "transition_matrices": _array_sha256(tree["transition_matrices"]),
            "innovation_scale_tril": _array_sha256(tree["innovation_scale_tril"]),
        }
    )
    record = {
        "schema": "bayesian-phystwin-prob4d-tree-sparse-binding-v1",
        "arrays": arrays,
        "view_ids": list(stack["view_ids"]),
        "factor_ids": list(stack["factor_ids"]),
        "correlation_group_ids": list(stack["groups"]),
        "gauge_ids": list(gauge_ids),
        "gauge_tree_prior_id": prior_id,
        "causal_frame_stop": causal_frame_stop,
    }
    return hashlib.sha256(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_claim_bearing_tree_sparse(
    validated: Any,
) -> tuple[
    str,
    str,
    Mapping[str, str],
    str,
    tuple[str, ...],
    str,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    envelope = validated.envelope
    observation = validated.observation
    manifest = observation.manifest
    factors = observation.factors
    artifact_id = _require_sha256(
        envelope.artifact_id,
        name="tree-sparse envelope artifact_id",
    )
    if (
        _require_sha256(validated.artifact_id, name="validated artifact_id")
        != artifact_id
    ):
        raise ValueError("validated tree-sparse artifact ID differs from its envelope")
    version = _require_integer(
        envelope.observation_artifact_schema_version,
        name="observation_artifact_schema_version",
        minimum=1,
    )
    if version != PROB4D_TREE_SPARSE_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            "claim-bearing tree-sparse observations require schema version 1"
        )
    repository = _require_string(envelope.source_repository, name="source_repository")
    if repository != PROB4D_FROZEN_FACTOR_REPOSITORY:
        raise ValueError("tree-sparse envelope changed the frozen Prob4D identity")
    source_revision = _require_revision(
        envelope.source_revision,
        name="source_revision",
    )
    causal_frame_stop = _require_integer(
        envelope.causal_frame_stop,
        name="causal_frame_stop",
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
    prior_artifact_id = _require_sha256(
        envelope.gauge_tree_prior_artifact_id,
        name="gauge_tree_prior_artifact_id",
    )
    prior_id = _require_sha256(
        envelope.gauge_tree_prior_id,
        name="gauge_tree_prior_id",
    )
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
        raise ValueError("tree-sparse runtime revision is not independently verified")
    _validate_provider_attestation(
        envelope,
        provider_manifest_id=provider_manifest_id,
        calibration_ids=calibration_ids,
        runtime_source=runtime_source,
        source_revision=source_revision,
    )
    _validate_extended_provider_manifest(
        envelope,
        provider_manifest_id=provider_manifest_id,
    )
    mirrored = {
        "observation_artifact_id": _require_sha256(
            manifest.artifact_id,
            name="observation manifest artifact_id",
        ),
        "sequence_id": _require_string(manifest.sequence_id, name="sequence_id"),
        "case_id": _require_string(manifest.case_id, name="case_id"),
        "stream_id": _require_string(manifest.stream_id, name="stream_id"),
        "source_repository": _require_string(
            manifest.source_repository,
            name="manifest source_repository",
        ),
        "source_revision": _require_revision(
            manifest.source_revision,
            name="manifest source_revision",
        ),
        "causal_frame_stop": _require_integer(
            manifest.causal_frame_stop,
            name="manifest causal_frame_stop",
            minimum=1,
        ),
        "observation_count": _require_integer(
            manifest.observation_count,
            name="manifest observation_count",
            minimum=1,
        ),
        "gauge_tree_prior_artifact_id": _require_sha256(
            manifest.gauge_tree_prior_artifact_id,
            name="manifest gauge_tree_prior_artifact_id",
        ),
        "gauge_tree_prior_id": _require_sha256(
            manifest.gauge_tree_prior_id,
            name="manifest gauge_tree_prior_id",
        ),
    }
    expected = {
        "observation_artifact_id": _require_sha256(
            envelope.observation_artifact_id,
            name="envelope observation_artifact_id",
        ),
        "sequence_id": _require_string(envelope.sequence_id, name="sequence_id"),
        "case_id": _require_string(envelope.case_id, name="case_id"),
        "stream_id": _require_string(envelope.stream_id, name="stream_id"),
        "source_repository": repository,
        "source_revision": source_revision,
        "causal_frame_stop": causal_frame_stop,
        "observation_count": observation_count,
        "gauge_tree_prior_artifact_id": prior_artifact_id,
        "gauge_tree_prior_id": prior_id,
    }
    for name, value in expected.items():
        if mirrored[name] != value:
            raise ValueError(
                f"tree-sparse observation differs from envelope field {name}"
            )
    manifest_gauges = _string_tuple(manifest.gauge_ids, name="manifest gauge_ids")
    if manifest_gauges != gauge_ids:
        raise ValueError(
            "tree-sparse observation gauge order differs from the envelope"
        )
    bounds = _lineage_bounds(
        envelope.causal_source_lineage,
        gauge_ids=gauge_ids,
        causal_frame_stop=causal_frame_stop,
    )
    stack = _validate_rows(
        factors,
        observation_count=observation_count,
        gauge_ids=gauge_ids,
        causal_frame_stop=causal_frame_stop,
        lineage_bounds=bounds,
    )
    tree = _validate_tree_prior(
        factors.gauge_tree_prior,
        gauge_ids=gauge_ids,
        prior_id=prior_id,
    )
    return (
        artifact_id,
        provider_manifest_id,
        calibration_ids,
        runtime_source,
        gauge_ids,
        prior_id,
        stack,
        tree,
    )


@dataclass(frozen=True, slots=True)
class ClaimBearingTreeSparseProb4DAdapterResult:
    """Strict tree-sparse inputs for the precision-form Bayesian update."""

    batch: GaugeAwareObservationBatch
    tree_gauge_design: TreeSparseGaugeDesignV1
    observation_artifact_id: str
    linearization_artifact_id: str
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    gauge_ids: tuple[str, ...]
    view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.batch, GaugeAwareObservationBatch):
            raise TypeError("batch must be a GaugeAwareObservationBatch")
        if not isinstance(self.tree_gauge_design, TreeSparseGaugeDesignV1):
            raise TypeError("tree_gauge_design must be a TreeSparseGaugeDesignV1")
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
            "gauge_ids",
            _string_tuple(self.gauge_ids, name="gauge_ids"),
        )
        object.__setattr__(
            self,
            "view_ids",
            _string_tuple(self.view_ids, name="view_ids"),
        )
        if self.tree_gauge_design.gauge_ids != self.gauge_ids:
            raise ValueError("adapter gauge IDs differ from the tree gauge design")


def build_claim_bearing_tree_sparse_prob4d_batch(
    validated_observation: Any,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
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
) -> ClaimBearingTreeSparseProb4DAdapterResult:
    """Validate and adapt one claim-bearing portable tree-sparse observation."""

    (
        artifact_id,
        provider_manifest_id,
        calibration_ids,
        runtime_source,
        gauge_ids,
        prior_id,
        stack,
        tree,
    ) = _validate_claim_bearing_tree_sparse(validated_observation)
    count = len(stack["mean"])
    _validate_linearization(
        linearization,
        observation_artifact_id=artifact_id,
        frame_indices=stack["frame_indices"],
        point_ids=stack["point_ids"],
        view_ids=stack["view_ids"],
        gauge_indices=stack["gauge_indices"],
    )
    physical_prediction = _float_array(
        physical_prediction_xyz_m,
        name="physical_prediction_xyz_m",
        shape=(count, 3),
    )
    shared = (
        np.zeros((count, 3, 0), dtype=np.float64)
        if shared_bias_jacobian is None
        else _design_array(
            shared_bias_jacobian,
            name="shared_bias_jacobian",
            count=count,
        )
    )
    view = (
        np.zeros((count, 3, 0), dtype=np.float64)
        if view_bias_jacobian is None
        else _design_array(
            view_bias_jacobian,
            name="view_bias_jacobian",
            count=count,
        )
    )
    tree_design = TreeSparseGaugeDesignV1(
        local_gauge_jacobian=stack["local_gauge"],
        gauge_indices=stack["gauge_indices"],
        parent_indices=tree["parent_indices"],
        transition_matrices=tree["transition_matrices"],
        innovation_scale_tril=tree["innovation_scale_tril"],
        gauge_ids=gauge_ids,
        prior_id=prior_id,
    )
    tree_hash = _tree_stack_sha256(
        stack,
        tree,
        gauge_ids=gauge_ids,
        prior_id=prior_id,
        causal_frame_stop=int(validated_observation.envelope.causal_frame_stop),
    )
    extra_metadata = frozen_finite_json_mapping(metadata)
    reserved_metadata: dict[str, Any] = {
        "observation_artifact_id": artifact_id,
        "linearization_artifact_id": linearization.artifact_id,
        "baseline_belief_id": linearization.baseline_belief_id,
        "action_prefix_id": linearization.action_prefix_id,
        "simulator_revision": linearization.simulator_revision,
        "row_alignment_verified": True,
        "prob4d_claim_bearing_provider_v2_validated": True,
        "prob4d_claim_bearing_tree_sparse_bridge_version": (
            TREE_SPARSE_EXPLICIT_GAUGE_BRIDGE_VERSION
        ),
        "prob4d_tree_sparse_observation_schema_version": (
            PROB4D_TREE_SPARSE_OBSERVATION_SCHEMA_VERSION
        ),
        "prob4d_tree_sparse_envelope_schema_version": (
            PROB4D_TREE_SPARSE_ENVELOPE_SCHEMA_VERSION
        ),
        "prob4d_claim_bearing_tree_sparse_binding_sha256": tree_hash,
        "prob4d_claim_bearing_provider_manifest_id": provider_manifest_id,
        "prob4d_claim_bearing_calibration_artifact_ids": dict(calibration_ids),
        "prob4d_claim_bearing_runtime_revision_source": runtime_source,
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
        "prob4d_gauge_tree_prior_id": prior_id,
        "prob4d_gauge_tree_prior_semantics": PROB4D_TREE_PRIOR_SEMANTICS,
        "prob4d_explicit_gauge_covariance_semantics": (
            "conditional-point-plus-causal-tree-gauge-prior-v1"
        ),
        "prob4d_marginal_point_covariance_consumed": False,
        "prob4d_dense_gauge_prior_covariance_materialized": False,
        "prob4d_dense_gauge_design_materialized": False,
        "prob4d_dense_gauge_design_avoided_bytes": (
            tree_design.equivalent_dense_design_bytes
        ),
        "prob4d_dense_gauge_prior_avoided_bytes": (
            tree_design.dense_gauge_prior_avoided_bytes
        ),
        "prob4d_tree_factor_storage_nbytes": tree_design.tree_factor_storage_nbytes,
        "prob4d_native_sparse_gauge_solver_version": (
            SPARSE_PRIOR_AWARE_GAUGE_SOLVER_VERSION
        ),
        "prob4d_association_probability_semantics": (
            "generalized-Bayes-row-power-not-source-reliability-v1"
        ),
        "prob4d_association_probability_sha256": _array_sha256(stack["association"]),
        "prob4d_source_reliability_sha256": _array_sha256(stack["reliability"]),
        "prob4d_prior_nominal_probability_sha256": _array_sha256(stack["nominal"]),
        "prob4d_provider_composite_weight_sha256": _array_sha256(stack["composite"]),
        "prob4d_gauge_ids": list(gauge_ids),
        "prob4d_view_ids_canonical_order": sorted(set(stack["view_ids"])),
        "physical_response_scale_source": (
            "PhysicalLinearizationV1.physical_response_m"
        ),
    }
    collisions = set(extra_metadata) & set(reserved_metadata)
    if collisions:
        raise ValueError(
            "metadata overrides reserved tree-sparse explicit-gauge fields: "
            f"{sorted(collisions)}"
        )
    extra_plain = plain_json(extra_metadata)
    if not isinstance(extra_plain, dict):
        raise TypeError("validated metadata lost its mapping type")
    batch = GaugeAwareObservationBatch(
        innovation_m=stack["mean"] - physical_prediction,
        observation_covariance_m2=stack["conditional"],
        state_jacobian=linearization.state_jacobian,
        gauge_jacobian=np.zeros((count, 3, 0), dtype=np.float64),
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        query_state_jacobian=linearization.query_state_jacobian,
        gauge_prior_covariance=np.zeros((0, 0), dtype=np.float64),
        correlation_group_ids=stack["groups"],
        prior_reliability=stack["reliability"],
        prior_nominal_probability=stack["nominal"],
        composite_weight=stack["composite"],
        association_probability=stack["association"],
        physical_response_scale_m=linearization.physical_response_scale_m,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        anchor_correlation_group_ids=anchor_correlation_group_ids,
        anchor_prior_reliability=anchor_prior_reliability,
        anchor_prior_nominal_probability=anchor_prior_nominal_probability,
        anchor_composite_weight=anchor_composite_weight,
        anchor_bias_jacobian=anchor_bias_jacobian,
        anchor_bias_prior_covariance=anchor_bias_prior_covariance,
        metadata={**cast(dict[str, Any], extra_plain), **reserved_metadata},
        composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    )
    return ClaimBearingTreeSparseProb4DAdapterResult(
        batch=batch,
        tree_gauge_design=tree_design,
        observation_artifact_id=artifact_id,
        linearization_artifact_id=linearization.artifact_id,
        provider_manifest_id=provider_manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source=runtime_source,
        gauge_ids=gauge_ids,
        view_ids=stack["view_ids"],
    )


def update_claim_bearing_tree_sparse_prob4d_from_artifacts(
    validated_observation: Any,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
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
    """Run one claim-bearing precision-form update from an admitted artifact."""

    adapted = build_claim_bearing_tree_sparse_prob4d_batch(
        validated_observation,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        anchor_correlation_group_ids=anchor_correlation_group_ids,
        anchor_prior_reliability=anchor_prior_reliability,
        anchor_prior_nominal_probability=anchor_prior_nominal_probability,
        anchor_composite_weight=anchor_composite_weight,
        anchor_bias_jacobian=anchor_bias_jacobian,
        anchor_bias_prior_covariance=anchor_bias_prior_covariance,
        metadata=metadata,
    )
    result = update_sparse_prior_aware_gauge_belief(
        adapted.batch,
        adapted.tree_gauge_design,
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


def load_claim_bearing_tree_sparse_prob4d(
    envelope_path: str | Path,
) -> Any:
    """Load through Prob4D's strict surface without widening package imports."""

    try:
        provider = importlib.import_module("prob4d.provider_v2_factors")
    except ImportError as error:
        raise ImportError(
            "loading a tree-sparse Prob4D artifact requires a compatible "
            "Prob4D installation"
        ) from error
    loader = getattr(
        provider,
        "load_claim_bearing_tree_sparse_observation",
        None,
    )
    if not callable(loader):
        raise ImportError("installed Prob4D lacks the claim-bearing tree-sparse loader")
    return loader(Path(envelope_path))


def update_claim_bearing_tree_sparse_prob4d_from_path(
    envelope_path: str | Path,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
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
    """Load a strict artifact lazily and run the precision-form update."""

    validated = load_claim_bearing_tree_sparse_prob4d(envelope_path)
    return update_claim_bearing_tree_sparse_prob4d_from_artifacts(
        validated,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        anchor_correlation_group_ids=anchor_correlation_group_ids,
        anchor_prior_reliability=anchor_prior_reliability,
        anchor_prior_nominal_probability=anchor_prior_nominal_probability,
        anchor_composite_weight=anchor_composite_weight,
        anchor_bias_jacobian=anchor_bias_jacobian,
        anchor_bias_prior_covariance=anchor_bias_prior_covariance,
        config=config,
        metadata=metadata,
    )


__all__ = [
    "ClaimBearingTreeSparseProb4DAdapterResult",
    "PROB4D_TREE_PRIOR_SEMANTICS",
    "PROB4D_TREE_SPARSE_ENVELOPE_SCHEMA_VERSION",
    "PROB4D_TREE_SPARSE_OBSERVATION_SCHEMA_VERSION",
    "TREE_SPARSE_EXPLICIT_GAUGE_BRIDGE_VERSION",
    "build_claim_bearing_tree_sparse_prob4d_batch",
    "load_claim_bearing_tree_sparse_prob4d",
    "update_claim_bearing_tree_sparse_prob4d_from_artifacts",
    "update_claim_bearing_tree_sparse_prob4d_from_path",
]
