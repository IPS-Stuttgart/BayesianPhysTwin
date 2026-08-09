"""Target-free object-level joint-sparse observability for Deform360.

Version 4 keeps every valid partial factor, marginalizes the registered nuisance
model, and decides support at the physical-object/query level.  The calculation
uses conditional covariances and Jacobians only; point estimates, residuals,
calibration outcomes, confirmation payloads, and target outcomes are outside the
contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    immutable_array,
    immutable_integer_array,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    sha256_digest,
    source_artifact_mapping,
)

DEFORM360_JOINT_SPARSE_VERSION: Final = 4
DEFORM360_JOINT_SPARSE_PROTOCOL_ID: Final = (
    "deform360-official-hub-joint-sparse-development-v4"
)
DEFORM360_JOINT_SPARSE_POLICY_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-observability-policy"
)
DEFORM360_JOINT_SPARSE_INPUT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-observability-input"
)
DEFORM360_JOINT_SPARSE_RESULT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-observability-result"
)
DEFORM360_JOINT_SPARSE_REPORT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-observability-report"
)
DEFORM360_JOINT_SPARSE_SEMANTICS: Final = (
    "target-free-partial-factor-object-level-query-observability-v4"
)
DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY: Final = (
    "Development-only target-free structural observability evidence. A valid "
    "artifact does not establish Prob4D calibration, BayesianPhysTwin physical-"
    "query benefit, contact benefit, confirmation accuracy, Causal4D benefit, "
    "deployment safety, or state of the art."
)

Stratum = Literal["sheet", "volumetric"]
ResultStatus = Literal["evaluated", "technical-failure-without-replacement"]

_DEVELOPMENT_BOUNDARY: Final = {
    "development_cohort_only": True,
    "prediction_point_values_used": False,
    "prediction_residuals_used": False,
    "calibration_outcomes_used": False,
    "confirmation_payloads_opened": False,
    "adaptive_confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
    "human_approval_required": False,
    "new_measurements_required": False,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _literal(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    _require(result == result.strip(), f"{name} has surrounding whitespace")
    return result


def _stratum(value: object) -> Stratum:
    _require(type(value) is str and value in {"sheet", "volumetric"}, "invalid stratum")
    return cast(Stratum, value)


def _positive(value: object, *, name: str, allow_zero: bool = False) -> float:
    _require(not isinstance(value, (bool, np.bool_)), f"{name} is not numeric")
    array = np.asarray(value)
    _require(array.shape == () and array.dtype.kind in "iuf", f"{name} is not scalar")
    result = float(array.item())
    _require(np.isfinite(result), f"{name} is not finite")
    _require(result >= 0.0 if allow_zero else result > 0.0, f"{name} is not positive")
    return result


def _fraction(value: object, *, name: str, strictly_positive: bool = False) -> float:
    result = _positive(value, name=name, allow_zero=not strictly_positive)
    _require(result <= 1.0, f"{name} exceeds one")
    return result


def _strings(
    values: Sequence[str], *, name: str, count: int | None = None, unique: bool = False
) -> tuple[str, ...]:
    _require(not isinstance(values, (str, bytes)), f"{name} is not a sequence")
    result = tuple(_literal(value, name=f"{name} item") for value in values)
    _require(count is None or len(result) == count, f"{name} length changed")
    _require(not unique or len(set(result)) == len(result), f"{name} repeats values")
    return result


def _float_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind in "iuf", f"{name} is not real")
    result = np.asarray(raw, dtype=np.dtype("<f8"), order="C")
    _require(result.ndim == ndim and np.all(np.isfinite(result)), f"invalid {name}")
    return immutable_array(result, dtype=np.dtype("<f8"))


def _integer_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    result = immutable_integer_array(value, name=name)
    _require(result.ndim == ndim, f"invalid {name}")
    return result


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _array_records(values: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _array_digest(array),
        }
        for name, array in sorted(values.items())
    }


def _boundary(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    supplied = dict(_DEVELOPMENT_BOUNDARY if value is None else value)
    _require(supplied == _DEVELOPMENT_BOUNDARY, "v4 information boundary changed")
    return frozen_finite_json_mapping(supplied, name="information_boundary")


def default_deform360_joint_sparse_information_boundary_v4() -> Mapping[str, Any]:
    """Return an immutable copy of the mandatory development-only boundary."""

    return _boundary(None)


@dataclass(frozen=True, slots=True)
class Deform360JointSparseObservabilityPolicyV4:
    """Frozen thresholds for a development-only object-level design gate."""

    minimum_distinct_cameras: int = 2
    minimum_distinct_windows: int = 2
    minimum_distinct_spatial_clusters: int = 8
    minimum_supported_objects: int = 8
    minimum_supported_objects_per_stratum: int = 4
    require_full_query_rank: bool = True
    minimum_query_precision_eigenvalue: float = 1e-9
    maximum_query_condition_number: float = 1e10
    maximum_single_camera_information_fraction: float = 0.85
    minimum_leave_one_camera_rank_fraction: float = 0.75
    minimum_leave_one_window_rank_fraction: float = 0.75
    effective_samples_per_correlation_group: float = 64.0
    shared_bias_prior_std_m: float = 0.02
    view_bias_prior_std_m: float = 0.01
    relative_rank_tolerance: float = 1e-9
    absolute_rank_tolerance: float = 1e-12
    information_boundary: Mapping[str, Any] = field(
        default_factory=default_deform360_joint_sparse_information_boundary_v4
    )
    protocol_id: str = DEFORM360_JOINT_SPARSE_PROTOCOL_ID
    policy_id: str | None = None

    def __post_init__(self) -> None:
        protocol = _literal(self.protocol_id, name="protocol_id")
        _require(protocol == DEFORM360_JOINT_SPARSE_PROTOCOL_ID, "protocol changed")
        integer_names = (
            "minimum_distinct_cameras",
            "minimum_distinct_windows",
            "minimum_distinct_spatial_clusters",
            "minimum_supported_objects",
            "minimum_supported_objects_per_stratum",
        )
        for name in integer_names:
            object.__setattr__(
                self,
                name,
                genuine_integer(getattr(self, name), name=name, minimum=1),
            )
        object.__setattr__(
            self,
            "require_full_query_rank",
            genuine_boolean(self.require_full_query_rank, name="require_full_query_rank"),
        )
        for name in (
            "minimum_query_precision_eigenvalue",
            "relative_rank_tolerance",
            "absolute_rank_tolerance",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name, allow_zero=True))
        for name in (
            "maximum_query_condition_number",
            "effective_samples_per_correlation_group",
            "shared_bias_prior_std_m",
            "view_bias_prior_std_m",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        for name in (
            "maximum_single_camera_information_fraction",
            "minimum_leave_one_camera_rank_fraction",
            "minimum_leave_one_window_rank_fraction",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name=name))
        object.__setattr__(self, "protocol_id", protocol)
        object.__setattr__(self, "information_boundary", _boundary(self.information_boundary))
        expected = content_id(self.identity_record())
        _require(self.policy_id is None or self.policy_id == expected, "policy_id changed")
        object.__setattr__(self, "policy_id", expected)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_JOINT_SPARSE_POLICY_SCHEMA,
            "schema_version": DEFORM360_JOINT_SPARSE_VERSION,
            "semantics": DEFORM360_JOINT_SPARSE_SEMANTICS,
            "protocol_id": self.protocol_id,
            "minimum_distinct_cameras": self.minimum_distinct_cameras,
            "minimum_distinct_windows": self.minimum_distinct_windows,
            "minimum_distinct_spatial_clusters": self.minimum_distinct_spatial_clusters,
            "minimum_supported_objects": self.minimum_supported_objects,
            "minimum_supported_objects_per_stratum": self.minimum_supported_objects_per_stratum,
            "require_full_query_rank": self.require_full_query_rank,
            "minimum_query_precision_eigenvalue": self.minimum_query_precision_eigenvalue,
            "maximum_query_condition_number": self.maximum_query_condition_number,
            "maximum_single_camera_information_fraction": self.maximum_single_camera_information_fraction,
            "minimum_leave_one_camera_rank_fraction": self.minimum_leave_one_camera_rank_fraction,
            "minimum_leave_one_window_rank_fraction": self.minimum_leave_one_window_rank_fraction,
            "effective_samples_per_correlation_group": self.effective_samples_per_correlation_group,
            "shared_bias_prior_std_m": self.shared_bias_prior_std_m,
            "view_bias_prior_std_m": self.view_bias_prior_std_m,
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "absolute_rank_tolerance": self.absolute_rank_tolerance,
            "information_boundary": plain_json(self.information_boundary),
            "claim_boundary": DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "policy_id": self.policy_id}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "Deform360JointSparseObservabilityPolicyV4":
        expected = {
            "schema",
            "schema_version",
            "semantics",
            "protocol_id",
            "minimum_distinct_cameras",
            "minimum_distinct_windows",
            "minimum_distinct_spatial_clusters",
            "minimum_supported_objects",
            "minimum_supported_objects_per_stratum",
            "require_full_query_rank",
            "minimum_query_precision_eigenvalue",
            "maximum_query_condition_number",
            "maximum_single_camera_information_fraction",
            "minimum_leave_one_camera_rank_fraction",
            "minimum_leave_one_window_rank_fraction",
            "effective_samples_per_correlation_group",
            "shared_bias_prior_std_m",
            "view_bias_prior_std_m",
            "relative_rank_tolerance",
            "absolute_rank_tolerance",
            "information_boundary",
            "claim_boundary",
            "policy_id",
        }
        _require(set(value) == expected, "policy fields changed")
        _require(value["schema"] == DEFORM360_JOINT_SPARSE_POLICY_SCHEMA, "policy schema changed")
        _require(value["schema_version"] == DEFORM360_JOINT_SPARSE_VERSION, "policy version changed")
        _require(value["semantics"] == DEFORM360_JOINT_SPARSE_SEMANTICS, "policy semantics changed")
        _require(value["claim_boundary"] == DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY, "claim boundary changed")
        return cls(**{key: value[key] for key in expected - {"schema", "schema_version", "semantics", "claim_boundary"}})


@dataclass(frozen=True, slots=True)
class Deform360JointSparseFactorBatchV4:
    """All active partial factors for one development object."""

    selection_artifact_sha256: str
    visual_provider_lock_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    implementation_revision: str
    object_id: str
    episode_id: int
    stratum: Stratum
    factor_ids: tuple[str, ...]
    camera_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    spatial_cluster_ids: tuple[str, ...]
    correlation_group_ids: tuple[str, ...]
    gauge_ids: tuple[str, ...]
    gauge_prior_id: str
    observation_covariance_m2: np.ndarray
    state_jacobian: np.ndarray
    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    parent_indices: np.ndarray
    transition_matrices: np.ndarray
    innovation_scale_tril: np.ndarray
    query_jacobian: np.ndarray
    prior_reliability: np.ndarray
    association_probability: np.ndarray
    composite_weight: np.ndarray
    shared_bias_jacobian: np.ndarray | None = None
    view_bias_jacobian: np.ndarray | None = None
    excluded_factor_count: int = 0
    source_artifacts: Mapping[str, str] = field(default_factory=dict)
    information_boundary: Mapping[str, Any] = field(
        default_factory=default_deform360_joint_sparse_information_boundary_v4
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_id: str = DEFORM360_JOINT_SPARSE_PROTOCOL_ID
    input_id: str | None = None

    def __post_init__(self) -> None:
        protocol = _literal(self.protocol_id, name="protocol_id")
        _require(protocol == DEFORM360_JOINT_SPARSE_PROTOCOL_ID, "protocol changed")
        for name in (
            "selection_artifact_sha256",
            "visual_provider_lock_id",
            "observation_artifact_id",
            "linearization_artifact_id",
            "gauge_prior_id",
        ):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "implementation_revision",
            exact_revision(self.implementation_revision, name="implementation_revision"),
        )
        object.__setattr__(self, "object_id", _literal(self.object_id, name="object_id"))
        object.__setattr__(self, "episode_id", genuine_integer(self.episode_id, name="episode_id", minimum=0))
        object.__setattr__(self, "stratum", _stratum(self.stratum))
        covariance = _float_array(self.observation_covariance_m2, name="observation_covariance_m2", ndim=3)
        state = _float_array(self.state_jacobian, name="state_jacobian", ndim=3)
        local = _float_array(self.local_gauge_jacobian, name="local_gauge_jacobian", ndim=3)
        count = len(covariance)
        _require(count > 0 and covariance.shape == (count, 3, 3), "invalid observation covariance shape")
        _require(state.shape[:2] == (count, 3) and state.shape[2] > 0, "invalid state Jacobian shape")
        _require(local.shape[:2] == (count, 3) and local.shape[2] > 0, "invalid gauge Jacobian shape")
        for matrix in covariance:
            _require(np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12), "nonsymmetric covariance")
            try:
                np.linalg.cholesky(0.5 * (matrix + matrix.T))
            except np.linalg.LinAlgError as error:
                raise ValueError("non-positive-definite covariance") from error
        factor_ids = tuple(sha256_digest(value, name="factor_id") for value in _strings(self.factor_ids, name="factor_ids", count=count, unique=True))
        camera_ids = _strings(self.camera_ids, name="camera_ids", count=count)
        window_ids = _strings(self.window_ids, name="window_ids", count=count)
        cluster_ids = _strings(self.spatial_cluster_ids, name="spatial_cluster_ids", count=count)
        group_ids = _strings(self.correlation_group_ids, name="correlation_group_ids", count=count)
        gauge_ids = _strings(self.gauge_ids, name="gauge_ids", unique=True)
        _require(bool(gauge_ids), "gauge_ids is empty")
        indices = _integer_array(self.gauge_indices, name="gauge_indices", ndim=1)
        parents = _integer_array(self.parent_indices, name="parent_indices", ndim=1)
        transitions = _float_array(self.transition_matrices, name="transition_matrices", ndim=3)
        scales = _float_array(self.innovation_scale_tril, name="innovation_scale_tril", ndim=3)
        gauge_count = len(gauge_ids)
        block_size = local.shape[2]
        _require(indices.shape == (count,) and np.all((indices >= 0) & (indices < gauge_count)), "invalid gauge indices")
        _require(parents.shape == (gauge_count,) and parents[0] == -1, "invalid gauge tree root")
        _require(all(0 <= int(parents[index]) < index for index in range(1, gauge_count)), "invalid gauge tree parent")
        _require(transitions.shape == (gauge_count, block_size, block_size), "invalid transition shape")
        _require(scales.shape == transitions.shape, "invalid innovation scale shape")
        _require(np.allclose(scales, np.tril(scales), atol=1e-14, rtol=0.0), "innovation scale is not triangular")
        _require(np.all(np.diagonal(scales, axis1=1, axis2=2) > 0.0), "innovation scale diagonal is not positive")
        query = _float_array(self.query_jacobian, name="query_jacobian", ndim=2)
        _require(query.shape[1] == state.shape[2] and query.shape[0] > 0, "invalid query Jacobian shape")
        singular = np.linalg.svd(query, compute_uv=False)
        tolerance = max(1e-12, 1e-10 * float(singular[0]))
        _require(int(np.count_nonzero(singular > tolerance)) == len(query), "query rows are dependent")
        probabilities: dict[str, np.ndarray] = {}
        for name, raw, strictly_positive in (
            ("prior_reliability", self.prior_reliability, False),
            ("association_probability", self.association_probability, False),
            ("composite_weight", self.composite_weight, True),
        ):
            values = _float_array(raw, name=name, ndim=1)
            _require(values.shape == (count,), f"{name} shape changed")
            lower = values > 0.0 if strictly_positive else values >= 0.0
            _require(np.all(lower & (values <= 1.0)), f"{name} leaves probability range")
            probabilities[name] = values
        shared = np.zeros((count, 3, 0), dtype=np.float64) if self.shared_bias_jacobian is None else _float_array(self.shared_bias_jacobian, name="shared_bias_jacobian", ndim=3)
        view = np.zeros((count, 3, 0), dtype=np.float64) if self.view_bias_jacobian is None else _float_array(self.view_bias_jacobian, name="view_bias_jacobian", ndim=3)
        _require(shared.shape[:2] == (count, 3), "shared-bias row shape changed")
        _require(view.shape[:2] == (count, 3), "view-bias row shape changed")
        object.__setattr__(self, "protocol_id", protocol)
        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "window_ids", window_ids)
        object.__setattr__(self, "spatial_cluster_ids", cluster_ids)
        object.__setattr__(self, "correlation_group_ids", group_ids)
        object.__setattr__(self, "gauge_ids", gauge_ids)
        object.__setattr__(self, "observation_covariance_m2", covariance)
        object.__setattr__(self, "state_jacobian", state)
        object.__setattr__(self, "local_gauge_jacobian", local)
        object.__setattr__(self, "gauge_indices", indices)
        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "transition_matrices", transitions)
        object.__setattr__(self, "innovation_scale_tril", scales)
        object.__setattr__(self, "query_jacobian", query)
        object.__setattr__(self, "shared_bias_jacobian", immutable_array(shared, dtype=np.dtype("<f8")))
        object.__setattr__(self, "view_bias_jacobian", immutable_array(view, dtype=np.dtype("<f8")))
        for name, values in probabilities.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "excluded_factor_count", genuine_integer(self.excluded_factor_count, name="excluded_factor_count", minimum=0))
        object.__setattr__(self, "source_artifacts", source_artifact_mapping(self.source_artifacts, name="source_artifacts", allow_empty=True))
        object.__setattr__(self, "information_boundary", _boundary(self.information_boundary))
        object.__setattr__(self, "metadata", frozen_finite_json_mapping(self.metadata, name="metadata"))
        expected = content_id(self.identity_record())
        _require(self.input_id is None or self.input_id == expected, "input_id changed")
        object.__setattr__(self, "input_id", expected)

    @property
    def state_dimension(self) -> int:
        return int(self.state_jacobian.shape[2])

    @property
    def query_dimension(self) -> int:
        return int(self.query_jacobian.shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "observation_covariance_m2": self.observation_covariance_m2,
            "state_jacobian": self.state_jacobian,
            "local_gauge_jacobian": self.local_gauge_jacobian,
            "gauge_indices": self.gauge_indices,
            "parent_indices": self.parent_indices,
            "transition_matrices": self.transition_matrices,
            "innovation_scale_tril": self.innovation_scale_tril,
            "query_jacobian": self.query_jacobian,
            "prior_reliability": self.prior_reliability,
            "association_probability": self.association_probability,
            "composite_weight": self.composite_weight,
            "shared_bias_jacobian": cast(np.ndarray, self.shared_bias_jacobian),
            "view_bias_jacobian": cast(np.ndarray, self.view_bias_jacobian),
        }

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_JOINT_SPARSE_INPUT_SCHEMA,
            "schema_version": DEFORM360_JOINT_SPARSE_VERSION,
            "semantics": DEFORM360_JOINT_SPARSE_SEMANTICS,
            "protocol_id": self.protocol_id,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "observation_artifact_id": self.observation_artifact_id,
            "linearization_artifact_id": self.linearization_artifact_id,
            "implementation_revision": self.implementation_revision,
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "stratum": self.stratum,
            "factor_ids": list(self.factor_ids),
            "camera_ids": list(self.camera_ids),
            "window_ids": list(self.window_ids),
            "spatial_cluster_ids": list(self.spatial_cluster_ids),
            "correlation_group_ids": list(self.correlation_group_ids),
            "gauge_ids": list(self.gauge_ids),
            "gauge_prior_id": self.gauge_prior_id,
            "excluded_factor_count": self.excluded_factor_count,
            "array_records": _array_records(self.arrays()),
            "source_artifacts": plain_json(self.source_artifacts),
            "information_boundary": plain_json(self.information_boundary),
            "metadata": plain_json(self.metadata),
            "claim_boundary": DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class _Spectrum:
    state_rank: int
    query_rank: int
    query_eigenvalues: np.ndarray
    minimum_eigenvalue: float
    condition_number: float | None
    trace_precision: float
    unobservable_fraction: float
    nuisance_dimension: int


@dataclass(frozen=True, slots=True)
class Deform360JointSparseObservabilityResultV4:
    """One object-level target-free observability decision."""

    input_id: str
    policy_id: str
    implementation_revision: str
    object_id: str
    episode_id: int
    stratum: Stratum
    status: ResultStatus
    factor_count: int
    excluded_factor_count: int
    distinct_camera_count: int
    distinct_window_count: int
    distinct_spatial_cluster_count: int
    distinct_correlation_group_count: int
    state_dimension: int | None
    query_dimension: int | None
    nuisance_dimension: int | None
    state_rank: int | None
    query_rank: int | None
    query_precision_eigenvalues: tuple[float, ...] | None
    minimum_query_precision_eigenvalue: float | None
    query_condition_number: float | None
    query_unobservable_fraction: float | None
    trace_query_precision: float | None
    single_camera_information_fraction: Mapping[str, float] | None
    leave_one_camera_rank_fraction: Mapping[str, float] | None
    leave_one_window_rank_fraction: Mapping[str, float] | None
    gate_checks: Mapping[str, bool] | None
    gate_passed: bool
    information_boundary: Mapping[str, Any]
    source_artifacts: Mapping[str, str]
    failure_reason: str | None = None
    failure_detail_sha256: str | None = None
    protocol_id: str = DEFORM360_JOINT_SPARSE_PROTOCOL_ID
    result_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_id", _literal(self.protocol_id, name="protocol_id"))
        _require(self.protocol_id == DEFORM360_JOINT_SPARSE_PROTOCOL_ID, "protocol changed")
        object.__setattr__(self, "input_id", sha256_digest(self.input_id, name="input_id"))
        object.__setattr__(self, "policy_id", sha256_digest(self.policy_id, name="policy_id"))
        object.__setattr__(self, "implementation_revision", exact_revision(self.implementation_revision, name="implementation_revision"))
        object.__setattr__(self, "object_id", _literal(self.object_id, name="object_id"))
        object.__setattr__(self, "episode_id", genuine_integer(self.episode_id, name="episode_id", minimum=0))
        object.__setattr__(self, "stratum", _stratum(self.stratum))
        _require(self.status in {"evaluated", "technical-failure-without-replacement"}, "invalid result status")
        for name in (
            "factor_count",
            "excluded_factor_count",
            "distinct_camera_count",
            "distinct_window_count",
            "distinct_spatial_cluster_count",
            "distinct_correlation_group_count",
        ):
            object.__setattr__(self, name, genuine_integer(getattr(self, name), name=name, minimum=0))
        object.__setattr__(self, "gate_passed", genuine_boolean(self.gate_passed, name="gate_passed"))
        object.__setattr__(self, "information_boundary", _boundary(self.information_boundary))
        object.__setattr__(self, "source_artifacts", source_artifact_mapping(self.source_artifacts, name="source_artifacts", allow_empty=True))
        if self.status == "evaluated":
            numerical = (
                self.state_dimension,
                self.query_dimension,
                self.nuisance_dimension,
                self.state_rank,
                self.query_rank,
                self.query_precision_eigenvalues,
                self.minimum_query_precision_eigenvalue,
                self.query_unobservable_fraction,
                self.trace_query_precision,
                self.single_camera_information_fraction,
                self.leave_one_camera_rank_fraction,
                self.leave_one_window_rank_fraction,
                self.gate_checks,
            )
            _require(all(value is not None for value in numerical), "evaluated result lacks diagnostics")
            _require(self.failure_reason is None and self.failure_detail_sha256 is None, "evaluated result contains failure")
            state_dimension = genuine_integer(self.state_dimension, name="state_dimension", minimum=1)
            query_dimension = genuine_integer(self.query_dimension, name="query_dimension", minimum=1)
            nuisance_dimension = genuine_integer(self.nuisance_dimension, name="nuisance_dimension", minimum=0)
            state_rank = genuine_integer(self.state_rank, name="state_rank", minimum=0)
            query_rank = genuine_integer(self.query_rank, name="query_rank", minimum=0)
            eigenvalues = tuple(_positive(value, name="query eigenvalue", allow_zero=True) for value in cast(tuple[float, ...], self.query_precision_eigenvalues))
            _require(len(eigenvalues) == query_dimension and tuple(sorted(eigenvalues)) == eigenvalues, "invalid query spectrum")
            minimum = _positive(self.minimum_query_precision_eigenvalue, name="minimum_query_precision_eigenvalue", allow_zero=True)
            _require(np.isclose(minimum, eigenvalues[0], rtol=1e-10, atol=1e-12), "minimum eigenvalue changed")
            condition = None if self.query_condition_number is None else _positive(self.query_condition_number, name="query_condition_number")
            _require((query_rank == query_dimension) == (condition is not None), "condition/rank mismatch")
            unobservable = _fraction(self.query_unobservable_fraction, name="query_unobservable_fraction")
            trace = _positive(self.trace_query_precision, name="trace_query_precision", allow_zero=True)
            camera = _fraction_mapping(cast(Mapping[str, float], self.single_camera_information_fraction), name="single_camera_information_fraction")
            leave_camera = _fraction_mapping(cast(Mapping[str, float], self.leave_one_camera_rank_fraction), name="leave_one_camera_rank_fraction")
            leave_window = _fraction_mapping(cast(Mapping[str, float], self.leave_one_window_rank_fraction), name="leave_one_window_rank_fraction")
            checks = _boolean_mapping(cast(Mapping[str, bool], self.gate_checks), name="gate_checks")
            _require(self.gate_passed == all(checks.values()), "gate decision changed")
            for name, value in (
                ("state_dimension", state_dimension),
                ("query_dimension", query_dimension),
                ("nuisance_dimension", nuisance_dimension),
                ("state_rank", state_rank),
                ("query_rank", query_rank),
                ("query_precision_eigenvalues", eigenvalues),
                ("minimum_query_precision_eigenvalue", minimum),
                ("query_condition_number", condition),
                ("query_unobservable_fraction", unobservable),
                ("trace_query_precision", trace),
                ("single_camera_information_fraction", frozen_finite_json_mapping(camera)),
                ("leave_one_camera_rank_fraction", frozen_finite_json_mapping(leave_camera)),
                ("leave_one_window_rank_fraction", frozen_finite_json_mapping(leave_window)),
                ("gate_checks", frozen_finite_json_mapping(checks)),
            ):
                object.__setattr__(self, name, value)
        else:
            numerical = (
                self.state_dimension,
                self.query_dimension,
                self.nuisance_dimension,
                self.state_rank,
                self.query_rank,
                self.query_precision_eigenvalues,
                self.minimum_query_precision_eigenvalue,
                self.query_condition_number,
                self.query_unobservable_fraction,
                self.trace_query_precision,
                self.single_camera_information_fraction,
                self.leave_one_camera_rank_fraction,
                self.leave_one_window_rank_fraction,
                self.gate_checks,
            )
            _require(all(value is None for value in numerical), "technical failure contains diagnostics")
            _require(not self.gate_passed, "technical failure passed gate")
            object.__setattr__(self, "failure_reason", _literal(self.failure_reason, name="failure_reason"))
            object.__setattr__(self, "failure_detail_sha256", sha256_digest(self.failure_detail_sha256, name="failure_detail_sha256"))
        expected = content_id(self.identity_record())
        _require(self.result_id is None or self.result_id == expected, "result_id changed")
        object.__setattr__(self, "result_id", expected)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_JOINT_SPARSE_RESULT_SCHEMA,
            "schema_version": DEFORM360_JOINT_SPARSE_VERSION,
            "semantics": DEFORM360_JOINT_SPARSE_SEMANTICS,
            "protocol_id": self.protocol_id,
            "input_id": self.input_id,
            "policy_id": self.policy_id,
            "implementation_revision": self.implementation_revision,
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "stratum": self.stratum,
            "status": self.status,
            "factor_count": self.factor_count,
            "excluded_factor_count": self.excluded_factor_count,
            "distinct_camera_count": self.distinct_camera_count,
            "distinct_window_count": self.distinct_window_count,
            "distinct_spatial_cluster_count": self.distinct_spatial_cluster_count,
            "distinct_correlation_group_count": self.distinct_correlation_group_count,
            "state_dimension": self.state_dimension,
            "query_dimension": self.query_dimension,
            "nuisance_dimension": self.nuisance_dimension,
            "state_rank": self.state_rank,
            "query_rank": self.query_rank,
            "query_precision_eigenvalues": None if self.query_precision_eigenvalues is None else list(self.query_precision_eigenvalues),
            "minimum_query_precision_eigenvalue": self.minimum_query_precision_eigenvalue,
            "query_condition_number": self.query_condition_number,
            "query_unobservable_fraction": self.query_unobservable_fraction,
            "trace_query_precision": self.trace_query_precision,
            "single_camera_information_fraction": None if self.single_camera_information_fraction is None else plain_json(self.single_camera_information_fraction),
            "leave_one_camera_rank_fraction": None if self.leave_one_camera_rank_fraction is None else plain_json(self.leave_one_camera_rank_fraction),
            "leave_one_window_rank_fraction": None if self.leave_one_window_rank_fraction is None else plain_json(self.leave_one_window_rank_fraction),
            "gate_checks": None if self.gate_checks is None else plain_json(self.gate_checks),
            "gate_passed": self.gate_passed,
            "failure_reason": self.failure_reason,
            "failure_detail_sha256": self.failure_detail_sha256,
            "information_boundary": plain_json(self.information_boundary),
            "source_artifacts": plain_json(self.source_artifacts),
            "claim_boundary": DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "result_id": self.result_id}


def _fraction_mapping(value: Mapping[str, float], *, name: str) -> dict[str, float]:
    _require(bool(value), f"{name} is empty")
    result = {_literal(key, name=f"{name} key"): _fraction(item, name=f"{name}.{key}") for key, item in value.items()}
    _require(len(result) == len(value), f"{name} repeats keys")
    return result


def _boolean_mapping(value: Mapping[str, bool], *, name: str) -> dict[str, bool]:
    _require(bool(value), f"{name} is empty")
    result = {_literal(key, name=f"{name} key"): genuine_boolean(item, name=f"{name}.{key}") for key, item in value.items()}
    _require(len(result) == len(value), f"{name} repeats keys")
    return result


def _tree_information(batch: Deform360JointSparseFactorBatchV4) -> np.ndarray:
    gauge_count = len(batch.gauge_ids)
    block_size = batch.local_gauge_jacobian.shape[2]
    information = np.zeros((gauge_count * block_size,) * 2, dtype=np.float64)
    identity = np.eye(block_size)
    for index in range(gauge_count):
        try:
            inverse = np.linalg.solve(batch.innovation_scale_tril[index], identity)
        except np.linalg.LinAlgError as error:
            raise ValueError("singular gauge innovation") from error
        precision = inverse.T @ inverse
        child = slice(index * block_size, (index + 1) * block_size)
        information[child, child] += precision
        parent_index = int(batch.parent_indices[index])
        if parent_index >= 0:
            parent = slice(parent_index * block_size, (parent_index + 1) * block_size)
            transition = batch.transition_matrices[index]
            information[child, parent] -= precision @ transition
            information[parent, child] -= transition.T @ precision
            information[parent, parent] += transition.T @ precision @ transition
    return 0.5 * (information + information.T)


def _row_weights(
    batch: Deform360JointSparseFactorBatchV4,
    policy: Deform360JointSparseObservabilityPolicyV4,
) -> np.ndarray:
    raw = batch.prior_reliability * batch.association_probability
    result = np.zeros(len(raw), dtype=np.float64)
    labels = np.asarray(batch.correlation_group_ids, dtype=object)
    for group in dict.fromkeys(batch.correlation_group_ids):
        selected = np.flatnonzero(labels == group)
        active = selected[raw[selected] > 0.0]
        if len(active):
            scale = min(policy.effective_samples_per_correlation_group, float(len(active))) / len(active)
            result[active] = raw[active] * float(batch.composite_weight[selected[0]]) * scale
    return result


def _whiten(value: np.ndarray) -> np.ndarray:
    try:
        factor = np.linalg.cholesky(0.5 * (value + value.T))
        return np.linalg.solve(factor, np.eye(len(value)))
    except np.linalg.LinAlgError as error:
        raise ValueError("observation covariance is not positive definite") from error


def _marginal_state_information(
    batch: Deform360JointSparseFactorBatchV4,
    policy: Deform360JointSparseObservabilityPolicyV4,
    mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    _require(mask.shape == (len(batch.factor_ids),) and mask.dtype.kind == "b", "invalid factor mask")
    state_dimension = batch.state_dimension
    gauge_dimension = len(batch.gauge_ids) * batch.local_gauge_jacobian.shape[2]
    shared_dimension = cast(np.ndarray, batch.shared_bias_jacobian).shape[2]
    view_dimension = cast(np.ndarray, batch.view_bias_jacobian).shape[2]
    nuisance_dimension = gauge_dimension + shared_dimension + view_dimension
    state_information = np.zeros((state_dimension, state_dimension))
    cross = np.zeros((state_dimension, nuisance_dimension))
    nuisance = np.zeros((nuisance_dimension, nuisance_dimension))
    nuisance[:gauge_dimension, :gauge_dimension] = _tree_information(batch)
    if shared_dimension:
        selected = slice(gauge_dimension, gauge_dimension + shared_dimension)
        nuisance[selected, selected] += np.eye(shared_dimension) / policy.shared_bias_prior_std_m**2
    if view_dimension:
        selected = slice(gauge_dimension + shared_dimension, nuisance_dimension)
        nuisance[selected, selected] += np.eye(view_dimension) / policy.view_bias_prior_std_m**2
    weights = _row_weights(batch, policy)
    block_size = batch.local_gauge_jacobian.shape[2]
    for index in np.flatnonzero(mask & (weights > 0.0)):
        whitener = _whiten(batch.observation_covariance_m2[index])
        state = whitener @ batch.state_jacobian[index]
        design = np.zeros((3, nuisance_dimension))
        gauge = int(batch.gauge_indices[index])
        gauge_slice = slice(gauge * block_size, (gauge + 1) * block_size)
        design[:, gauge_slice] = whitener @ batch.local_gauge_jacobian[index]
        shared_start = gauge_dimension
        shared_stop = shared_start + shared_dimension
        if shared_dimension:
            design[:, shared_start:shared_stop] = whitener @ cast(np.ndarray, batch.shared_bias_jacobian)[index]
        if view_dimension:
            design[:, shared_stop:] = whitener @ cast(np.ndarray, batch.view_bias_jacobian)[index]
        weight = float(weights[index])
        state_information += weight * state.T @ state
        cross += weight * state.T @ design
        nuisance += weight * design.T @ design
    try:
        marginal = state_information - cross @ np.linalg.solve(0.5 * (nuisance + nuisance.T), cross.T)
    except np.linalg.LinAlgError as error:
        raise ValueError("joint nuisance information is singular") from error
    marginal = 0.5 * (marginal + marginal.T)
    eigenvalues, eigenvectors = np.linalg.eigh(marginal)
    tolerance = max(policy.absolute_rank_tolerance, policy.relative_rank_tolerance * max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0))
    _require(np.all(eigenvalues >= -tolerance), "marginal state information is indefinite")
    clipped = np.maximum(eigenvalues, 0.0)
    return 0.5 * ((eigenvectors * clipped) @ eigenvectors.T + ((eigenvectors * clipped) @ eigenvectors.T).T), nuisance_dimension


def _nullspace(value: np.ndarray, tolerance: float) -> np.ndarray:
    _left, singular, right = np.linalg.svd(value, full_matrices=True)
    rank = int(np.count_nonzero(singular > tolerance))
    return right[rank:].T


def _spectrum(
    information: np.ndarray,
    query: np.ndarray,
    policy: Deform360JointSparseObservabilityPolicyV4,
    nuisance_dimension: int,
) -> _Spectrum:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (information + information.T))
    tolerance = max(policy.absolute_rank_tolerance, policy.relative_rank_tolerance * max(float(eigenvalues[-1]), 0.0))
    positive = eigenvalues > tolerance
    state_rank = int(np.count_nonzero(positive))
    null = eigenvectors[:, ~positive]
    query_null = query @ null
    singular = np.linalg.svd(query_null.T, compute_uv=False)
    null_tolerance = max(policy.absolute_rank_tolerance, policy.relative_rank_tolerance * float(singular[0] if len(singular) else 0.0))
    combinations = np.eye(len(query)) if null.shape[1] == 0 else _nullspace(query_null.T, null_tolerance)
    if state_rank == 0 or combinations.shape[1] == 0:
        query_information = np.zeros((len(query), len(query)))
    else:
        inverse_state = (eigenvectors[:, positive] * (1.0 / eigenvalues[positive])) @ eigenvectors[:, positive].T
        covariance = combinations.T @ query @ inverse_state @ query.T @ combinations
        covariance = 0.5 * (covariance + covariance.T)
        try:
            reduced_precision = np.linalg.solve(covariance, np.eye(len(covariance)))
        except np.linalg.LinAlgError as error:
            raise ValueError("identifiable query covariance is singular") from error
        query_information = combinations @ reduced_precision @ combinations.T
    query_eigenvalues = np.linalg.eigvalsh(0.5 * (query_information + query_information.T))
    query_scale = max(float(query_eigenvalues[-1]), 0.0)
    query_tolerance = max(policy.absolute_rank_tolerance, policy.relative_rank_tolerance * query_scale)
    _require(np.all(query_eigenvalues >= -query_tolerance), "query information is indefinite")
    query_eigenvalues = np.maximum(query_eigenvalues, 0.0)
    query_rank = int(np.count_nonzero(query_eigenvalues > query_tolerance))
    positive_query = query_eigenvalues[query_eigenvalues > query_tolerance]
    condition = None if query_rank < len(query) else float(positive_query[-1] / positive_query[0])
    query_norm = float(np.linalg.norm(query))
    unobservable = 0.0 if query_norm == 0.0 else min(1.0, float(np.linalg.norm(query_null) / query_norm))
    return _Spectrum(
        state_rank=state_rank,
        query_rank=query_rank,
        query_eigenvalues=immutable_array(query_eigenvalues, dtype=np.dtype("<f8")),
        minimum_eigenvalue=float(query_eigenvalues[0]),
        condition_number=condition,
        trace_precision=float(np.sum(query_eigenvalues)),
        unobservable_fraction=unobservable,
        nuisance_dimension=nuisance_dimension,
    )


def _masked_spectrum(
    batch: Deform360JointSparseFactorBatchV4,
    policy: Deform360JointSparseObservabilityPolicyV4,
    mask: np.ndarray,
) -> _Spectrum:
    information, nuisance_dimension = _marginal_state_information(batch, policy, mask)
    return _spectrum(information, batch.query_jacobian, policy, nuisance_dimension)


def evaluate_deform360_joint_sparse_observability_v4(
    batch: Deform360JointSparseFactorBatchV4,
    policy: Deform360JointSparseObservabilityPolicyV4,
    *,
    implementation_revision: str,
) -> Deform360JointSparseObservabilityResultV4:
    """Evaluate complementary partial factors jointly for one object."""

    _require(batch.protocol_id == policy.protocol_id, "batch/policy protocol mismatch")
    revision = exact_revision(implementation_revision, name="implementation_revision")
    all_rows = np.ones(len(batch.factor_ids), dtype=np.bool_)
    full = _masked_spectrum(batch, policy, all_rows)
    cameras = sorted(set(batch.camera_ids))
    windows = sorted(set(batch.window_ids))
    camera_array = np.asarray(batch.camera_ids, dtype=object)
    window_array = np.asarray(batch.window_ids, dtype=object)
    single_camera: dict[str, float] = {}
    leave_camera: dict[str, float] = {}
    for camera in cameras:
        selected = camera_array == camera
        only = _masked_spectrum(batch, policy, selected)
        ratio = 0.0 if full.trace_precision == 0.0 else only.trace_precision / full.trace_precision
        _require(-1e-12 <= ratio <= 1.0 + 1e-8, "camera information is not monotone")
        single_camera[camera] = float(np.clip(ratio, 0.0, 1.0))
        leave_camera[camera] = _masked_spectrum(batch, policy, ~selected).query_rank / batch.query_dimension
    leave_window = {
        window: _masked_spectrum(batch, policy, window_array != window).query_rank / batch.query_dimension
        for window in windows
    }
    max_camera = max(single_camera.values())
    min_leave_camera = min(leave_camera.values())
    min_leave_window = min(leave_window.values())
    full_rank = full.query_rank == batch.query_dimension
    checks = {
        "minimum_distinct_cameras": len(cameras) >= policy.minimum_distinct_cameras,
        "minimum_distinct_windows": len(windows) >= policy.minimum_distinct_windows,
        "minimum_distinct_spatial_clusters": len(set(batch.spatial_cluster_ids)) >= policy.minimum_distinct_spatial_clusters,
        "query_rank": full_rank if policy.require_full_query_rank else full.query_rank > 0,
        "minimum_query_precision_eigenvalue": full.minimum_eigenvalue >= policy.minimum_query_precision_eigenvalue,
        "maximum_query_condition_number": full.condition_number is not None and full.condition_number <= policy.maximum_query_condition_number,
        "maximum_single_camera_information_fraction": max_camera <= policy.maximum_single_camera_information_fraction,
        "minimum_leave_one_camera_rank_fraction": min_leave_camera >= policy.minimum_leave_one_camera_rank_fraction,
        "minimum_leave_one_window_rank_fraction": min_leave_window >= policy.minimum_leave_one_window_rank_fraction,
    }
    return Deform360JointSparseObservabilityResultV4(
        input_id=cast(str, batch.input_id),
        policy_id=cast(str, policy.policy_id),
        implementation_revision=revision,
        object_id=batch.object_id,
        episode_id=batch.episode_id,
        stratum=batch.stratum,
        status="evaluated",
        factor_count=len(batch.factor_ids),
        excluded_factor_count=batch.excluded_factor_count,
        distinct_camera_count=len(cameras),
        distinct_window_count=len(windows),
        distinct_spatial_cluster_count=len(set(batch.spatial_cluster_ids)),
        distinct_correlation_group_count=len(set(batch.correlation_group_ids)),
        state_dimension=batch.state_dimension,
        query_dimension=batch.query_dimension,
        nuisance_dimension=full.nuisance_dimension,
        state_rank=full.state_rank,
        query_rank=full.query_rank,
        query_precision_eigenvalues=tuple(map(float, full.query_eigenvalues)),
        minimum_query_precision_eigenvalue=full.minimum_eigenvalue,
        query_condition_number=full.condition_number,
        query_unobservable_fraction=full.unobservable_fraction,
        trace_query_precision=full.trace_precision,
        single_camera_information_fraction=single_camera,
        leave_one_camera_rank_fraction=leave_camera,
        leave_one_window_rank_fraction=leave_window,
        gate_checks=checks,
        gate_passed=all(checks.values()),
        information_boundary=batch.information_boundary,
        source_artifacts=batch.source_artifacts,
    )


def technical_failure_deform360_joint_sparse_result_v4(
    batch: Deform360JointSparseFactorBatchV4,
    policy: Deform360JointSparseObservabilityPolicyV4,
    *,
    implementation_revision: str,
    reason: str,
    detail: str,
) -> Deform360JointSparseObservabilityResultV4:
    """Retain an object-level technical failure without replacement."""

    return Deform360JointSparseObservabilityResultV4(
        input_id=cast(str, batch.input_id),
        policy_id=cast(str, policy.policy_id),
        implementation_revision=implementation_revision,
        object_id=batch.object_id,
        episode_id=batch.episode_id,
        stratum=batch.stratum,
        status="technical-failure-without-replacement",
        factor_count=len(batch.factor_ids),
        excluded_factor_count=batch.excluded_factor_count,
        distinct_camera_count=len(set(batch.camera_ids)),
        distinct_window_count=len(set(batch.window_ids)),
        distinct_spatial_cluster_count=len(set(batch.spatial_cluster_ids)),
        distinct_correlation_group_count=len(set(batch.correlation_group_ids)),
        state_dimension=None,
        query_dimension=None,
        nuisance_dimension=None,
        state_rank=None,
        query_rank=None,
        query_precision_eigenvalues=None,
        minimum_query_precision_eigenvalue=None,
        query_condition_number=None,
        query_unobservable_fraction=None,
        trace_query_precision=None,
        single_camera_information_fraction=None,
        leave_one_camera_rank_fraction=None,
        leave_one_window_rank_fraction=None,
        gate_checks=None,
        gate_passed=False,
        information_boundary=batch.information_boundary,
        source_artifacts=batch.source_artifacts,
        failure_reason=reason,
        failure_detail_sha256=hashlib.sha256(detail.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class Deform360JointSparseDevelopmentReportV4:
    """Object-balanced v4 development report; confirmation is always closed."""

    selection_artifact_sha256: str
    visual_provider_lock_id: str
    policy_id: str
    implementation_revision: str
    results: tuple[Deform360JointSparseObservabilityResultV4, ...]
    source_artifacts: Mapping[str, str] = field(default_factory=dict)
    information_boundary: Mapping[str, Any] = field(
        default_factory=default_deform360_joint_sparse_information_boundary_v4
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_id: str = DEFORM360_JOINT_SPARSE_PROTOCOL_ID
    report_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_artifact_sha256", sha256_digest(self.selection_artifact_sha256, name="selection_artifact_sha256"))
        object.__setattr__(self, "visual_provider_lock_id", sha256_digest(self.visual_provider_lock_id, name="visual_provider_lock_id"))
        object.__setattr__(self, "policy_id", sha256_digest(self.policy_id, name="policy_id"))
        object.__setattr__(self, "implementation_revision", exact_revision(self.implementation_revision, name="implementation_revision"))
        object.__setattr__(self, "protocol_id", _literal(self.protocol_id, name="protocol_id"))
        _require(self.protocol_id == DEFORM360_JOINT_SPARSE_PROTOCOL_ID, "protocol changed")
        results = tuple(self.results)
        _require(bool(results) and all(isinstance(value, Deform360JointSparseObservabilityResultV4) for value in results), "invalid report results")
        identities = [(value.object_id, value.episode_id) for value in results]
        _require(identities == sorted(identities) and len(identities) == len(set(identities)), "report results are unsorted or repeated")
        _require(all(value.policy_id == self.policy_id and value.implementation_revision == self.implementation_revision for value in results), "report result lineage changed")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "source_artifacts", source_artifact_mapping(self.source_artifacts, name="source_artifacts", allow_empty=True))
        object.__setattr__(self, "information_boundary", _boundary(self.information_boundary))
        object.__setattr__(self, "metadata", frozen_finite_json_mapping(self.metadata, name="metadata"))
        expected = content_id(self.identity_record())
        _require(self.report_id is None or self.report_id == expected, "report_id changed")
        object.__setattr__(self, "report_id", expected)

    def summary(self) -> dict[str, object]:
        by_stratum = {
            stratum: {
                "object_count": sum(value.stratum == stratum for value in self.results),
                "supported_object_count": sum(value.stratum == stratum and value.gate_passed for value in self.results),
                "technical_failure_object_count": sum(value.stratum == stratum and value.status != "evaluated" for value in self.results),
            }
            for stratum in ("sheet", "volumetric")
        }
        return {
            "object_count": len(self.results),
            "supported_object_count": sum(value.gate_passed for value in self.results),
            "technical_failure_object_count": sum(value.status != "evaluated" for value in self.results),
            "by_stratum": by_stratum,
        }

    def support_gate(self, policy: Deform360JointSparseObservabilityPolicyV4) -> dict[str, object]:
        _require(policy.policy_id == self.policy_id, "report/policy identity mismatch")
        summary = self.summary()
        by_stratum = cast(Mapping[str, Mapping[str, int]], summary["by_stratum"])
        checks = {
            "minimum_supported_objects": cast(int, summary["supported_object_count"]) >= policy.minimum_supported_objects,
            "minimum_supported_sheet_objects": by_stratum["sheet"]["supported_object_count"] >= policy.minimum_supported_objects_per_stratum,
            "minimum_supported_volumetric_objects": by_stratum["volumetric"]["supported_object_count"] >= policy.minimum_supported_objects_per_stratum,
            "no_technical_failures": summary["technical_failure_object_count"] == 0,
        }
        return {"checks": checks, "passed": all(checks.values())}

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_JOINT_SPARSE_REPORT_SCHEMA,
            "schema_version": DEFORM360_JOINT_SPARSE_VERSION,
            "semantics": DEFORM360_JOINT_SPARSE_SEMANTICS,
            "protocol_id": self.protocol_id,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "policy_id": self.policy_id,
            "implementation_revision": self.implementation_revision,
            "results": [value.to_record() for value in self.results],
            "summary": self.summary(),
            "confirmation_access_authorized": False,
            "source_artifacts": plain_json(self.source_artifacts),
            "information_boundary": plain_json(self.information_boundary),
            "metadata": plain_json(self.metadata),
            "claim_boundary": DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
        }

    def to_record(self, policy: Deform360JointSparseObservabilityPolicyV4) -> dict[str, object]:
        gate = self.support_gate(policy)
        status = "development-design-supported" if gate["passed"] else "development-design-not-supported"
        if self.summary()["technical_failure_object_count"]:
            status = "development-technical-failures-retained"
        return {**self.identity_record(), "report_id": self.report_id, "support_gate": gate, "status": status}


def build_deform360_joint_sparse_factor_batch_from_tree_sparse_v4(
    adapted: Any,
    *,
    selection_artifact_sha256: str,
    visual_provider_lock_id: str,
    implementation_revision: str,
    object_id: str,
    episode_id: int,
    stratum: Stratum,
    factor_ids: Sequence[str],
    spatial_cluster_ids: Sequence[str],
    query_jacobian: np.ndarray,
    excluded_factor_count: int = 0,
    source_artifacts: Mapping[str, str] | None = None,
    information_boundary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Deform360JointSparseFactorBatchV4:
    """Adapt the existing claim-bearing tree-sparse bridge without residuals."""

    batch = adapted.batch
    design = adapted.tree_gauge_design
    count = len(batch.observation_covariance_m2)
    _require(len(adapted.view_ids) == count, "tree-sparse adapter view IDs changed")
    window_ids = tuple(design.gauge_ids[int(index)] for index in design.gauge_indices)
    lineage = {
        "provider_manifest_id": adapted.provider_manifest_id,
        "calibration_artifact_ids": dict(adapted.calibration_artifact_ids),
        "runtime_revision_source": adapted.runtime_revision_source,
        "source_adapter": "ClaimBearingTreeSparseProb4DAdapterResult",
        **dict(metadata or {}),
    }
    return Deform360JointSparseFactorBatchV4(
        selection_artifact_sha256=selection_artifact_sha256,
        visual_provider_lock_id=visual_provider_lock_id,
        observation_artifact_id=adapted.observation_artifact_id,
        linearization_artifact_id=adapted.linearization_artifact_id,
        implementation_revision=implementation_revision,
        object_id=object_id,
        episode_id=episode_id,
        stratum=stratum,
        factor_ids=tuple(factor_ids),
        camera_ids=tuple(adapted.view_ids),
        window_ids=window_ids,
        spatial_cluster_ids=tuple(spatial_cluster_ids),
        correlation_group_ids=tuple(batch.correlation_group_ids),
        gauge_ids=tuple(design.gauge_ids),
        gauge_prior_id=design.prior_id,
        observation_covariance_m2=batch.observation_covariance_m2,
        state_jacobian=batch.state_jacobian,
        local_gauge_jacobian=design.local_gauge_jacobian,
        gauge_indices=design.gauge_indices,
        parent_indices=design.parent_indices,
        transition_matrices=design.transition_matrices,
        innovation_scale_tril=design.innovation_scale_tril,
        query_jacobian=query_jacobian,
        prior_reliability=batch.prior_reliability,
        association_probability=batch.association_probability,
        composite_weight=batch.composite_weight,
        shared_bias_jacobian=batch.shared_bias_jacobian,
        view_bias_jacobian=batch.view_bias_jacobian,
        excluded_factor_count=excluded_factor_count,
        source_artifacts=source_artifacts or {},
        information_boundary=_boundary(information_boundary),
        metadata=lineage,
    )


__all__ = [
    "DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY",
    "DEFORM360_JOINT_SPARSE_INPUT_SCHEMA",
    "DEFORM360_JOINT_SPARSE_POLICY_SCHEMA",
    "DEFORM360_JOINT_SPARSE_PROTOCOL_ID",
    "DEFORM360_JOINT_SPARSE_REPORT_SCHEMA",
    "DEFORM360_JOINT_SPARSE_RESULT_SCHEMA",
    "DEFORM360_JOINT_SPARSE_SEMANTICS",
    "DEFORM360_JOINT_SPARSE_VERSION",
    "Deform360JointSparseDevelopmentReportV4",
    "Deform360JointSparseFactorBatchV4",
    "Deform360JointSparseObservabilityPolicyV4",
    "Deform360JointSparseObservabilityResultV4",
    "build_deform360_joint_sparse_factor_batch_from_tree_sparse_v4",
    "default_deform360_joint_sparse_information_boundary_v4",
    "evaluate_deform360_joint_sparse_observability_v4",
    "technical_failure_deform360_joint_sparse_result_v4",
]
