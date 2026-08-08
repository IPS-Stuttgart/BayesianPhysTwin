"""Content-addressed physical linearizations and nonlinear-closure diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    immutable_array,
    immutable_integer_array,
    integer_array,
    plain_json,
)

PHYSICAL_LINEARIZATION_SCHEMA = "bayesian_phystwin.physical_linearization"
PHYSICAL_LINEARIZATION_VERSION = 1
NONLINEAR_CLOSURE_SCHEMA = "bayesian_phystwin.nonlinear_closure"
NONLINEAR_CLOSURE_VERSION = 1


def _canonical_json(values: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _readonly(values: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    return immutable_array(values, dtype=dtype)


def _readonly_integer(values: object, *, name: str) -> np.ndarray:
    return immutable_integer_array(values, name=name)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _artifact_id(
    descriptor: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(descriptor))
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(_array_sha256(values).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PhysicalLinearizationV1:
    """Row-bound state/query Jacobians generated from one immutable baseline."""

    observation_artifact_id: str
    baseline_belief_id: str
    action_prefix_id: str
    simulator_revision: str
    frame_ids: np.ndarray
    entity_ids: np.ndarray
    view_indices: np.ndarray
    window_indices: np.ndarray
    state_jacobian: np.ndarray
    query_state_jacobian: np.ndarray
    physical_response_m: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_artifact_id", self.observation_artifact_id),
            ("baseline_belief_id", self.baseline_belief_id),
            ("action_prefix_id", self.action_prefix_id),
        ):
            _validate_sha256(value, name=name)
        if not isinstance(self.simulator_revision, str) or not self.simulator_revision:
            raise ValueError("simulator_revision must be nonempty")
        frame_ids = _readonly_integer(self.frame_ids, name="frame_ids")
        entity_ids = _readonly_integer(self.entity_ids, name="entity_ids")
        view_indices = _readonly_integer(self.view_indices, name="view_indices")
        window_indices = _readonly_integer(self.window_indices, name="window_indices")
        state = _readonly(self.state_jacobian, dtype=np.float64)
        query = _readonly(self.query_state_jacobian, dtype=np.float64)
        response = _readonly(self.physical_response_m, dtype=np.float64)
        count = len(frame_ids)
        if frame_ids.shape != (count,):
            raise ValueError(f"frame_ids must have shape ({count},)")
        for name, values in (
            ("entity_ids", entity_ids),
            ("view_indices", view_indices),
            ("window_indices", window_indices),
        ):
            if values.shape != (count,):
                raise ValueError(f"{name} must have shape ({count},)")
        if count == 0 or np.any(frame_ids < 0) or np.any(entity_ids < 0):
            raise ValueError(
                "linearization row identities must be nonempty and nonnegative"
            )
        if np.any(view_indices < 0) or np.any(window_indices < 0):
            raise ValueError("linearization view/window identities must be nonnegative")
        if state.ndim != 3 or state.shape[:2] != (count, 3) or state.shape[2] < 1:
            raise ValueError("state_jacobian must have shape (N, 3, S) with S >= 1")
        state_count = state.shape[2]
        if query.ndim != 3 or query.shape[1:] != (3, state_count) or len(query) == 0:
            raise ValueError("query_state_jacobian must have shape (Q, 3, S)")
        if response.shape != query.shape[:2]:
            raise ValueError("physical_response_m must have shape (Q, 3)")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(query)):
            raise ValueError("linearization Jacobians must be finite")
        if not np.all(np.isfinite(response)):
            raise ValueError("physical response must be finite")
        if np.max(np.linalg.norm(response, axis=1), initial=0.0) <= 0.0:
            raise ValueError(
                "physical response must contain a nonzero query displacement"
            )
        order = np.lexsort((window_indices, view_indices, entity_ids, frame_ids))
        keys = np.column_stack(
            (
                frame_ids[order],
                entity_ids[order],
                view_indices[order],
                window_indices[order],
            )
        )
        if len(keys) > 1 and np.any(np.all(keys[1:] == keys[:-1], axis=1)):
            raise ValueError("linearization row identities must be unique")
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "entity_ids", entity_ids)
        object.__setattr__(self, "view_indices", view_indices)
        object.__setattr__(self, "window_indices", window_indices)
        object.__setattr__(self, "state_jacobian", state)
        object.__setattr__(self, "query_state_jacobian", query)
        object.__setattr__(self, "physical_response_m", response)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata),
        )

    @property
    def physical_response_scale_m(self) -> float:
        return float(np.max(np.linalg.norm(self.physical_response_m, axis=1)))

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": PHYSICAL_LINEARIZATION_SCHEMA,
            "schema_version": PHYSICAL_LINEARIZATION_VERSION,
            "observation_artifact_id": self.observation_artifact_id,
            "baseline_belief_id": self.baseline_belief_id,
            "action_prefix_id": self.action_prefix_id,
            "simulator_revision": self.simulator_revision,
            "metadata": plain_json(self.metadata),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "frame_ids": self.frame_ids,
            "entity_ids": self.entity_ids,
            "view_indices": self.view_indices,
            "window_indices": self.window_indices,
            "state_jacobian": self.state_jacobian,
            "query_state_jacobian": self.query_state_jacobian,
            "physical_response_m": self.physical_response_m,
        }

    @property
    def artifact_id(self) -> str:
        return _artifact_id(self.descriptor(), self.arrays())


def validate_observation_linearization_alignment(
    observation_belief: Any,
    linearization: PhysicalLinearizationV1,
) -> None:
    """Fail closed on artifact or row-order mismatches."""

    if str(observation_belief.artifact_id) != linearization.observation_artifact_id:
        raise ValueError("linearization does not identify this observation artifact")
    for name in ("frame_ids", "entity_ids", "view_indices", "window_indices"):
        observed = integer_array(
            getattr(observation_belief, name),
            name=f"observation {name}",
        )
        expected = np.asarray(getattr(linearization, name), dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise ValueError(f"observation and linearization {name} differ")


def build_gauge_aware_batch_from_artifacts(
    observation_belief: Any,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    **anchor_dependence: Any,
) -> Any:
    """Build a gauge-aware batch after identity-bound row validation."""

    validate_observation_linearization_alignment(observation_belief, linearization)
    from .observation_belief_gauge_adapter import (
        build_gauge_aware_batch_from_observation_belief,
    )

    adapted = build_gauge_aware_batch_from_observation_belief(
        observation_belief,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        state_jacobian=linearization.state_jacobian,
        query_state_jacobian=linearization.query_state_jacobian,
        physical_response_scale_m=linearization.physical_response_scale_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        **anchor_dependence,
    )
    metadata = dict(adapted.batch.metadata)
    metadata.update(
        {
            "linearization_artifact_id": linearization.artifact_id,
            "baseline_belief_id": linearization.baseline_belief_id,
            "action_prefix_id": linearization.action_prefix_id,
            "simulator_revision": linearization.simulator_revision,
            "physical_response_scale_source": (
                "PhysicalLinearizationV1.physical_response_m"
            ),
            "row_alignment_verified": True,
        }
    )
    return replace(adapted, batch=replace(adapted.batch, metadata=metadata))


def save_physical_linearization(
    path: str | Path, linearization: PhysicalLinearizationV1
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = linearization.descriptor()
    descriptor["artifact_id"] = linearization.artifact_id
    archive_payload: dict[str, Any] = {
        "descriptor_json": np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    }
    archive_payload.update(linearization.arrays())
    np.savez_compressed(target, **archive_payload)


def load_physical_linearization(path: str | Path) -> PhysicalLinearizationV1:
    with np.load(path, allow_pickle=False) as archive:
        if "descriptor_json" not in archive:
            raise ValueError("physical linearization has no descriptor_json")
        descriptor = json.loads(str(archive["descriptor_json"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }
    if descriptor.get("schema_name") != PHYSICAL_LINEARIZATION_SCHEMA:
        raise ValueError("unsupported physical-linearization schema")
    version = genuine_integer(
        descriptor.get("schema_version"),
        name="physical-linearization schema_version",
        minimum=0,
    )
    if version != PHYSICAL_LINEARIZATION_VERSION:
        raise ValueError("unsupported physical-linearization version")
    required = {
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "state_jacobian",
        "query_state_jacobian",
        "physical_response_m",
    }
    if set(arrays) != required:
        raise ValueError("physical-linearization array set changed")
    result = PhysicalLinearizationV1(
        observation_artifact_id=str(descriptor["observation_artifact_id"]),
        baseline_belief_id=str(descriptor["baseline_belief_id"]),
        action_prefix_id=str(descriptor["action_prefix_id"]),
        simulator_revision=str(descriptor["simulator_revision"]),
        metadata=descriptor.get("metadata", {}),
        **arrays,
    )
    expected = str(descriptor.get("artifact_id", ""))
    _validate_sha256(expected, name="artifact_id")
    if result.artifact_id != expected:
        raise ValueError("physical-linearization digest does not match its payload")
    return result


@dataclass(frozen=True)
class NonlinearClosureV1:
    """Comparison between a local query linearization and nonlinear replay."""

    linearization_artifact_id: str
    absolute_error_m: float
    relative_error: float
    absolute_tolerance_m: float
    relative_tolerance: float
    candidate_valid: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_sha256(
            self.linearization_artifact_id,
            name="linearization_artifact_id",
        )
        values = (
            self.absolute_error_m,
            self.relative_error,
            self.absolute_tolerance_m,
            self.relative_tolerance,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("nonlinear-closure values must be finite and nonnegative")
        candidate_valid = genuine_boolean(
            self.candidate_valid,
            name="candidate_valid",
        )
        expected = (
            self.absolute_error_m <= self.absolute_tolerance_m
            or self.relative_error <= self.relative_tolerance
        )
        if candidate_valid != expected:
            raise ValueError("candidate_valid does not match the closure tolerances")
        object.__setattr__(self, "candidate_valid", candidate_valid)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata),
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": NONLINEAR_CLOSURE_SCHEMA,
            "schema_version": NONLINEAR_CLOSURE_VERSION,
            "linearization_artifact_id": self.linearization_artifact_id,
            "absolute_error_m": self.absolute_error_m,
            "relative_error": self.relative_error,
            "absolute_tolerance_m": self.absolute_tolerance_m,
            "relative_tolerance": self.relative_tolerance,
            "candidate_valid": self.candidate_valid,
            "metadata": plain_json(self.metadata),
        }

    @property
    def closure_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()


def evaluate_nonlinear_closure(
    linearization_artifact_id: str,
    *,
    baseline_query_m: np.ndarray,
    linearized_query_m: np.ndarray,
    nonlinear_query_m: np.ndarray,
    absolute_tolerance_m: float,
    relative_tolerance: float,
    denominator_floor_m: float = 1e-12,
    metadata: Mapping[str, Any] | None = None,
) -> NonlinearClosureV1:
    """Measure the local-model remainder before a candidate reaches the guard."""

    baseline = np.asarray(baseline_query_m, dtype=np.float64)
    linearized = np.asarray(linearized_query_m, dtype=np.float64)
    nonlinear = np.asarray(nonlinear_query_m, dtype=np.float64)
    if baseline.shape != linearized.shape or baseline.shape != nonlinear.shape:
        raise ValueError("closure query arrays must have identical shapes")
    if baseline.ndim != 2 or baseline.shape[1] != 3:
        raise ValueError("closure query arrays must have shape (Q, 3)")
    if (
        not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(linearized))
        or not np.all(np.isfinite(nonlinear))
    ):
        raise ValueError("closure query arrays must be finite")
    if (
        not np.isfinite(absolute_tolerance_m)
        or not np.isfinite(relative_tolerance)
        or not np.isfinite(denominator_floor_m)
        or absolute_tolerance_m < 0.0
        or relative_tolerance < 0.0
        or denominator_floor_m <= 0.0
    ):
        raise ValueError(
            "closure tolerances must be finite and nonnegative and floor positive"
        )
    remainder = nonlinear - linearized
    absolute_error = float(np.max(np.linalg.norm(remainder, axis=1), initial=0.0))
    predicted_change = linearized - baseline
    denominator = max(float(np.linalg.norm(predicted_change)), denominator_floor_m)
    relative_error = float(np.linalg.norm(remainder) / denominator)
    candidate_valid = (
        absolute_error <= absolute_tolerance_m or relative_error <= relative_tolerance
    )
    return NonlinearClosureV1(
        linearization_artifact_id=linearization_artifact_id,
        absolute_error_m=absolute_error,
        relative_error=relative_error,
        absolute_tolerance_m=float(absolute_tolerance_m),
        relative_tolerance=float(relative_tolerance),
        candidate_valid=candidate_valid,
        metadata=metadata or {},
    )


__all__ = [
    "NONLINEAR_CLOSURE_SCHEMA",
    "NONLINEAR_CLOSURE_VERSION",
    "NonlinearClosureV1",
    "PHYSICAL_LINEARIZATION_SCHEMA",
    "PHYSICAL_LINEARIZATION_VERSION",
    "PhysicalLinearizationV1",
    "build_gauge_aware_batch_from_artifacts",
    "evaluate_nonlinear_closure",
    "load_physical_linearization",
    "save_physical_linearization",
    "validate_observation_linearization_alignment",
]
