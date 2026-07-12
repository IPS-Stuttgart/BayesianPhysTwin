"""Typed, provenance-complete artifacts for Causal4D inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping

import numpy as np


CONTRACT_VERSION = 1


def array_sha256(values: np.ndarray) -> str:
    """Hash an array including its dtype and shape."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _readonly_array(
    values: np.ndarray,
    *,
    dtype: np.dtype[Any] | type | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _validated_weights(values: np.ndarray, *, name: str) -> np.ndarray:
    weights = _readonly_array(values, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must sum to one")
    return weights


def _validated_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be finite JSON data") from error


@dataclass(frozen=True)
class ObservationWindow:
    """One explicitly identified observation interval ``[start, stop)``."""

    case_id: str
    stream_id: str
    frame_start: int
    frame_stop: int
    content_sha256: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.stream_id:
            raise ValueError("observation case_id and stream_id must be nonempty")
        if self.frame_start < 0 or self.frame_stop <= self.frame_start:
            raise ValueError("observation interval must be nonempty and nonnegative")
        _validate_sha256(self.content_sha256, name="observation content_sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ObservationWindow:
        return cls(
            case_id=str(values["case_id"]),
            stream_id=str(values["stream_id"]),
            frame_start=int(values["frame_start"]),
            frame_stop=int(values["frame_stop"]),
            content_sha256=str(values["content_sha256"]),
        )


@dataclass(frozen=True)
class ActionWindow:
    """One commanded action interval with factual/counterfactual provenance."""

    action_id: str
    case_id: str
    frame_start: int
    frame_stop: int
    trajectory_sha256: str
    provenance: str

    def __post_init__(self) -> None:
        if not self.action_id or not self.case_id or not self.provenance:
            raise ValueError("action id, case id, and provenance must be nonempty")
        if self.frame_start < 0 or self.frame_stop <= self.frame_start:
            raise ValueError("action interval must be nonempty and nonnegative")
        _validate_sha256(self.trajectory_sha256, name="action trajectory_sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "case_id": self.case_id,
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "trajectory_sha256": self.trajectory_sha256,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ActionWindow:
        return cls(
            action_id=str(values["action_id"]),
            case_id=str(values["case_id"]),
            frame_start=int(values["frame_start"]),
            frame_stop=int(values["frame_stop"]),
            trajectory_sha256=str(values["trajectory_sha256"]),
            provenance=str(values["provenance"]),
        )


@dataclass(frozen=True)
class CausalContext:
    """The four data/action identities required by every Causal4D artifact."""

    protocol_id: str
    o_minus: ObservationWindow
    o_plus: ObservationWindow
    u_obs: ActionWindow
    u_cf: ActionWindow

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id must be nonempty")
        case_ids = {
            self.o_minus.case_id,
            self.o_plus.case_id,
            self.u_obs.case_id,
            self.u_cf.case_id,
        }
        if len(case_ids) != 1:
            raise ValueError("O-, O+, u_obs, and u_cf must identify the same case")
        if self.o_minus.frame_stop > self.o_plus.frame_start:
            raise ValueError("O- must not overlap O+")
        if self.u_obs.frame_stop > self.o_plus.frame_stop:
            raise ValueError("u_obs must not extend beyond the factual observation window")
        if self.u_cf.frame_start < self.o_minus.frame_stop:
            raise ValueError("u_cf must begin at or after the pre-intervention window")

    @property
    def case_id(self) -> str:
        return self.o_minus.case_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "o_minus": self.o_minus.as_dict(),
            "o_plus": self.o_plus.as_dict(),
            "u_obs": self.u_obs.as_dict(),
            "u_cf": self.u_cf.as_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CausalContext:
        return cls(
            protocol_id=str(values["protocol_id"]),
            o_minus=ObservationWindow.from_dict(values["o_minus"]),
            o_plus=ObservationWindow.from_dict(values["o_plus"]),
            u_obs=ActionWindow.from_dict(values["u_obs"]),
            u_cf=ActionWindow.from_dict(values["u_cf"]),
        )


def build_causal_context(
    *,
    protocol_id: str,
    case_id: str,
    observations: np.ndarray,
    observed_actions: np.ndarray,
    counterfactual_actions: np.ndarray,
    intervention_frame: int,
    stream_id: str = "object_points_m",
    observed_action_id: str = "u_obs",
    counterfactual_action_id: str = "u_cf",
    observed_action_provenance: str = "recorded controller trajectory",
    counterfactual_action_provenance: str = "counterfactual controller trajectory",
) -> CausalContext:
    """Build a context while hashing only the declared frame windows."""

    observation_array = np.asarray(observations)
    observed_action_array = np.asarray(observed_actions)
    counterfactual_action_array = np.asarray(counterfactual_actions)
    frame_count = len(observation_array)
    if not 1 <= intervention_frame < frame_count:
        raise ValueError("intervention_frame must split the observation sequence")
    if len(observed_action_array) < frame_count:
        raise ValueError("observed actions must cover the factual observation interval")
    if len(counterfactual_action_array) < frame_count:
        raise ValueError("counterfactual actions must cover the requested future interval")
    return CausalContext(
        protocol_id=protocol_id,
        o_minus=ObservationWindow(
            case_id=case_id,
            stream_id=stream_id,
            frame_start=0,
            frame_stop=intervention_frame,
            content_sha256=array_sha256(observation_array[:intervention_frame]),
        ),
        o_plus=ObservationWindow(
            case_id=case_id,
            stream_id=stream_id,
            frame_start=intervention_frame,
            frame_stop=frame_count,
            content_sha256=array_sha256(observation_array[intervention_frame:]),
        ),
        u_obs=ActionWindow(
            action_id=observed_action_id,
            case_id=case_id,
            frame_start=0,
            frame_stop=frame_count,
            trajectory_sha256=array_sha256(observed_action_array[:frame_count]),
            provenance=observed_action_provenance,
        ),
        u_cf=ActionWindow(
            action_id=counterfactual_action_id,
            case_id=case_id,
            frame_start=intervention_frame,
            frame_stop=frame_count,
            trajectory_sha256=array_sha256(
                counterfactual_action_array[intervention_frame:frame_count]
            ),
            provenance=counterfactual_action_provenance,
        ),
    )


class _Contract:
    contract_type: ClassVar[str]
    context: CausalContext

    def _scalar_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def _array_payload(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @property
    def artifact_id(self) -> str:
        digest = hashlib.sha256()
        descriptor = {
            "contract_version": CONTRACT_VERSION,
            "contract_type": self.contract_type,
            "context": self.context.as_dict(),
            "payload": self._scalar_payload(),
        }
        digest.update(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, values in sorted(self._array_payload().items()):
            digest.update(name.encode("utf-8"))
            digest.update(array_sha256(values).encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class TwinBelief(_Contract):
    """Particle belief over endpoint state, physics, and model discrepancy."""

    contract_type: ClassVar[str] = "TwinBelief"

    context: CausalContext
    endpoint_frame: int
    particle_ids: tuple[str, ...]
    theta_names: tuple[str, ...]
    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    theta: np.ndarray
    discrepancy_mean_m: np.ndarray
    discrepancy_variance_m2: np.ndarray
    weights: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = _readonly_array(self.endpoint_position_m, dtype=float)
        velocity = _readonly_array(self.endpoint_velocity_mps, dtype=float)
        theta = _readonly_array(self.theta, dtype=float)
        discrepancy = _readonly_array(self.discrepancy_mean_m, dtype=float)
        variance = _readonly_array(self.discrepancy_variance_m2, dtype=float)
        weights = _validated_weights(self.weights, name="TwinBelief weights")
        particle_count = len(weights)
        if self.endpoint_frame != self.context.o_minus.frame_stop - 1:
            raise ValueError("TwinBelief endpoint must be the final O- frame")
        if len(self.particle_ids) != particle_count or len(set(self.particle_ids)) != particle_count:
            raise ValueError("particle_ids must uniquely identify every particle")
        if not self.theta_names:
            raise ValueError("theta_names must be nonempty")
        if position.ndim != 3 or position.shape[0] != particle_count or position.shape[2] != 3:
            raise ValueError("endpoint_position_m must have shape (P, N, 3)")
        expected_state = position.shape
        if velocity.shape != expected_state or discrepancy.shape != expected_state:
            raise ValueError("velocity and discrepancy means must match endpoint positions")
        if variance.shape != expected_state:
            raise ValueError("discrepancy_variance_m2 must have shape (P, N, 3)")
        if theta.shape != (particle_count, len(self.theta_names)):
            raise ValueError("theta must have shape (P, len(theta_names))")
        arrays = (position, velocity, theta, discrepancy, variance)
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("TwinBelief arrays must be finite")
        if np.any(variance < 0.0):
            raise ValueError("discrepancy variances must be nonnegative")
        object.__setattr__(self, "endpoint_position_m", position)
        object.__setattr__(self, "endpoint_velocity_mps", velocity)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "discrepancy_mean_m", discrepancy)
        object.__setattr__(self, "discrepancy_variance_m2", variance)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "endpoint_frame": self.endpoint_frame,
            "particle_ids": list(self.particle_ids),
            "theta_names": list(self.theta_names),
            "metadata": self.metadata,
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "endpoint_position_m": self.endpoint_position_m,
            "endpoint_velocity_mps": self.endpoint_velocity_mps,
            "theta": self.theta,
            "discrepancy_mean_m": self.discrepancy_mean_m,
            "discrepancy_variance_m2": self.discrepancy_variance_m2,
            "weights": self.weights,
        }


@dataclass(frozen=True)
class FactualIntervention(_Contract):
    """Posterior over persistent actuation and factual event variables."""

    contract_type: ClassVar[str] = "FactualIntervention"

    context: CausalContext
    component_ids: tuple[str, ...]
    phi_names: tuple[str, ...]
    kappa_names: tuple[str, ...]
    phi: np.ndarray
    kappa_obs: np.ndarray
    hypothesis_indices: np.ndarray
    twin_particle_indices: np.ndarray
    weights: np.ndarray
    evidence_frame_stop: int
    source_twin_belief_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phi = _readonly_array(self.phi, dtype=float)
        kappa = _readonly_array(self.kappa_obs, dtype=float)
        hypotheses = _readonly_array(self.hypothesis_indices, dtype=np.int64)
        particles = _readonly_array(self.twin_particle_indices, dtype=np.int64)
        weights = _validated_weights(self.weights, name="FactualIntervention weights")
        count = len(weights)
        if len(self.component_ids) != count or len(set(self.component_ids)) != count:
            raise ValueError("component_ids must uniquely identify every component")
        if phi.shape != (count, len(self.phi_names)):
            raise ValueError("phi must have shape (K, len(phi_names))")
        if kappa.shape != (count, len(self.kappa_names)):
            raise ValueError("kappa_obs must have shape (K, len(kappa_names))")
        if hypotheses.shape != (count,) or particles.shape != (count,):
            raise ValueError("hypothesis and twin-particle indices must match support")
        if np.any(hypotheses < 0) or np.any(particles < 0):
            raise ValueError("support indices must be nonnegative")
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(kappa)):
            raise ValueError("intervention variables must be finite")
        if not self.context.o_plus.frame_start < self.evidence_frame_stop <= self.context.o_plus.frame_stop:
            raise ValueError("evidence_frame_stop must be a nonempty O+ prefix")
        _validate_sha256(self.source_twin_belief_id, name="source_twin_belief_id")
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "kappa_obs", kappa)
        object.__setattr__(self, "hypothesis_indices", hypotheses)
        object.__setattr__(self, "twin_particle_indices", particles)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "component_ids": list(self.component_ids),
            "phi_names": list(self.phi_names),
            "kappa_names": list(self.kappa_names),
            "evidence_frame_stop": self.evidence_frame_stop,
            "source_twin_belief_id": self.source_twin_belief_id,
            "metadata": self.metadata,
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "phi": self.phi,
            "kappa_obs": self.kappa_obs,
            "hypothesis_indices": self.hypothesis_indices,
            "twin_particle_indices": self.twin_particle_indices,
            "weights": self.weights,
        }


@dataclass(frozen=True)
class CounterfactualQuery(_Contract):
    """Explicit ``do(u_cf)`` query and contact-resampling policy."""

    contract_type: ClassVar[str] = "CounterfactualQuery"

    context: CausalContext
    controller_points_m: np.ndarray
    horizon_frames: int
    contact_policy: Literal["same_grasp", "new_contact"]
    source_factual_intervention_id: str
    language: str | None = None
    query_node_indices: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        controls = _readonly_array(self.controller_points_m, dtype=float)
        if controls.ndim != 3 or controls.shape[2] != 3 or not np.all(np.isfinite(controls)):
            raise ValueError("controller_points_m must have finite shape (T, C, 3)")
        if self.horizon_frames < 1 or len(controls) != self.horizon_frames:
            raise ValueError("horizon_frames must match the counterfactual controls")
        if self.context.u_cf.frame_stop - self.context.u_cf.frame_start != self.horizon_frames:
            raise ValueError("counterfactual context interval must match horizon_frames")
        if array_sha256(controls) != self.context.u_cf.trajectory_sha256:
            raise ValueError("counterfactual controls disagree with the u_cf digest")
        if self.contact_policy not in {"same_grasp", "new_contact"}:
            raise ValueError("contact_policy must be 'same_grasp' or 'new_contact'")
        _validate_sha256(
            self.source_factual_intervention_id,
            name="source_factual_intervention_id",
        )
        nodes = None
        if self.query_node_indices is not None:
            nodes = _readonly_array(self.query_node_indices, dtype=np.int64)
            if nodes.ndim != 1 or len(nodes) == 0 or np.any(nodes < 0):
                raise ValueError("query_node_indices must be a nonempty nonnegative vector")
        object.__setattr__(self, "controller_points_m", controls)
        object.__setattr__(self, "query_node_indices", nodes)
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "horizon_frames": self.horizon_frames,
            "contact_policy": self.contact_policy,
            "language": self.language,
            "source_factual_intervention_id": self.source_factual_intervention_id,
            "metadata": self.metadata,
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        arrays = {"controller_points_m": self.controller_points_m}
        if self.query_node_indices is not None:
            arrays["query_node_indices"] = self.query_node_indices
        return arrays


@dataclass(frozen=True)
class PhysicalPosterior(_Contract):
    """Physical-only posterior over dense counterfactual rollouts."""

    contract_type: ClassVar[str] = "PhysicalPosterior"

    context: CausalContext
    component_ids: tuple[str, ...]
    state_trajectories_m: np.ndarray
    readout_trajectories_m: np.ndarray
    readout_variance_m2: np.ndarray
    weights: np.ndarray
    phi: np.ndarray
    kappa_cf: np.ndarray
    hypothesis_indices: np.ndarray
    twin_particle_indices: np.ndarray
    phi_names: tuple[str, ...]
    kappa_names: tuple[str, ...]
    source_twin_belief_id: str
    source_factual_intervention_id: str
    source_query_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = _readonly_array(self.state_trajectories_m, dtype=np.float32)
        readout = _readonly_array(self.readout_trajectories_m, dtype=np.float32)
        variance = _readonly_array(self.readout_variance_m2, dtype=np.float32)
        weights = _validated_weights(self.weights, name="PhysicalPosterior weights")
        phi = _readonly_array(self.phi, dtype=float)
        kappa = _readonly_array(self.kappa_cf, dtype=float)
        hypotheses = _readonly_array(self.hypothesis_indices, dtype=np.int64)
        particles = _readonly_array(self.twin_particle_indices, dtype=np.int64)
        count = len(weights)
        if len(self.component_ids) != count or len(set(self.component_ids)) != count:
            raise ValueError("component_ids must uniquely identify every rollout")
        if state.ndim != 4 or state.shape[0] != count or state.shape[3] != 3:
            raise ValueError("state_trajectories_m must have shape (K, T, N, 3)")
        if readout.shape != state.shape:
            raise ValueError("readout trajectories must match state trajectories")
        if variance.shape != (count, state.shape[2], state.shape[3]):
            raise ValueError("readout_variance_m2 must have shape (K, N, 3)")
        if phi.shape != (count, len(self.phi_names)) or kappa.shape != (
            count,
            len(self.kappa_names),
        ):
            raise ValueError("phi and kappa_cf must identify every rollout component")
        if hypotheses.shape != (count,) or particles.shape != (count,):
            raise ValueError("hypothesis and twin-particle indices must match support")
        if np.any(hypotheses < 0) or np.any(particles < 0):
            raise ValueError("support indices must be nonnegative")
        if (
            not np.all(np.isfinite(state))
            or not np.all(np.isfinite(readout))
            or not np.all(np.isfinite(variance))
            or not np.all(np.isfinite(phi))
            or not np.all(np.isfinite(kappa))
        ):
            raise ValueError("PhysicalPosterior arrays must be finite")
        if np.any(variance < 0.0):
            raise ValueError("readout variances must be nonnegative")
        for name, value in (
            ("source_twin_belief_id", self.source_twin_belief_id),
            ("source_factual_intervention_id", self.source_factual_intervention_id),
            ("source_query_id", self.source_query_id),
        ):
            _validate_sha256(value, name=name)
        object.__setattr__(self, "state_trajectories_m", state)
        object.__setattr__(self, "readout_trajectories_m", readout)
        object.__setattr__(self, "readout_variance_m2", variance)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "kappa_cf", kappa)
        object.__setattr__(self, "hypothesis_indices", hypotheses)
        object.__setattr__(self, "twin_particle_indices", particles)
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "component_ids": list(self.component_ids),
            "phi_names": list(self.phi_names),
            "kappa_names": list(self.kappa_names),
            "source_twin_belief_id": self.source_twin_belief_id,
            "source_factual_intervention_id": self.source_factual_intervention_id,
            "source_query_id": self.source_query_id,
            "metadata": self.metadata,
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "state_trajectories_m": self.state_trajectories_m,
            "readout_trajectories_m": self.readout_trajectories_m,
            "readout_variance_m2": self.readout_variance_m2,
            "weights": self.weights,
            "phi": self.phi,
            "kappa_cf": self.kappa_cf,
            "hypothesis_indices": self.hypothesis_indices,
            "twin_particle_indices": self.twin_particle_indices,
        }


@dataclass(frozen=True)
class TaskPosterior(_Contract):
    """Semantic reweighting of, never a replacement for, a physical posterior."""

    contract_type: ClassVar[str] = "TaskPosterior"

    context: CausalContext
    physical_posterior_id: str
    component_ids: tuple[str, ...]
    physical_weights: np.ndarray
    task_weights: np.ndarray
    semantic_log_scores: np.ndarray
    beta: float
    query_node_indices: np.ndarray
    semantic_source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        physical = _validated_weights(self.physical_weights, name="physical_weights")
        task = _validated_weights(self.task_weights, name="task_weights")
        scores = _readonly_array(self.semantic_log_scores, dtype=float)
        nodes = _readonly_array(self.query_node_indices, dtype=np.int64)
        count = len(physical)
        if len(self.component_ids) != count or len(set(self.component_ids)) != count:
            raise ValueError("component_ids must uniquely identify every component")
        if task.shape != physical.shape or scores.shape != physical.shape:
            raise ValueError("task weights and semantic scores must match physical support")
        if not np.all(np.isfinite(scores)):
            raise ValueError("semantic scores must be finite")
        if self.beta < 0.0 or not np.isfinite(self.beta):
            raise ValueError("beta must be finite and nonnegative")
        if nodes.ndim != 1 or len(nodes) == 0 or np.any(nodes < 0):
            raise ValueError("query_node_indices must identify sparse physical readouts")
        if not self.semantic_source:
            raise ValueError("semantic_source must be nonempty")
        _validate_sha256(self.physical_posterior_id, name="physical_posterior_id")
        if self.beta == 0.0 and not np.array_equal(task, physical):
            raise ValueError("beta=0 must preserve physical weights bit-for-bit")
        object.__setattr__(self, "physical_weights", physical)
        object.__setattr__(self, "task_weights", task)
        object.__setattr__(self, "semantic_log_scores", scores)
        object.__setattr__(self, "query_node_indices", nodes)
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "physical_posterior_id": self.physical_posterior_id,
            "component_ids": list(self.component_ids),
            "beta": self.beta,
            "semantic_source": self.semantic_source,
            "metadata": self.metadata,
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "physical_weights": self.physical_weights,
            "task_weights": self.task_weights,
            "semantic_log_scores": self.semantic_log_scores,
            "query_node_indices": self.query_node_indices,
        }


Contract = TwinBelief | FactualIntervention | CounterfactualQuery | PhysicalPosterior | TaskPosterior


def save_contract(path: str | Path, artifact: Contract) -> None:
    """Write a contract as JSON metadata plus non-pickled NumPy arrays."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "contract_version": CONTRACT_VERSION,
        "contract_type": artifact.contract_type,
        "artifact_id": artifact.artifact_id,
        "context": artifact.context.as_dict(),
        "payload": artifact._scalar_payload(),
    }
    arrays = artifact._array_payload()
    np.savez_compressed(
        target,
        descriptor_json=np.asarray(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        ),
        **arrays,
    )


def load_contract(path: str | Path) -> Contract:
    """Load and revalidate any Causal4D contract artifact."""

    with np.load(path, allow_pickle=False) as archive:
        descriptor = json.loads(str(archive["descriptor_json"]))
        if int(descriptor["contract_version"]) != CONTRACT_VERSION:
            raise ValueError("unsupported Causal4D contract version")
        context = CausalContext.from_dict(descriptor["context"])
        payload = descriptor["payload"]
        kind = str(descriptor["contract_type"])
        arrays = {name: np.asarray(archive[name]) for name in archive.files if name != "descriptor_json"}
    if kind == TwinBelief.contract_type:
        artifact: Contract = TwinBelief(
            context=context,
            endpoint_frame=int(payload["endpoint_frame"]),
            particle_ids=tuple(map(str, payload["particle_ids"])),
            theta_names=tuple(map(str, payload["theta_names"])),
            metadata=payload["metadata"],
            **arrays,
        )
    elif kind == FactualIntervention.contract_type:
        artifact = FactualIntervention(
            context=context,
            component_ids=tuple(map(str, payload["component_ids"])),
            phi_names=tuple(map(str, payload["phi_names"])),
            kappa_names=tuple(map(str, payload["kappa_names"])),
            evidence_frame_stop=int(payload["evidence_frame_stop"]),
            source_twin_belief_id=str(payload["source_twin_belief_id"]),
            metadata=payload["metadata"],
            **arrays,
        )
    elif kind == CounterfactualQuery.contract_type:
        artifact = CounterfactualQuery(
            context=context,
            horizon_frames=int(payload["horizon_frames"]),
            contact_policy=str(payload["contact_policy"]),
            language=payload["language"],
            source_factual_intervention_id=payload["source_factual_intervention_id"],
            metadata=payload["metadata"],
            query_node_indices=arrays.pop("query_node_indices", None),
            **arrays,
        )
    elif kind == PhysicalPosterior.contract_type:
        artifact = PhysicalPosterior(
            context=context,
            component_ids=tuple(map(str, payload["component_ids"])),
            phi_names=tuple(map(str, payload["phi_names"])),
            kappa_names=tuple(map(str, payload["kappa_names"])),
            source_twin_belief_id=str(payload["source_twin_belief_id"]),
            source_factual_intervention_id=str(payload["source_factual_intervention_id"]),
            source_query_id=str(payload["source_query_id"]),
            metadata=payload["metadata"],
            **arrays,
        )
    elif kind == TaskPosterior.contract_type:
        artifact = TaskPosterior(
            context=context,
            physical_posterior_id=str(payload["physical_posterior_id"]),
            component_ids=tuple(map(str, payload["component_ids"])),
            beta=float(payload["beta"]),
            semantic_source=str(payload["semantic_source"]),
            metadata=payload["metadata"],
            **arrays,
        )
    else:
        raise ValueError(f"unknown Causal4D contract type {kind!r}")
    if artifact.artifact_id != descriptor["artifact_id"]:
        raise ValueError("Causal4D artifact digest does not match its payload")
    return artifact
