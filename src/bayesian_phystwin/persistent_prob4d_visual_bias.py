"""Persistent shared-bias inference for recursive Prob4D observations.

The solver carries one visual-bias posterior through an ordered Prob4D factor
stream. It never reinstantiates the source-calibrated bias prior at a later
update. The physical block may contain the complete local linearized state,
including explicit gauge variables, while the persistent bias is represented in
a covariance-root latent coordinate system that remains well defined for
singular source priors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from .prob4d_visual_bias_stream import (
    Prob4DVisualBiasStreamConsumptionBindingV1,
)
from .prob4d_visual_bias_update import (
    _array_descriptor,
    _canonical_id,
    _finite_real,
    _immutable_array,
    _sha256,
    _symmetric_psd,
)

PERSISTENT_VISUAL_BIAS_SOLVER_SCHEMA = (
    "bayesian_phystwin.persistent_prob4d_visual_bias_solver"
)
PERSISTENT_VISUAL_BIAS_SOLVER_VERSION = 1
PERSISTENT_VISUAL_BIAS_BELIEF_SCHEMA = (
    "bayesian_phystwin.persistent_prob4d_visual_bias_belief"
)
PERSISTENT_VISUAL_BIAS_BELIEF_VERSION = 1
PERSISTENT_VISUAL_BIAS_CANDIDATE_SCHEMA = (
    "bayesian_phystwin.persistent_prob4d_visual_bias_candidate"
)
PERSISTENT_VISUAL_BIAS_CANDIDATE_VERSION = 1
PERSISTENT_VISUAL_BIAS_EVENT_SCHEMA = (
    "bayesian_phystwin.persistent_prob4d_visual_bias_event"
)
PERSISTENT_VISUAL_BIAS_EVENT_VERSION = 1
PERSISTENT_VISUAL_BIAS_RUN_SCHEMA = (
    "bayesian_phystwin.persistent_prob4d_visual_bias_run"
)
PERSISTENT_VISUAL_BIAS_RUN_VERSION = 1
PERSISTENT_VISUAL_BIAS_UPDATE_INPUT_SCHEMA = (
    "bayesian_phystwin.persistent_prob4d_visual_bias_update_input"
)
PERSISTENT_VISUAL_BIAS_PREDICTION_INPUT_SCHEMA = (
    "bayesian_phystwin.persistent_prob4d_visual_bias_prediction_input"
)
PERSISTENT_VISUAL_BIAS_EVENT_TYPES = ("prediction", "measurement")
PERSISTENT_VISUAL_BIAS_CLAIM_BOUNDARY = (
    "This solver preserves one source-calibrated visual-bias posterior across "
    "ordered recursive updates and exact fallback decisions. It does not "
    "establish provider competence, complete bias coverage, target calibration, "
    "physical-state identifiability, guarded-query improvement, deployment "
    "safety, Causal4D intervention benefit, or state of the art."
)

EventType = Literal["prediction", "measurement"]


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _float64_vector(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a nonempty one-dimensional float64 array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _float64_matrix(
    value: object,
    *,
    name: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional float64 array")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _symmetric_positive_definite(
    value: object,
    *,
    name: str,
    dimension: int,
) -> np.ndarray:
    symmetric = _symmetric_psd(value, name=name, dimension=dimension)
    try:
        np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return symmetric


def _covariance_root(value: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    root = 0.5 * (root + root.T)
    if not np.allclose(root @ root.T, value, atol=1e-10, rtol=1e-10):
        raise ValueError("visual-bias covariance root failed reconstruction")
    return _immutable_array(root, dtype=np.dtype(np.float64))


def _joint_mean(belief: PersistentVisualBiasBeliefV1) -> np.ndarray:
    return np.concatenate((belief.physical_mean, belief.bias_latent_mean))


def _require_compatible_posterior(
    prior: PersistentVisualBiasBeliefV1,
    posterior: PersistentVisualBiasBeliefV1,
) -> None:
    if posterior.stream_binding_id != prior.stream_binding_id:
        raise ValueError("candidate posterior uses a different stream binding")
    if posterior.visual_bias_model_id != prior.visual_bias_model_id:
        raise ValueError("candidate posterior uses a different visual-bias model")
    if posterior.physical_state_domain_id != prior.physical_state_domain_id:
        raise ValueError("candidate posterior uses a different physical state domain")
    if posterior.physical_dimension != prior.physical_dimension:
        raise ValueError("candidate posterior physical dimension differs")
    if posterior.bias_dimension != prior.bias_dimension:
        raise ValueError("candidate posterior bias dimension differs")
    if not np.array_equal(
        posterior.bias_covariance_root,
        prior.bias_covariance_root,
    ):
        raise ValueError(
            "candidate posterior uses a different visual-bias covariance root"
        )


def _require_measurement_covariance_contraction(
    prior_covariance: np.ndarray,
    posterior_covariance: np.ndarray,
) -> None:
    reduction = prior_covariance - posterior_covariance
    reduction = 0.5 * (reduction + reduction.T)
    eigenvalues = np.linalg.eigvalsh(reduction)
    scale = max(
        1.0,
        float(np.linalg.norm(prior_covariance, ord=2)),
        float(np.linalg.norm(posterior_covariance, ord=2)),
    )
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError(
            "candidate posterior covariance is not a measurement contraction"
        )


def _information_gain_nats(
    prior_covariance: np.ndarray,
    posterior_covariance: np.ndarray,
) -> float:
    prior_sign, prior_logdet = np.linalg.slogdet(prior_covariance)
    posterior_sign, posterior_logdet = np.linalg.slogdet(posterior_covariance)
    if prior_sign <= 0.0 or posterior_sign <= 0.0:
        raise ValueError("belief covariance determinant must be positive")
    gain = 0.5 * float(prior_logdet - posterior_logdet)
    if gain < -1e-10:
        raise ValueError("measurement update increased joint covariance volume")
    return max(gain, 0.0)


def _row_latent_design(
    binding: Prob4DVisualBiasStreamConsumptionBindingV1,
    *,
    row_index: int,
    covariance_root: np.ndarray,
) -> np.ndarray:
    stream = binding.visual_bias_stream
    latent_dimension = stream.latent_dimension
    basis_dimension = len(stream.basis_names)
    scope_index = int(stream.row_bias_indices[row_index])
    provider_design: np.ndarray = np.zeros(
        (3, latent_dimension),
        dtype=np.float64,
    )
    start = scope_index * basis_dimension
    provider_design[:, start : start + basis_dimension] = stream.bias_jacobian[
        row_index
    ]
    return provider_design @ covariance_root


@dataclass(frozen=True, slots=True)
class PersistentVisualBiasBeliefV1:
    """Joint Gaussian belief over a physical block and persistent bias latent."""

    stream_binding_id: str
    visual_bias_model_id: str
    physical_state_domain_id: str
    physical_mean: np.ndarray
    bias_latent_mean: np.ndarray
    joint_covariance: np.ndarray
    bias_covariance_root: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    belief_id: str | None = None

    def __post_init__(self) -> None:
        binding_id = _sha256(self.stream_binding_id, name="stream_binding_id")
        model_id = _sha256(self.visual_bias_model_id, name="visual_bias_model_id")
        domain_id = _nonempty_string(
            self.physical_state_domain_id,
            name="physical_state_domain_id",
        )
        physical_mean = _float64_vector(self.physical_mean, name="physical_mean")
        bias_mean = _float64_vector(
            self.bias_latent_mean,
            name="bias_latent_mean",
        )
        root = _symmetric_psd(
            self.bias_covariance_root,
            name="bias_covariance_root",
            dimension=bias_mean.size,
        )
        joint_dimension = physical_mean.size + bias_mean.size
        covariance = _symmetric_positive_definite(
            self.joint_covariance,
            name="joint_covariance",
            dimension=joint_dimension,
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="persistent visual-bias belief metadata",
        )

        object.__setattr__(self, "stream_binding_id", binding_id)
        object.__setattr__(self, "visual_bias_model_id", model_id)
        object.__setattr__(self, "physical_state_domain_id", domain_id)
        object.__setattr__(
            self,
            "physical_mean",
            _immutable_array(physical_mean, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "bias_latent_mean",
            _immutable_array(bias_mean, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "joint_covariance",
            _immutable_array(covariance, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "bias_covariance_root",
            _immutable_array(root, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "metadata", metadata)

        expected_id = _canonical_id(self.identity_record())
        supplied_id = self.belief_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="belief_id")
            if supplied_id != expected_id:
                raise ValueError("persistent visual-bias belief ID mismatch")
        object.__setattr__(self, "belief_id", expected_id)

    @property
    def physical_dimension(self) -> int:
        return int(self.physical_mean.size)

    @property
    def bias_dimension(self) -> int:
        return int(self.bias_latent_mean.size)

    @property
    def physical_covariance(self) -> np.ndarray:
        stop = self.physical_dimension
        return self.joint_covariance[:stop, :stop]

    @property
    def physical_bias_cross_covariance(self) -> np.ndarray:
        stop = self.physical_dimension
        return self.joint_covariance[:stop, stop:]

    @property
    def bias_latent_covariance(self) -> np.ndarray:
        start = self.physical_dimension
        return self.joint_covariance[start:, start:]

    @property
    def provider_bias_mean(self) -> np.ndarray:
        mean = self.bias_covariance_root @ self.bias_latent_mean
        return _immutable_array(mean, dtype=np.dtype(np.float64))

    @property
    def provider_bias_covariance(self) -> np.ndarray:
        covariance = (
            self.bias_covariance_root
            @ self.bias_latent_covariance
            @ self.bias_covariance_root.T
        )
        covariance = 0.5 * (covariance + covariance.T)
        return _immutable_array(covariance, dtype=np.dtype(np.float64))

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PERSISTENT_VISUAL_BIAS_BELIEF_SCHEMA,
            "schema_version": PERSISTENT_VISUAL_BIAS_BELIEF_VERSION,
            "stream_binding_id": self.stream_binding_id,
            "visual_bias_model_id": self.visual_bias_model_id,
            "physical_state_domain_id": self.physical_state_domain_id,
            "physical_mean": _array_descriptor(np.asarray(self.physical_mean)),
            "bias_latent_mean": _array_descriptor(np.asarray(self.bias_latent_mean)),
            "joint_covariance": _array_descriptor(np.asarray(self.joint_covariance)),
            "bias_covariance_root": _array_descriptor(
                np.asarray(self.bias_covariance_root)
            ),
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PersistentVisualBiasCandidateV1:
    """One content-bound candidate measurement update."""

    stream_binding_id: str
    visual_bias_stream_update_id: str
    factor_stream_update_id: str
    observation_binding_id: str
    physical_linearization_id: str
    update_index: int
    update_input_id: str
    prior_belief_id: str
    posterior_belief: PersistentVisualBiasBeliefV1
    conditional_innovation_quadratic_per_dimension: float
    information_gain_nats: float
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "stream_binding_id",
            "visual_bias_stream_update_id",
            "factor_stream_update_id",
            "observation_binding_id",
            "physical_linearization_id",
            "update_input_id",
            "prior_belief_id",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        index = genuine_integer(self.update_index, name="update_index", minimum=0)
        if not isinstance(self.posterior_belief, PersistentVisualBiasBeliefV1):
            raise TypeError("posterior_belief must be a PersistentVisualBiasBeliefV1")
        if self.posterior_belief.stream_binding_id != self.stream_binding_id:
            raise ValueError("candidate posterior uses a different stream binding")
        posterior_lineage = self.posterior_belief.metadata
        if posterior_lineage.get("source_update_index") != index:
            raise ValueError("candidate posterior source update index differs")
        if (
            posterior_lineage.get("physical_linearization_id")
            != self.physical_linearization_id
        ):
            raise ValueError("candidate posterior physical linearization differs")
        quadratic = _finite_real(
            self.conditional_innovation_quadratic_per_dimension,
            name="conditional_innovation_quadratic_per_dimension",
            minimum=0.0,
        )
        information_gain = _finite_real(
            self.information_gain_nats,
            name="information_gain_nats",
            minimum=0.0,
        )
        object.__setattr__(self, "update_index", index)
        object.__setattr__(
            self,
            "conditional_innovation_quadratic_per_dimension",
            quadratic,
        )
        object.__setattr__(self, "information_gain_nats", information_gain)
        expected_id = _canonical_id(self.identity_record())
        supplied_id = self.candidate_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="candidate_id")
            if supplied_id != expected_id:
                raise ValueError("persistent visual-bias candidate ID mismatch")
        object.__setattr__(self, "candidate_id", expected_id)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PERSISTENT_VISUAL_BIAS_CANDIDATE_SCHEMA,
            "schema_version": PERSISTENT_VISUAL_BIAS_CANDIDATE_VERSION,
            "solver_schema": PERSISTENT_VISUAL_BIAS_SOLVER_SCHEMA,
            "solver_version": PERSISTENT_VISUAL_BIAS_SOLVER_VERSION,
            "stream_binding_id": self.stream_binding_id,
            "visual_bias_stream_update_id": self.visual_bias_stream_update_id,
            "factor_stream_update_id": self.factor_stream_update_id,
            "observation_binding_id": self.observation_binding_id,
            "physical_linearization_id": self.physical_linearization_id,
            "update_index": self.update_index,
            "update_input_id": self.update_input_id,
            "prior_belief_id": self.prior_belief_id,
            "posterior_belief_id": self.posterior_belief.belief_id,
            "conditional_innovation_quadratic_per_dimension": (
                self.conditional_innovation_quadratic_per_dimension
            ),
            "information_gain_nats": self.information_gain_nats,
            "claim_boundary": PERSISTENT_VISUAL_BIAS_CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class PersistentVisualBiasEventV1:
    """Prediction or accept/fallback event in one persistent solver run."""

    event_type: EventType
    update_index: int
    prior_belief_id: str
    candidate_belief_id: str
    selected_belief_id: str
    event_input_id: str
    reason: str
    visual_bias_stream_update_id: str | None = None
    candidate_id: str | None = None
    accepted: bool | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.event_type) is not str or self.event_type not in (
            PERSISTENT_VISUAL_BIAS_EVENT_TYPES
        ):
            raise ValueError("event_type must be prediction or measurement")
        index = genuine_integer(self.update_index, name="update_index", minimum=0)
        for name in (
            "prior_belief_id",
            "candidate_belief_id",
            "selected_belief_id",
            "event_input_id",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        reason = _nonempty_string(self.reason, name="reason")
        update_id = self.visual_bias_stream_update_id
        candidate_id = self.candidate_id
        accepted = self.accepted
        if self.event_type == "prediction":
            if (
                update_id is not None
                or candidate_id is not None
                or accepted is not None
            ):
                raise ValueError("prediction events cannot carry measurement fields")
            if self.selected_belief_id != self.candidate_belief_id:
                raise ValueError("prediction events must select the predicted belief")
        else:
            if update_id is None or candidate_id is None or accepted is None:
                raise ValueError(
                    "measurement events require update and decision fields"
                )
            update_id = _sha256(
                update_id,
                name="visual_bias_stream_update_id",
            )
            candidate_id = _sha256(candidate_id, name="candidate_id")
            accepted = genuine_boolean(accepted, name="accepted")
            expected = self.candidate_belief_id if accepted else self.prior_belief_id
            if self.selected_belief_id != expected:
                raise ValueError("measurement event violates exact accept/fallback")
        object.__setattr__(self, "update_index", index)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "visual_bias_stream_update_id", update_id)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "accepted", accepted)
        expected_id = _canonical_id(self.identity_record())
        supplied_id = self.event_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="event_id")
            if supplied_id != expected_id:
                raise ValueError("persistent visual-bias event ID mismatch")
        object.__setattr__(self, "event_id", expected_id)

    @property
    def exact_fallback_reproduced(self) -> bool | None:
        if self.event_type != "measurement" or self.accepted:
            return None
        return self.selected_belief_id == self.prior_belief_id

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PERSISTENT_VISUAL_BIAS_EVENT_SCHEMA,
            "schema_version": PERSISTENT_VISUAL_BIAS_EVENT_VERSION,
            "event_type": self.event_type,
            "update_index": self.update_index,
            "prior_belief_id": self.prior_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "selected_belief_id": self.selected_belief_id,
            "event_input_id": self.event_input_id,
            "reason": self.reason,
            "visual_bias_stream_update_id": self.visual_bias_stream_update_id,
            "candidate_id": self.candidate_id,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class PersistentVisualBiasRunV1:
    """Ordered persistent-bias run with exact accept/fallback history."""

    stream_binding: Prob4DVisualBiasStreamConsumptionBindingV1
    initial_belief_id: str
    belief: PersistentVisualBiasBeliefV1
    events: tuple[PersistentVisualBiasEventV1, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.stream_binding,
            Prob4DVisualBiasStreamConsumptionBindingV1,
        ):
            raise TypeError(
                "stream_binding must be a Prob4DVisualBiasStreamConsumptionBindingV1"
            )
        binding_id = self.stream_binding.binding_id
        if binding_id is None:
            raise ValueError("stream binding lacks a binding ID")
        initial_id = _sha256(self.initial_belief_id, name="initial_belief_id")
        if not isinstance(self.belief, PersistentVisualBiasBeliefV1):
            raise TypeError("belief must be a PersistentVisualBiasBeliefV1")
        if self.belief.stream_binding_id != binding_id:
            raise ValueError("run belief uses a different stream binding")
        stream = self.stream_binding.visual_bias_stream
        if self.belief.visual_bias_model_id != stream.bias_model_id:
            raise ValueError("run belief uses a different visual-bias model")
        if self.belief.bias_dimension != stream.latent_dimension:
            raise ValueError("run belief bias dimension differs from the stream")
        reconstructed_bias_covariance = (
            self.belief.bias_covariance_root @ self.belief.bias_covariance_root.T
        )
        if not np.allclose(
            reconstructed_bias_covariance,
            stream.joint_bias_covariance,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("run belief uses a different visual-bias prior root")
        if type(self.events) is not tuple or any(
            not isinstance(event, PersistentVisualBiasEventV1) for event in self.events
        ):
            raise TypeError("events must be a canonical tuple of solver events")
        expected_prior = initial_id
        next_update_index = 0
        for event in self.events:
            if event.prior_belief_id != expected_prior:
                raise ValueError("persistent visual-bias event belief chain is broken")
            if event.update_index != next_update_index:
                raise ValueError("persistent visual-bias event update index differs")
            if event.event_type == "measurement":
                if next_update_index >= len(stream.updates):
                    raise ValueError("persistent visual-bias run has extra updates")
                expected_update_id = stream.updates[next_update_index].update_id
                if event.visual_bias_stream_update_id != expected_update_id:
                    raise ValueError("persistent visual-bias update order differs")
                next_update_index += 1
            expected_prior = event.selected_belief_id
        if expected_prior != self.belief.belief_id:
            raise ValueError("run belief differs from the terminal event selection")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="persistent visual-bias run metadata",
        )
        object.__setattr__(self, "initial_belief_id", initial_id)
        object.__setattr__(self, "metadata", metadata)
        expected_id = _canonical_id(self.descriptor())
        supplied_id = self.run_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="run_id")
            if supplied_id != expected_id:
                raise ValueError("persistent visual-bias run ID mismatch")
        object.__setattr__(self, "run_id", expected_id)

    @property
    def next_update_index(self) -> int:
        return sum(event.event_type == "measurement" for event in self.events)

    @property
    def complete(self) -> bool:
        return self.next_update_index == len(
            self.stream_binding.visual_bias_stream.updates
        )

    @property
    def claim_bearing_execution_admissible(self) -> bool:
        return True

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": PERSISTENT_VISUAL_BIAS_RUN_SCHEMA,
            "schema_version": PERSISTENT_VISUAL_BIAS_RUN_VERSION,
            "solver_schema": PERSISTENT_VISUAL_BIAS_SOLVER_SCHEMA,
            "solver_version": PERSISTENT_VISUAL_BIAS_SOLVER_VERSION,
            "stream_binding_id": self.stream_binding.binding_id,
            "visual_bias_stream_artifact_id": (
                self.stream_binding.visual_bias_stream.artifact_id
            ),
            "initial_belief_id": self.initial_belief_id,
            "current_belief_id": self.belief.belief_id,
            "next_update_index": self.next_update_index,
            "complete": self.complete,
            "events": [
                {**event.identity_record(), "event_id": event.event_id}
                for event in self.events
            ],
            "metadata": plain_json(self.metadata),
            "claim_boundary": PERSISTENT_VISUAL_BIAS_CLAIM_BOUNDARY,
        }


def start_persistent_visual_bias_run(
    stream_binding: Prob4DVisualBiasStreamConsumptionBindingV1,
    *,
    physical_state_domain_id: str,
    physical_mean: np.ndarray,
    physical_covariance: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
) -> PersistentVisualBiasRunV1:
    """Start one solver run with the visual-bias prior represented exactly once."""

    if not isinstance(
        stream_binding,
        Prob4DVisualBiasStreamConsumptionBindingV1,
    ):
        raise TypeError(
            "stream_binding must be a Prob4DVisualBiasStreamConsumptionBindingV1"
        )
    binding_id = stream_binding.binding_id
    model_id = stream_binding.visual_bias_stream.bias_model_id
    if binding_id is None or model_id is None:
        raise ValueError("validated stream binding lacks immutable identities")
    physical_mean_array = _float64_vector(
        physical_mean,
        name="physical_mean",
    )
    physical_covariance_array = _symmetric_positive_definite(
        physical_covariance,
        name="physical_covariance",
        dimension=physical_mean_array.size,
    )
    bias_covariance = np.asarray(
        stream_binding.visual_bias_stream.joint_bias_covariance
    )
    root = _covariance_root(bias_covariance)
    bias_dimension = root.shape[0]
    joint_covariance = np.zeros(
        (
            physical_mean_array.size + bias_dimension,
            physical_mean_array.size + bias_dimension,
        ),
        dtype=np.float64,
    )
    joint_covariance[
        : physical_mean_array.size,
        : physical_mean_array.size,
    ] = physical_covariance_array
    joint_covariance[
        physical_mean_array.size :,
        physical_mean_array.size :,
    ] = np.eye(bias_dimension, dtype=np.float64)
    belief = PersistentVisualBiasBeliefV1(
        stream_binding_id=binding_id,
        visual_bias_model_id=model_id,
        physical_state_domain_id=physical_state_domain_id,
        physical_mean=physical_mean_array,
        bias_latent_mean=np.zeros(bias_dimension, dtype=np.float64),
        joint_covariance=joint_covariance,
        bias_covariance_root=root,
        metadata={"initial_visual_bias_prior_instantiations": 1},
    )
    belief_id = cast(str, belief.belief_id)
    return PersistentVisualBiasRunV1(
        stream_binding=stream_binding,
        initial_belief_id=belief_id,
        belief=belief,
        metadata={} if metadata is None else metadata,
    )


def predict_persistent_visual_bias_run(
    run: PersistentVisualBiasRunV1,
    *,
    physical_transition: np.ndarray,
    process_covariance: np.ndarray,
    transition_id: str,
    physical_offset: np.ndarray | None = None,
    reason: str = "physical_prediction",
) -> PersistentVisualBiasRunV1:
    """Propagate the physical block while retaining the same bias posterior."""

    if not isinstance(run, PersistentVisualBiasRunV1):
        raise TypeError("run must be a PersistentVisualBiasRunV1")
    transition_digest = _sha256(transition_id, name="transition_id")
    belief = run.belief
    physical_dimension = belief.physical_dimension
    transition = _float64_matrix(
        physical_transition,
        name="physical_transition",
        shape=(physical_dimension, physical_dimension),
    )
    process = _symmetric_psd(
        process_covariance,
        name="process_covariance",
        dimension=physical_dimension,
    )
    offset: np.ndarray
    if physical_offset is None:
        offset = np.zeros(physical_dimension, dtype=np.float64)
    else:
        offset = _float64_vector(physical_offset, name="physical_offset")
        if offset.size != physical_dimension:
            raise ValueError("physical_offset dimension differs from the belief")

    bias_dimension = belief.bias_dimension
    affine = np.eye(physical_dimension + bias_dimension, dtype=np.float64)
    affine[:physical_dimension, :physical_dimension] = transition
    joint_mean = _joint_mean(belief)
    predicted_mean = affine @ joint_mean
    predicted_mean[:physical_dimension] += offset
    predicted_covariance = affine @ belief.joint_covariance @ affine.T
    predicted_covariance[:physical_dimension, :physical_dimension] += process
    predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
    candidate = PersistentVisualBiasBeliefV1(
        stream_binding_id=belief.stream_binding_id,
        visual_bias_model_id=belief.visual_bias_model_id,
        physical_state_domain_id=belief.physical_state_domain_id,
        physical_mean=predicted_mean[:physical_dimension],
        bias_latent_mean=predicted_mean[physical_dimension:],
        joint_covariance=predicted_covariance,
        bias_covariance_root=belief.bias_covariance_root,
        metadata={"prediction_transition_id": transition_digest},
    )
    event_input_id = _canonical_id(
        {
            "schema": PERSISTENT_VISUAL_BIAS_PREDICTION_INPUT_SCHEMA,
            "transition_id": transition_digest,
            "physical_transition": _array_descriptor(transition),
            "process_covariance": _array_descriptor(process),
            "physical_offset": _array_descriptor(offset),
        }
    )
    event = PersistentVisualBiasEventV1(
        event_type="prediction",
        update_index=run.next_update_index,
        prior_belief_id=cast(str, belief.belief_id),
        candidate_belief_id=cast(str, candidate.belief_id),
        selected_belief_id=cast(str, candidate.belief_id),
        event_input_id=event_input_id,
        reason=reason,
    )
    return PersistentVisualBiasRunV1(
        stream_binding=run.stream_binding,
        initial_belief_id=run.initial_belief_id,
        belief=candidate,
        events=(*run.events, event),
        metadata=run.metadata,
    )


def propose_persistent_visual_bias_update(
    run: PersistentVisualBiasRunV1,
    *,
    innovation_xyz: np.ndarray,
    physical_jacobian: np.ndarray,
    conditional_covariance: np.ndarray,
    physical_linearization_id: str,
) -> PersistentVisualBiasCandidateV1:
    """Form one matrix-free joint physical and persistent-bias update."""

    if not isinstance(run, PersistentVisualBiasRunV1):
        raise TypeError("run must be a PersistentVisualBiasRunV1")
    if run.complete:
        raise ValueError("persistent visual-bias run is already complete")
    linearization_id = _sha256(
        physical_linearization_id,
        name="physical_linearization_id",
    )
    index = run.next_update_index
    binding = run.stream_binding
    stream = binding.visual_bias_stream
    update = stream.updates[index]
    row_count = update.observation_count
    physical_dimension = run.belief.physical_dimension
    innovation = np.asarray(innovation_xyz)
    jacobian = np.asarray(physical_jacobian)
    covariance = np.asarray(conditional_covariance)
    if innovation.dtype != np.dtype(np.float64) or innovation.shape != (
        row_count,
        3,
    ):
        raise ValueError("innovation_xyz must be float64 with shape (N, 3)")
    if jacobian.dtype != np.dtype(np.float64) or jacobian.shape != (
        row_count,
        3,
        physical_dimension,
    ):
        raise ValueError(
            "physical_jacobian must be float64 with shape (N, 3, physical_dimension)"
        )
    if covariance.dtype != np.dtype(np.float64) or covariance.shape != (
        row_count,
        3,
        3,
    ):
        raise ValueError("conditional_covariance must be float64 with shape (N, 3, 3)")
    if not np.all(np.isfinite(innovation)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("innovation and physical Jacobian must be finite")

    prior_covariance = np.asarray(run.belief.joint_covariance)
    joint_dimension = prior_covariance.shape[0]
    information = np.linalg.solve(
        prior_covariance,
        np.eye(joint_dimension, dtype=np.float64),
    )
    information = 0.5 * (information + information.T)
    score = np.zeros(joint_dimension, dtype=np.float64)
    innovation_quadratic = 0.0
    for local_row in range(row_count):
        row_covariance = _symmetric_positive_definite(
            covariance[local_row],
            name=f"conditional_covariance[{local_row}]",
            dimension=3,
        )
        global_row = update.row_start + local_row
        bias_design = _row_latent_design(
            binding,
            row_index=global_row,
            covariance_root=run.belief.bias_covariance_root,
        )
        design = np.concatenate((jacobian[local_row], bias_design), axis=1)
        weighted_design = np.linalg.solve(row_covariance, design)
        weighted_innovation = np.linalg.solve(
            row_covariance,
            innovation[local_row],
        )
        information += design.T @ weighted_design
        score += design.T @ weighted_innovation
        innovation_quadratic += float(innovation[local_row] @ weighted_innovation)
    information = 0.5 * (information + information.T)
    posterior_covariance = np.linalg.solve(
        information,
        np.eye(joint_dimension, dtype=np.float64),
    )
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    delta = np.linalg.solve(information, score)
    posterior_mean = _joint_mean(run.belief) + delta
    posterior = PersistentVisualBiasBeliefV1(
        stream_binding_id=run.belief.stream_binding_id,
        visual_bias_model_id=run.belief.visual_bias_model_id,
        physical_state_domain_id=run.belief.physical_state_domain_id,
        physical_mean=posterior_mean[:physical_dimension],
        bias_latent_mean=posterior_mean[physical_dimension:],
        joint_covariance=posterior_covariance,
        bias_covariance_root=run.belief.bias_covariance_root,
        metadata={
            "source_update_index": index,
            "physical_linearization_id": linearization_id,
        },
    )
    update_id = update.update_id
    if update_id is None:
        raise ValueError("visual-bias stream update lacks an update ID")
    update_input_id = _canonical_id(
        {
            "schema": PERSISTENT_VISUAL_BIAS_UPDATE_INPUT_SCHEMA,
            "stream_binding_id": binding.binding_id,
            "visual_bias_stream_update_id": update_id,
            "factor_stream_update_id": binding.factor_stream_update_ids[index],
            "observation_binding_id": binding.observation_binding_ids[index],
            "physical_linearization_id": linearization_id,
            "innovation_xyz": _array_descriptor(innovation),
            "physical_jacobian": _array_descriptor(jacobian),
            "conditional_covariance": _array_descriptor(covariance),
        }
    )
    return PersistentVisualBiasCandidateV1(
        stream_binding_id=cast(str, binding.binding_id),
        visual_bias_stream_update_id=update_id,
        factor_stream_update_id=binding.factor_stream_update_ids[index],
        observation_binding_id=binding.observation_binding_ids[index],
        physical_linearization_id=linearization_id,
        update_index=index,
        update_input_id=update_input_id,
        prior_belief_id=cast(str, run.belief.belief_id),
        posterior_belief=posterior,
        conditional_innovation_quadratic_per_dimension=(
            innovation_quadratic / float(3 * row_count)
        ),
        information_gain_nats=_information_gain_nats(
            prior_covariance,
            posterior_covariance,
        ),
    )


def select_persistent_visual_bias_candidate(
    run: PersistentVisualBiasRunV1,
    candidate: PersistentVisualBiasCandidateV1,
    *,
    innovation_xyz: np.ndarray,
    physical_jacobian: np.ndarray,
    conditional_covariance: np.ndarray,
    accepted: bool,
    reason: str,
) -> PersistentVisualBiasRunV1:
    """Commit an accepted posterior or retain the exact prior belief object."""

    if not isinstance(run, PersistentVisualBiasRunV1):
        raise TypeError("run must be a PersistentVisualBiasRunV1")
    if not isinstance(candidate, PersistentVisualBiasCandidateV1):
        raise TypeError("candidate must be a PersistentVisualBiasCandidateV1")
    accept = genuine_boolean(accepted, name="accepted")
    binding = run.stream_binding
    if candidate.stream_binding_id != binding.binding_id:
        raise ValueError("candidate uses a different stream binding")
    index = run.next_update_index
    if candidate.update_index != index:
        raise ValueError("candidate update index is stale or out of order")
    if candidate.prior_belief_id != run.belief.belief_id:
        raise ValueError("candidate prior belief is stale")
    update = binding.visual_bias_stream.updates[index]
    if candidate.visual_bias_stream_update_id != update.update_id:
        raise ValueError("candidate identifies a different visual-bias update")
    if candidate.factor_stream_update_id != binding.factor_stream_update_ids[index]:
        raise ValueError(
            "candidate factor-stream update differs from the active stream member"
        )
    if candidate.observation_binding_id != binding.observation_binding_ids[index]:
        raise ValueError(
            "candidate observation binding differs from the active stream member"
        )
    posterior = candidate.posterior_belief
    _require_compatible_posterior(run.belief, posterior)
    _require_measurement_covariance_contraction(
        run.belief.joint_covariance,
        posterior.joint_covariance,
    )
    expected_information_gain = _information_gain_nats(
        run.belief.joint_covariance,
        posterior.joint_covariance,
    )
    gain_scale = max(1.0, abs(expected_information_gain))
    if not np.isclose(
        candidate.information_gain_nats,
        expected_information_gain,
        rtol=1e-10,
        atol=1e-12 * gain_scale,
    ):
        raise ValueError(
            "candidate information gain does not match posterior covariance"
        )
    reproduced = propose_persistent_visual_bias_update(
        run,
        innovation_xyz=innovation_xyz,
        physical_jacobian=physical_jacobian,
        conditional_covariance=conditional_covariance,
        physical_linearization_id=candidate.physical_linearization_id,
    )
    if reproduced.candidate_id != candidate.candidate_id:
        raise ValueError("candidate does not match canonical solver reproduction")
    selected = posterior if accept else run.belief
    event = PersistentVisualBiasEventV1(
        event_type="measurement",
        update_index=run.next_update_index,
        prior_belief_id=cast(str, run.belief.belief_id),
        candidate_belief_id=cast(str, candidate.posterior_belief.belief_id),
        selected_belief_id=cast(str, selected.belief_id),
        event_input_id=candidate.update_input_id,
        reason=reason,
        visual_bias_stream_update_id=candidate.visual_bias_stream_update_id,
        candidate_id=candidate.candidate_id,
        accepted=accept,
    )
    return PersistentVisualBiasRunV1(
        stream_binding=run.stream_binding,
        initial_belief_id=run.initial_belief_id,
        belief=selected,
        events=(*run.events, event),
        metadata=run.metadata,
    )


def apply_persistent_visual_bias_update(
    run: PersistentVisualBiasRunV1,
    *,
    innovation_xyz: np.ndarray,
    physical_jacobian: np.ndarray,
    conditional_covariance: np.ndarray,
    physical_linearization_id: str,
    accepted: bool,
    reason: str,
) -> tuple[PersistentVisualBiasRunV1, PersistentVisualBiasCandidateV1]:
    """Propose and transactionally select one ordered stream update."""

    candidate = propose_persistent_visual_bias_update(
        run,
        innovation_xyz=innovation_xyz,
        physical_jacobian=physical_jacobian,
        conditional_covariance=conditional_covariance,
        physical_linearization_id=physical_linearization_id,
    )
    selected = select_persistent_visual_bias_candidate(
        run,
        candidate,
        innovation_xyz=innovation_xyz,
        physical_jacobian=physical_jacobian,
        conditional_covariance=conditional_covariance,
        accepted=accepted,
        reason=reason,
    )
    return selected, candidate


__all__ = [
    "PERSISTENT_VISUAL_BIAS_BELIEF_SCHEMA",
    "PERSISTENT_VISUAL_BIAS_BELIEF_VERSION",
    "PERSISTENT_VISUAL_BIAS_CANDIDATE_SCHEMA",
    "PERSISTENT_VISUAL_BIAS_CANDIDATE_VERSION",
    "PERSISTENT_VISUAL_BIAS_CLAIM_BOUNDARY",
    "PERSISTENT_VISUAL_BIAS_EVENT_SCHEMA",
    "PERSISTENT_VISUAL_BIAS_EVENT_TYPES",
    "PERSISTENT_VISUAL_BIAS_EVENT_VERSION",
    "PERSISTENT_VISUAL_BIAS_RUN_SCHEMA",
    "PERSISTENT_VISUAL_BIAS_RUN_VERSION",
    "PERSISTENT_VISUAL_BIAS_SOLVER_SCHEMA",
    "PERSISTENT_VISUAL_BIAS_SOLVER_VERSION",
    "PersistentVisualBiasBeliefV1",
    "PersistentVisualBiasCandidateV1",
    "PersistentVisualBiasEventV1",
    "PersistentVisualBiasRunV1",
    "apply_persistent_visual_bias_update",
    "predict_persistent_visual_bias_run",
    "propose_persistent_visual_bias_update",
    "select_persistent_visual_bias_candidate",
    "start_persistent_visual_bias_run",
]
