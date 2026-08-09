"""Independent binding of Prob4D recursive visual-bias streams.

Prob4D may bind several observation-factor updates to one source-calibrated
visual-bias latent. This module copies and independently validates that stream,
binds every member to BayesianPhysTwin's admitted observations and recursive
factor stream, and requires the persistent-explicit-state nuisance policy.

The current BayesianPhysTwin solver can consume one visual-bias sidecar. It
cannot yet propagate one shared bias posterior through several physical updates.
Accordingly, a multi-update binding is valid evidence plumbing but is
fail-closed for claim-bearing execution rather than being approximated by
repeated independent one-shot updates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    plain_json,
)
from ._prob4d_recursive_policy import RecursiveNuisancePolicyV1
from ._prob4d_stream_binding import (
    Prob4DStreamObservationBindingV1,
    bind_prob4d_stream_observation,
)
from ._prob4d_stream_manifest import Prob4DObservationFactorStreamV1
from .observation_belief import ObservationBeliefV1
from .prob4d_visual_bias_update import (
    PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
    Prob4DVisualBiasBindingV1,
    _array_descriptor,
    _canonical_id,
    _canonical_strings,
    _finite_real,
    _immutable_array,
    _sha256,
    _symmetric_psd,
    validate_prob4d_visual_bias_nuisance,
)

PROB4D_VISUAL_BIAS_MODEL_SCHEMA = "prob4d.visual-bias-model.v1"
PROB4D_VISUAL_BIAS_STREAM_SCHEMA = "prob4d.visual-bias-nuisance-stream"
PROB4D_VISUAL_BIAS_STREAM_VERSION = 1
PROB4D_VISUAL_BIAS_STREAM_UPDATE_SCHEMA = "prob4d.visual-bias-stream-update.v1"
BPT_PROB4D_VISUAL_BIAS_STREAM_BINDING_SCHEMA = (
    "bayesian_phystwin.prob4d_visual_bias_stream_binding"
)
BPT_PROB4D_VISUAL_BIAS_STREAM_BINDING_VERSION = 1
PROB4D_VISUAL_BIAS_NUISANCE_FAMILY_PREFIX = "prob4d-visual-bias:"
PROB4D_VISUAL_BIAS_STREAM_CLAIM_BOUNDARY = (
    "This artifact binds several causal observation updates to one persistent "
    "source-calibrated visual-bias latent and prior. It does not establish provider "
    "competence, target calibration, physical-state identifiability, guarded-query "
    "benefit, Causal4D intervention benefit, deployment safety, or state of the art."
)
BPT_VISUAL_BIAS_STREAM_CLAIM_BOUNDARY = (
    "BayesianPhysTwin independently validated the Prob4D stream, member identities, "
    "sidecars, shared covariance, and persistent nuisance policy. Multi-update "
    "claim-bearing inference remains rejected until a persistent visual-bias state "
    "solver is used; repeated one-shot priors are not treated as independent evidence."
)


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _exact_tuple(value: object, *, name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a canonical tuple")
    return cast(tuple[object, ...], value)


@dataclass(frozen=True, slots=True)
class Prob4DVisualBiasStreamUpdateBindingV1:
    """Independent copy of one Prob4D visual-bias stream update."""

    bias_model_id: str
    observation_stream_update_id: str
    visual_bias_artifact_id: str
    observation_artifact_id: str
    observation_identity_sha256: str
    frame_start: int
    frame_stop_exclusive: int
    row_start: int
    row_stop_exclusive: int
    maximum_gauge_projection: float
    previous_update_id: str | None
    update_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "bias_model_id",
            "observation_stream_update_id",
            "visual_bias_artifact_id",
            "observation_artifact_id",
            "observation_identity_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        frame_start = genuine_integer(
            self.frame_start,
            name="frame_start",
            minimum=0,
        )
        frame_stop = genuine_integer(
            self.frame_stop_exclusive,
            name="frame_stop_exclusive",
            minimum=1,
        )
        row_start = genuine_integer(self.row_start, name="row_start", minimum=0)
        row_stop = genuine_integer(
            self.row_stop_exclusive,
            name="row_stop_exclusive",
            minimum=1,
        )
        if frame_stop <= frame_start:
            raise ValueError("visual-bias stream frame interval must be nonempty")
        if row_stop <= row_start:
            raise ValueError("visual-bias stream row interval must be nonempty")
        maximum = _finite_real(
            self.maximum_gauge_projection,
            name="maximum_gauge_projection",
            minimum=0.0,
        )
        previous = self.previous_update_id
        if previous is not None:
            previous = _sha256(previous, name="previous_update_id")
        object.__setattr__(self, "frame_start", frame_start)
        object.__setattr__(self, "frame_stop_exclusive", frame_stop)
        object.__setattr__(self, "row_start", row_start)
        object.__setattr__(self, "row_stop_exclusive", row_stop)
        object.__setattr__(self, "maximum_gauge_projection", maximum)
        object.__setattr__(self, "previous_update_id", previous)
        expected_id = _canonical_id(self.identity_record())
        supplied_id = self.update_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="update_id")
            if supplied_id != expected_id:
                raise ValueError("Prob4D visual-bias stream update ID mismatch")
        object.__setattr__(self, "update_id", expected_id)

    @property
    def observation_count(self) -> int:
        return self.row_stop_exclusive - self.row_start

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PROB4D_VISUAL_BIAS_STREAM_UPDATE_SCHEMA,
            "bias_model_id": self.bias_model_id,
            "observation_stream_update_id": self.observation_stream_update_id,
            "visual_bias_artifact_id": self.visual_bias_artifact_id,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_identity_sha256": self.observation_identity_sha256,
            "frame_start": self.frame_start,
            "frame_stop_exclusive": self.frame_stop_exclusive,
            "row_start": self.row_start,
            "row_stop_exclusive": self.row_stop_exclusive,
            "maximum_gauge_projection": self.maximum_gauge_projection,
            "previous_update_id": self.previous_update_id,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "update_id": self.update_id}


@dataclass(frozen=True, slots=True)
class ValidatedProb4DVisualBiasStreamV1:
    """Independent BayesianPhysTwin reconstruction of one producer stream."""

    stream_key: str
    bias_ids: tuple[str, ...]
    basis_names: tuple[str, ...]
    orthogonalization_semantics: str
    gauge_projection_tolerance: float
    updates: tuple[Prob4DVisualBiasStreamUpdateBindingV1, ...]
    row_update_indices: np.ndarray
    row_bias_indices: np.ndarray
    bias_jacobian: np.ndarray
    joint_bias_covariance: np.ndarray
    model_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    bias_model_id: str | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        stream_key = _nonempty_string(self.stream_key, name="stream_key")
        bias_ids = _canonical_strings(self.bias_ids, name="bias_ids")
        basis_names = _canonical_strings(self.basis_names, name="basis_names")
        if (
            type(self.orthogonalization_semantics) is not str
            or self.orthogonalization_semantics != PROB4D_VISUAL_BIAS_ORTHOGONALIZATION
        ):
            raise ValueError(
                "claim-bearing recursive visual bias requires the global "
                "gauge-orthogonalized basis"
            )
        tolerance = _finite_real(
            self.gauge_projection_tolerance,
            name="gauge_projection_tolerance",
            strictly_positive=True,
        )
        updates_raw = _exact_tuple(self.updates, name="updates")
        if not updates_raw or any(
            not isinstance(update, Prob4DVisualBiasStreamUpdateBindingV1)
            for update in updates_raw
        ):
            raise ValueError(
                "updates must contain Prob4DVisualBiasStreamUpdateBindingV1 objects"
            )
        updates = cast(
            tuple[Prob4DVisualBiasStreamUpdateBindingV1, ...],
            updates_raw,
        )

        row_update = np.asarray(self.row_update_indices)
        row_bias = np.asarray(self.row_bias_indices)
        jacobian = np.asarray(self.bias_jacobian)
        if row_update.dtype != np.dtype(np.int64) or row_update.ndim != 1:
            raise ValueError("row_update_indices must be one-dimensional int64")
        if row_bias.dtype != np.dtype(np.int64) or row_bias.shape != row_update.shape:
            raise ValueError("row_bias_indices must match row_update_indices as int64")
        if jacobian.dtype != np.dtype(np.float64) or jacobian.shape != (
            row_update.size,
            3,
            len(basis_names),
        ):
            raise ValueError("bias_jacobian shape or dtype differs from the stream")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("bias_jacobian must be finite")
        if np.any(row_bias < 0) or np.any(row_bias >= len(bias_ids)):
            raise ValueError("row_bias_indices refer to an unknown bias scope")
        latent_dimension = len(bias_ids) * len(basis_names)
        covariance = _symmetric_psd(
            self.joint_bias_covariance,
            name="joint_bias_covariance",
            dimension=latent_dimension,
        )
        model_metadata = frozen_finite_json_mapping(
            self.model_metadata,
            name="Prob4D visual-bias stream model metadata",
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="Prob4D visual-bias stream metadata",
        )

        expected_row_start = 0
        previous: Prob4DVisualBiasStreamUpdateBindingV1 | None = None
        unique_fields = (
            "observation_stream_update_id",
            "visual_bias_artifact_id",
            "observation_artifact_id",
            "observation_identity_sha256",
        )
        for name in unique_fields:
            values = tuple(getattr(update, name) for update in updates)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} values must be unique")
        for index, update in enumerate(updates):
            if update.row_start != expected_row_start:
                raise ValueError("visual-bias stream row intervals must be contiguous")
            if update.bias_model_id != updates[0].bias_model_id:
                raise ValueError("visual-bias stream bias_model_id changed")
            expected_previous = None if previous is None else previous.update_id
            if update.previous_update_id != expected_previous:
                raise ValueError("visual-bias stream update hash chain is broken")
            if (
                previous is not None
                and update.frame_start < previous.frame_stop_exclusive
            ):
                raise ValueError("visual-bias stream frame intervals overlap")
            if update.maximum_gauge_projection > tolerance:
                raise ValueError(
                    "visual-bias update exceeds gauge projection tolerance"
                )
            expected_rows: np.ndarray = np.full(
                update.observation_count,
                index,
                dtype=np.int64,
            )
            if not np.array_equal(
                row_update[update.row_start : update.row_stop_exclusive],
                expected_rows,
            ):
                raise ValueError("row_update_indices differ from update intervals")
            expected_row_start = update.row_stop_exclusive
            previous = update
        if expected_row_start != row_update.size:
            raise ValueError("visual-bias stream updates do not cover every row")

        object.__setattr__(self, "stream_key", stream_key)
        object.__setattr__(self, "bias_ids", bias_ids)
        object.__setattr__(self, "basis_names", basis_names)
        object.__setattr__(
            self,
            "orthogonalization_semantics",
            PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
        )
        object.__setattr__(self, "gauge_projection_tolerance", tolerance)
        object.__setattr__(self, "updates", updates)
        object.__setattr__(
            self,
            "row_update_indices",
            _immutable_array(row_update, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "row_bias_indices",
            _immutable_array(row_bias, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "bias_jacobian",
            _immutable_array(jacobian, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "joint_bias_covariance",
            _immutable_array(covariance, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "model_metadata", model_metadata)
        object.__setattr__(self, "metadata", metadata)

        expected_model_id = _canonical_id(self.model_record())
        supplied_model_id = self.bias_model_id
        if supplied_model_id is not None:
            supplied_model_id = _sha256(
                supplied_model_id,
                name="bias_model_id",
            )
            if supplied_model_id != expected_model_id:
                raise ValueError("Prob4D visual-bias model ID mismatch")
        if any(update.bias_model_id != expected_model_id for update in updates):
            raise ValueError("visual-bias updates identify a different bias model")
        object.__setattr__(self, "bias_model_id", expected_model_id)

        expected_artifact_id = _canonical_id(self.identity_record())
        supplied_artifact_id = self.artifact_id
        if supplied_artifact_id is not None:
            supplied_artifact_id = _sha256(
                supplied_artifact_id,
                name="artifact_id",
            )
            if supplied_artifact_id != expected_artifact_id:
                raise ValueError("Prob4D visual-bias stream artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_artifact_id)

    @property
    def observation_count(self) -> int:
        return int(self.row_update_indices.size)

    @property
    def latent_dimension(self) -> int:
        return len(self.bias_ids) * len(self.basis_names)

    @property
    def nuisance_family_id(self) -> str:
        return prob4d_visual_bias_nuisance_family_id(cast(str, self.bias_model_id))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "row_update_indices": np.asarray(self.row_update_indices),
            "row_bias_indices": np.asarray(self.row_bias_indices),
            "bias_jacobian": np.asarray(self.bias_jacobian),
            "joint_bias_covariance": np.asarray(self.joint_bias_covariance),
        }

    def array_descriptors(self) -> dict[str, dict[str, object]]:
        return {name: _array_descriptor(value) for name, value in self.arrays().items()}

    def model_record(self) -> dict[str, object]:
        return {
            "schema": PROB4D_VISUAL_BIAS_MODEL_SCHEMA,
            "bias_ids": list(self.bias_ids),
            "basis_names": list(self.basis_names),
            "joint_bias_covariance": _array_descriptor(
                np.asarray(self.joint_bias_covariance)
            ),
            "orthogonalization_semantics": self.orthogonalization_semantics,
            "gauge_projection_tolerance": self.gauge_projection_tolerance,
            "model_metadata": plain_json(self.model_metadata),
        }

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PROB4D_VISUAL_BIAS_STREAM_SCHEMA,
            "schema_version": PROB4D_VISUAL_BIAS_STREAM_VERSION,
            "stream_key": self.stream_key,
            "bias_model_id": self.bias_model_id,
            "bias_ids": list(self.bias_ids),
            "basis_names": list(self.basis_names),
            "orthogonalization_semantics": self.orthogonalization_semantics,
            "gauge_projection_tolerance": self.gauge_projection_tolerance,
            "updates": [update.to_record() for update in self.updates],
            "arrays": self.array_descriptors(),
            "model_metadata": plain_json(self.model_metadata),
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROB4D_VISUAL_BIAS_STREAM_CLAIM_BOUNDARY,
        }

    def global_design(self) -> np.ndarray:
        result: np.ndarray = np.zeros(
            (self.observation_count, 3, self.latent_dimension),
            dtype=np.float64,
        )
        width = len(self.basis_names)
        for row, bias_index in enumerate(self.row_bias_indices):
            start = int(bias_index) * width
            result[row, :, start : start + width] = self.bias_jacobian[row]
        return _immutable_array(result, dtype=np.dtype(np.float64))


@dataclass(frozen=True, slots=True)
class Prob4DVisualBiasStreamConsumptionBindingV1:
    """Binding of one validated producer stream to BPT recursive artifacts."""

    visual_bias_stream: ValidatedProb4DVisualBiasStreamV1
    factor_stream_artifact_id: str
    factor_stream_update_ids: tuple[str, ...]
    observation_binding_ids: tuple[str, ...]
    recursive_nuisance_policy_id: str
    nuisance_family_id: str
    binding_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.visual_bias_stream,
            ValidatedProb4DVisualBiasStreamV1,
        ):
            raise TypeError(
                "visual_bias_stream must be a ValidatedProb4DVisualBiasStreamV1"
            )
        factor_stream_id = _sha256(
            self.factor_stream_artifact_id,
            name="factor_stream_artifact_id",
        )
        update_ids = _canonical_strings(
            self.factor_stream_update_ids,
            name="factor_stream_update_ids",
        )
        observation_ids = _canonical_strings(
            self.observation_binding_ids,
            name="observation_binding_ids",
        )
        expected_count = len(self.visual_bias_stream.updates)
        if len(update_ids) != expected_count or len(observation_ids) != expected_count:
            raise ValueError("stream binding member counts differ")
        for index, update in enumerate(self.visual_bias_stream.updates):
            if update_ids[index] != update.observation_stream_update_id:
                raise ValueError("factor-stream update order differs from visual bias")
            _sha256(observation_ids[index], name="observation_binding_id")
        policy_id = _sha256(
            self.recursive_nuisance_policy_id,
            name="recursive_nuisance_policy_id",
        )
        family_id = _nonempty_string(
            self.nuisance_family_id,
            name="nuisance_family_id",
        )
        if family_id != self.visual_bias_stream.nuisance_family_id:
            raise ValueError("nuisance family does not identify the visual-bias model")

        object.__setattr__(self, "factor_stream_artifact_id", factor_stream_id)
        object.__setattr__(self, "factor_stream_update_ids", update_ids)
        object.__setattr__(self, "observation_binding_ids", observation_ids)
        object.__setattr__(self, "recursive_nuisance_policy_id", policy_id)
        object.__setattr__(self, "nuisance_family_id", family_id)
        expected_id = _canonical_id(self.descriptor())
        supplied_id = self.binding_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="binding_id")
            if supplied_id != expected_id:
                raise ValueError("visual-bias stream binding ID mismatch")
        object.__setattr__(self, "binding_id", expected_id)

    @property
    def update_count(self) -> int:
        return len(self.visual_bias_stream.updates)

    @property
    def claim_bearing_execution_admissible(self) -> bool:
        return self.update_count == 1

    @property
    def execution_reason(self) -> str:
        if self.claim_bearing_execution_admissible:
            return "single_update_supported_by_visual_bias_v2"
        return "persistent_visual_bias_state_solver_required"

    def require_claim_bearing_execution(self) -> None:
        """Reject repeated one-shot use of one shared recursive bias prior."""

        if not self.claim_bearing_execution_admissible:
            raise ValueError(
                "multi-update visual-bias streams require a persistent explicit-state "
                "solver; repeated one-shot visual-bias updates would duplicate "
                "the prior"
            )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": BPT_PROB4D_VISUAL_BIAS_STREAM_BINDING_SCHEMA,
            "schema_version": BPT_PROB4D_VISUAL_BIAS_STREAM_BINDING_VERSION,
            "visual_bias_stream_artifact_id": self.visual_bias_stream.artifact_id,
            "visual_bias_model_id": self.visual_bias_stream.bias_model_id,
            "factor_stream_artifact_id": self.factor_stream_artifact_id,
            "factor_stream_update_ids": list(self.factor_stream_update_ids),
            "observation_binding_ids": list(self.observation_binding_ids),
            "recursive_nuisance_policy_id": self.recursive_nuisance_policy_id,
            "nuisance_family_id": self.nuisance_family_id,
            "claim_bearing_execution_admissible": (
                self.claim_bearing_execution_admissible
            ),
            "execution_reason": self.execution_reason,
            "claim_boundary": BPT_VISUAL_BIAS_STREAM_CLAIM_BOUNDARY,
        }


def prob4d_visual_bias_nuisance_family_id(bias_model_id: str) -> str:
    """Return the exact recursive-policy family ID for one bias model."""

    return PROB4D_VISUAL_BIAS_NUISANCE_FAMILY_PREFIX + _sha256(
        bias_model_id,
        name="bias_model_id",
    )


def _copy_prob4d_visual_bias_stream(
    nuisance_stream: object,
) -> ValidatedProb4DVisualBiasStreamV1:
    source = cast(Any, nuisance_stream)
    try:
        raw_updates = _exact_tuple(source.updates, name="updates")
        copied_updates: list[Prob4DVisualBiasStreamUpdateBindingV1] = []
        for raw_update in raw_updates:
            update = cast(Any, raw_update)
            copied_updates.append(
                Prob4DVisualBiasStreamUpdateBindingV1(
                    bias_model_id=update.bias_model_id,
                    observation_stream_update_id=update.observation_stream_update_id,
                    visual_bias_artifact_id=update.visual_bias_artifact_id,
                    observation_artifact_id=update.observation_artifact_id,
                    observation_identity_sha256=update.observation_identity_sha256,
                    frame_start=update.frame_start,
                    frame_stop_exclusive=update.frame_stop_exclusive,
                    row_start=update.row_start,
                    row_stop_exclusive=update.row_stop_exclusive,
                    maximum_gauge_projection=update.maximum_gauge_projection,
                    previous_update_id=update.previous_update_id,
                    update_id=update.update_id,
                )
            )
        updates = tuple(copied_updates)
        return ValidatedProb4DVisualBiasStreamV1(
            stream_key=source.stream_key,
            bias_ids=source.bias_ids,
            basis_names=source.basis_names,
            orthogonalization_semantics=source.orthogonalization_semantics,
            gauge_projection_tolerance=source.gauge_projection_tolerance,
            updates=updates,
            row_update_indices=source.row_update_indices,
            row_bias_indices=source.row_bias_indices,
            bias_jacobian=source.bias_jacobian,
            joint_bias_covariance=source.joint_bias_covariance,
            model_metadata=source.model_metadata,
            metadata=source.metadata,
            bias_model_id=source.bias_model_id,
            artifact_id=source.artifact_id,
        )
    except AttributeError as error:
        raise TypeError(
            "nuisance_stream is not a Prob4D VisualBiasNuisanceStreamV1 object"
        ) from error


def _validate_member_sidecar(
    *,
    stream: ValidatedProb4DVisualBiasStreamV1,
    update_index: int,
    observation: ObservationBeliefV1,
    nuisance: object,
) -> Prob4DVisualBiasBindingV1:
    update = stream.updates[update_index]
    sidecar = validate_prob4d_visual_bias_nuisance(
        observation,
        nuisance,
        require_gauge_orthogonalized=True,
    )
    expected = {
        "visual_bias_artifact_id": sidecar.artifact_id,
        "observation_artifact_id": sidecar.observation_artifact_id,
        "observation_identity_sha256": sidecar.observation_identity_sha256,
    }
    for name, value in expected.items():
        if getattr(update, name) != value:
            raise ValueError(f"visual-bias stream update differs in {name}")
    if sidecar.bias_ids != stream.bias_ids:
        raise ValueError("visual-bias sidecar bias IDs differ from the stream")
    if sidecar.basis_names != stream.basis_names:
        raise ValueError("visual-bias sidecar basis names differ from the stream")
    if not np.array_equal(
        sidecar.joint_bias_covariance,
        stream.joint_bias_covariance,
    ):
        raise ValueError("visual-bias sidecar covariance differs from the stream")
    row_slice = slice(update.row_start, update.row_stop_exclusive)
    if not np.array_equal(
        sidecar.row_bias_indices,
        stream.row_bias_indices[row_slice],
    ):
        raise ValueError("visual-bias sidecar row scopes differ from the stream")
    if not np.array_equal(
        sidecar.bias_jacobian,
        stream.bias_jacobian[row_slice],
    ):
        raise ValueError("visual-bias sidecar Jacobian differs from the stream")
    if sidecar.maximum_gauge_projection != update.maximum_gauge_projection:
        raise ValueError("visual-bias sidecar gauge projection differs from the stream")
    return sidecar


def validate_prob4d_visual_bias_nuisance_stream(
    factor_stream: Prob4DObservationFactorStreamV1,
    observations: Sequence[ObservationBeliefV1],
    visual_bias_nuisances: Sequence[object],
    nuisance_stream: object,
    nuisance_policy: RecursiveNuisancePolicyV1,
) -> Prob4DVisualBiasStreamConsumptionBindingV1:
    """Validate and bind one persistent Prob4D visual-bias stream.

    Multi-update streams are intentionally not executed by the existing one-shot
    visual-bias solver. The returned binding records this as a fail-closed
    execution decision while preserving the independently validated evidence.
    """

    if not isinstance(factor_stream, Prob4DObservationFactorStreamV1):
        raise TypeError("factor_stream must be a Prob4DObservationFactorStreamV1")
    if not isinstance(nuisance_policy, RecursiveNuisancePolicyV1):
        raise TypeError("nuisance_policy must be a RecursiveNuisancePolicyV1")
    if nuisance_policy.mode != "persistent_explicit_state":
        raise ValueError(
            "shared recursive visual bias requires persistent_explicit_state; "
            "conditionally independent increments would duplicate the prior"
        )
    stream = _copy_prob4d_visual_bias_stream(nuisance_stream)
    observations_tuple = tuple(observations)
    nuisances_tuple = tuple(visual_bias_nuisances)
    update_count = len(stream.updates)
    if len(factor_stream.updates) != update_count:
        raise ValueError("factor and visual-bias stream update counts differ")
    if len(observations_tuple) != update_count:
        raise ValueError("observation count differs from visual-bias stream updates")
    if len(nuisances_tuple) != update_count:
        raise ValueError("sidecar count differs from visual-bias stream updates")
    if stream.nuisance_family_id not in nuisance_policy.nuisance_family_ids:
        raise ValueError(
            "recursive nuisance policy does not name the exact visual-bias model"
        )

    observation_bindings: list[Prob4DStreamObservationBindingV1] = []
    for index, (observation, nuisance) in enumerate(
        zip(observations_tuple, nuisances_tuple, strict=True)
    ):
        if not isinstance(observation, ObservationBeliefV1):
            raise TypeError("observations must contain ObservationBeliefV1 values")
        factor_update = factor_stream.updates[index]
        visual_update = stream.updates[index]
        if visual_update.observation_stream_update_id != factor_update.update_id:
            raise ValueError(
                "visual-bias update identifies a different factor-stream update"
            )
        if visual_update.frame_start != factor_update.admitted_frame_start:
            raise ValueError("visual-bias and factor-stream frame starts differ")
        if visual_update.frame_stop_exclusive != factor_update.causal_frame_stop:
            raise ValueError("visual-bias and factor-stream frame stops differ")
        if visual_update.observation_identity_sha256 != (
            factor_update.observation_identity_sha256
        ):
            raise ValueError("visual-bias and factor-stream row identities differ")
        if visual_update.observation_count != factor_update.observation_count:
            raise ValueError("visual-bias and factor-stream row counts differ")
        observation_binding = bind_prob4d_stream_observation(
            factor_stream,
            index,
            observation,
        )
        if visual_update.observation_artifact_id != (
            observation_binding.observation_artifact_id
        ):
            raise ValueError(
                "visual-bias update identifies a different BPT observation artifact"
            )
        _validate_member_sidecar(
            stream=stream,
            update_index=index,
            observation=observation,
            nuisance=nuisance,
        )
        observation_bindings.append(observation_binding)

    factor_stream_id = factor_stream.artifact_id
    if factor_stream_id is None:
        raise ValueError("factor stream lacks an artifact ID")
    policy_id = nuisance_policy.policy_id
    if policy_id is None:
        raise ValueError("recursive nuisance policy lacks a policy ID")
    return Prob4DVisualBiasStreamConsumptionBindingV1(
        visual_bias_stream=stream,
        factor_stream_artifact_id=factor_stream_id,
        factor_stream_update_ids=tuple(
            cast(str, update.update_id) for update in factor_stream.updates
        ),
        observation_binding_ids=tuple(
            cast(str, binding.binding_id) for binding in observation_bindings
        ),
        recursive_nuisance_policy_id=policy_id,
        nuisance_family_id=stream.nuisance_family_id,
    )


__all__ = [
    "BPT_PROB4D_VISUAL_BIAS_STREAM_BINDING_SCHEMA",
    "BPT_PROB4D_VISUAL_BIAS_STREAM_BINDING_VERSION",
    "BPT_VISUAL_BIAS_STREAM_CLAIM_BOUNDARY",
    "PROB4D_VISUAL_BIAS_NUISANCE_FAMILY_PREFIX",
    "PROB4D_VISUAL_BIAS_STREAM_CLAIM_BOUNDARY",
    "PROB4D_VISUAL_BIAS_STREAM_SCHEMA",
    "PROB4D_VISUAL_BIAS_STREAM_UPDATE_SCHEMA",
    "PROB4D_VISUAL_BIAS_STREAM_VERSION",
    "Prob4DVisualBiasStreamConsumptionBindingV1",
    "Prob4DVisualBiasStreamUpdateBindingV1",
    "ValidatedProb4DVisualBiasStreamV1",
    "prob4d_visual_bias_nuisance_family_id",
    "validate_prob4d_visual_bias_nuisance_stream",
]
