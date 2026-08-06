"""Claim-bearing consumption of Prob4D coherent visual-bias sidecars.

Prob4D's ``VisualBiasNuisanceV1`` stores row-local bias bases together with one
complete cross-scope covariance. BayesianPhysTwin's frozen V1 solver uses an
isotropic prior for explicit shared-bias coordinates. This module preserves
both boundaries by independently validating the producer object and
reparameterizing its complete covariance through a symmetric square root.

The V2 update is additive: existing V1 update identities and solver semantics
remain unchanged, and the coherent covariance is never added to the local point
covariance while it is retained as an explicit nuisance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from ._gauge_aware_contracts import GaugeAwareBeliefResult
from ._prob4d_stream_binding import prob4d_observation_identity_summary
from .claim_bearing_prob4d import (
    build_claim_bearing_gauge_aware_batch_from_artifacts,
)
from .observation_belief import ObservationBeliefV1
from .physical_linearization import PhysicalLinearizationV1
from .prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)
from .prospective_prob4d_update import ClaimBearingProb4DUpdateV1

PROB4D_VISUAL_BIAS_NUISANCE_SCHEMA = "prob4d.visual-bias-nuisance"
PROB4D_VISUAL_BIAS_NUISANCE_VERSION = 1
PROB4D_VISUAL_BIAS_ORTHOGONALIZATION = "conditional-whitened-global-gauge-projection-v1"
PROB4D_VISUAL_BIAS_REPARAMETERIZATION = (
    "full-joint-covariance-root-to-isotropic-shared-bias-v1"
)
CLAIM_BEARING_PROB4D_VISUAL_BIAS_UPDATE_VERSION = 2


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a literal string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _canonical_strings(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a canonical tuple")
    result = cast(tuple[object, ...], value)
    if not result or any(type(item) is not str or not item for item in result):
        raise ValueError(f"{name} must contain nonempty literal strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return cast(tuple[str, ...], result)


def _immutable_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Return a C-contiguous array backed by immutable ``bytes`` storage."""

    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.dtype.hasobject:
        raise TypeError("contract arrays must not contain Python objects")
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=array.dtype).reshape(array.shape)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }


def _symmetric_psd(value: object, *, name: str, dimension: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must be float64 with shape ({dimension}, {dimension})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(array, array.T, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (array + array.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def _validated_calibration_ids(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("claim-bearing calibration artifact IDs are missing")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if type(name) is not str or not name:
            raise ValueError("calibration artifact names must be nonempty strings")
        result[name] = _sha256(digest, name=f"calibration artifact {name}")
    return MappingProxyType(dict(sorted(result.items())))


def _runtime_revision_source(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("runtime_revision_source must be a nonempty literal string")
    return value


@dataclass(frozen=True, slots=True)
class Prob4DVisualBiasBindingV1:
    """Independent BayesianPhysTwin validation of one Prob4D bias sidecar."""

    observation_artifact_id: str
    observation_identity_sha256: str
    bias_ids: tuple[str, ...]
    basis_names: tuple[str, ...]
    row_bias_indices: np.ndarray
    bias_jacobian: np.ndarray
    joint_bias_covariance: np.ndarray
    orthogonalization_semantics: str
    maximum_gauge_projection: float
    gauge_projection_tolerance: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        observation_id = _sha256(
            self.observation_artifact_id,
            name="observation_artifact_id",
        )
        identity_sha = _sha256(
            self.observation_identity_sha256,
            name="observation_identity_sha256",
        )
        bias_ids = _canonical_strings(self.bias_ids, name="bias_ids")
        basis_names = _canonical_strings(self.basis_names, name="basis_names")

        row_indices = np.asarray(self.row_bias_indices)
        if row_indices.dtype != np.dtype(np.int64) or row_indices.ndim != 1:
            raise ValueError("row_bias_indices must be a one-dimensional int64 array")
        if row_indices.size < 1:
            raise ValueError("visual-bias sidecar requires observation rows")
        if np.any(row_indices < 0) or np.any(row_indices >= len(bias_ids)):
            raise ValueError("row_bias_indices refer to an unknown bias ID")

        jacobian = np.asarray(self.bias_jacobian)
        expected_shape = (row_indices.size, 3, len(basis_names))
        if jacobian.dtype != np.dtype(np.float64) or jacobian.shape != expected_shape:
            raise ValueError(
                f"bias_jacobian must be float64 with shape {expected_shape}"
            )
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("bias_jacobian must be finite")

        latent_dimension = len(bias_ids) * len(basis_names)
        covariance = _symmetric_psd(
            self.joint_bias_covariance,
            name="joint_bias_covariance",
            dimension=latent_dimension,
        )
        if type(self.orthogonalization_semantics) is not str:
            raise ValueError("orthogonalization_semantics must be a literal string")
        semantics = self.orthogonalization_semantics
        if semantics not in {
            "not-orthogonalized",
            PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
        }:
            raise ValueError("unsupported visual-bias orthogonalization semantics")
        maximum_projection = _finite_real(
            self.maximum_gauge_projection,
            name="maximum_gauge_projection",
            minimum=0.0,
        )
        projection_tolerance = _finite_real(
            self.gauge_projection_tolerance,
            name="gauge_projection_tolerance",
            strictly_positive=True,
        )
        if (
            semantics == PROB4D_VISUAL_BIAS_ORTHOGONALIZATION
            and maximum_projection > projection_tolerance
        ):
            raise ValueError(
                "orthogonalized bias basis exceeds its gauge projection tolerance"
            )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="Prob4D visual-bias metadata",
        )

        object.__setattr__(self, "observation_artifact_id", observation_id)
        object.__setattr__(self, "observation_identity_sha256", identity_sha)
        object.__setattr__(self, "bias_ids", bias_ids)
        object.__setattr__(self, "basis_names", basis_names)
        object.__setattr__(
            self,
            "row_bias_indices",
            _immutable_array(row_indices, dtype=np.dtype(np.int64)),
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
        object.__setattr__(self, "orthogonalization_semantics", semantics)
        object.__setattr__(self, "maximum_gauge_projection", maximum_projection)
        object.__setattr__(
            self,
            "gauge_projection_tolerance",
            projection_tolerance,
        )
        object.__setattr__(self, "metadata", metadata)

        expected_id = _canonical_id(self.identity_record())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("Prob4D visual-bias artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def observation_count(self) -> int:
        return int(self.row_bias_indices.size)

    @property
    def basis_dimension(self) -> int:
        return len(self.basis_names)

    @property
    def latent_dimension(self) -> int:
        return len(self.bias_ids) * self.basis_dimension

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return tuple(
            f"{bias_id}:{basis_name}"
            for bias_id in self.bias_ids
            for basis_name in self.basis_names
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "row_bias_indices": np.asarray(self.row_bias_indices),
            "bias_jacobian": np.asarray(self.bias_jacobian),
            "joint_bias_covariance": np.asarray(self.joint_bias_covariance),
        }

    def array_descriptors(self) -> dict[str, dict[str, object]]:
        return {name: _array_descriptor(value) for name, value in self.arrays().items()}

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PROB4D_VISUAL_BIAS_NUISANCE_SCHEMA,
            "schema_version": PROB4D_VISUAL_BIAS_NUISANCE_VERSION,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_identity_sha256": self.observation_identity_sha256,
            "bias_ids": list(self.bias_ids),
            "basis_names": list(self.basis_names),
            "orthogonalization_semantics": self.orthogonalization_semantics,
            "maximum_gauge_projection": self.maximum_gauge_projection,
            "gauge_projection_tolerance": self.gauge_projection_tolerance,
            "arrays": self.array_descriptors(),
            "metadata": plain_json(self.metadata),
        }

    def global_design(self) -> np.ndarray:
        """Return the block-sparse row design with shape ``(N, 3, S*R)``."""

        result: np.ndarray = np.zeros(
            (self.observation_count, 3, self.latent_dimension),
            dtype=np.float64,
        )
        width = self.basis_dimension
        for row, bias_index in enumerate(self.row_bias_indices):
            start = int(bias_index) * width
            result[row, :, start : start + width] = self.bias_jacobian[row]
        return _immutable_array(result, dtype=np.dtype(np.float64))

    def symmetric_covariance_root(self) -> np.ndarray:
        """Return the unique symmetric PSD root of the complete covariance."""

        eigenvalues, eigenvectors = np.linalg.eigh(self.joint_bias_covariance)
        root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
        root = 0.5 * (root + root.T)
        if not np.allclose(
            root @ root.T,
            self.joint_bias_covariance,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("visual-bias covariance root failed reconstruction")
        return _immutable_array(root, dtype=np.dtype(np.float64))

    def reparameterized_design(
        self,
        *,
        shared_bias_prior_std_m: float,
    ) -> np.ndarray:
        """Map the complete covariance into the frozen isotropic prior."""

        scale = _finite_real(
            shared_bias_prior_std_m,
            name="shared_bias_prior_std_m",
            strictly_positive=True,
        )
        design = np.einsum(
            "nck,kr->ncr",
            self.global_design(),
            self.symmetric_covariance_root(),
        )
        return _immutable_array(design / scale, dtype=np.dtype(np.float64))

    def summary(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_identity_sha256": self.observation_identity_sha256,
            "observation_count": self.observation_count,
            "bias_ids": list(self.bias_ids),
            "basis_names": list(self.basis_names),
            "latent_dimension": self.latent_dimension,
            "orthogonalization_semantics": self.orthogonalization_semantics,
            "maximum_gauge_projection": self.maximum_gauge_projection,
            "gauge_projection_tolerance": self.gauge_projection_tolerance,
        }


def validate_prob4d_visual_bias_nuisance(
    observation: ObservationBeliefV1,
    nuisance: object,
    *,
    require_gauge_orthogonalized: bool = True,
) -> Prob4DVisualBiasBindingV1:
    """Copy and independently validate a loaded Prob4D bias sidecar."""

    if not isinstance(observation, ObservationBeliefV1):
        raise TypeError("observation must be an ObservationBeliefV1")
    require_orthogonal = genuine_boolean(
        require_gauge_orthogonalized,
        name="require_gauge_orthogonalized",
    )
    source = cast(Any, nuisance)
    try:
        binding = Prob4DVisualBiasBindingV1(
            observation_artifact_id=source.observation_artifact_id,
            observation_identity_sha256=source.observation_identity_sha256,
            bias_ids=source.bias_ids,
            basis_names=source.basis_names,
            row_bias_indices=source.row_bias_indices,
            bias_jacobian=source.bias_jacobian,
            joint_bias_covariance=source.joint_bias_covariance,
            orthogonalization_semantics=source.orthogonalization_semantics,
            maximum_gauge_projection=source.maximum_gauge_projection,
            gauge_projection_tolerance=source.gauge_projection_tolerance,
            metadata=source.metadata,
            artifact_id=source.artifact_id,
        )
    except AttributeError as error:
        raise TypeError(
            "nuisance is not a Prob4D VisualBiasNuisanceV1 object"
        ) from error

    if binding.observation_artifact_id != observation.artifact_id:
        raise ValueError(
            "visual-bias sidecar identifies a different observation artifact"
        )
    _, observation_count, identity_sha = prob4d_observation_identity_summary(
        observation
    )
    if binding.observation_count != observation_count:
        raise ValueError("visual-bias sidecar row count differs from the observation")
    if binding.observation_identity_sha256 != identity_sha:
        raise ValueError("visual-bias row identity digest differs from the observation")
    if (
        require_orthogonal
        and binding.orthogonalization_semantics != PROB4D_VISUAL_BIAS_ORTHOGONALIZATION
    ):
        raise ValueError(
            "claim-bearing visual-bias use requires the global "
            "gauge-orthogonalized basis"
        )
    return binding


def _maximum_conditional_gauge_projection(
    bias_design: np.ndarray,
    gauge_design: np.ndarray,
    conditional_covariance: np.ndarray,
) -> float:
    """Measure the maximum whitened bias-column projection onto gauge span."""

    bias = np.asarray(bias_design, dtype=np.float64)
    gauge = np.asarray(gauge_design, dtype=np.float64)
    covariance = np.asarray(conditional_covariance, dtype=np.float64)
    if bias.ndim != 3 or bias.shape[1] != 3:
        raise ValueError("bias_design must have shape (N, 3, B)")
    if gauge.ndim != 3 or gauge.shape[:2] != bias.shape[:2]:
        raise ValueError("gauge_design must have shape (N, 3, G)")
    if covariance.shape != (len(bias), 3, 3):
        raise ValueError("conditional covariance differs from bias rows")
    if not np.all(np.isfinite(bias)) or not np.all(np.isfinite(gauge)):
        raise ValueError("bias and gauge designs must be finite")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("conditional covariance must be finite")
    if gauge.shape[2] == 0:
        return 0.0

    white_bias: list[np.ndarray] = []
    white_gauge: list[np.ndarray] = []
    for index, matrix in enumerate(covariance):
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10):
            raise ValueError(f"conditional covariance {index} must be symmetric")
        try:
            factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"conditional covariance {index} must be positive definite"
            ) from error
        white_bias.append(np.linalg.solve(factor, bias[index]))
        white_gauge.append(np.linalg.solve(factor, gauge[index]))
    stacked_bias = np.concatenate(white_bias, axis=0)
    stacked_gauge = np.concatenate(white_gauge, axis=0)
    left, singular_values, _ = np.linalg.svd(stacked_gauge, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return 0.0
    threshold = (
        max(stacked_gauge.shape) * np.finfo(np.float64).eps * float(singular_values[0])
    )
    gauge_basis = left[:, singular_values > threshold]
    if gauge_basis.shape[1] == 0:
        return 0.0

    maximum = 0.0
    for column in range(stacked_bias.shape[1]):
        vector = stacked_bias[:, column]
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float64).eps:
            continue
        projection = float(np.linalg.norm(gauge_basis.T @ vector) / norm)
        maximum = max(maximum, projection)
    return maximum


@dataclass(frozen=True, slots=True)
class ClaimBearingProb4DVisualBiasUpdateV2:
    """V2 claim-bearing update with a bound coherent visual-bias prior."""

    base_update: ClaimBearingProb4DUpdateV1
    visual_bias: Prob4DVisualBiasBindingV1
    shared_bias_prior_std_m: float
    measured_gauge_projection: float

    def __post_init__(self) -> None:
        if not isinstance(self.base_update, ClaimBearingProb4DUpdateV1):
            raise TypeError("base_update must be a ClaimBearingProb4DUpdateV1")
        if not isinstance(self.visual_bias, Prob4DVisualBiasBindingV1):
            raise TypeError("visual_bias must be a Prob4DVisualBiasBindingV1")
        scale = _finite_real(
            self.shared_bias_prior_std_m,
            name="shared_bias_prior_std_m",
            strictly_positive=True,
        )
        measured_projection = _finite_real(
            self.measured_gauge_projection,
            name="measured_gauge_projection",
            minimum=0.0,
        )
        if measured_projection > self.visual_bias.gauge_projection_tolerance:
            raise ValueError("visual-bias basis overlaps the BPT gauge design")
        if (
            self.base_update.observation_artifact_id
            != self.visual_bias.observation_artifact_id
        ):
            raise ValueError(
                "visual bias and base update identify different observations"
            )
        result = self.base_update.result
        if len(result.shared_bias_coefficients) != self.visual_bias.latent_dimension:
            raise ValueError(
                "shared-bias coefficient dimension differs from the sidecar"
            )
        if len(result.view_bias_coefficients) != 0:
            raise ValueError(
                "V2 visual-bias update must disable the legacy view-bias block"
            )

        lineage = result.input_lineage
        expected = {
            "prob4d_visual_bias_artifact_id": self.visual_bias.artifact_id,
            "prob4d_visual_bias_observation_identity_sha256": (
                self.visual_bias.observation_identity_sha256
            ),
            "prob4d_visual_bias_reparameterization": (
                PROB4D_VISUAL_BIAS_REPARAMETERIZATION
            ),
            "prob4d_visual_bias_prior_std_m": scale,
            "prob4d_visual_bias_gauge_orthogonalized": True,
            "prob4d_visual_bias_measured_gauge_projection": measured_projection,
            "prob4d_visual_bias_marginal_covariance_added": False,
        }
        for name, value in expected.items():
            if lineage.get(name) != value:
                raise ValueError(f"result lineage does not bind {name}")
        object.__setattr__(self, "shared_bias_prior_std_m", scale)
        object.__setattr__(self, "measured_gauge_projection", measured_projection)

    @property
    def result(self) -> GaugeAwareBeliefResult:
        return self.base_update.result

    @property
    def inference_admissible(self) -> bool:
        return self.base_update.inference_admissible

    @property
    def provider_bias_coefficients(self) -> np.ndarray:
        """Map posterior shared-bias coordinates back to Prob4D coefficients."""

        coefficients = (
            self.visual_bias.symmetric_covariance_root()
            @ self.result.shared_bias_coefficients
            / self.shared_bias_prior_std_m
        )
        return _immutable_array(coefficients, dtype=np.dtype(np.float64))

    @property
    def provider_bias_covariance(self) -> np.ndarray:
        """Return the posterior covariance in Prob4D coefficient space."""

        state_count = len(self.result.state_coefficients)
        gauge_count = len(self.result.gauge_delta)
        start = state_count + gauge_count
        stop = start + self.visual_bias.latent_dimension
        transformed = self.result.posterior_covariance[start:stop, start:stop]
        root = self.visual_bias.symmetric_covariance_root()
        covariance = root @ transformed @ root.T / self.shared_bias_prior_std_m**2
        covariance = 0.5 * (covariance + covariance.T)
        return _immutable_array(covariance, dtype=np.dtype(np.float64))

    @property
    def update_id(self) -> str:
        return _canonical_id(
            {
                "schema": ("bayesian_phystwin.claim_bearing_prob4d_visual_bias_update"),
                "schema_version": (CLAIM_BEARING_PROB4D_VISUAL_BIAS_UPDATE_VERSION),
                "base_update_id": self.base_update.update_id,
                "visual_bias_artifact_id": self.visual_bias.artifact_id,
                "visual_bias_observation_identity_sha256": (
                    self.visual_bias.observation_identity_sha256
                ),
                "visual_bias_coefficient_names": list(
                    self.visual_bias.coefficient_names
                ),
                "shared_bias_prior_std_m": self.shared_bias_prior_std_m,
                "measured_gauge_projection": self.measured_gauge_projection,
                "reparameterization": PROB4D_VISUAL_BIAS_REPARAMETERIZATION,
                "inference_admissible": self.inference_admissible,
                "reason": self.result.reason,
            }
        )


def update_claim_bearing_prob4d_with_visual_bias_from_artifacts(
    observation_belief: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    *,
    visual_bias_nuisance: object,
    physical_prediction_xyz_m: np.ndarray,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    config: PriorAwareGaugeConfigV1 | None = None,
    **anchor_dependence: Any,
) -> ClaimBearingProb4DVisualBiasUpdateV2:
    """Validate, bind, reparameterize, and solve one coherent-bias update.

    The sidecar must first be loaded through Prob4D's strict loader. This
    function independently reconstructs its identity, verifies the exact BPT
    observation and row order, requires the producer's gauge orthogonalization,
    rechecks it against BPT's admitted gauge design, and preserves the complete
    joint covariance.
    """

    visual_bias = validate_prob4d_visual_bias_nuisance(
        observation_belief,
        visual_bias_nuisance,
        require_gauge_orthogonalized=True,
    )
    if config is not None and not isinstance(config, PriorAwareGaugeConfigV1):
        raise TypeError("config must be a PriorAwareGaugeConfigV1")
    cfg = config or PriorAwareGaugeConfigV1()
    shared_design = visual_bias.reparameterized_design(
        shared_bias_prior_std_m=cfg.shared_bias_prior_std_m,
    )
    empty_view_design = np.zeros(
        (observation_belief.observation_count, 3, 0),
        dtype=np.float64,
    )
    adapted = build_claim_bearing_gauge_aware_batch_from_artifacts(
        observation_belief,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_design,
        view_bias_jacobian=empty_view_design,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        **anchor_dependence,
    )
    measured_projection = _maximum_conditional_gauge_projection(
        visual_bias.global_design(),
        adapted.batch.gauge_jacobian,
        adapted.batch.observation_covariance_m2,
    )
    if measured_projection > visual_bias.gauge_projection_tolerance:
        raise ValueError(
            "visual-bias basis exceeds its gauge projection tolerance against "
            "the admitted BayesianPhysTwin gauge design"
        )

    metadata = dict(adapted.batch.metadata or {})
    metadata.update(
        {
            "prob4d_visual_bias_artifact_id": visual_bias.artifact_id,
            "prob4d_visual_bias_observation_identity_sha256": (
                visual_bias.observation_identity_sha256
            ),
            "prob4d_visual_bias_reparameterization": (
                PROB4D_VISUAL_BIAS_REPARAMETERIZATION
            ),
            "prob4d_visual_bias_prior_std_m": cfg.shared_bias_prior_std_m,
            "prob4d_visual_bias_gauge_orthogonalized": True,
            "prob4d_visual_bias_measured_gauge_projection": measured_projection,
            "prob4d_visual_bias_marginal_covariance_added": False,
            "prob4d_visual_bias_latent_dimension": visual_bias.latent_dimension,
            "prob4d_visual_bias_bias_ids": list(visual_bias.bias_ids),
            "prob4d_visual_bias_basis_names": list(visual_bias.basis_names),
        }
    )
    provider_manifest_id = _sha256(
        metadata.get("prob4d_claim_bearing_provider_manifest_id"),
        name="provider_manifest_id",
    )
    calibration_ids = _validated_calibration_ids(
        metadata.get("prob4d_claim_bearing_calibration_artifact_ids")
    )
    runtime_source = _runtime_revision_source(
        metadata.get("prob4d_claim_bearing_runtime_revision_source")
    )
    if (
        metadata.get("prob4d_claim_bearing_runtime_revision_independently_verified")
        is not True
    ):
        raise ValueError(
            "claim-bearing Prob4D runtime revision was not independently verified"
        )

    batch = replace(adapted.batch, metadata=metadata)
    result = update_prior_aware_gauge_belief(batch, config=cfg)
    base_update = ClaimBearingProb4DUpdateV1(
        result=result,
        observation_artifact_id=adapted.observation_artifact_id,
        linearization_artifact_id=linearization.artifact_id,
        provider_manifest_id=provider_manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source=runtime_source,
        runtime_revision_independently_verified=True,
    )
    return ClaimBearingProb4DVisualBiasUpdateV2(
        base_update=base_update,
        visual_bias=visual_bias,
        shared_bias_prior_std_m=cfg.shared_bias_prior_std_m,
        measured_gauge_projection=measured_projection,
    )


__all__ = [
    "CLAIM_BEARING_PROB4D_VISUAL_BIAS_UPDATE_VERSION",
    "PROB4D_VISUAL_BIAS_NUISANCE_SCHEMA",
    "PROB4D_VISUAL_BIAS_NUISANCE_VERSION",
    "PROB4D_VISUAL_BIAS_ORTHOGONALIZATION",
    "PROB4D_VISUAL_BIAS_REPARAMETERIZATION",
    "ClaimBearingProb4DVisualBiasUpdateV2",
    "Prob4DVisualBiasBindingV1",
    "update_claim_bearing_prob4d_with_visual_bias_from_artifacts",
    "validate_prob4d_visual_bias_nuisance",
]
