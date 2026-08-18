"""Same-mean heavy-tailed predictive beliefs for physical queries.

The module preserves a caller-owned point prediction exactly and attaches a
source-frozen two-component Gaussian scale mixture.  It is intended for future
proper-score and coverage--sharpness experiments where deterministic point
prediction must remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Final

import numpy as np

SAME_MEAN_GAUSSIAN_MIXTURE_SCHEMA: Final = (
    "bayesian-phystwin-same-mean-gaussian-mixture-record-v1"
)
SAME_MEAN_GAUSSIAN_MIXTURE_VERSION: Final = 1
SAME_MEAN_GAUSSIAN_MIXTURE_CANDIDATE_SCHEMA: Final = (
    "bayesian-phystwin-same-mean-gaussian-mixture-candidate-v1"
)
SAME_MEAN_GAUSSIAN_MIXTURE_SELECTION_SCHEMA: Final = (
    "bayesian-phystwin-same-mean-gaussian-mixture-selection-v1"
)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
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
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _open_probability(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


def _shape(value: Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty integer shape")
    source = tuple(value)
    if not source:
        raise ValueError(f"{name} must be a nonempty integer shape")
    result: list[int] = []
    for index, item in enumerate(source):
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"{name}[{index}] must be a positive integer")
        dimension = int(item)
        if dimension < 1:
            raise ValueError(f"{name}[{index}] must be a positive integer")
        result.append(dimension)
    return tuple(result)


def _plain_json(value: object, *, name: str = "value") -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar_result = float(value)
        if not math.isfinite(scalar_result):
            raise ValueError(f"{name} must contain only finite JSON values")
        return scalar_result
    if isinstance(value, Mapping):
        mapping_result: dict[str, Any] = {}
        for key, item in value.items():
            canonical_key = _canonical_string(key, name=f"{name} key")
            if canonical_key in mapping_result:
                raise ValueError(f"{name} contains a duplicate key")
            mapping_result[canonical_key] = _plain_json(
                item,
                name=f"{name}.{canonical_key}",
            )
        return mapping_result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _plain_json(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{name} must contain only finite JSON values")


def _content_id(values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _plain_json(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _immutable_float(value: object) -> np.ndarray:
    canonical = np.array(value, dtype=np.dtype("<f8"), copy=True, order="C")
    return np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=np.dtype("<f8"),
    ).reshape(canonical.shape)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _reference_mean(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("reference_mean_m must be a NumPy array to preserve identity")
    if value.dtype != np.dtype(np.float64):
        raise ValueError("reference_mean_m must have dtype float64")
    if value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError("reference_mean_m must have shape (..., dimension)")
    if not value.flags.c_contiguous:
        raise ValueError("reference_mean_m must be C-contiguous")
    if not np.all(np.isfinite(value)):
        raise ValueError("reference_mean_m must be finite")
    return value


def _real_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _covariance_stack(
    value: object,
    *,
    mean_shape: tuple[int, ...],
    name: str,
    tolerance: float,
) -> np.ndarray:
    covariance = _real_array(value, name=name)
    expected = mean_shape + (mean_shape[-1],)
    if covariance.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    transposed = np.swapaxes(covariance, -1, -2)
    if not np.allclose(covariance, transposed, atol=tolerance, rtol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (covariance + transposed)
    flat = symmetric.reshape(-1, mean_shape[-1], mean_shape[-1])
    for index, matrix in enumerate(flat):
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"{name}[{index}] must be positive definite for density scoring"
            ) from error
    return np.array(symmetric, dtype=np.float64, copy=True, order="C")


def _probability_schedule(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("nominal_probability must contain real numeric values")
    try:
        broadcast = np.broadcast_to(np.asarray(raw, dtype=np.float64), shape)
    except ValueError as error:
        raise ValueError(f"nominal_probability must broadcast to {shape}") from error
    if (
        not np.all(np.isfinite(broadcast))
        or np.any(broadcast <= 0.0)
        or np.any(broadcast >= 1.0)
    ):
        raise ValueError("nominal_probability must lie strictly inside (0, 1)")
    return np.array(broadcast, dtype=np.float64, copy=True, order="C")


def _tail_dominates_nominal(
    nominal: np.ndarray,
    tail: np.ndarray,
    *,
    tolerance: float,
) -> None:
    excess = 0.5 * ((tail - nominal) + np.swapaxes(tail - nominal, -1, -2))
    minimum = float(np.min(np.linalg.eigvalsh(excess), initial=0.0))
    if minimum < -tolerance:
        raise ValueError(
            "tail_covariance_m2 must dominate nominal_covariance_m2 in PSD order"
        )


def _residual_array(value: object, *, mean_shape: tuple[int, ...]) -> np.ndarray:
    residual = _real_array(value, name="residual_m")
    if residual.shape != mean_shape:
        raise ValueError(f"residual_m must have shape {mean_shape}")
    return residual


def _gaussian_log_density(
    residual: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    dimension = residual.shape[-1]
    flat_residual = residual.reshape(-1, dimension)
    flat_covariance = covariance.reshape(-1, dimension, dimension)
    result: np.ndarray = np.empty(len(flat_residual), dtype=np.float64)
    constant = dimension * math.log(2.0 * math.pi)
    for index, (vector, matrix) in enumerate(
        zip(flat_residual, flat_covariance, strict=True)
    ):
        cholesky = np.linalg.cholesky(matrix)
        whitened = np.linalg.solve(cholesky, vector)
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
        result[index] = -0.5 * (constant + log_determinant + float(whitened @ whitened))
    return result.reshape(residual.shape[:-1])


@dataclass(frozen=True, slots=True)
class SameMeanGaussianMixtureCandidateV1:
    """Source-frozen scalar parameterization of one mixture candidate."""

    nominal_probability: float = 0.90
    tail_covariance_scale: float = 4.0
    tail_isotropic_variance_m2: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        probability = _open_probability(
            self.nominal_probability,
            name="nominal_probability",
        )
        scale = _finite_real(
            self.tail_covariance_scale,
            name="tail_covariance_scale",
            minimum=1.0,
        )
        nugget = _finite_real(
            self.tail_isotropic_variance_m2,
            name="tail_isotropic_variance_m2",
            minimum=0.0,
        )
        metadata = _plain_json(self.metadata, name="metadata")
        object.__setattr__(self, "nominal_probability", probability)
        object.__setattr__(self, "tail_covariance_scale", scale)
        object.__setattr__(self, "tail_isotropic_variance_m2", nugget)
        object.__setattr__(self, "metadata", metadata)
        expected = _content_id(self.descriptor())
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", expected)
        elif _sha256(self.candidate_id, name="candidate_id") != expected:
            raise ValueError("candidate_id does not match the mixture candidate")

    @property
    def is_gaussian_reference(self) -> bool:
        return (
            self.tail_covariance_scale == 1.0 and self.tail_isotropic_variance_m2 == 0.0
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": SAME_MEAN_GAUSSIAN_MIXTURE_CANDIDATE_SCHEMA,
            "schema_version": SAME_MEAN_GAUSSIAN_MIXTURE_VERSION,
            "nominal_probability": self.nominal_probability,
            "tail_covariance_scale": self.tail_covariance_scale,
            "tail_isotropic_variance_m2": self.tail_isotropic_variance_m2,
            "metadata": _plain_json(self.metadata, name="metadata"),
        }


@dataclass(frozen=True, slots=True)
class SameMeanGaussianMixtureRecordV1:
    """Content-addressed proof that a mixture changed no point prediction."""

    reference_predictor_id: str
    nominal_covariance_id: str
    tail_covariance_id: str
    mean_shape: Sequence[int]
    covariance_shape: Sequence[int]
    reference_mean_sha256: str
    nominal_covariance_sha256: str
    tail_covariance_sha256: str
    nominal_probability_sha256: str
    minimum_nominal_probability: float
    maximum_nominal_probability: float
    density_floor_variance_m2: float
    tail_dominates_nominal: bool
    mean_object_identity_preserved: bool
    point_prediction_changed: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        mean_shape = _shape(self.mean_shape, name="mean_shape")
        covariance_shape = _shape(self.covariance_shape, name="covariance_shape")
        if covariance_shape != mean_shape + (mean_shape[-1],):
            raise ValueError("covariance_shape is incompatible with mean_shape")
        if self.tail_dominates_nominal is not True:
            raise ValueError("tail_dominates_nominal must remain true")
        if self.mean_object_identity_preserved is not True:
            raise ValueError("mean_object_identity_preserved must remain true")
        if self.point_prediction_changed is not False:
            raise ValueError("point_prediction_changed must remain false")
        minimum = _open_probability(
            self.minimum_nominal_probability,
            name="minimum_nominal_probability",
        )
        maximum = _open_probability(
            self.maximum_nominal_probability,
            name="maximum_nominal_probability",
        )
        if maximum < minimum:
            raise ValueError(
                "maximum_nominal_probability must not be smaller than minimum"
            )
        floor = _finite_real(
            self.density_floor_variance_m2,
            name="density_floor_variance_m2",
            minimum=0.0,
        )
        object.__setattr__(
            self,
            "reference_predictor_id",
            _canonical_string(
                self.reference_predictor_id,
                name="reference_predictor_id",
            ),
        )
        object.__setattr__(
            self,
            "nominal_covariance_id",
            _canonical_string(
                self.nominal_covariance_id,
                name="nominal_covariance_id",
            ),
        )
        object.__setattr__(
            self,
            "tail_covariance_id",
            _canonical_string(
                self.tail_covariance_id,
                name="tail_covariance_id",
            ),
        )
        object.__setattr__(self, "mean_shape", mean_shape)
        object.__setattr__(self, "covariance_shape", covariance_shape)
        for field_name in (
            "reference_mean_sha256",
            "nominal_covariance_sha256",
            "tail_covariance_sha256",
            "nominal_probability_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(self, "minimum_nominal_probability", minimum)
        object.__setattr__(self, "maximum_nominal_probability", maximum)
        object.__setattr__(self, "density_floor_variance_m2", floor)
        object.__setattr__(
            self,
            "metadata",
            _plain_json(self.metadata, name="metadata"),
        )
        expected = _content_id(self.descriptor())
        if self.artifact_id is None:
            object.__setattr__(self, "artifact_id", expected)
        elif _sha256(self.artifact_id, name="artifact_id") != expected:
            raise ValueError("artifact_id does not match the mixture record")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": SAME_MEAN_GAUSSIAN_MIXTURE_SCHEMA,
            "schema_version": SAME_MEAN_GAUSSIAN_MIXTURE_VERSION,
            "reference_predictor_id": self.reference_predictor_id,
            "nominal_covariance_id": self.nominal_covariance_id,
            "tail_covariance_id": self.tail_covariance_id,
            "mean_shape": list(self.mean_shape),
            "covariance_shape": list(self.covariance_shape),
            "reference_mean_sha256": self.reference_mean_sha256,
            "nominal_covariance_sha256": self.nominal_covariance_sha256,
            "tail_covariance_sha256": self.tail_covariance_sha256,
            "nominal_probability_sha256": self.nominal_probability_sha256,
            "minimum_nominal_probability": self.minimum_nominal_probability,
            "maximum_nominal_probability": self.maximum_nominal_probability,
            "density_floor_variance_m2": self.density_floor_variance_m2,
            "tail_dominates_nominal": self.tail_dominates_nominal,
            "mean_object_identity_preserved": self.mean_object_identity_preserved,
            "point_prediction_changed": self.point_prediction_changed,
            "metadata": _plain_json(self.metadata, name="metadata"),
        }


@dataclass(frozen=True, slots=True)
class SameMeanGaussianMixturePredictionV1:
    """Two-component query density whose mean is the exact reference object."""

    mean_m: np.ndarray
    nominal_covariance_m2: np.ndarray
    tail_covariance_m2: np.ndarray
    nominal_probability: np.ndarray
    record: SameMeanGaussianMixtureRecordV1


def compose_same_mean_gaussian_mixture(
    reference_mean_m: np.ndarray,
    nominal_covariance_m2: object,
    tail_covariance_m2: object,
    *,
    reference_predictor_id: str,
    nominal_covariance_id: str,
    tail_covariance_id: str,
    nominal_probability: object = 0.90,
    density_floor_variance_m2: float = 0.0,
    covariance_tolerance: float = 1e-10,
    metadata: Mapping[str, Any] | None = None,
) -> SameMeanGaussianMixturePredictionV1:
    """Attach a broad-tail density while retaining the exact point mean.

    ``density_floor_variance_m2`` is explicit and applied equally to both
    components.  No hidden jitter, clipping, or pseudoinverse is used.
    """

    mean = _reference_mean(reference_mean_m)
    tolerance = _finite_real(
        covariance_tolerance,
        name="covariance_tolerance",
        strictly_positive=True,
    )
    floor = _finite_real(
        density_floor_variance_m2,
        name="density_floor_variance_m2",
        minimum=0.0,
    )
    dimension = mean.shape[-1]
    eye = np.eye(dimension, dtype=np.float64)
    nominal_raw = _real_array(
        nominal_covariance_m2,
        name="nominal_covariance_m2",
    )
    tail_raw = _real_array(tail_covariance_m2, name="tail_covariance_m2")
    with np.errstate(over="ignore", invalid="ignore"):
        nominal_floored = nominal_raw + floor * eye
        tail_floored = tail_raw + floor * eye
    if not np.all(np.isfinite(nominal_floored)) or not np.all(
        np.isfinite(tail_floored)
    ):
        raise ValueError("density covariance must remain finite after the floor")
    nominal = _covariance_stack(
        nominal_floored,
        mean_shape=mean.shape,
        name="nominal_covariance_m2",
        tolerance=tolerance,
    )
    tail = _covariance_stack(
        tail_floored,
        mean_shape=mean.shape,
        name="tail_covariance_m2",
        tolerance=tolerance,
    )
    _tail_dominates_nominal(nominal, tail, tolerance=tolerance)
    probability = _probability_schedule(
        nominal_probability,
        shape=mean.shape[:-1],
    )
    nominal_output = _immutable_float(nominal)
    tail_output = _immutable_float(tail)
    probability_output = _immutable_float(probability)
    record = SameMeanGaussianMixtureRecordV1(
        reference_predictor_id=reference_predictor_id,
        nominal_covariance_id=nominal_covariance_id,
        tail_covariance_id=tail_covariance_id,
        mean_shape=mean.shape,
        covariance_shape=nominal_output.shape,
        reference_mean_sha256=_array_sha256(mean),
        nominal_covariance_sha256=_array_sha256(nominal_output),
        tail_covariance_sha256=_array_sha256(tail_output),
        nominal_probability_sha256=_array_sha256(probability_output),
        minimum_nominal_probability=float(np.min(probability_output)),
        maximum_nominal_probability=float(np.max(probability_output)),
        density_floor_variance_m2=floor,
        tail_dominates_nominal=True,
        mean_object_identity_preserved=True,
        point_prediction_changed=False,
        metadata={} if metadata is None else metadata,
    )
    result = SameMeanGaussianMixturePredictionV1(
        mean_m=mean,
        nominal_covariance_m2=nominal_output,
        tail_covariance_m2=tail_output,
        nominal_probability=probability_output,
        record=record,
    )
    if result.mean_m is not reference_mean_m:
        raise AssertionError("mixture composition copied the reference mean")
    return result


def compose_candidate_same_mean_gaussian_mixture(
    reference_mean_m: np.ndarray,
    nominal_covariance_m2: object,
    candidate: SameMeanGaussianMixtureCandidateV1,
    *,
    reference_predictor_id: str,
    nominal_covariance_id: str,
    density_floor_variance_m2: float = 0.0,
    covariance_tolerance: float = 1e-10,
    metadata: Mapping[str, Any] | None = None,
) -> SameMeanGaussianMixturePredictionV1:
    """Construct the tail covariance from one frozen scalar candidate."""

    if not isinstance(candidate, SameMeanGaussianMixtureCandidateV1):
        raise TypeError("candidate must be a SameMeanGaussianMixtureCandidateV1")
    mean = _reference_mean(reference_mean_m)
    nominal = _real_array(
        nominal_covariance_m2,
        name="nominal_covariance_m2",
    )
    expected = mean.shape + (mean.shape[-1],)
    if nominal.shape != expected:
        raise ValueError(f"nominal_covariance_m2 must have shape {expected}")
    eye = np.eye(mean.shape[-1], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        tail = (
            candidate.tail_covariance_scale * nominal
            + candidate.tail_isotropic_variance_m2 * eye
        )
    if not np.all(np.isfinite(tail)):
        raise ValueError("candidate tail covariance must be finite")
    combined_metadata = {
        "candidate": candidate.descriptor(),
        "candidate_id": candidate.candidate_id,
    }
    if metadata is not None:
        combined_metadata["caller"] = _plain_json(metadata, name="metadata")
    return compose_same_mean_gaussian_mixture(
        mean,
        nominal,
        tail,
        reference_predictor_id=reference_predictor_id,
        nominal_covariance_id=nominal_covariance_id,
        tail_covariance_id=f"same-mean-mixture-tail:{candidate.candidate_id}",
        nominal_probability=candidate.nominal_probability,
        density_floor_variance_m2=density_floor_variance_m2,
        covariance_tolerance=covariance_tolerance,
        metadata=combined_metadata,
    )


def gaussian_mixture_negative_log_density(
    residual_m: object,
    prediction: SameMeanGaussianMixturePredictionV1,
) -> np.ndarray:
    """Return endpoint-wise negative log density for residuals from the mean."""

    if not isinstance(prediction, SameMeanGaussianMixturePredictionV1):
        raise TypeError("prediction must be a SameMeanGaussianMixturePredictionV1")
    residual = _residual_array(residual_m, mean_shape=prediction.mean_m.shape)
    nominal_log = _gaussian_log_density(
        residual,
        prediction.nominal_covariance_m2,
    )
    tail_log = _gaussian_log_density(residual, prediction.tail_covariance_m2)
    probability = np.asarray(prediction.nominal_probability, dtype=np.float64)
    log_density = np.logaddexp(
        np.log(probability) + nominal_log,
        np.log1p(-probability) + tail_log,
    )
    if not np.all(np.isfinite(log_density)):
        raise ValueError("mixture log density must be finite")
    return -log_density


def group_gaussian_mixture_negative_log_score(
    residual_m: object,
    prediction: SameMeanGaussianMixturePredictionV1,
) -> float:
    """Return one mean log score for a complete physical group."""

    scores = gaussian_mixture_negative_log_density(residual_m, prediction)
    if scores.size == 0:
        raise ValueError("a physical group must contain at least one endpoint")
    return float(np.mean(scores))


def gaussian_mixture_moment_covariance(
    prediction: SameMeanGaussianMixturePredictionV1,
) -> np.ndarray:
    """Return the exact same-mean mixture covariance without collapsing density."""

    if not isinstance(prediction, SameMeanGaussianMixturePredictionV1):
        raise TypeError("prediction must be a SameMeanGaussianMixturePredictionV1")
    probability = prediction.nominal_probability[..., None, None]
    moment = (
        probability * prediction.nominal_covariance_m2
        + (1.0 - probability) * prediction.tail_covariance_m2
    )
    if not np.all(np.isfinite(moment)):
        raise ValueError("mixture moment covariance must be finite")
    return _immutable_float(moment)


def gaussian_mixture_rms_marginal_standard_deviation(
    prediction: SameMeanGaussianMixturePredictionV1,
) -> float:
    """Return a scalar sharpness proxy for source-only candidate selection."""

    covariance = gaussian_mixture_moment_covariance(prediction)
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    if np.any(diagonal <= 0.0):
        raise ValueError("mixture marginal variances must be positive")
    return float(np.sqrt(np.mean(diagonal)))


def _draw_bank(
    normal_draws: object,
    component_uniforms: object,
    *,
    dimension: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    normals = _real_array(normal_draws, name=f"{name}_normal_draws")
    uniforms = _real_array(component_uniforms, name=f"{name}_component_uniforms")
    if normals.ndim != 2 or normals.shape[1] != dimension or len(normals) < 2:
        raise ValueError(
            f"{name}_normal_draws must have shape (K, {dimension}) with K >= 2"
        )
    if uniforms.shape != (len(normals),):
        raise ValueError(
            f"{name}_component_uniforms must contain one value per normal draw"
        )
    if np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
        raise ValueError(f"{name}_component_uniforms must lie in [0, 1)")
    return normals, uniforms


def group_gaussian_mixture_energy_score(
    residual_m: object,
    prediction: SameMeanGaussianMixturePredictionV1,
    *,
    normal_draws_a: object,
    component_uniforms_a: object,
    normal_draws_b: object,
    component_uniforms_b: object,
) -> float:
    """Return a deterministic paired-Monte-Carlo energy score.

    Draw banks are caller supplied so candidate comparisons can bind and reuse
    exactly the same random numbers without hidden runtime randomness.
    """

    residual = _residual_array(residual_m, mean_shape=prediction.mean_m.shape)
    dimension = residual.shape[-1]
    normals_a, uniforms_a = _draw_bank(
        normal_draws_a,
        component_uniforms_a,
        dimension=dimension,
        name="a",
    )
    normals_b, uniforms_b = _draw_bank(
        normal_draws_b,
        component_uniforms_b,
        dimension=dimension,
        name="b",
    )
    if len(normals_a) != len(normals_b):
        raise ValueError("energy-score draw banks must have the same size")
    flat_residual = residual.reshape(-1, dimension)
    flat_nominal = prediction.nominal_covariance_m2.reshape(
        -1,
        dimension,
        dimension,
    )
    flat_tail = prediction.tail_covariance_m2.reshape(
        -1,
        dimension,
        dimension,
    )
    flat_probability = prediction.nominal_probability.reshape(-1)
    endpoint_scores: np.ndarray = np.empty(len(flat_residual), dtype=np.float64)
    for index, (observed, nominal, tail, probability) in enumerate(
        zip(
            flat_residual,
            flat_nominal,
            flat_tail,
            flat_probability,
            strict=True,
        )
    ):
        nominal_cholesky = np.linalg.cholesky(nominal)
        tail_cholesky = np.linalg.cholesky(tail)
        samples_a = np.empty_like(normals_a)
        samples_b = np.empty_like(normals_b)
        for draw_index in range(len(normals_a)):
            factor_a = (
                nominal_cholesky
                if uniforms_a[draw_index] < probability
                else tail_cholesky
            )
            factor_b = (
                nominal_cholesky
                if uniforms_b[draw_index] < probability
                else tail_cholesky
            )
            samples_a[draw_index] = factor_a @ normals_a[draw_index]
            samples_b[draw_index] = factor_b @ normals_b[draw_index]
        endpoint_scores[index] = float(
            np.mean(np.linalg.norm(samples_a - observed, axis=1))
            - 0.5 * np.mean(np.linalg.norm(samples_a - samples_b, axis=1))
        )
    if not np.all(np.isfinite(endpoint_scores)):
        raise ValueError("energy score must be finite")
    return float(np.mean(endpoint_scores))


def _canonical_group_ids(value: object, *, count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("development_group_ids must be a sequence of strings")
    result = tuple(
        _canonical_string(item, name=f"development_group_ids[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if len(result) != count:
        raise ValueError("development_group_ids length must match group arrays")
    if len(set(result)) != len(result):
        raise ValueError("development_group_ids must be unique")
    return result


def _candidate_sequence(
    value: object,
) -> tuple[SameMeanGaussianMixtureCandidateV1, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("candidates must be a sequence")
    result = tuple(value)
    if not result or any(
        not isinstance(candidate, SameMeanGaussianMixtureCandidateV1)
        for candidate in result
    ):
        raise ValueError(
            "candidates must contain SameMeanGaussianMixtureCandidateV1 values"
        )
    identifiers = [candidate.candidate_id for candidate in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate IDs must be unique")
    return result


def _selection_winner(
    candidate_ids: Sequence[str],
    scores: np.ndarray,
    widths: np.ndarray,
    *,
    reference_index: int,
    maximum_worst_group_regret: float,
    maximum_width_ratio: float,
) -> str:
    reference_scores = scores[reference_index]
    reference_widths = widths[reference_index]
    eligible: list[int] = [reference_index]
    for index in range(len(candidate_ids)):
        if index == reference_index:
            continue
        worst_regret = float(np.max(scores[index] - reference_scores))
        worst_width_ratio = float(np.max(widths[index] / reference_widths))
        if (
            worst_regret <= maximum_worst_group_regret
            and worst_width_ratio <= maximum_width_ratio
        ):
            eligible.append(index)
    selected = min(
        eligible,
        key=lambda index: (
            float(np.mean(scores[index])),
            float(np.max(scores[index])),
            float(np.median(scores[index])),
            float(np.mean(widths[index])),
            candidate_ids[index],
        ),
    )
    return candidate_ids[selected]


@dataclass(frozen=True, slots=True)
class SameMeanGaussianMixtureSelectionV1:
    """Group-balanced source selection with exact Gaussian fallback."""

    predictor_id: str
    query_set_id: str
    grouping_rule_id: str
    development_evidence_id: str
    development_group_ids: Sequence[str]
    candidates: Sequence[SameMeanGaussianMixtureCandidateV1]
    group_negative_log_scores: np.ndarray
    group_rms_marginal_standard_deviations: np.ndarray
    reference_candidate_id: str
    selected_candidate_id: str | None
    maximum_worst_group_regret: float
    maximum_width_ratio: float
    density_floor_variance_m2: float
    grid_frozen_before_development_scores: bool
    target_outcomes_used: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "predictor_id",
            "query_set_id",
            "grouping_rule_id",
            "development_evidence_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        candidates = _candidate_sequence(self.candidates)
        group_count = len(tuple(self.development_group_ids))
        group_ids = _canonical_group_ids(
            self.development_group_ids,
            count=group_count,
        )
        score_raw = np.asarray(self.group_negative_log_scores)
        width_raw = np.asarray(self.group_rms_marginal_standard_deviations)
        expected_shape = (len(candidates), len(group_ids))
        if score_raw.dtype.kind not in "iuf" or score_raw.shape != expected_shape:
            raise ValueError(
                "group_negative_log_scores must have shape "
                f"{expected_shape} and contain real values"
            )
        if width_raw.dtype.kind not in "iuf" or width_raw.shape != expected_shape:
            raise ValueError(
                "group_rms_marginal_standard_deviations must have shape "
                f"{expected_shape} and contain real values"
            )
        scores = np.array(score_raw, dtype=np.float64, copy=True, order="C")
        widths = np.array(width_raw, dtype=np.float64, copy=True, order="C")
        if not np.all(np.isfinite(scores)):
            raise ValueError("group negative log scores must be finite")
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
            raise ValueError("group marginal standard deviations must be positive")
        reference_id = _sha256(
            self.reference_candidate_id,
            name="reference_candidate_id",
        )
        candidate_ids = [str(candidate.candidate_id) for candidate in candidates]
        if reference_id not in candidate_ids:
            raise ValueError("reference_candidate_id must name one candidate")
        regret = _finite_real(
            self.maximum_worst_group_regret,
            name="maximum_worst_group_regret",
            minimum=0.0,
        )
        width_ratio = _finite_real(
            self.maximum_width_ratio,
            name="maximum_width_ratio",
            minimum=1.0,
        )
        floor = _finite_real(
            self.density_floor_variance_m2,
            name="density_floor_variance_m2",
            minimum=0.0,
        )
        if self.grid_frozen_before_development_scores is not True:
            raise ValueError(
                "the candidate grid must be frozen before development scores"
            )
        if self.target_outcomes_used is not False:
            raise ValueError("target outcomes cannot be used for source selection")
        candidate_order = np.argsort(np.asarray(candidate_ids, dtype=object))
        group_order = np.argsort(np.asarray(group_ids, dtype=object))
        candidates = tuple(candidates[int(index)] for index in candidate_order)
        candidate_ids = [str(candidate.candidate_id) for candidate in candidates]
        group_ids = tuple(group_ids[int(index)] for index in group_order)
        scores = scores[candidate_order][:, group_order]
        widths = widths[candidate_order][:, group_order]
        reference_index = candidate_ids.index(reference_id)
        expected_selected = _selection_winner(
            candidate_ids,
            scores,
            widths,
            reference_index=reference_index,
            maximum_worst_group_regret=regret,
            maximum_width_ratio=width_ratio,
        )
        selected = (
            expected_selected
            if self.selected_candidate_id is None
            else _sha256(
                self.selected_candidate_id,
                name="selected_candidate_id",
            )
        )
        if selected != expected_selected:
            raise ValueError("selected_candidate_id does not match frozen selection")
        object.__setattr__(self, "development_group_ids", group_ids)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "group_negative_log_scores",
            _immutable_float(scores),
        )
        object.__setattr__(
            self,
            "group_rms_marginal_standard_deviations",
            _immutable_float(widths),
        )
        object.__setattr__(self, "reference_candidate_id", reference_id)
        object.__setattr__(self, "selected_candidate_id", selected)
        object.__setattr__(self, "maximum_worst_group_regret", regret)
        object.__setattr__(self, "maximum_width_ratio", width_ratio)
        object.__setattr__(self, "density_floor_variance_m2", floor)
        object.__setattr__(
            self,
            "metadata",
            _plain_json(self.metadata, name="metadata"),
        )
        expected_id = _content_id(self.descriptor())
        if self.artifact_id is None:
            object.__setattr__(self, "artifact_id", expected_id)
        elif _sha256(self.artifact_id, name="artifact_id") != expected_id:
            raise ValueError("artifact_id does not match the mixture selection")

    @property
    def selected_candidate(self) -> SameMeanGaussianMixtureCandidateV1:
        for candidate in self.candidates:
            if candidate.candidate_id == self.selected_candidate_id:
                return candidate
        raise AssertionError("selected candidate disappeared")

    @property
    def selected_reference(self) -> bool:
        return self.selected_candidate_id == self.reference_candidate_id

    def descriptor(self) -> dict[str, Any]:
        candidate_ids = [str(candidate.candidate_id) for candidate in self.candidates]
        return {
            "schema": SAME_MEAN_GAUSSIAN_MIXTURE_SELECTION_SCHEMA,
            "schema_version": SAME_MEAN_GAUSSIAN_MIXTURE_VERSION,
            "predictor_id": self.predictor_id,
            "query_set_id": self.query_set_id,
            "grouping_rule_id": self.grouping_rule_id,
            "development_evidence_id": self.development_evidence_id,
            "development_group_ids": list(self.development_group_ids),
            "candidates": [candidate.descriptor() for candidate in self.candidates],
            "candidate_ids": candidate_ids,
            "group_negative_log_scores": self.group_negative_log_scores.tolist(),
            "group_rms_marginal_standard_deviations": (
                self.group_rms_marginal_standard_deviations.tolist()
            ),
            "reference_candidate_id": self.reference_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "maximum_worst_group_regret": self.maximum_worst_group_regret,
            "maximum_width_ratio": self.maximum_width_ratio,
            "density_floor_variance_m2": self.density_floor_variance_m2,
            "grid_frozen_before_development_scores": (
                self.grid_frozen_before_development_scores
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": _plain_json(self.metadata, name="metadata"),
        }


def select_same_mean_gaussian_mixture(
    *,
    development_group_ids: Sequence[str],
    residual_groups: Sequence[np.ndarray],
    nominal_covariance_groups: Sequence[np.ndarray],
    candidates: Sequence[SameMeanGaussianMixtureCandidateV1],
    predictor_id: str,
    query_set_id: str,
    grouping_rule_id: str,
    development_evidence_id: str,
    reference_candidate_id: str,
    maximum_worst_group_regret: float = 0.0,
    maximum_width_ratio: float = 2.0,
    density_floor_variance_m2: float = 0.0,
    covariance_tolerance: float = 1e-10,
    grid_frozen_before_development_scores: bool = True,
    target_outcomes_used: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> SameMeanGaussianMixtureSelectionV1:
    """Select one mixture on complete source/development physical groups.

    The reference candidate is always eligible.  Non-reference candidates must
    satisfy the frozen worst-group log-score and moment-width guards.  The
    resulting candidate still requires disjoint calibration and confirmation.
    """

    residual_values = tuple(residual_groups)
    covariance_values = tuple(nominal_covariance_groups)
    if not residual_values or len(residual_values) != len(covariance_values):
        raise ValueError(
            "residual_groups and nominal_covariance_groups must have equal "
            "nonzero length"
        )
    group_ids = _canonical_group_ids(
        development_group_ids,
        count=len(residual_values),
    )
    candidate_values = _candidate_sequence(candidates)
    scores: np.ndarray = np.empty(
        (len(candidate_values), len(group_ids)), dtype=np.float64
    )
    widths = np.empty_like(scores)
    for group_index, (residual_value, covariance_value) in enumerate(
        zip(residual_values, covariance_values, strict=True)
    ):
        residual = _real_array(
            residual_value,
            name=f"residual_groups[{group_index}]",
        )
        if residual.ndim == 1:
            residual = residual[None, :]
        if residual.ndim != 2 or residual.shape[0] == 0 or residual.shape[1] == 0:
            raise ValueError(f"residual_groups[{group_index}] must have shape (M, D)")
        mean = np.zeros_like(residual, dtype=np.float64, order="C")
        covariance = _real_array(
            covariance_value,
            name=f"nominal_covariance_groups[{group_index}]",
        )
        expected = residual.shape + (residual.shape[-1],)
        if covariance.shape != expected:
            raise ValueError(
                f"nominal_covariance_groups[{group_index}] must have shape {expected}"
            )
        for candidate_index, candidate in enumerate(candidate_values):
            prediction = compose_candidate_same_mean_gaussian_mixture(
                mean,
                covariance,
                candidate,
                reference_predictor_id=predictor_id,
                nominal_covariance_id=(
                    f"development-nominal-covariance:{development_evidence_id}:"
                    f"{group_ids[group_index]}"
                ),
                density_floor_variance_m2=density_floor_variance_m2,
                covariance_tolerance=covariance_tolerance,
                metadata={"development_group_id": group_ids[group_index]},
            )
            scores[candidate_index, group_index] = (
                group_gaussian_mixture_negative_log_score(
                    residual,
                    prediction,
                )
            )
            widths[candidate_index, group_index] = (
                gaussian_mixture_rms_marginal_standard_deviation(prediction)
            )
    return SameMeanGaussianMixtureSelectionV1(
        predictor_id=predictor_id,
        query_set_id=query_set_id,
        grouping_rule_id=grouping_rule_id,
        development_evidence_id=development_evidence_id,
        development_group_ids=group_ids,
        candidates=candidate_values,
        group_negative_log_scores=scores,
        group_rms_marginal_standard_deviations=widths,
        reference_candidate_id=reference_candidate_id,
        selected_candidate_id=None,
        maximum_worst_group_regret=maximum_worst_group_regret,
        maximum_width_ratio=maximum_width_ratio,
        density_floor_variance_m2=density_floor_variance_m2,
        grid_frozen_before_development_scores=(grid_frozen_before_development_scores),
        target_outcomes_used=target_outcomes_used,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "SAME_MEAN_GAUSSIAN_MIXTURE_SCHEMA",
    "SAME_MEAN_GAUSSIAN_MIXTURE_VERSION",
    "SameMeanGaussianMixtureCandidateV1",
    "SameMeanGaussianMixturePredictionV1",
    "SameMeanGaussianMixtureRecordV1",
    "SameMeanGaussianMixtureSelectionV1",
    "compose_candidate_same_mean_gaussian_mixture",
    "compose_same_mean_gaussian_mixture",
    "gaussian_mixture_moment_covariance",
    "gaussian_mixture_negative_log_density",
    "gaussian_mixture_rms_marginal_standard_deviation",
    "group_gaussian_mixture_energy_score",
    "group_gaussian_mixture_negative_log_score",
    "select_same_mean_gaussian_mixture",
]
