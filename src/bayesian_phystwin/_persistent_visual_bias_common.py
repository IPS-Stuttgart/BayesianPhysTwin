"""Shared contracts and covariance helpers for persistent visual bias."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Literal, cast

import numpy as np

from ._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from .prob4d_visual_bias_stream import ValidatedProb4DVisualBiasStreamV1
from .prob4d_visual_bias_update import (
    _canonical_id,
    _finite_real,
    _immutable_array,
    _sha256,
)

PERSISTENT_VISUAL_BIAS_CONFIG_SCHEMA = (
    "bayesian_phystwin.persistent_visual_bias_config"
)
PERSISTENT_VISUAL_BIAS_CONFIG_VERSION = 1
PERSISTENT_VISUAL_BIAS_STATE_SCHEMA = (
    "bayesian_phystwin.persistent_visual_bias_information_state"
)
PERSISTENT_VISUAL_BIAS_STATE_VERSION = 1
PERSISTENT_VISUAL_BIAS_UPDATE_SCHEMA = (
    "bayesian_phystwin.persistent_visual_bias_update"
)
PERSISTENT_VISUAL_BIAS_UPDATE_VERSION = 1
PERSISTENT_VISUAL_BIAS_REPARAMETERIZATION = (
    "full-joint-covariance-root-to-persistent-isotropic-bias-v1"
)
PERSISTENT_VISUAL_BIAS_CLAIM_BOUNDARY = (
    "The persistent solver inserts one source-calibrated coherent visual-bias "
    "prior, retains physical-bias cross-covariance across causal updates, and "
    "returns an exact zero physical increment when the registered query is not "
    "admissible. It does not establish provider competence, target calibration, "
    "guarded physical-query benefit, deployment safety, Causal4D benefit, or "
    "state of the art."
)

LikelihoodMode = Literal["grouped_student_t_mixture", "gaussian"]
_LIKELIHOOD_MODES = frozenset({"grouped_student_t_mixture", "gaussian"})


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _canonical_strings(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a canonical tuple")
    result = cast(tuple[object, ...], value)
    if any(type(item) is not str or not item for item in result):
        raise ValueError(f"{name} must contain nonempty literal strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return cast(tuple[str, ...], result)


def _finite_vector(value: object, *, name: str, nonempty: bool = True) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional float64 array")
    if nonempty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _symmetric_psd(
    value: object,
    *,
    name: str,
    dimension: int | None = None,
    tolerance: float = 1e-10,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional float64 array")
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    if dimension is not None and array.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(array, array.T, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (array + array.T)
    if len(symmetric):
        eigenvalues = np.linalg.eigvalsh(symmetric)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(np.min(eigenvalues)) < -tolerance * scale:
            raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def _project_psd(value: np.ndarray, *, name: str, tolerance: float) -> np.ndarray:
    symmetric = _symmetric_psd(value, name=name, tolerance=tolerance)
    if not len(symmetric):
        return symmetric
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -tolerance * scale:
        raise ValueError(f"{name} is not positive semidefinite")
    projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return 0.5 * (projected + projected.T)


def _config_record(config: PriorAwareGaugeConfigV1) -> dict[str, object]:
    return {item.name: getattr(config, item.name) for item in fields(config)}


def _stream_covariance_root(stream: ValidatedProb4DVisualBiasStreamV1) -> np.ndarray:
    covariance = np.asarray(stream.joint_bias_covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError("visual-bias stream covariance must be positive semidefinite")
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    root = 0.5 * (root + root.T)
    if not np.allclose(root @ root.T, covariance, atol=1e-10, rtol=1e-10):
        raise ValueError("visual-bias stream covariance root failed reconstruction")
    return _immutable_array(root, dtype=np.dtype(np.float64))


def _stream_reparameterized_design(
    stream: ValidatedProb4DVisualBiasStreamV1,
    *,
    shared_bias_prior_std_m: float,
) -> np.ndarray:
    scale = _finite_real(
        shared_bias_prior_std_m,
        name="shared_bias_prior_std_m",
        strictly_positive=True,
    )
    design = np.einsum(
        "nck,kr->ncr",
        stream.global_design(),
        _stream_covariance_root(stream),
    )
    return _immutable_array(design / scale, dtype=np.dtype(np.float64))


@dataclass(frozen=True, slots=True)
class PersistentVisualBiasConfigV1:
    """Frozen numerical policy for persistent coherent-bias inference."""

    inference: PriorAwareGaugeConfigV1 = field(
        default_factory=PriorAwareGaugeConfigV1
    )
    likelihood_mode: LikelihoodMode = "grouped_student_t_mixture"
    covariance_psd_tolerance: float = 1e-10
    config_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inference, PriorAwareGaugeConfigV1):
            raise TypeError("inference must be a PriorAwareGaugeConfigV1")
        if type(self.likelihood_mode) is not str or self.likelihood_mode not in (
            _LIKELIHOOD_MODES
        ):
            raise ValueError(
                "likelihood_mode must be grouped_student_t_mixture or gaussian"
            )
        tolerance = _finite_real(
            self.covariance_psd_tolerance,
            name="covariance_psd_tolerance",
            strictly_positive=True,
        )
        object.__setattr__(self, "covariance_psd_tolerance", tolerance)
        expected_id = _canonical_id(self.descriptor())
        supplied_id = self.config_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="config_id")
            if supplied_id != expected_id:
                raise ValueError("persistent visual-bias config ID mismatch")
        object.__setattr__(self, "config_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": PERSISTENT_VISUAL_BIAS_CONFIG_SCHEMA,
            "schema_version": PERSISTENT_VISUAL_BIAS_CONFIG_VERSION,
            "inference": _config_record(self.inference),
            "likelihood_mode": self.likelihood_mode,
            "covariance_psd_tolerance": self.covariance_psd_tolerance,
        }
