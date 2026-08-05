"""Versioned Deform360 contact anchors for gauge-aware physical updates.

The released Deform360 tactile grids are unitless peak-relative responses.  This
contract does not reinterpret taxels as Cartesian measurements.  A caller must
first map synchronized tactile/proprioceptive evidence through a frozen contact
linearization into displacement-equivalent residuals and state Jacobians.  The
resulting rows remain independent of the camera gauge while retaining their own
correlation groups and optional sensor-family bias.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar, cast

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_integer,
    integer_array,
    plain_json,
)

DEFORM360_CONTACT_ANCHOR_SCHEMA = "bayesian-phystwin.deform360-contact-anchor"
DEFORM360_CONTACT_ANCHOR_VERSION = 1
DEFORM360_CONTACT_ANCHOR_SEMANTICS = (
    "prefix-tactile-proprioceptive-contact-linearization-v1"
)
DEFORM360_TACTILE_SOURCE_UNITS = "unitless-peak-relative"
DEFORM360_CONTACT_ANCHOR_UNITS = "displacement-equivalent-m"
DEFORM360_SOURCE_REPOSITORY = "brownu/deform360"


class _ContactAnchorBatch(Protocol):
    state_jacobian: np.ndarray


_BatchT = TypeVar("_BatchT", bound=_ContactAnchorBatch)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(values: object, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy(order="C")
    result.setflags(write=False)
    return result


def _finite_array(values: object, *, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    _require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


def _probability_vector(
    values: object | None,
    count: int,
    *,
    name: str,
    strictly_positive: bool = False,
) -> np.ndarray:
    result = (
        np.ones(count, dtype=np.float64)
        if values is None
        else np.asarray(values, dtype=np.float64)
    )
    _require(result.shape == (count,), f"{name} must have shape ({count},)")
    lower = result > 0.0 if strictly_positive else result >= 0.0
    interval = "(0, 1]" if strictly_positive else "[0, 1]"
    _require(
        np.all(np.isfinite(result)) and np.all(lower & (result <= 1.0)),
        f"{name} must lie in {interval}",
    )
    return result


def _validate_covariances(values: np.ndarray, *, name: str) -> None:
    for index, matrix in enumerate(values):
        _require(
            np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0),
            f"{name} {index} must be symmetric",
        )
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name} {index} must be positive definite") from error


def _validate_square_psd(values: np.ndarray, *, name: str) -> None:
    _require(values.shape[0] == values.shape[1], f"{name} must be square")
    _require(
        np.allclose(values, values.T, atol=1e-12, rtol=0.0),
        f"{name} must be symmetric",
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (values + values.T))
    _require(np.all(eigenvalues >= -1e-12), f"{name} must be positive semidefinite")


def _require_sha256(value: object, *, name: str) -> str:
    result = str(value)
    _require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return result


def _require_revision(value: object, *, name: str) -> str:
    result = str(value)
    _require(
        len(result) in {40, 64}
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be an exact lowercase revision",
    )
    return result


def _array_record(values: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(values)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _canonical_json(values: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class Deform360ContactAnchorV1:
    """Independent contact rows mapped into physical-state coordinates.

    ``innovation_m`` and ``state_jacobian`` must already be expressed in metres
    and state-coefficient coordinates.  Raw tactile taxels are provenance only;
    they are never admitted as independent Cartesian rows by this contract.
    """

    object_id: str
    episode_id: int
    causal_frame_stop: int
    sensor_names: Sequence[str]
    frame_ids: np.ndarray
    innovation_m: np.ndarray
    covariance_m2: np.ndarray
    state_jacobian: np.ndarray
    correlation_group_ids: Sequence[str]
    source_revision: str
    source_artifacts: Mapping[str, str]
    prior_reliability: np.ndarray | None = None
    prior_nominal_probability: np.ndarray | None = None
    composite_weight: np.ndarray | None = None
    bias_jacobian: np.ndarray | None = None
    bias_prior_covariance: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _require(
            isinstance(self.object_id, str) and bool(self.object_id),
            "object_id must be a nonempty string",
        )
        episode_id = genuine_integer(self.episode_id, name="episode_id", minimum=0)
        causal_frame_stop = genuine_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        frame_ids = integer_array(self.frame_ids, name="frame_ids")
        _require(frame_ids.ndim == 1 and len(frame_ids) > 0, "frame_ids must be 1-D")
        _require(
            np.all((frame_ids >= 0) & (frame_ids < causal_frame_stop)),
            "contact anchor contains a post-cutoff frame",
        )
        count = len(frame_ids)
        sensor_names = canonical_string_tuple(
            self.sensor_names,
            name="sensor_names",
            allow_empty=False,
        )
        groups = canonical_string_tuple(
            self.correlation_group_ids,
            name="correlation_group_ids",
            allow_empty=False,
        )
        _require(len(sensor_names) == count, "sensor_names must identify every row")
        _require(len(groups) == count, "correlation_group_ids must identify every row")

        innovation = _finite_array(self.innovation_m, name="innovation_m", ndim=2)
        covariance = _finite_array(self.covariance_m2, name="covariance_m2", ndim=3)
        state = _finite_array(self.state_jacobian, name="state_jacobian", ndim=3)
        _require(innovation.shape == (count, 3), "innovation_m must have shape (A, 3)")
        _require(
            covariance.shape == (count, 3, 3),
            "covariance_m2 must have shape (A, 3, 3)",
        )
        _require(
            state.shape[:2] == (count, 3) and state.shape[2] >= 1,
            "state_jacobian must have shape (A, 3, S) with S >= 1",
        )
        _validate_covariances(covariance, name="contact covariance")

        reliability = _probability_vector(
            self.prior_reliability,
            count,
            name="prior_reliability",
        )
        nominal_probability = _probability_vector(
            self.prior_nominal_probability,
            count,
            name="prior_nominal_probability",
        )
        composite_weight = _probability_vector(
            self.composite_weight,
            count,
            name="composite_weight",
            strictly_positive=True,
        )

        bias: np.ndarray | None = None
        bias_prior: np.ndarray | None = None
        if self.bias_jacobian is None:
            _require(
                self.bias_prior_covariance is None,
                "bias_prior_covariance requires bias_jacobian",
            )
        else:
            bias = _finite_array(self.bias_jacobian, name="bias_jacobian", ndim=3)
            _require(
                bias.shape[:2] == (count, 3),
                "bias_jacobian must have shape (A, 3, B)",
            )
            _require(
                self.bias_prior_covariance is not None,
                "bias_jacobian requires bias_prior_covariance",
            )
            bias_prior = _finite_array(
                self.bias_prior_covariance,
                name="bias_prior_covariance",
                ndim=2,
            )
            _require(
                bias_prior.shape == (bias.shape[2], bias.shape[2]),
                "bias_prior_covariance has changed shape",
            )
            _validate_square_psd(bias_prior, name="bias_prior_covariance")

        source_revision = _require_revision(
            self.source_revision,
            name="source_revision",
        )
        _require(
            isinstance(self.source_artifacts, Mapping) and bool(self.source_artifacts),
            "source_artifacts must be a nonempty mapping",
        )
        source_artifacts: dict[str, str] = {}
        for path, digest in self.source_artifacts.items():
            _require(
                isinstance(path, str) and bool(path),
                "source_artifacts keys must be nonempty paths",
            )
            source_artifacts[path] = _require_sha256(
                digest,
                name=f"source artifact {path}",
            )
        metadata = frozen_finite_json_mapping(self.metadata, name="anchor metadata")

        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)
        object.__setattr__(self, "sensor_names", sensor_names)
        object.__setattr__(self, "frame_ids", _readonly(frame_ids, dtype=np.int64))
        object.__setattr__(
            self, "innovation_m", _readonly(innovation, dtype=np.float64)
        )
        object.__setattr__(
            self, "covariance_m2", _readonly(covariance, dtype=np.float64)
        )
        object.__setattr__(self, "state_jacobian", _readonly(state, dtype=np.float64))
        object.__setattr__(self, "correlation_group_ids", groups)
        object.__setattr__(
            self,
            "prior_reliability",
            _readonly(reliability, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "prior_nominal_probability",
            _readonly(nominal_probability, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "composite_weight",
            _readonly(composite_weight, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "bias_jacobian",
            None if bias is None else _readonly(bias, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "bias_prior_covariance",
            None if bias_prior is None else _readonly(bias_prior, dtype=np.float64),
        )
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(
            self,
            "source_artifacts",
            frozen_finite_json_mapping(source_artifacts, name="source_artifacts"),
        )
        object.__setattr__(self, "metadata", metadata)

    @property
    def state_count(self) -> int:
        """Number of physical-state coefficients used by the anchor."""

        return int(self.state_jacobian.shape[2])

    @property
    def row_count(self) -> int:
        """Number of reduced, correlation-aware contact rows."""

        return int(len(self.innovation_m))

    @property
    def artifact_id(self) -> str:
        """Return a content address over descriptors, provenance, and arrays."""

        arrays: dict[str, object] = {
            "frame_ids": _array_record(self.frame_ids),
            "innovation_m": _array_record(self.innovation_m),
            "covariance_m2": _array_record(self.covariance_m2),
            "state_jacobian": _array_record(self.state_jacobian),
            "prior_reliability": _array_record(
                cast(np.ndarray, self.prior_reliability)
            ),
            "prior_nominal_probability": _array_record(
                cast(np.ndarray, self.prior_nominal_probability)
            ),
            "composite_weight": _array_record(cast(np.ndarray, self.composite_weight)),
        }
        if self.bias_jacobian is not None:
            arrays["bias_jacobian"] = _array_record(
                cast(np.ndarray, self.bias_jacobian)
            )
            arrays["bias_prior_covariance"] = _array_record(
                cast(np.ndarray, self.bias_prior_covariance)
            )
        descriptor = {
            "schema": DEFORM360_CONTACT_ANCHOR_SCHEMA,
            "schema_version": DEFORM360_CONTACT_ANCHOR_VERSION,
            "semantics": DEFORM360_CONTACT_ANCHOR_SEMANTICS,
            "source_repository": DEFORM360_SOURCE_REPOSITORY,
            "source_revision": self.source_revision,
            "source_artifacts": self.source_artifacts,
            "source_units": DEFORM360_TACTILE_SOURCE_UNITS,
            "anchor_units": DEFORM360_CONTACT_ANCHOR_UNITS,
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "causal_frame_stop": self.causal_frame_stop,
            "sensor_names": self.sensor_names,
            "correlation_group_ids": self.correlation_group_ids,
            "metadata": self.metadata,
            "arrays": arrays,
        }
        return hashlib.sha256(_canonical_json(descriptor)).hexdigest()

    def summary(self) -> dict[str, object]:
        """Return compact lineage suitable for a gauge-aware batch."""

        return {
            "schema": DEFORM360_CONTACT_ANCHOR_SCHEMA,
            "schema_version": DEFORM360_CONTACT_ANCHOR_VERSION,
            "semantics": DEFORM360_CONTACT_ANCHOR_SEMANTICS,
            "artifact_id": self.artifact_id,
            "source_repository": DEFORM360_SOURCE_REPOSITORY,
            "source_revision": self.source_revision,
            "source_units": DEFORM360_TACTILE_SOURCE_UNITS,
            "anchor_units": DEFORM360_CONTACT_ANCHOR_UNITS,
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "causal_frame_stop": self.causal_frame_stop,
            "row_count": self.row_count,
            "state_count": self.state_count,
            "correlation_group_count": len(set(self.correlation_group_ids)),
            "sensor_count": len(set(self.sensor_names)),
            "raw_taxels_used_as_independent_rows": False,
            "camera_gauge_present_in_anchor": False,
            "anchor_bias_parameter_count": (
                0 if self.bias_jacobian is None else self.bias_jacobian.shape[2]
            ),
        }


def attach_deform360_contact_anchor(
    batch: _BatchT,
    anchor: Deform360ContactAnchorV1,
) -> _BatchT:
    """Attach one independent contact family to an unanchored batch.

    The returned object is a dataclass replacement.  Existing anchor rows fail
    closed rather than being concatenated without an explicit dependence model.
    """

    for name in (
        "anchor_innovation_m",
        "anchor_covariance_m2",
        "anchor_state_jacobian",
    ):
        _require(getattr(batch, name, None) is None, "batch already contains anchors")
    state = np.asarray(batch.state_jacobian, dtype=np.float64)
    _require(
        state.ndim == 3 and state.shape[2] == anchor.state_count,
        "contact anchor state dimension differs from the visual batch",
    )
    batch_metadata = plain_json(getattr(batch, "metadata", None) or {})
    _require(
        "deform360_contact_anchor" not in batch_metadata,
        "contact anchor lineage is already present",
    )
    visual_cutoff = batch_metadata.get("observation_causal_frame_stop")
    if visual_cutoff is not None:
        _require(
            visual_cutoff == anchor.causal_frame_stop,
            "visual and contact causal cutoffs differ",
        )
    batch_metadata["deform360_contact_anchor"] = anchor.summary()
    return cast(
        _BatchT,
        replace(
            cast(Any, batch),
            anchor_innovation_m=anchor.innovation_m,
            anchor_covariance_m2=anchor.covariance_m2,
            anchor_state_jacobian=anchor.state_jacobian,
            anchor_correlation_group_ids=anchor.correlation_group_ids,
            anchor_prior_reliability=anchor.prior_reliability,
            anchor_prior_nominal_probability=anchor.prior_nominal_probability,
            anchor_composite_weight=anchor.composite_weight,
            anchor_bias_jacobian=anchor.bias_jacobian,
            anchor_bias_prior_covariance=anchor.bias_prior_covariance,
            metadata=batch_metadata,
        ),
    )


__all__ = [
    "DEFORM360_CONTACT_ANCHOR_SCHEMA",
    "DEFORM360_CONTACT_ANCHOR_SEMANTICS",
    "DEFORM360_CONTACT_ANCHOR_UNITS",
    "DEFORM360_CONTACT_ANCHOR_VERSION",
    "DEFORM360_SOURCE_REPOSITORY",
    "DEFORM360_TACTILE_SOURCE_UNITS",
    "Deform360ContactAnchorV1",
    "attach_deform360_contact_anchor",
]
