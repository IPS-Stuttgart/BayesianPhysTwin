"""Validation helpers for graph-modal discrepancy tournament artifacts."""

from __future__ import annotations

import hashlib
import re
from numbers import Integral, Real
from typing import Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import (
    immutable_array,
    immutable_integer_array,
)

GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_SCHEMA: Final = (
    "bayesian_phystwin.graph_dynamic_tournament_prediction"
)
GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_VERSION: Final = 1
GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_SCHEMA: Final = (
    "bayesian_phystwin.graph_dynamic_tournament_prediction_bundle"
)
GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_VERSION: Final = 1
GRAPH_DYNAMIC_TOURNAMENT_SCORE_SCHEMA: Final = (
    "bayesian_phystwin.graph_dynamic_tournament_scored_bundle"
)
GRAPH_DYNAMIC_TOURNAMENT_SCORE_VERSION: Final = 1
GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_SCHEMA: Final = (
    "bayesian_phystwin.graph_dynamic_tournament_scoring_policy"
)
GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_VERSION: Final = 1
GRAPH_DYNAMIC_TOURNAMENT_FAMILY: Final = "graph-modal"
GRAPH_DYNAMIC_TOURNAMENT_BOUNDARY: Final = (
    "Source-only candidate records from predictions sealed before scored outcomes. "
    "They do not establish fresh-object transfer, calibrated deployment "
    "uncertainty, intervention benefit, deployment safety, or state of the art."
)
_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?\Z")

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def identifier(value: object, *, name: str) -> str:
    result = canonical_string(value, name=name)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase identifier")
    return result


def genuine_boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def genuine_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def float_array(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=np.dtype("<f8"))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return cast(FloatArray, immutable_array(array, dtype=np.dtype("<f8")))


def integer_vector(value: object, *, name: str) -> IntArray:
    array = immutable_integer_array(value, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return cast(IntArray, array)


def array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def validated_covariance(
    value: object,
    *,
    name: str,
    dimension: int,
) -> FloatArray:
    matrix = float_array(value, name=name)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} shape changed")
    if not np.allclose(matrix, matrix.T, atol=1e-11, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(np.linalg.norm(symmetric, ord=2)))
    tolerance = 16.0 * max(1, dimension) * np.finfo(np.float64).eps * scale
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return cast(
        FloatArray,
        immutable_array(symmetric, dtype=np.dtype("<f8")),
    )


__all__ = [
    "GRAPH_DYNAMIC_TOURNAMENT_BOUNDARY",
    "GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_SCHEMA",
    "GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_VERSION",
    "GRAPH_DYNAMIC_TOURNAMENT_FAMILY",
    "GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_SCHEMA",
    "GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_VERSION",
    "GRAPH_DYNAMIC_TOURNAMENT_SCORE_SCHEMA",
    "GRAPH_DYNAMIC_TOURNAMENT_SCORE_VERSION",
    "GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_SCHEMA",
    "GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_VERSION",
    "FloatArray",
    "IntArray",
    "array_record",
    "canonical_string",
    "finite_real",
    "float_array",
    "genuine_boolean",
    "genuine_integer",
    "identifier",
    "integer_vector",
    "validated_covariance",
]
