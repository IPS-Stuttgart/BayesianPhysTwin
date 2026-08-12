"""Shared validation and canonicalization for independent-group inference v1."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from numbers import Real
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import immutable_array

INDEPENDENT_GROUP_INFERENCE_SCHEMA = "bayesian_phystwin.independent_group_inference"
INDEPENDENT_GROUP_INFERENCE_VERSION = 1
EFFECT_DIRECTION = "negative_candidate_minus_comparator_is_better"
RESAMPLING_UNIT = "complete_independent_physical_object_or_session"
GROUP_WEIGHTING = "equal"
SIGN_FLIP_ASSUMPTION = "joint_group_effect_sign_symmetry"
SIGN_FLIP_STATISTIC = "mean_divided_by_root_mean_square_group_effect"
POINTWISE_INTERVAL_METHOD = "paired_group_bootstrap_percentile_linear"
SIMULTANEOUS_INTERVAL_METHOD = (
    "paired_group_bootstrap_max_standardized_deviation_higher"
)
BOOTSTRAP_RNG = "numpy.PCG64"
BOOTSTRAP_INDEX_DIGEST_DTYPE = "uint32-little-endian-c-order"
BOOTSTRAP_MEAN_DIGEST_DTYPE = "float64-little-endian-c-order"
SIGN_PATTERN_DIGEST_DTYPE = "int8-c-order-pattern-id-lsb-first"
BOOTSTRAP_CHUNK_SIZE = 8192
MAXIMUM_EXACT_GROUPS = 20
MAXIMUM_ESTIMANDS = 64
MAXIMUM_BOOTSTRAP_REPLICATES = 1_000_000
MAXIMUM_BOOTSTRAP_DRAWS = 10_000_000
MAXIMUM_BOOTSTRAP_RESULT_VALUES = 10_000_000
DEFAULT_BOOTSTRAP_REPLICATES = 100_000
DEFAULT_BOOTSTRAP_SEED = 20260812
DEFAULT_CONFIDENCE = 0.95
SIGN_FLIP_COMPARISON_EPSILON_MULTIPLIER = 32.0

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_id",
        "family_id",
        "statistical_unit",
        "within_group_aggregation",
        "effect_direction",
        "resampling_unit",
        "group_weighting",
        "sign_flip_assumption",
        "sign_flip_statistic",
        "sign_flip_comparison_epsilon_multiplier",
        "pointwise_interval_method",
        "simultaneous_interval_method",
        "bootstrap_rng",
        "bootstrap_index_digest_dtype",
        "bootstrap_mean_digest_dtype",
        "sign_pattern_digest_dtype",
        "bootstrap_chunk_size",
        "group_ids",
        "estimand_ids",
        "group_effects",
        "confidence",
        "bootstrap_replicates",
        "bootstrap_seed",
        "maximum_exact_groups",
        "sign_pattern_count",
        "observed_mean",
        "standard_error",
        "root_mean_square_scale",
        "standardized_mean",
        "exact_unadjusted_p_value",
        "exact_familywise_p_value",
        "exact_global_family_p_value",
        "pointwise_interval_lower",
        "pointwise_interval_upper",
        "simultaneous_interval_lower",
        "simultaneous_interval_upper",
        "simultaneous_superiority_upper",
        "simultaneous_two_sided_critical_value",
        "simultaneous_one_sided_critical_value",
        "win_count",
        "tie_count",
        "harm_count",
        "best_group_effect",
        "worst_group_effect",
        "median_group_effect",
        "bootstrap_index_sha256",
        "bootstrap_mean_sha256",
        "sign_pattern_sha256",
        "metadata",
        "artifact_id",
    }
)


def canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def canonical_identifiers(
    values: Sequence[str],
    *,
    name: str,
    minimum_count: int,
    maximum_count: int | None = None,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of exact strings")
    result = tuple(
        canonical_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(values)
    )
    if len(result) < minimum_count:
        raise ValueError(f"{name} must contain at least {minimum_count} entries")
    if maximum_count is not None and len(result) > maximum_count:
        raise ValueError(f"{name} must contain at most {maximum_count} entries")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def finite_real(value: object, *, name: str, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def strict_integer(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    integer = cast(int, value)
    if not minimum <= integer <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return integer


def finite_effect_matrix(
    value: object,
    *,
    group_count: int,
    estimand_count: int,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("group_effects must contain real numeric values")
    effects = np.asarray(raw, dtype=np.float64)
    if effects.shape != (group_count, estimand_count):
        raise ValueError(
            "group_effects must have shape "
            f"({group_count}, {estimand_count})"
        )
    if not np.all(np.isfinite(effects)):
        raise ValueError("group_effects must contain only finite values")
    return effects


def canonical_axis_order(
    identifiers: tuple[str, ...],
    values: FloatArray,
    *,
    axis: int,
) -> tuple[tuple[str, ...], FloatArray]:
    order = tuple(sorted(range(len(identifiers)), key=identifiers.__getitem__))
    index = np.asarray(order, dtype=np.int64)
    ordered = np.take(values, index, axis=axis)
    return tuple(identifiers[position] for position in order), ordered


def immutable_float(value: object) -> FloatArray:
    return cast(FloatArray, immutable_array(value, dtype=np.float64))


def immutable_int(value: object) -> IntArray:
    return cast(IntArray, immutable_array(value, dtype=np.int64))


def sha256_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> str:
    canonical = np.asarray(value, dtype=dtype, order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def array_payload(value: np.ndarray) -> list[Any]:
    return cast(list[Any], np.asarray(value).tolist())
