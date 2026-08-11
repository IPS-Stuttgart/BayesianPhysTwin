"""Derived numerical-compatibility identities for exact runtime profiles.

``NumericalEnvironmentV1.profile_id`` remains the exact replay identity. This
module derives a narrower identity for deciding whether two validated profiles
share the solver-relevant state declared by this contract version.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from .numerical_environment_v1 import NumericalEnvironmentV1

NUMERICAL_COMPATIBILITY_SCHEMA = "bayesian_phystwin.numerical_compatibility_profile"
NUMERICAL_COMPATIBILITY_SCHEMA_VERSION = 1
NUMERICAL_COMPATIBILITY_RECORD_SCHEMA = (
    "bayesian_phystwin.numerical_compatibility_record"
)
NUMERICAL_COMPATIBILITY_RECORD_VERSION = 1

_VERSION_PREFIX = re.compile(r"^([0-9]+)\.([0-9]+)(?:\.|$)")
_POSITIVE_INTEGER = re.compile(r"^[1-9][0-9]*$")
_FALSE_CONTROL_VALUES = frozenset({"0", "false", "no", "off"})
_THREAD_COUNT_CONTROLS = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_RECORD_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_profile_id",
        "numerical_compatibility_id",
        "compatibility_descriptor",
    }
)


def _require_profile(value: object) -> NumericalEnvironmentV1:
    if type(value) is not NumericalEnvironmentV1:
        raise ValueError("profile must be NumericalEnvironmentV1")
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a literal boolean")
    return value


def _python_major_minor(version: str) -> str:
    match = _VERSION_PREFIX.match(version)
    if match is None:
        raise ValueError("Python version must start with major.minor")
    return f"{int(match.group(1))}.{int(match.group(2))}"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _thread_counts_fully_pinned(profile: NumericalEnvironmentV1) -> bool:
    controls = profile.execution_controls
    count_controls_are_positive_integers = all(
        isinstance(controls[name], str)
        and _POSITIVE_INTEGER.fullmatch(controls[name]) is not None
        for name in _THREAD_COUNT_CONTROLS
    )
    omp_dynamic = controls["OMP_DYNAMIC"]
    dynamic_teams_disabled = (
        isinstance(omp_dynamic, str)
        and omp_dynamic.lower() in _FALSE_CONTROL_VALUES
    )
    return count_controls_are_positive_integers and dynamic_teams_disabled


def _implicit_parallelism_descriptor(
    profile: NumericalEnvironmentV1,
) -> dict[str, object]:
    fully_pinned = _thread_counts_fully_pinned(profile)
    cpu_count = None if fully_pinned else profile.logical_cpu_count
    if not fully_pinned and cpu_count is None:
        raise ValueError(
            "numerical compatibility requires logical CPU count unless all "
            "registered thread counts are fully pinned"
        )
    return {
        "thread_counts_fully_pinned": fully_pinned,
        "logical_cpu_count": cpu_count,
    }


def numerical_compatibility_descriptor_v1(
    profile: NumericalEnvironmentV1,
    *,
    require_dependency_lock: bool = True,
) -> dict[str, object]:
    """Return the solver-relevant compatibility descriptor for ``profile``.

    The exact installed-distribution inventory and Python patch/compiler details
    remain bound by ``profile.profile_id`` but are intentionally excluded here.
    Logical CPU count is excluded only when every registered CPU thread count is
    an explicit positive integer and OpenMP dynamic teams are explicitly off.
    The dependency lock is required by default so claim-bearing comparisons
    cannot silently rely on an unbound resolver result.
    """

    exact = _require_profile(profile)
    require_lock = _require_boolean(
        require_dependency_lock,
        name="require_dependency_lock",
    )
    lock = exact.dependency_lock
    if require_lock and lock is None:
        raise ValueError("numerical compatibility requires a dependency lock")
    lock_descriptor: dict[str, object] | None = None
    if lock is not None:
        lock_descriptor = {
            "sha256": lock.sha256,
            "size_bytes": lock.size_bytes,
        }
    return {
        "schema_name": NUMERICAL_COMPATIBILITY_SCHEMA,
        "schema_version": NUMERICAL_COMPATIBILITY_SCHEMA_VERSION,
        "python": {
            "implementation": exact.python_implementation,
            "major_minor": _python_major_minor(exact.python_version),
        },
        "numpy": {
            "version": exact.numpy_version,
            "configuration_sha256": exact.numpy_configuration_sha256,
        },
        "scipy_version": exact.scipy_version,
        "byte_order": exact.byte_order,
        "execution_controls": {
            name: exact.execution_controls[name]
            for name in sorted(exact.execution_controls)
        },
        "implicit_parallelism": _implicit_parallelism_descriptor(exact),
        "dependency_lock": lock_descriptor,
    }


def numerical_compatibility_id_v1(
    profile: NumericalEnvironmentV1,
    *,
    require_dependency_lock: bool = True,
) -> str:
    """Return the content identity of the derived compatibility descriptor."""

    return _sha256(
        numerical_compatibility_descriptor_v1(
            profile,
            require_dependency_lock=require_dependency_lock,
        )
    )


def numerical_compatibility_record_v1(
    profile: NumericalEnvironmentV1,
    *,
    require_dependency_lock: bool = True,
) -> dict[str, object]:
    """Bind one compatibility identity to its exact source profile."""

    exact = _require_profile(profile)
    descriptor = numerical_compatibility_descriptor_v1(
        exact,
        require_dependency_lock=require_dependency_lock,
    )
    return {
        "schema_name": NUMERICAL_COMPATIBILITY_RECORD_SCHEMA,
        "schema_version": NUMERICAL_COMPATIBILITY_RECORD_VERSION,
        "source_profile_id": exact.profile_id,
        "numerical_compatibility_id": _sha256(descriptor),
        "compatibility_descriptor": descriptor,
    }


def validate_numerical_compatibility_record_v1(
    value: Mapping[str, Any],
    profile: NumericalEnvironmentV1,
    *,
    require_dependency_lock: bool = True,
) -> dict[str, object]:
    """Validate a serialized record against one exact environment profile."""

    if not isinstance(value, Mapping):
        raise ValueError("numerical compatibility record must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError("numerical compatibility record keys must be strings")
    if frozenset(value) != _RECORD_FIELDS:
        raise ValueError("numerical compatibility record fields changed")
    expected = numerical_compatibility_record_v1(
        profile,
        require_dependency_lock=require_dependency_lock,
    )
    try:
        detached = json.loads(
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "numerical compatibility record must contain finite JSON values"
        ) from error
    if detached != expected:
        raise ValueError(
            "numerical compatibility record does not match its exact profile"
        )
    return expected


def numerically_compatible_v1(
    left: NumericalEnvironmentV1,
    right: NumericalEnvironmentV1,
    *,
    require_dependency_lock: bool = True,
) -> bool:
    """Return whether two exact profiles share compatibility identity v1."""

    return numerical_compatibility_id_v1(
        left,
        require_dependency_lock=require_dependency_lock,
    ) == numerical_compatibility_id_v1(
        right,
        require_dependency_lock=require_dependency_lock,
    )


__all__ = [
    "NUMERICAL_COMPATIBILITY_RECORD_SCHEMA",
    "NUMERICAL_COMPATIBILITY_RECORD_VERSION",
    "NUMERICAL_COMPATIBILITY_SCHEMA",
    "NUMERICAL_COMPATIBILITY_SCHEMA_VERSION",
    "numerical_compatibility_descriptor_v1",
    "numerical_compatibility_id_v1",
    "numerical_compatibility_record_v1",
    "numerically_compatible_v1",
    "validate_numerical_compatibility_record_v1",
]
