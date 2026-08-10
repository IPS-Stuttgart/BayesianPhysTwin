from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.numerical_environment_v1 as numerical_environment
from bayesian_phystwin.numerical_environment_v1 import (
    DependencyLockV1,
    InstalledDistributionV1,
    NumericalEnvironmentV1,
    numerical_environment_from_dict,
)

def _controls() -> dict[str, str | None]:
    return {
        name: None for name in numerical_environment._EXECUTION_CONTROL_NAMES
    }


def _lock(*, digest: str = "a" * 64) -> DependencyLockV1:
    return DependencyLockV1(
        name="requirements.lock",
        sha256=digest,
        size_bytes=123,
    )


def _profile(
    *,
    configuration: str = "Build Dependencies:\r\n  blas: test   \r\n\r\n",
    controls: dict[str, str | None] | None = None,
    lock: DependencyLockV1 | None = None,
) -> NumericalEnvironmentV1:
    return NumericalEnvironmentV1(
        python_implementation="CPython",
        python_version="3.12.1",
        python_compiler="GCC 13.2.0",
        numpy_version=np.__version__,
        numpy_configuration_text=configuration,
        scipy_version=None,
        logical_cpu_count=8,
        byte_order="little",
        execution_controls=_controls() if controls is None else controls,
        installed_distributions=(
            InstalledDistributionV1(name="numpy", version=np.__version__),
        ),
        dependency_lock=lock,
    )



@pytest.mark.parametrize(
    ("value", "message"),
    (
        (cast(Any, []), "must be a JSON object"),
        (cast(Any, {1: "value"}), "literal string keys"),
    ),
)
def test_mapping_validator_rejects_nonobjects_and_nonstring_keys(
    value: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        numerical_environment._require_mapping(value, name="record")


@pytest.mark.parametrize("value", (cast(Any, "items"), cast(Any, 1)))
def test_sequence_validator_rejects_scalars(value: list[object]) -> None:
    with pytest.raises(ValueError, match="must be a JSON array"):
        numerical_environment._require_sequence(value, name="items")


def test_exact_field_validation_reports_missing_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="missing.*unknown"):
        numerical_environment._require_exact_fields(
            {"unknown": 1},
            expected=frozenset({"required"}),
            name="record",
        )


@pytest.mark.parametrize(
    ("value", "kwargs", "message"),
    (
        (" value", {}, "canonical text"),
        ("", {}, "nonempty"),
        ("abcd", {"maximum_length": 3}, "exceeds"),
        ("a\tb", {}, "control character"),
    ),
)
def test_scalar_text_validation_is_bounded_and_canonical(
    value: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        numerical_environment._require_text(value, name="value", **kwargs)


def test_multiline_configuration_validation_covers_empty_control_and_size() -> None:
    with pytest.raises(ValueError, match="nonempty text"):
        numerical_environment._normalize_numpy_configuration("")
    with pytest.raises(ValueError, match="control character"):
        numerical_environment._normalize_numpy_configuration("bad\x00value")

    normalized = numerical_environment._normalize_numpy_configuration("\r\n")
    assert normalized == "(no NumPy build configuration reported)\n"

    with pytest.raises(ValueError, match="exceeds one MiB"):
        numerical_environment._normalize_numpy_configuration(
            "x" * (1024 * 1024 + 1)
        )


def test_distribution_and_profile_boundary_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="PEP-503"):
        InstalledDistributionV1("invalid name", "1.0")

    with pytest.raises(ValueError, match="byte order"):
        NumericalEnvironmentV1(
            python_implementation="CPython",
            python_version="3.12",
            python_compiler="",
            numpy_version=np.__version__,
            numpy_configuration_text="configuration\n",
            scipy_version=None,
            logical_cpu_count=None,
            byte_order="middle",
            execution_controls=_controls(),
            installed_distributions=(
                InstalledDistributionV1("numpy", np.__version__),
            ),
        )

    long_control = _controls()
    long_control["OMP_NUM_THREADS"] = "x" * 257
    with pytest.raises(ValueError, match="too long"):
        _profile(controls=long_control)

    with pytest.raises(ValueError, match="must be nonempty"):
        NumericalEnvironmentV1(
            python_implementation="CPython",
            python_version="3.12",
            python_compiler="",
            numpy_version=np.__version__,
            numpy_configuration_text="configuration\n",
            scipy_version=None,
            logical_cpu_count=None,
            byte_order="little",
            execution_controls=_controls(),
            installed_distributions=(),
        )

    monkeypatch.setattr(numerical_environment, "_MAX_DISTRIBUTIONS", 1)
    with pytest.raises(ValueError, match="inventory is too large"):
        NumericalEnvironmentV1(
            python_implementation="CPython",
            python_version="3.12",
            python_compiler="",
            numpy_version=np.__version__,
            numpy_configuration_text="configuration\n",
            scipy_version=None,
            logical_cpu_count=None,
            byte_order="little",
            execution_controls=_controls(),
            installed_distributions=(
                InstalledDistributionV1("numpy", np.__version__),
                InstalledDistributionV1("test-package", "1.0"),
            ),
        )

    with pytest.raises(ValueError, match="DependencyLockV1"):
        NumericalEnvironmentV1(
            python_implementation="CPython",
            python_version="3.12",
            python_compiler="",
            numpy_version=np.__version__,
            numpy_configuration_text="configuration\n",
            scipy_version=None,
            logical_cpu_count=None,
            byte_order="little",
            execution_controls=_controls(),
            installed_distributions=(
                InstalledDistributionV1("numpy", np.__version__),
            ),
            dependency_lock=cast(Any, {}),
        )


def test_optional_scipy_and_cpu_fields_round_trip() -> None:
    profile = NumericalEnvironmentV1(
        python_implementation="CPython",
        python_version="3.12",
        python_compiler="",
        numpy_version=np.__version__,
        numpy_configuration_text="configuration\n",
        scipy_version="1.2.3",
        logical_cpu_count=None,
        byte_order="big",
        execution_controls=_controls(),
        installed_distributions=(
            InstalledDistributionV1("numpy", np.__version__),
            InstalledDistributionV1("scipy", "1.2.3"),
        ),
    )

    restored = numerical_environment_from_dict(profile.as_dict())

    assert restored.scipy_version == "1.2.3"
    assert restored.logical_cpu_count is None
    assert restored.byte_order == "big"
