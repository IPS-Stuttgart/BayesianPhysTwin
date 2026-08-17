from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.numerical_environment_v1 as numerical_environment
from bayesian_phystwin.numerical_environment_v1 import (
    DependencyLockV1,
    InstalledDistributionV1,
    NumericalEnvironmentV1,
    embed_numerical_environment_v1,
    numerical_environment_from_dict,
    validate_embedded_numerical_environment_v1,
)


def _controls() -> dict[str, str | None]:
    return {name: None for name in numerical_environment._EXECUTION_CONTROL_NAMES}


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


def test_runtime_versions_must_match_distribution_inventory() -> None:
    with pytest.raises(ValueError, match="NumPy runtime version"):
        NumericalEnvironmentV1(
            python_implementation="CPython",
            python_version="3.12.1",
            python_compiler="GCC",
            numpy_version="0.0",
            numpy_configuration_text="configuration\n",
            scipy_version=None,
            logical_cpu_count=1,
            byte_order="little",
            execution_controls=_controls(),
            installed_distributions=(InstalledDistributionV1("numpy", np.__version__),),
        )

    with pytest.raises(ValueError, match="SciPy runtime version"):
        NumericalEnvironmentV1(
            python_implementation="CPython",
            python_version="3.12.1",
            python_compiler="GCC",
            numpy_version=np.__version__,
            numpy_configuration_text="configuration\n",
            scipy_version="1.0",
            logical_cpu_count=1,
            byte_order="little",
            execution_controls=_controls(),
            installed_distributions=(InstalledDistributionV1("numpy", np.__version__),),
        )


def test_distribution_inventory_must_be_sorted_unique_and_typed() -> None:
    numpy_distribution = InstalledDistributionV1("numpy", np.__version__)
    test_distribution = InstalledDistributionV1("test-package", "1.0")

    with pytest.raises(ValueError, match="must be sorted"):
        NumericalEnvironmentV1(
            python_implementation="CPython",
            python_version="3.12.1",
            python_compiler="GCC",
            numpy_version=np.__version__,
            numpy_configuration_text="configuration\n",
            scipy_version=None,
            logical_cpu_count=1,
            byte_order="little",
            execution_controls=_controls(),
            installed_distributions=(test_distribution, numpy_distribution),
        )

    with pytest.raises(ValueError, match="must be unique"):
        NumericalEnvironmentV1(
            python_implementation="CPython",
            python_version="3.12.1",
            python_compiler="GCC",
            numpy_version=np.__version__,
            numpy_configuration_text="configuration\n",
            scipy_version=None,
            logical_cpu_count=1,
            byte_order="little",
            execution_controls=_controls(),
            installed_distributions=(numpy_distribution, numpy_distribution),
        )

    with pytest.raises(ValueError, match="InstalledDistributionV1"):
        NumericalEnvironmentV1(
            python_implementation="CPython",
            python_version="3.12.1",
            python_compiler="GCC",
            numpy_version=np.__version__,
            numpy_configuration_text="configuration\n",
            scipy_version=None,
            logical_cpu_count=1,
            byte_order="little",
            execution_controls=_controls(),
            installed_distributions=cast(
                Any,
                ({"name": "numpy", "version": np.__version__},),
            ),
        )


def test_dependency_lock_contract_is_strict() -> None:
    with pytest.raises(ValueError, match="portable basename"):
        DependencyLockV1("locks/requirements.txt", "a" * 64, 1)
    with pytest.raises(ValueError, match="integer"):
        DependencyLockV1("requirements.txt", "a" * 64, cast(Any, True))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        DependencyLockV1("requirements.txt", "A" * 64, 1)

    payload = _profile().as_dict()
    with pytest.raises(ValueError, match="requires a dependency lock"):
        numerical_environment_from_dict(
            payload,
            require_dependency_lock=True,
        )


def test_embed_and_validate_are_fail_closed_and_detached() -> None:
    runtime = {"accelerator": {"model": "test"}}
    profile = _profile(lock=_lock())

    embedded = embed_numerical_environment_v1(runtime, profile)
    cast(dict[str, str], runtime["accelerator"])["model"] = "changed"
    restored = validate_embedded_numerical_environment_v1(
        embedded,
        require_profile=True,
        require_dependency_lock=True,
    )

    assert restored == profile
    assert embedded["accelerator"] == {"model": "test"}

    with pytest.raises(ValueError, match="already contains"):
        embed_numerical_environment_v1(embedded, profile)
    with pytest.raises(ValueError, match="non-finite number"):
        embed_numerical_environment_v1({"value": float("nan")}, profile)
