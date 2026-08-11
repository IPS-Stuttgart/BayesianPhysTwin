from __future__ import annotations

import json
from typing import Any, cast

import pytest

import bayesian_phystwin.numerical_environment_v1 as numerical_environment
from bayesian_phystwin.numerical_compatibility_v1 import (
    numerical_compatibility_descriptor_v1,
    numerical_compatibility_id_v1,
    numerical_compatibility_record_v1,
    numerically_compatible_v1,
    validate_numerical_compatibility_record_v1,
)
from bayesian_phystwin.numerical_environment_v1 import (
    DependencyLockV1,
    InstalledDistributionV1,
    NumericalEnvironmentV1,
)

_THREAD_COUNT_CONTROLS = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _controls(
    *,
    omp_threads: str | None = None,
    fully_pinned: bool = False,
) -> dict[str, str | None]:
    controls = {name: None for name in numerical_environment._EXECUTION_CONTROL_NAMES}
    if fully_pinned:
        for name in _THREAD_COUNT_CONTROLS:
            controls[name] = "1"
        controls["OMP_DYNAMIC"] = "FALSE"
    if omp_threads is not None:
        controls["OMP_NUM_THREADS"] = omp_threads
    return controls


def _lock(*, digest: str = "a" * 64) -> DependencyLockV1:
    return DependencyLockV1(
        name="requirements.lock",
        sha256=digest,
        size_bytes=123,
    )


def _profile(
    *,
    python_version: str = "3.12.1",
    python_compiler: str = "GCC 13.2.0",
    numpy_version: str = "2.2.0",
    numpy_configuration: str = "Build Dependencies:\n  blas: openblas\n",
    scipy_version: str | None = None,
    cpu_count: int | None = 8,
    byte_order: str = "little",
    controls: dict[str, str | None] | None = None,
    lock: DependencyLockV1 | None = None,
    unrelated_distribution: bool = False,
) -> NumericalEnvironmentV1:
    distributions = [
        InstalledDistributionV1(name="numpy", version=numpy_version),
    ]
    if scipy_version is not None:
        distributions.append(
            InstalledDistributionV1(name="scipy", version=scipy_version)
        )
    if unrelated_distribution:
        distributions.append(
            InstalledDistributionV1(name="unrelated-package", version="9.9")
        )
    return NumericalEnvironmentV1(
        python_implementation="CPython",
        python_version=python_version,
        python_compiler=python_compiler,
        numpy_version=numpy_version,
        numpy_configuration_text=numpy_configuration,
        scipy_version=scipy_version,
        logical_cpu_count=cpu_count,
        byte_order=byte_order,
        execution_controls=_controls() if controls is None else controls,
        installed_distributions=tuple(sorted(distributions)),
        dependency_lock=lock,
    )


def test_exact_runtime_drift_can_preserve_compatibility_identity() -> None:
    pinned = _controls(fully_pinned=True)
    baseline = _profile(controls=pinned, lock=_lock())
    changed = _profile(
        python_version="3.12.9",
        python_compiler="Clang 19",
        cpu_count=64,
        controls=pinned,
        unrelated_distribution=True,
        lock=_lock(),
    )

    assert changed.profile_id != baseline.profile_id
    assert numerical_compatibility_id_v1(changed) == (
        numerical_compatibility_id_v1(baseline)
    )
    assert numerically_compatible_v1(baseline, changed)


@pytest.mark.parametrize(
    ("case", "changed"),
    (
        ("python-minor", {"python_version": "3.13.0"}),
        ("numpy", {"numpy_version": "2.3.0"}),
        ("numpy-configuration", {"numpy_configuration": "blas: mkl\n"}),
        ("scipy", {"scipy_version": "1.15.0"}),
        ("byte-order", {"byte_order": "big"}),
        ("execution-control", {"controls": _controls(omp_threads="2")}),
        ("dependency-lock", {"lock": _lock(digest="b" * 64)}),
    ),
)
def test_solver_relevant_drift_changes_compatibility_identity(
    case: str,
    changed: dict[str, object],
) -> None:
    assert case
    baseline = _profile(lock=_lock())
    candidate_arguments: dict[str, object] = {"lock": _lock()}
    candidate_arguments.update(changed)
    candidate = _profile(**candidate_arguments)  # type: ignore[arg-type]

    assert numerical_compatibility_id_v1(candidate) != (
        numerical_compatibility_id_v1(baseline)
    )
    assert not numerically_compatible_v1(baseline, candidate)


def test_unpinned_parallelism_binds_logical_cpu_count() -> None:
    baseline = _profile(cpu_count=8, lock=_lock())
    changed = _profile(cpu_count=64, lock=_lock())

    assert numerical_compatibility_id_v1(baseline) != (
        numerical_compatibility_id_v1(changed)
    )
    descriptor = numerical_compatibility_descriptor_v1(baseline)
    parallelism = cast(dict[str, Any], descriptor["implicit_parallelism"])
    assert parallelism == {
        "thread_counts_fully_pinned": False,
        "logical_cpu_count": 8,
    }


def test_partial_thread_pinning_still_binds_logical_cpu_count() -> None:
    controls = _controls(omp_threads="2")
    baseline = _profile(cpu_count=8, controls=controls, lock=_lock())
    changed = _profile(cpu_count=64, controls=controls, lock=_lock())

    assert not numerically_compatible_v1(baseline, changed)


def test_missing_cpu_count_requires_fully_pinned_thread_counts() -> None:
    with pytest.raises(ValueError, match="requires logical CPU count"):
        numerical_compatibility_descriptor_v1(
            _profile(cpu_count=None, lock=_lock())
        )

    descriptor = numerical_compatibility_descriptor_v1(
        _profile(
            cpu_count=None,
            controls=_controls(fully_pinned=True),
            lock=_lock(),
        )
    )
    parallelism = cast(dict[str, Any], descriptor["implicit_parallelism"])
    assert parallelism == {
        "thread_counts_fully_pinned": True,
        "logical_cpu_count": None,
    }


def test_lock_is_required_by_default_but_optional_for_diagnostics() -> None:
    profile = _profile()

    with pytest.raises(ValueError, match="requires a dependency lock"):
        numerical_compatibility_descriptor_v1(profile)

    descriptor = numerical_compatibility_descriptor_v1(
        profile,
        require_dependency_lock=False,
    )
    assert descriptor["dependency_lock"] is None
    assert (
        len(
            numerical_compatibility_id_v1(
                profile,
                require_dependency_lock=False,
            )
        )
        == 64
    )


def test_compatibility_descriptor_excludes_exact_only_state_when_pinned() -> None:
    descriptor = numerical_compatibility_descriptor_v1(
        _profile(
            python_compiler="different compiler",
            cpu_count=128,
            controls=_controls(fully_pinned=True),
            unrelated_distribution=True,
            lock=_lock(),
        )
    )

    assert "logical_cpu_count" not in descriptor
    assert "installed_distributions" not in descriptor
    python_record = cast(dict[str, Any], descriptor["python"])
    parallelism = cast(dict[str, Any], descriptor["implicit_parallelism"])
    assert "compiler" not in python_record
    assert python_record["major_minor"] == "3.12"
    assert parallelism["thread_counts_fully_pinned"] is True
    assert parallelism["logical_cpu_count"] is None


def test_record_binds_compatibility_to_the_exact_source_profile() -> None:
    profile = _profile(lock=_lock())
    record = numerical_compatibility_record_v1(profile)
    restored = validate_numerical_compatibility_record_v1(
        json.loads(json.dumps(record)),
        profile,
    )

    assert restored == record
    assert record["source_profile_id"] == profile.profile_id
    assert len(cast(str, record["numerical_compatibility_id"])) == 64


def test_record_tampering_and_wrong_source_profile_fail_closed() -> None:
    pinned = _controls(fully_pinned=True)
    profile = _profile(controls=pinned, lock=_lock())
    record = numerical_compatibility_record_v1(profile)
    tampered = json.loads(json.dumps(record))
    tampered["numerical_compatibility_id"] = "f" * 64

    with pytest.raises(ValueError, match="does not match"):
        validate_numerical_compatibility_record_v1(tampered, profile)

    compatible_but_not_exact = _profile(
        python_version="3.12.8",
        cpu_count=32,
        controls=pinned,
        unrelated_distribution=True,
        lock=_lock(),
    )
    assert numerically_compatible_v1(profile, compatible_but_not_exact)
    with pytest.raises(ValueError, match="does not match"):
        validate_numerical_compatibility_record_v1(
            record,
            compatible_but_not_exact,
        )


def test_record_fields_and_boolean_switch_are_strict() -> None:
    profile = _profile(lock=_lock())
    record = numerical_compatibility_record_v1(profile)
    record["unexpected"] = True

    with pytest.raises(ValueError, match="fields changed"):
        validate_numerical_compatibility_record_v1(record, profile)

    with pytest.raises(ValueError, match="literal boolean"):
        numerical_compatibility_descriptor_v1(
            profile,
            require_dependency_lock=cast(Any, 1),
        )


def test_python_version_must_expose_major_minor() -> None:
    profile = _profile(python_version="development", lock=_lock())

    with pytest.raises(ValueError, match="major.minor"):
        numerical_compatibility_descriptor_v1(profile)
