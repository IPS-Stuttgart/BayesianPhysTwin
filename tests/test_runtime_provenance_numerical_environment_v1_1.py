from __future__ import annotations

import copy
import importlib.metadata
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.numerical_environment_v1 as numerical_environment
from bayesian_phystwin.numerical_environment_v1 import (
    NUMERICAL_ENVIRONMENT_RUNTIME_KEY,
    DependencyLockV1,
    InstalledDistributionV1,
    NumericalEnvironmentV1,
    capture_numerical_environment_v1,
    embed_numerical_environment_v1,
    main,
    numerical_environment_from_dict,
    validate_embedded_numerical_environment_v1,
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



def test_profile_round_trip_is_canonical_and_content_addressed() -> None:
    profile = _profile(lock=_lock())

    restored = numerical_environment_from_dict(
        json.loads(json.dumps(profile.as_dict()))
    )

    assert restored == profile
    assert restored.profile_id == profile.profile_id
    assert profile.numpy_configuration_text == (
        "Build Dependencies:\n  blas: test\n"
    )
    assert len(profile.numpy_configuration_sha256) == 64
    assert len(profile.installed_distributions_sha256) == 64


@pytest.mark.parametrize(
    "mutation",
    (
        "configuration",
        "execution-control",
        "dependency-lock",
    ),
)
def test_profile_identity_changes_for_numerically_relevant_state(
    mutation: str,
) -> None:
    baseline = _profile(lock=_lock())
    controls = _controls()
    configuration = baseline.numpy_configuration_text
    lock = baseline.dependency_lock
    if mutation == "configuration":
        configuration = configuration.replace("test", "other")
    elif mutation == "execution-control":
        controls["OMP_NUM_THREADS"] = "2"
    elif mutation == "dependency-lock":
        lock = _lock(digest="b" * 64)
    else:
        raise AssertionError("unknown test mutation")

    changed = _profile(
        configuration=configuration,
        controls=controls,
        lock=lock,
    )

    assert changed.profile_id != baseline.profile_id


def test_tampered_numpy_configuration_digest_is_rejected() -> None:
    payload = _profile().as_dict()
    numpy_record = cast(dict[str, Any], payload["numpy"])
    numpy_record["configuration_text"] = "different\n"

    with pytest.raises(ValueError, match="configuration digest"):
        numerical_environment_from_dict(payload)


def test_tampered_distribution_digest_is_rejected() -> None:
    payload = _profile().as_dict()
    distributions = cast(list[dict[str, str]], payload["installed_distributions"])
    distributions.append({"name": "test-package", "version": "1.0"})

    with pytest.raises(ValueError, match="installed-distribution digest"):
        numerical_environment_from_dict(payload)


def test_tampered_profile_identity_is_rejected() -> None:
    payload = _profile().as_dict()
    payload["profile_id"] = "f" * 64

    with pytest.raises(ValueError, match="profile ID"):
        numerical_environment_from_dict(payload)


def test_schema_and_records_are_closed_and_strictly_typed() -> None:
    payload = _profile().as_dict()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema version"):
        numerical_environment_from_dict(payload)

    payload = _profile().as_dict()
    payload["unexpected"] = "value"
    with pytest.raises(ValueError, match="unknown"):
        numerical_environment_from_dict(payload)

    payload = _profile().as_dict()
    python_record = cast(dict[str, Any], payload["python"])
    python_record["unexpected"] = "value"
    with pytest.raises(ValueError, match="Python record"):
        numerical_environment_from_dict(payload)


def test_execution_controls_require_the_complete_registered_set() -> None:
    controls = _controls()
    controls.pop("OMP_NUM_THREADS")
    with pytest.raises(ValueError, match="complete registered set"):
        _profile(controls=controls)

    payload = _profile().as_dict()
    execution_controls = cast(dict[str, Any], payload["execution_controls"])
    execution_controls["OMP_NUM_THREADS"] = 1
    with pytest.raises(ValueError, match="must be text or null"):
        numerical_environment_from_dict(payload)


def test_execution_controls_are_detached_and_scalar() -> None:
    controls = _controls()
    profile = _profile(controls=controls)
    controls["OMP_NUM_THREADS"] = "99"

    assert profile.execution_controls["OMP_NUM_THREADS"] is None

    invalid = _controls()
    invalid["OMP_NUM_THREADS"] = "1\n2"
    with pytest.raises(ValueError, match="control character"):
        _profile(controls=invalid)
