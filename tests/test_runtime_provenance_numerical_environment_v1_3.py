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



def test_missing_embedded_profile_can_be_optional_or_required() -> None:
    assert validate_embedded_numerical_environment_v1({}) is None
    with pytest.raises(ValueError, match="requires a numerical profile"):
        validate_embedded_numerical_environment_v1({}, require_profile=True)
    with pytest.raises(ValueError, match="requires a numerical profile"):
        validate_embedded_numerical_environment_v1(
            {},
            require_dependency_lock=True,
        )


def test_capture_binds_real_inventory_controls_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "constraints.txt"
    lock_path.write_text("numpy==" + np.__version__ + "\n", encoding="utf-8")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")

    profile = capture_numerical_environment_v1(
        dependency_lock=lock_path,
        dependency_lock_name="evidence.lock",
    )
    restored = numerical_environment_from_dict(
        profile.as_dict(),
        require_dependency_lock=True,
    )

    inventory = {item.name: item.version for item in profile.installed_distributions}
    assert inventory["numpy"] == importlib.metadata.version("numpy")
    assert profile.numpy_version == np.__version__
    assert profile.execution_controls["OMP_NUM_THREADS"] == "7"
    assert profile.dependency_lock is not None
    assert profile.dependency_lock.name == "evidence.lock"
    assert restored.profile_id == profile.profile_id


def test_capture_rejects_incomplete_lock_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires dependency_lock"):
        capture_numerical_environment_v1(
            dependency_lock_name="requirements.lock"
        )
    with pytest.raises(FileNotFoundError):
        capture_numerical_environment_v1(
            dependency_lock=tmp_path / "missing.lock"
        )


def test_module_cli_captures_validates_and_does_not_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text("numpy==" + np.__version__ + "\n", encoding="utf-8")
    output = tmp_path / "runtime.json"

    assert (
        main(
            [
                "capture",
                str(output),
                "--dependency-lock",
                str(lock_path),
            ]
        )
        == 0
    )
    capture_summary = json.loads(capsys.readouterr().out)
    assert capture_summary["dependency_lock"]["name"] == "requirements.lock"

    runtime = json.loads(output.read_text(encoding="utf-8"))
    assert NUMERICAL_ENVIRONMENT_RUNTIME_KEY in runtime
    assert (
        main(
            [
                "validate",
                str(output),
                "--require-dependency-lock",
            ]
        )
        == 0
    )
    validate_summary = json.loads(capsys.readouterr().out)
    assert validate_summary["status"] == "valid"

    with pytest.raises(FileExistsError):
        main(["capture", str(output)])


def test_runtime_fragment_loader_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        numerical_environment._load_runtime_fragment(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        numerical_environment._load_runtime_fragment(nonfinite)


def test_serialized_payload_mutation_does_not_change_original_profile() -> None:
    profile = _profile(lock=_lock())
    payload = copy.deepcopy(profile.as_dict())
    execution_controls = cast(dict[str, Any], payload["execution_controls"])
    execution_controls["OMP_NUM_THREADS"] = "8"

    assert profile.execution_controls["OMP_NUM_THREADS"] is None
    assert profile.as_dict() != payload
