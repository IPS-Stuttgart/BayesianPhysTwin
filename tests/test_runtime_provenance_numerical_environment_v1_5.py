from __future__ import annotations

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
    embed_numerical_environment_v1,
    main,
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



class _FakeDistribution:
    def __init__(self, name: str | None, version: str) -> None:
        self.metadata = {} if name is None else {"Name": name}
        self.version = version


def test_distribution_capture_rejects_missing_and_conflicting_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        numerical_environment.importlib.metadata,
        "distributions",
        lambda: [_FakeDistribution(None, "1.0")],
    )
    with pytest.raises(ValueError, match="missing its Name"):
        numerical_environment._capture_installed_distributions()

    monkeypatch.setattr(
        numerical_environment.importlib.metadata,
        "distributions",
        lambda: [
            _FakeDistribution("Test_Package", "1.0"),
            _FakeDistribution("test-package", "2.0"),
        ],
    )
    with pytest.raises(ValueError, match="conflicting versions"):
        numerical_environment._capture_installed_distributions()

    monkeypatch.setattr(
        numerical_environment.importlib.metadata,
        "distributions",
        lambda: [_FakeDistribution("Test_Package", "1.0")],
    )
    assert numerical_environment._capture_installed_distributions() == (
        InstalledDistributionV1("test-package", "1.0"),
    )


def test_empty_numpy_show_config_is_recorded_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(numerical_environment.np, "show_config", lambda: None)

    assert numerical_environment._capture_numpy_configuration() == (
        "(no NumPy build configuration reported)\n"
    )


def test_serialized_profile_rejects_wrong_shapes_and_versions() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        numerical_environment_from_dict(cast(Any, []))

    payload = _profile().as_dict()
    payload.pop("byte_order")
    with pytest.raises(ValueError, match="missing"):
        numerical_environment_from_dict(payload)

    payload = _profile().as_dict()
    payload["schema_name"] = "other"
    with pytest.raises(ValueError, match="unsupported.*schema"):
        numerical_environment_from_dict(payload)

    payload = _profile().as_dict()
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported.*version"):
        numerical_environment_from_dict(payload)

    payload = _profile().as_dict()
    payload["installed_distributions"] = "not-an-array"
    with pytest.raises(ValueError, match="must be a JSON array"):
        numerical_environment_from_dict(payload)

    payload = _profile().as_dict()
    payload["dependency_lock"] = {"name": "lock"}
    with pytest.raises(ValueError, match="dependency lock"):
        numerical_environment_from_dict(payload)


def test_strict_runtime_copy_rejects_cycles_keys_and_nonjson_values() -> None:
    profile = _profile()
    assert embed_numerical_environment_v1(
        {"finite": 1.5, "sequence": (True, None, "text")},
        profile,
    )["sequence"] == [True, None, "text"]

    with pytest.raises(ValueError, match="literal string keys"):
        embed_numerical_environment_v1(cast(Any, {1: "value"}), profile)

    circular_mapping: dict[str, object] = {}
    circular_mapping["self"] = circular_mapping
    with pytest.raises(ValueError, match="circular mapping"):
        embed_numerical_environment_v1(circular_mapping, profile)

    circular_sequence: list[object] = []
    circular_sequence.append(circular_sequence)
    with pytest.raises(ValueError, match="circular sequence"):
        embed_numerical_environment_v1(
            {"sequence": circular_sequence},
            profile,
        )

    with pytest.raises(ValueError, match="non-JSON value"):
        embed_numerical_environment_v1({"value": {1, 2}}, profile)
    with pytest.raises(ValueError, match="must be a JSON object"):
        numerical_environment._strict_json_mapping([], name="runtime")
    with pytest.raises(ValueError, match="profile must be"):
        embed_numerical_environment_v1({}, cast(Any, {}))


def test_runtime_fragment_loader_rejects_io_shape_size_and_syntax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="cannot be read"):
        numerical_environment._load_runtime_fragment(tmp_path / "missing.json")

    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="cannot be read"):
        numerical_environment._load_runtime_fragment(invalid_utf8)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        numerical_environment._load_runtime_fragment(malformed)

    array = tmp_path / "array.json"
    array.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        numerical_environment._load_runtime_fragment(array)

    oversized = tmp_path / "oversized.json"
    oversized.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(numerical_environment, "_MAX_RUNTIME_FRAGMENT_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds 16 MiB"):
        numerical_environment._load_runtime_fragment(oversized)


def test_force_capture_replaces_an_existing_runtime_fragment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "runtime.json"
    output.write_text("old\n", encoding="utf-8")

    assert main(["capture", str(output), "--force"]) == 0
    capsys.readouterr()
    runtime = json.loads(output.read_text(encoding="utf-8"))

    assert NUMERICAL_ENVIRONMENT_RUNTIME_KEY in runtime
