from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.numerical_environment_v1 import (
    NUMERICAL_ENVIRONMENT_SCHEMA,
    NUMERICAL_ENVIRONMENT_SCHEMA_VERSION,
    NUMERICAL_EXECUTION_CONTROL_NAMES,
    NumericalDependencyLockV1,
    NumericalEnvironmentProfileV1,
    capture_numerical_environment_profile,
    load_numerical_environment_profile,
    numerical_environment_runtime_binding,
    write_numerical_environment_profile,
)


def _controls(**updates: str | None) -> dict[str, str | None]:
    values = {name: None for name in NUMERICAL_EXECUTION_CONTROL_NAMES}
    values.update(updates)
    return values


def _profile(**updates: object) -> NumericalEnvironmentProfileV1:
    values: dict[str, object] = {
        "python_implementation": "CPython",
        "python_version": "3.12.4",
        "python_compiler": "GCC 13.2.0",
        "operating_system": "Linux-6.8.0-x86_64-with-glibc2.39",
        "machine": "x86_64",
        "processor": "x86_64",
        "byte_order": "little",
        "numpy_version": "2.1.0",
        "numpy_config_text": (
            "Build Dependencies:\n"
            "  blas:\n"
            "    name: scipy-openblas\n"
            "    version: 0.3.27"
        ),
        "execution_controls": _controls(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        ),
        "packages": {
            "numpy": "2.1.0",
            "scipy": "1.14.1",
        },
        "dependency_lock": NumericalDependencyLockV1(
            name="requirements.lock",
            sha256="1" * 64,
            size_bytes=137,
        ),
        "container_image_digest": f"sha256:{'2' * 64}",
    }
    values.update(updates)
    return NumericalEnvironmentProfileV1(**values)  # type: ignore[arg-type]


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_profile_identity_is_canonical_and_order_independent() -> None:
    first = _profile(packages={"scipy": "1.14.1", "numpy": "2.1.0"})
    second = _profile(packages={"numpy": "2.1.0", "scipy": "1.14.1"})

    assert first.profile_id == second.profile_id
    assert first.as_dict() == second.as_dict()
    assert list(first.packages) == ["numpy", "scipy"]
    assert first.as_dict()["schema_name"] == NUMERICAL_ENVIRONMENT_SCHEMA
    assert (
        first.as_dict()["schema_version"]
        == NUMERICAL_ENVIRONMENT_SCHEMA_VERSION
    )


@pytest.mark.parametrize(
    "replacement",
    [
        {"numpy_config_text": "blas=openblas\nlapack=openblas"},
        {"execution_controls": _controls(OMP_NUM_THREADS="2")},
        {"packages": {"numpy": "2.1.0", "scipy": "1.14.2"}},
        {
            "dependency_lock": NumericalDependencyLockV1(
                name="requirements.lock",
                sha256="3" * 64,
                size_bytes=137,
            )
        },
        {"container_image_digest": f"sha256:{'4' * 64}"},
    ],
)
def test_profile_identity_covers_numerical_inputs(
    replacement: dict[str, object],
) -> None:
    assert _profile().profile_id != _profile(**replacement).profile_id


def test_profile_detaches_and_freezes_mutable_inputs() -> None:
    controls = _controls(OMP_NUM_THREADS="1")
    packages = {"numpy": "2.1.0"}
    profile = _profile(
        execution_controls=controls,
        packages=packages,
    )

    controls["OMP_NUM_THREADS"] = "8"
    packages["numpy"] = "9.9.9"

    assert profile.execution_controls["OMP_NUM_THREADS"] == "1"
    assert profile.packages["numpy"] == "2.1.0"
    with pytest.raises(TypeError):
        profile.execution_controls["OMP_NUM_THREADS"] = "4"  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.packages["numpy"] = "2.2.0"  # type: ignore[index]


def test_round_trip_and_no_clobber_publication(tmp_path: Path) -> None:
    destination = tmp_path / "numerical-environment.json"
    profile = _profile()
    write_numerical_environment_profile(destination, profile)

    loaded = load_numerical_environment_profile(destination)
    assert loaded.as_dict() == profile.as_dict()

    with pytest.raises(FileExistsError):
        write_numerical_environment_profile(destination, _profile(machine="arm64"))

    replacement = _profile(machine="arm64")
    write_numerical_environment_profile(
        destination,
        replacement,
        overwrite=True,
    )
    assert load_numerical_environment_profile(destination).profile_id == (
        replacement.profile_id
    )


def test_loader_rejects_content_address_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "profile.json"
    payload = _profile().as_dict()
    payload["machine"] = "arm64"
    _write_payload(destination, payload)

    with pytest.raises(ValueError, match="digest does not match payload"):
        load_numerical_environment_profile(destination)


def test_loader_rejects_numpy_config_digest_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "profile.json"
    payload = _profile().as_dict()
    payload["numpy_config_sha256"] = "0" * 64
    _write_payload(destination, payload)

    with pytest.raises(ValueError, match="configuration digest"):
        load_numerical_environment_profile(destination)


def test_loader_rejects_unknown_missing_and_duplicate_fields(tmp_path: Path) -> None:
    destination = tmp_path / "profile.json"
    payload = _profile().as_dict()
    payload["unexpected"] = True
    _write_payload(destination, payload)
    with pytest.raises(ValueError, match="unknown"):
        load_numerical_environment_profile(destination)

    payload = _profile().as_dict()
    del payload["machine"]
    _write_payload(destination, payload)
    with pytest.raises(ValueError, match="missing"):
        load_numerical_environment_profile(destination)

    destination.write_text(
        '{"profile_id":"' + "0" * 64 + '","profile_id":"' + "1" * 64 + '"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_numerical_environment_profile(destination)


def test_loader_rejects_nonfinite_json(tmp_path: Path) -> None:
    destination = tmp_path / "profile.json"
    destination.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_numerical_environment_profile(destination)


def test_profile_validates_closed_controls_and_package_inventory() -> None:
    incomplete = _controls()
    del incomplete["OMP_NUM_THREADS"]
    with pytest.raises(ValueError, match="complete allowlist"):
        _profile(execution_controls=incomplete)

    unknown = _controls()
    unknown["SECRET_TOKEN"] = "not-recorded"
    with pytest.raises(ValueError, match="unknown"):
        _profile(execution_controls=unknown)

    with pytest.raises(ValueError, match="canonical normalized"):
        _profile(packages={"NumPy": "2.1.0"})
    with pytest.raises(ValueError, match="must include numpy"):
        _profile(packages={"scipy": "1.14.1"})
    with pytest.raises(ValueError, match="must match"):
        _profile(packages={"numpy": "2.0.0"})


def test_dependency_lock_and_container_digest_are_strict() -> None:
    with pytest.raises(ValueError, match="one file name"):
        NumericalDependencyLockV1(
            name="locks/requirements.lock",
            sha256="1" * 64,
            size_bytes=1,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        NumericalDependencyLockV1(
            name="requirements.lock",
            sha256="ABC",
            size_bytes=1,
        )
    with pytest.raises(ValueError, match="sha256"):
        _profile(container_image_digest="latest")


def test_capture_hashes_lock_and_collects_only_allowlisted_controls(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "requirements.lock"
    lock_bytes = b"numpy==2.1.0 --hash=sha256:example\n"
    lock.write_bytes(lock_bytes)
    profile = capture_numerical_environment_profile(
        dependency_lock=lock,
        package_names=("NumPy",),
        environment={
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "SECRET_TOKEN": "must-not-be-recorded",
        },
    )

    assert profile.dependency_lock is not None
    assert profile.dependency_lock.name == "requirements.lock"
    assert profile.dependency_lock.sha256 == hashlib.sha256(lock_bytes).hexdigest()
    assert profile.dependency_lock.size_bytes == len(lock_bytes)
    assert profile.execution_controls["OMP_NUM_THREADS"] == "1"
    assert profile.execution_controls["MKL_NUM_THREADS"] is None
    assert "SECRET_TOKEN" not in profile.execution_controls
    assert profile.packages == {"numpy": profile.numpy_version}
    assert profile.numpy_config_text
    assert profile.numpy_config_sha256 == hashlib.sha256(
        profile.numpy_config_text.encode("utf-8")
    ).hexdigest()


def test_capture_requires_lock_for_custom_lock_name() -> None:
    with pytest.raises(ValueError, match="requires dependency_lock"):
        capture_numerical_environment_profile(
            dependency_lock_name="requirements.lock",
            package_names=("numpy",),
        )


def test_runtime_binding_is_small_and_content_addressed() -> None:
    profile = _profile()
    binding = numerical_environment_runtime_binding(profile)

    assert binding == {
        "numerical_environment": {
            "schema_name": NUMERICAL_ENVIRONMENT_SCHEMA,
            "schema_version": NUMERICAL_ENVIRONMENT_SCHEMA_VERSION,
            "profile_id": profile.profile_id,
        }
    }


def test_numpy_config_text_is_canonicalized() -> None:
    profile = _profile(numpy_config_text="\r\nblas=openblas   \r\n\r\n")
    assert profile.numpy_config_text == "blas=openblas"
    assert replace(profile).profile_id == profile.profile_id
