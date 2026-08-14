from __future__ import annotations

import json
import stat
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.structured_point_covariance_io as covariance_io
from bayesian_phystwin.structured_point_covariance import (
    StructuredPointCovarianceV1,
)
from bayesian_phystwin.structured_point_covariance_io import (
    StructuredPointCovarianceIOLimitsV1,
    load_structured_point_covariance,
    write_structured_point_covariance,
)

_SOURCE_ID = "1" * 64


def _artifact() -> StructuredPointCovarianceV1:
    local = np.repeat(
        (np.eye(3, dtype=np.float64) * 0.01)[None, :, :],
        2,
        axis=0,
    )
    return StructuredPointCovarianceV1(
        point_ids=("point-0", "point-1"),
        local_covariance_m2=local,
        shared_factors_m={
            "gauge": np.ones((2, 3, 1), dtype=np.float64) / 100.0,
            "process": np.ones((2, 3, 1), dtype=np.float64) / 200.0,
        },
        coordinate_frame="world",
        source_artifact_id=_SOURCE_ID,
        metadata={"purpose": "adversarial-archive-coverage"},
    )


def _payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _write_payload(path: Path, payload: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _descriptor(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    value = json.loads(str(payload["descriptor_json"].item()))
    assert isinstance(value, dict)
    return value


def _replace_descriptor(
    payload: dict[str, np.ndarray],
    descriptor: dict[str, Any],
) -> None:
    payload["descriptor_json"] = np.asarray(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _rewrite_descriptor(
    path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = _payload(path)
    descriptor = _descriptor(payload)
    mutate(descriptor)
    _replace_descriptor(payload, descriptor)
    _write_payload(path, payload)


def test_private_scalar_and_descriptor_validators_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite number"):
        StructuredPointCovarianceIOLimitsV1(
            maximum_compression_ratio=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="valid JSON"):
        covariance_io._strict_json_object("{")
    with pytest.raises(ValueError, match="root must be an object"):
        covariance_io._strict_json_object("[]")
    with pytest.raises(ValueError, match="fields changed"):
        covariance_io._exact_fields(
            {"actual": 1},
            expected=frozenset({"expected"}),
            name="fixture",
        )
    with pytest.raises(ValueError, match="must be an integer"):
        covariance_io._genuine_integer(True, name="fixture")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        covariance_io._lower_sha256("short", name="fixture")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        covariance_io._lower_sha256("G" * 64, name="fixture")
    with pytest.raises(ValueError, match="must be an object"):
        covariance_io._component_names({"shared_factors_m": []})
    with pytest.raises(ValueError, match="must be sorted"):
        covariance_io._component_names(
            {"shared_factors_m": {"process": {}, "gauge": {}}}
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (np.asarray(["{}"]), "scalar array"),
        (np.asarray(b"\xff"), "valid UTF-8"),
        (np.asarray(7), "contain a string"),
        (np.asarray("{"), "valid JSON"),
        (np.asarray("[]"), "root must be an object"),
    ],
)
def test_descriptor_member_container_is_strict(
    tmp_path: Path,
    replacement: np.ndarray,
    message: str,
) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())
    payload = _payload(path)
    payload["descriptor_json"] = replacement
    _write_payload(path, payload)

    with pytest.raises(ValueError, match=message):
        load_structured_point_covariance(path)


def test_archive_descriptor_fields_and_identities_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"

    mutations: tuple[tuple[Callable[[dict[str, Any]], None], str], ...] = (
        (lambda value: value.update(extra=True), "fields changed"),
        (lambda value: value.__setitem__("schema", "wrong"), "schema changed"),
        (
            lambda value: value.__setitem__("schema_version", True),
            "must be an integer",
        ),
        (
            lambda value: value.__setitem__("semantics", "wrong"),
            "semantics changed",
        ),
        (
            lambda value: value.__setitem__("covariance_descriptor", []),
            "must be an object",
        ),
        (
            lambda value: value.__setitem__("artifact_id", "short"),
            "lowercase SHA-256",
        ),
        (
            lambda value: value.__setitem__("artifact_id", "G" * 64),
            "lowercase SHA-256",
        ),
    )
    for mutate, message in mutations:
        write_structured_point_covariance(path, _artifact(), overwrite=path.exists())
        _rewrite_descriptor(path, mutate)
        with pytest.raises(ValueError, match=message):
            load_structured_point_covariance(path)


def test_covariance_descriptor_fields_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"

    def nested(key: str, value: object) -> Callable[[dict[str, Any]], None]:
        def mutate(descriptor: dict[str, Any]) -> None:
            covariance = descriptor["covariance_descriptor"]
            assert isinstance(covariance, dict)
            covariance[key] = value

        return mutate

    mutations: tuple[tuple[Callable[[dict[str, Any]], None], str], ...] = (
        (nested("schema", "wrong"), "covariance schema changed"),
        (nested("schema_version", True), "must be an integer"),
        (nested("semantics", "wrong"), "covariance semantics changed"),
        (nested("shared_factors_m", []), "must be an object"),
        (nested("point_ids", "wrong"), "point_ids must be"),
        (nested("metadata", []), "metadata must be an object"),
        (nested("calibration_artifact_id", 7), "string or null"),
    )
    for mutate, message in mutations:
        write_structured_point_covariance(path, _artifact(), overwrite=path.exists())
        _rewrite_descriptor(path, mutate)
        with pytest.raises(ValueError, match=message):
            load_structured_point_covariance(path)


def test_zip_preflight_rejects_empty_duplicate_and_noncanonical_members(
    tmp_path: Path,
) -> None:
    path = tmp_path / "covariance.npz"
    with zipfile.ZipFile(path, "w"):
        pass
    with pytest.raises(ValueError, match="archive is empty"):
        load_structured_point_covariance(path)

    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("descriptor_json.npy", b"x")
            archive.writestr("descriptor_json.npy", b"x")
            archive.writestr("local_covariance_m2.npy", b"x")
    with pytest.raises(ValueError, match="duplicate members"):
        load_structured_point_covariance(path)

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("folder/", b"")
    with pytest.raises(ValueError, match="invalid member"):
        load_structured_point_covariance(path)

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested/descriptor_json.npy", b"x")
        archive.writestr("local_covariance_m2.npy", b"x")
    with pytest.raises(ValueError, match="not canonical"):
        load_structured_point_covariance(path)

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("descriptor_json.npy", b"x")
    with pytest.raises(ValueError, match="lacks required members"):
        load_structured_point_covariance(path)


def test_additional_archive_resource_budgets_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())

    with pytest.raises(ValueError, match="uncompressed byte budget"):
        load_structured_point_covariance(
            path,
            limits=StructuredPointCovarianceIOLimitsV1(maximum_uncompressed_bytes=1),
        )
    with pytest.raises(ValueError, match="compression-ratio budget"):
        load_structured_point_covariance(
            path,
            limits=StructuredPointCovarianceIOLimitsV1(maximum_compression_ratio=1.0),
        )


def test_missing_and_open_failure_paths_are_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="cannot inspect"):
        load_structured_point_covariance(tmp_path / "missing.npz")

    path = tmp_path / "covariance.npz"
    path.write_bytes(b"x")

    def fail_open(*args: object, **kwargs: object) -> int:
        raise OSError("denied")

    monkeypatch.setattr(covariance_io.os, "open", fail_open)
    with pytest.raises(ValueError, match="cannot open"):
        covariance_io._ordinary_input(path)


def test_opened_archive_must_remain_regular_and_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "covariance.npz"
    path.write_bytes(b"x")
    before = path.lstat()

    monkeypatch.setattr(
        covariance_io.os,
        "fstat",
        lambda descriptor: SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_dev=before.st_dev,
            st_ino=before.st_ino,
        ),
    )
    with pytest.raises(ValueError, match="must remain an ordinary file"):
        covariance_io._ordinary_input(path)

    monkeypatch.setattr(
        covariance_io.os,
        "fstat",
        lambda descriptor: SimpleNamespace(
            st_mode=before.st_mode,
            st_dev=before.st_dev,
            st_ino=before.st_ino + 1,
        ),
    )
    with pytest.raises(ValueError, match="changed before opening"):
        covariance_io._ordinary_input(path)


def test_decode_and_snapshot_failures_are_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())

    def fail_load(*args: object, **kwargs: object) -> object:
        raise KeyError("missing")

    monkeypatch.setattr(covariance_io.np, "load", fail_load)
    with pytest.raises(ValueError, match="cannot decode"):
        load_structured_point_covariance(path)

    monkeypatch.undo()
    monkeypatch.setattr(
        covariance_io, "_snapshot_unchanged", lambda first, second: False
    )
    with pytest.raises(ValueError, match="changed while being read"):
        load_structured_point_covariance(path)


def test_reconstruction_and_publication_verification_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())

    class MismatchedDescriptor:
        def descriptor(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(
        covariance_io,
        "StructuredPointCovarianceV1",
        lambda **kwargs: MismatchedDescriptor(),
    )
    with pytest.raises(ValueError, match="changed during reconstruction"):
        load_structured_point_covariance(path)

    monkeypatch.undo()
    monkeypatch.setattr(
        covariance_io,
        "load_structured_point_covariance",
        lambda *args, **kwargs: SimpleNamespace(artifact_id="0" * 64),
    )
    with pytest.raises(ValueError, match="changed identity"):
        write_structured_point_covariance(
            tmp_path / "publication.npz",
            _artifact(),
        )


def test_directory_fsync_failures_do_not_break_completed_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(*args: object, **kwargs: object) -> int:
        raise OSError("unsupported")

    monkeypatch.setattr(covariance_io.os, "open", fail_open)
    covariance_io._fsync_directory(tmp_path)

    monkeypatch.undo()

    def fail_fsync(descriptor: int) -> None:
        raise OSError("unsupported")

    monkeypatch.setattr(covariance_io.os, "fsync", fail_fsync)
    covariance_io._fsync_directory(tmp_path)


def test_writer_limit_type_is_not_coerced(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="limits"):
        write_structured_point_covariance(
            tmp_path / "covariance.npz",
            _artifact(),
            limits=object(),  # type: ignore[arg-type]
        )
