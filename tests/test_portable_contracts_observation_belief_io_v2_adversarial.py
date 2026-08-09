from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin._observation_belief_io_v2 as helper
import bayesian_phystwin.observation_belief_io_v2 as io_v2
from bayesian_phystwin.observation_belief import save_observation_belief
from bayesian_phystwin.observation_belief_io_v2 import (
    ObservationBeliefIOLimitsV2,
    load_observation_belief_bounded_v2,
    save_observation_belief_atomic_v2,
)
from tests.test_portable_contracts_observation_belief_io_v2 import _belief


def _payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _write(path: Path, payload: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _descriptor(payload: dict[str, np.ndarray]) -> dict[str, object]:
    return json.loads(str(payload["descriptor_json"].item()))


def _replace_descriptor(
    payload: dict[str, np.ndarray],
    descriptor: object,
    *,
    encode: bool = False,
) -> None:
    text = descriptor if isinstance(descriptor, str) else json.dumps(descriptor)
    payload["descriptor_json"] = np.asarray(
        text.encode("utf-8") if encode else text
    )


def _headers(path: Path) -> object:
    with path.open("rb") as stream:
        return helper.preflight_archive(
            stream,
            limits=ObservationBeliefIOLimitsV2(),
        )


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_limits_reject_descriptor_budget_above_member_budget() -> None:
    with pytest.raises(ValueError, match="descriptor_bytes cannot exceed"):
        ObservationBeliefIOLimitsV2(
            maximum_uncompressed_bytes=10,
            maximum_member_bytes=10,
            maximum_descriptor_bytes=11,
        )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("{", "not valid JSON"),
        ('{"value":NaN}', "non-finite constant"),
        ("[]", "must be a JSON object"),
    ],
)
def test_strict_descriptor_json_rejects_noncanonical_inputs(
    text: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        helper.strict_descriptor_json(text)


def test_strict_descriptor_json_rejects_nonstring_input() -> None:
    with pytest.raises(ValueError, match="strict finite JSON"):
        helper.strict_descriptor_json(None)  # type: ignore[arg-type]


def test_validate_sha256_rejects_noncanonical_digest() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        helper.validate_sha256("ABC", name="digest")


def test_no_follow_fallback_rejects_symlink_and_missing_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    link = tmp_path / "link.bin"
    link.symlink_to(source)
    monkeypatch.delattr(helper.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(ValueError, match="symbolic link"):
        helper.ordinary_input_stream(link)
    with pytest.raises(ValueError, match="does not exist"):
        helper.ordinary_input_stream(tmp_path / "missing.bin")
    stream, metadata = helper.ordinary_input_stream(source)
    with stream:
        assert metadata.st_size == len(b"source")


def test_ordinary_input_stream_rejects_nonregular_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        helper.ordinary_input_stream(tmp_path)


class _HeaderArchive:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def open(self, _: object, __: str) -> io.BytesIO:
        return io.BytesIO(self.payload)


class _HeaderInfo:
    filename = "member.npy"

    def __init__(self, file_size: int) -> None:
        self.file_size = file_size


def _header_reader(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    consumed: int,
):
    def read(stream: io.BytesIO, **_: object) -> tuple[tuple[int, ...], bool, object]:
        stream.seek(consumed)
        return shape, False, dtype

    return read


def test_npy_header_version_two_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(helper.np.lib.format, "read_magic", lambda _: (2, 0))
    monkeypatch.setattr(
        helper.np.lib.format,
        "read_array_header_2_0",
        _header_reader((1,), np.dtype(np.float64), consumed=2),
    )
    result = helper._read_npy_header(
        _HeaderArchive(b"x" * 10),
        _HeaderInfo(10),
        limits=ObservationBeliefIOLimitsV2(),
    )
    assert result.payload_bytes == 8


@pytest.mark.parametrize(
    ("version", "shape", "dtype", "consumed", "size", "limits", "match"),
    [
        ((3, 0), (1,), np.dtype(np.float64), 2, 10, None, "invalid NPY header"),
        ((1, 0), (-1,), np.dtype(np.float64), 2, 2, None, "negative array"),
        ((1, 0), (1,), np.dtype(object), 2, 10, None, "object dtype"),
        (
            (1, 0),
            (2,),
            np.dtype(np.float64),
            2,
            18,
            ObservationBeliefIOLimitsV2(
                maximum_uncompressed_bytes=8,
                maximum_member_bytes=8,
                maximum_descriptor_bytes=8,
            ),
            "decoded member budget",
        ),
        ((1, 0), (1,), np.dtype(np.float64), 1, 10, None, "size disagrees"),
    ],
)
def test_npy_header_rejects_malformed_or_overbudget_member(
    version: tuple[int, int],
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    consumed: int,
    size: int,
    limits: ObservationBeliefIOLimitsV2 | None,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(helper.np.lib.format, "read_magic", lambda _: version)
    monkeypatch.setattr(
        helper.np.lib.format,
        "read_array_header_1_0",
        _header_reader(shape, dtype, consumed=consumed),
    )
    with pytest.raises(ValueError, match=match):
        helper._read_npy_header(
            _HeaderArchive(b"x" * max(size, consumed)),
            _HeaderInfo(size),
            limits=limits or ObservationBeliefIOLimitsV2(),
        )


def test_preflight_rejects_missing_and_extra_member(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    payload = _payload(path)
    payload.pop("frame_ids")
    payload["unexpected"] = np.asarray([1], dtype=np.int64)
    _write(path, payload)
    with pytest.raises(ValueError, match="members changed"):
        load_observation_belief_bounded_v2(path)


def test_preflight_rejects_unsupported_zip_compression(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    members = _members(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_BZIP2) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="unsupported ZIP compression"):
        load_observation_belief_bounded_v2(path)


def test_preflight_rejects_member_and_total_budgets(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    with zipfile.ZipFile(path, "r") as archive:
        largest = max(entry.file_size for entry in archive.infolist())
    with pytest.raises(ValueError, match="member budget"):
        load_observation_belief_bounded_v2(
            path,
            limits=ObservationBeliefIOLimitsV2(
                maximum_uncompressed_bytes=largest,
                maximum_member_bytes=largest - 1,
                maximum_descriptor_bytes=min(1024, largest - 1),
            ),
        )
    with pytest.raises(ValueError, match="uncompressed byte budget"):
        load_observation_belief_bounded_v2(
            path,
            limits=ObservationBeliefIOLimitsV2(
                maximum_uncompressed_bytes=largest,
                maximum_member_bytes=largest,
                maximum_descriptor_bytes=min(1024, largest),
            ),
        )


def test_preflight_rejects_bad_zip(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    path.write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="valid ZIP archive"):
        load_observation_belief_bounded_v2(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("descriptor_json", np.asarray(["{}"]), "scalar byte or Unicode"),
        ("mean_xyz_m", np.zeros((3, 3), dtype=np.float32), "exact dtype"),
        ("mean_xyz_m", np.zeros((3, 1), dtype=np.float64), r"shape \(N, 3\)"),
        (
            "low_rank_factor_m",
            np.zeros((3, 3), dtype=np.float64),
            r"shape \(N, 3, R\)",
        ),
        ("frame_ids", np.zeros((3, 1), dtype=np.int64), "must declare shape"),
        ("declared_frame_ids", np.zeros((1, 2), dtype=np.int64), "one dimension"),
        ("group_ids", np.zeros((1, 1), dtype=np.int64), "one dimension"),
    ],
)
def test_preflight_rejects_invalid_dtype_and_shape(
    field: str,
    value: np.ndarray,
    match: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"{field}.npz"
    save_observation_belief(path, _belief())
    payload = _payload(path)
    payload[field] = value
    _write(path, payload)
    with pytest.raises(ValueError, match=match):
        load_observation_belief_bounded_v2(path)


def test_preflight_rejects_descriptor_frame_and_group_budgets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    with pytest.raises(ValueError, match="descriptor_json exceeds"):
        load_observation_belief_bounded_v2(
            path,
            limits=ObservationBeliefIOLimitsV2(maximum_descriptor_bytes=1),
        )
    with pytest.raises(ValueError, match="declared frame count"):
        load_observation_belief_bounded_v2(
            path,
            limits=ObservationBeliefIOLimitsV2(maximum_declared_frame_count=1),
        )
    payload = _payload(path)
    payload["group_ids"] = np.asarray([0, 1], dtype=np.int64)
    payload["group_prior_nominal_probability"] = np.asarray([0.5, 0.5])
    payload["group_composite_weight"] = np.asarray([0.5, 0.5])
    _write(path, payload)
    with pytest.raises(ValueError, match="group count"):
        load_observation_belief_bounded_v2(
            path,
            limits=ObservationBeliefIOLimitsV2(maximum_group_count=1),
        )


def test_preflight_rejects_npy_member_with_trailing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    members = _members(path)
    members["frame_ids.npy"] += b"trailing"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="size disagrees"):
        load_observation_belief_bounded_v2(path)


def test_bytes_descriptor_and_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    source = _belief()
    save_observation_belief(path, source)
    payload = _payload(path)
    text = str(payload["descriptor_json"].item())
    payload["descriptor_json"] = np.asarray(text.encode("utf-8"))
    _write(path, payload)
    assert load_observation_belief_bounded_v2(path).artifact_id == source.artifact_id
    payload["descriptor_json"] = np.asarray(b"\xff")
    _write(path, payload)
    with pytest.raises(ValueError, match="valid UTF-8"):
        load_observation_belief_bounded_v2(path)


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"unexpected": True}, "fields changed"),
        ({"schema_name": "other.schema"}, "unsupported observation-belief schema"),
        ({"schema_version": True}, "schema_version changed type"),
        ({"schema_version": 99}, "unsupported observation-belief version"),
    ],
)
def test_loader_rejects_descriptor_contract_drift(
    update: dict[str, object],
    match: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    payload = _payload(path)
    descriptor = _descriptor(payload)
    descriptor.update(update)
    _replace_descriptor(payload, descriptor)
    _write(path, payload)
    with pytest.raises(ValueError, match=match):
        load_observation_belief_bounded_v2(path)


def test_post_preflight_array_checks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    headers = _headers(path)
    monkeypatch.setattr(io_v2, "preflight_archive", lambda *_a, **_k: headers)
    payload = _payload(path)
    payload["mean_xyz_m"] = payload["mean_xyz_m"].astype(np.float32)
    _write(path, payload)
    with pytest.raises(ValueError, match="exact dtype"):
        load_observation_belief_bounded_v2(path)
    payload = _payload(path)
    payload["mean_xyz_m"] = np.zeros((2, 3), dtype=np.float64)
    _write(path, payload)
    with pytest.raises(ValueError, match="shape changed"):
        load_observation_belief_bounded_v2(path)


def test_post_preflight_descriptor_checks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    headers = _headers(path)
    monkeypatch.setattr(io_v2, "preflight_archive", lambda *_a, **_k: headers)
    payload = _payload(path)
    payload["descriptor_json"] = np.asarray(["{}"])
    _write(path, payload)
    with pytest.raises(ValueError, match="scalar array"):
        load_observation_belief_bounded_v2(path)
    payload["descriptor_json"] = np.asarray(1, dtype=np.int64)
    _write(path, payload)
    with pytest.raises(ValueError, match="must contain a string"):
        load_observation_belief_bounded_v2(path)


def test_loader_wraps_nonvalidation_decode_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())

    def fail(*_: object, **__: object) -> object:
        raise OSError("injected decoder error")

    monkeypatch.setattr(io_v2.np, "load", fail)
    with pytest.raises(ValueError, match="cannot decode"):
        load_observation_belief_bounded_v2(path)


def test_loader_rejects_mutating_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    monkeypatch.setattr(io_v2, "snapshot_unchanged", lambda *_: False)
    with pytest.raises(ValueError, match="changed while"):
        load_observation_belief_bounded_v2(path)


@pytest.mark.parametrize("artifact_id", ["invalid", "0" * 64])
def test_loader_rejects_invalid_or_mismatched_artifact_id(
    artifact_id: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    payload = _payload(path)
    descriptor = _descriptor(payload)
    descriptor["artifact_id"] = artifact_id
    _replace_descriptor(payload, descriptor)
    _write(path, payload)
    with pytest.raises(ValueError, match="artifact"):
        load_observation_belief_bounded_v2(path)


def test_loader_and_writer_reject_wrong_limit_types(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    with pytest.raises(TypeError, match="limits"):
        load_observation_belief_bounded_v2(
            path,
            limits=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="limits"):
        save_observation_belief_atomic_v2(
            path,
            _belief(),
            limits=object(),  # type: ignore[arg-type]
        )


def test_writer_rejects_wrong_belief_and_directory_target(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="belief"):
        save_observation_belief_atomic_v2(
            tmp_path / "belief.npz",
            object(),  # type: ignore[arg-type]
        )
    target = tmp_path / "directory"
    target.mkdir()
    with pytest.raises(ValueError, match="must not be a directory"):
        save_observation_belief_atomic_v2(target, _belief())


def test_writer_rejects_verified_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DifferentIdentity:
        artifact_id = "0" * 64

    monkeypatch.setattr(
        io_v2,
        "load_observation_belief_bounded_v2",
        lambda *_a, **_k: _DifferentIdentity(),
    )
    target = tmp_path / "belief.npz"
    with pytest.raises(ValueError, match="changed identity"):
        save_observation_belief_atomic_v2(target, _belief())
    assert not target.exists()


def test_directory_sync_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(*_: object, **__: object) -> int:
        raise OSError("injected directory-open failure")

    monkeypatch.setattr(io_v2.os, "open", fail_open)
    io_v2._fsync_directory(tmp_path)
