"""Bounded result-side NPZ support loading for Deform360 geometric v4."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._portable_contracts import load_strict_json_object, sha256_digest
from .deform360_joint_sparse_geometric_common_v4 import (
    METRIC_ARRAY_MEMBERS,
    MOTIONCRAFTER_INTEGRITY_SCHEMA,
    PREDICTION_ALLOWED_MEMBERS,
    PREDICTION_REQUIRED_MEMBERS,
    _array_digest,
    _canonical_bytes,
    _confined_file,
    _integer,
    _literal,
    _require,
    _safe_relative,
    _sha256_file,
)


def _zip_members(path: Path) -> dict[str, zipfile.ZipInfo]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot inspect NPZ archive {path}") from error
    names = [info.filename for info in infos]
    _require(len(names) == len(set(names)), "NPZ archive repeats members")
    _require(
        all(not info.is_dir() for info in infos), "NPZ archive contains a directory"
    )
    return {info.filename: info for info in infos}


def _load_npy_member(
    path: Path, member: str, *, maximum_uncompressed_bytes: int
) -> np.ndarray:
    infos = _zip_members(path)
    _require(member in infos, f"NPZ member {member!r} is missing")
    info = infos[member]
    _require(
        info.file_size <= maximum_uncompressed_bytes,
        f"NPZ member {member!r} exceeds its bound",
    )
    try:
        with zipfile.ZipFile(path) as archive, archive.open(info) as stream:
            result = np.lib.format.read_array(stream, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot load NPZ member {member!r}") from error
    return np.asarray(result)


def _read_exact(stream: Any, count: int) -> bytes:
    blocks: list[bytes] = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        _require(bool(block), "truncated NPY member")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def _read_npy_header(
    stream: Any,
    version: tuple[int, int],
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    """Read a bounded NPY header without relying on NumPy private APIs."""

    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError(f"unsupported metric NPY format version {version!r}")


@dataclass(frozen=True, slots=True)
class _MetricFrame:
    rows: np.ndarray
    columns: np.ndarray
    points_world_m: np.ndarray


def _load_metric_sparse_frames(
    path: Path,
    *,
    causal_range: tuple[int, int],
) -> tuple[dict[int, _MetricFrame], tuple[int, int]]:
    infos = _zip_members(path)
    _require(
        set(infos) == METRIC_ARRAY_MEMBERS, "metric-prefix NPZ member roster changed"
    )
    start, stop = causal_range
    expected_frames: np.ndarray = np.arange(start, stop, dtype=np.int64)
    frames = _load_npy_member(
        path, "frame_indices.npy", maximum_uncompressed_bytes=1024 * 1024
    )
    _require(
        frames.dtype.kind in "iu" and np.array_equal(frames, expected_frames),
        "metric frame indices changed",
    )
    valid = _load_npy_member(
        path, "valid_mask.npy", maximum_uncompressed_bytes=256 * 1024 * 1024
    )
    _require(
        valid.dtype.kind == "b" and valid.ndim == 3 and len(valid) == len(frames),
        "metric valid mask changed",
    )
    height, width = map(int, valid.shape[1:])
    info = infos["points_world_m.npy"]
    maximum_points_bytes = len(frames) * height * width * 3 * 8 + 1024 * 1024
    _require(
        info.file_size <= maximum_points_bytes,
        "metric point member exceeds expected size",
    )
    output: dict[int, _MetricFrame] = {}
    try:
        with zipfile.ZipFile(path) as archive, archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran_order, dtype = _read_npy_header(stream, version)
            _require(
                shape == (len(frames), height, width, 3), "metric point shape changed"
            )
            _require(not fortran_order, "metric point array changed storage order")
            _require(
                np.dtype(dtype) == np.dtype(np.float64), "metric point dtype changed"
            )
            frame_bytes = height * width * 3 * np.dtype(np.float64).itemsize
            for local_index, frame in enumerate(frames):
                payload = _read_exact(stream, frame_bytes)
                points = np.frombuffer(payload, dtype=np.float64).reshape(
                    height, width, 3
                )
                mask = valid[local_index]
                rows, columns = np.nonzero(mask)
                selected = np.array(points[rows, columns], dtype=np.float64, copy=True)
                _require(
                    np.all(np.isfinite(selected)), "valid metric points are non-finite"
                )
                output[int(frame)] = _MetricFrame(
                    rows=np.asarray(rows, dtype=np.int64),
                    columns=np.asarray(columns, dtype=np.int64),
                    points_world_m=selected,
                )
            _require(not stream.read(1), "metric point member contains trailing data")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("cannot stream metric point member") from error
    return output, (height, width)


def _load_camera_center(
    path: Path, *, object_id: str, episode_id: int, camera_id: str
) -> tuple[np.ndarray, str]:
    value = cast(
        dict[str, Any], load_strict_json_object(path, label="metric calibration")
    )
    _require(
        value.get("schema") == "bayesian-phystwin.deform360-robot-metric-calibration",
        "metric calibration schema changed",
    )
    _require(value.get("schema_version") == 1, "metric calibration version changed")
    _require(
        value.get("object_id") == object_id and value.get("episode_id") == episode_id,
        "metric calibration object changed",
    )
    _require(value.get("camera_id") == camera_id, "metric calibration camera changed")
    matrix = np.asarray(value.get("camera_to_world"), dtype=np.float64)
    _require(
        matrix.shape == (4, 4) and np.all(np.isfinite(matrix)),
        "camera transform changed",
    )
    _require(
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0.0),
        "camera transform is not homogeneous",
    )
    rotation = matrix[:3, :3]
    _require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0),
        "camera rotation changed",
    )
    calibration_id = sha256_digest(value.get("calibration_id"), name="calibration_id")
    return np.asarray(matrix[:3, 3], dtype=np.float64), calibration_id


@dataclass(frozen=True, slots=True)
class _SupportWindow:
    window_id: str
    start_frame: int
    stop_frame: int
    frame_indices: np.ndarray
    valid_mask: np.ndarray
    support_digest: str
    source_member_sha256: str


def _load_prediction_support_windows(
    path: Path,
    *,
    causal_range: tuple[int, int],
    image_shape: tuple[int, int],
    expected_motioncrafter_revision: str,
) -> tuple[list[_SupportWindow], str]:
    manifest = cast(
        dict[str, Any], load_strict_json_object(path, label="prediction manifest")
    )
    _require(manifest.get("format_version") == 1, "prediction manifest version changed")
    _require(
        manifest.get("motioncrafter_commit") == expected_motioncrafter_revision,
        "prediction MotionCrafter revision changed",
    )
    raw_windows = manifest.get("overlap_windows")
    _require(
        isinstance(raw_windows, list) and bool(raw_windows),
        "prediction overlap windows changed",
    )
    integrity = manifest.get("artifact_integrity")
    _require(isinstance(integrity, Mapping), "prediction integrity block changed")
    integrity = cast(Mapping[str, Any], integrity)
    _require(
        integrity.get("schema") == MOTIONCRAFTER_INTEGRITY_SCHEMA,
        "prediction integrity schema changed",
    )
    run_spec = integrity.get("run_spec")
    _require(isinstance(run_spec, Mapping), "prediction run specification changed")
    run_spec_sha = sha256_digest(
        integrity.get("run_spec_sha256"), name="prediction run_spec_sha256"
    )
    _require(
        hashlib.sha256(_canonical_bytes(cast(Mapping[str, Any], run_spec))).hexdigest()
        == run_spec_sha,
        "prediction run specification SHA-256 changed",
    )
    raw_members = integrity.get("members")
    _require(
        isinstance(raw_members, list) and bool(raw_members),
        "prediction integrity member roster changed",
    )
    descriptors: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(cast(list[object], raw_members)):
        _require(
            isinstance(raw, Mapping), f"prediction integrity member {index} changed"
        )
        descriptor = cast(Mapping[str, Any], raw)
        relative = _safe_relative(descriptor.get("path"), name="prediction member path")
        _require(relative not in descriptors, "prediction member path repeats")
        sha256_digest(descriptor.get("sha256"), name="prediction member sha256")
        _integer(descriptor.get("bytes"), name="prediction member bytes")
        _literal(descriptor.get("kind"), name="prediction member kind")
        descriptors[relative] = descriptor

    start, stop = causal_range
    output: list[_SupportWindow] = []
    observed_ids: set[str] = set()
    for index, raw in enumerate(cast(list[object], raw_windows)):
        _require(isinstance(raw, Mapping), f"prediction window {index} changed")
        record = cast(Mapping[str, Any], raw)
        window_id = _literal(record.get("window_id"), name="window_id")
        _require(window_id not in observed_ids, "prediction window ID repeats")
        observed_ids.add(window_id)
        relative = _safe_relative(record.get("path"), name="prediction window path")
        descriptor_value = descriptors.get(relative)
        _require(
            descriptor_value is not None,
            "prediction window lacks integrity descriptor",
        )
        descriptor = cast(Mapping[str, Any], descriptor_value)
        _require(
            descriptor.get("kind") == "independently_decoded_overlap_window",
            "prediction window kind changed",
        )
        member_path = _confined_file(path.parent, relative, name="prediction window")
        _require(
            member_path.stat().st_size == descriptor.get("bytes"),
            "prediction window byte count changed",
        )
        _require(
            _sha256_file(member_path) == descriptor.get("sha256"),
            "prediction window SHA-256 changed",
        )
        window_start = _integer(record.get("start_frame"), name="window start")
        window_stop = _integer(record.get("stop_frame"), name="window stop", minimum=1)
        _require(
            start <= window_start < window_stop <= stop,
            "prediction window crosses causal range",
        )
        infos = _zip_members(member_path)
        _require(
            PREDICTION_REQUIRED_MEMBERS <= set(infos),
            "prediction support members are missing",
        )
        _require(
            set(infos) <= PREDICTION_ALLOWED_MEMBERS,
            "prediction window member roster changed",
        )
        frame_indices = _load_npy_member(
            member_path, "frame_indices.npy", maximum_uncompressed_bytes=1024 * 1024
        )
        expected_frames: np.ndarray = np.arange(
            window_start, window_stop, dtype=np.int64
        )
        _require(
            frame_indices.dtype.kind in "iu"
            and np.array_equal(frame_indices, expected_frames),
            "prediction support frames changed",
        )
        valid = _load_npy_member(
            member_path, "valid_mask.npy", maximum_uncompressed_bytes=256 * 1024 * 1024
        )
        _require(
            valid.dtype.kind == "b"
            and valid.shape == (len(expected_frames), *image_shape),
            "prediction support shape changed",
        )
        stored_window_id = _load_npy_member(
            member_path, "window_id.npy", maximum_uncompressed_bytes=1024 * 1024
        )
        _require(
            stored_window_id.shape == () and str(stored_window_id.item()) == window_id,
            "prediction support window ID changed",
        )
        support_digest = hashlib.sha256(
            _array_digest(np.asarray(frame_indices, dtype=np.int64)).encode("ascii")
            + _array_digest(np.asarray(valid, dtype=np.bool_)).encode("ascii")
        ).hexdigest()
        output.append(
            _SupportWindow(
                window_id=window_id,
                start_frame=window_start,
                stop_frame=window_stop,
                frame_indices=np.asarray(frame_indices, dtype=np.int64),
                valid_mask=np.asarray(valid, dtype=np.bool_),
                support_digest=support_digest,
                source_member_sha256=cast(str, descriptor["sha256"]),
            )
        )
    output.sort(key=lambda item: (item.start_frame, item.stop_frame, item.window_id))
    _require(
        output[0].start_frame == start, "prediction windows miss causal prefix start"
    )
    covered_frames = {
        int(frame)
        for window in output
        for frame in np.asarray(window.frame_indices, dtype=np.int64)
    }
    _require(
        covered_frames == set(range(start, stop)),
        "prediction windows do not cover the complete causal range",
    )
    return output, run_spec_sha
