#!/usr/bin/env python3
"""Probe a locked Deform360 visual-hull cohort without reading point values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

PROBE_SCHEMA = "bayesian-phystwin/deform360-source-hull-probe-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-source-hull-contract-probe-protocol-v1"
_REQUIRED_MEMBERS = (
    "frame_indices.npy",
    "point_offsets.npy",
    "points_world_m.npy",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_lower_hex_identity(value: object, *, lengths: tuple[int, ...]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _signed_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_probe_sha256", None)
    payload.pop("probe_sha256", None)
    payload.pop("repository_revision", None)
    payload.pop("dataset_root", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def load_probe_protocol(path: Path) -> dict[str, Any]:
    """Load and validate the content-addressed source-hull probe lock."""

    payload = _load_json(path.resolve())
    _require(payload.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    _require(payload.get("schema_version") == 1, "unsupported probe protocol version")
    config = payload.get("config")
    _require(isinstance(config, dict), "probe protocol lacks config")
    expected = payload.get("config_sha256")
    _require(
        isinstance(expected, str)
        and len(expected) == 64
        and hashlib.sha256(_canonical_bytes(config)).hexdigest() == expected,
        "probe protocol config checksum changed",
    )
    _require(
        config.get("protocol_id") == "deform360-source-hull-contract-probe-v1",
        "unexpected probe protocol identity",
    )
    _require(
        config.get("status") == "locked-before-source-hull-payload-metadata-access",
        "probe protocol is not locked before payload metadata access",
    )
    cohort = config.get("cohort")
    _require(isinstance(cohort, dict), "probe cohort is missing")
    entries = cohort.get("entries")
    _require(isinstance(entries, list) and entries, "probe cohort is empty")
    _require(
        cohort.get("reserved_target_object_count") == 0,
        "probe cohort includes reserved targets",
    )
    _require(
        cohort.get("unit_of_replication") == "physical object",
        "probe unit of replication changed",
    )
    paths: list[str] = []
    objects: set[str] = set()
    for record in entries:
        _require(isinstance(record, dict), "probe cohort record must be an object")
        _require(
            record.get("classification") == "prior_open_or_reserved",
            "probe cohort contains a non-source object",
        )
        _require(
            record.get("representation") == "packed_visual_hulls",
            "probe cohort representation changed",
        )
        object_id = record.get("object_id")
        episode_id = record.get("episode_id")
        relative_path = record.get("relative_path")
        _require(
            isinstance(object_id, str) and object_id,
            "probe object_id must be a nonempty string",
        )
        _require(
            isinstance(episode_id, int) and not isinstance(episode_id, bool),
            "probe episode_id must be an integer",
        )
        _require(
            isinstance(relative_path, str)
            and relative_path.endswith("/sampled_hulls.npz"),
            "probe path must name sampled_hulls.npz",
        )
        path_parts = Path(relative_path).parts
        _require(
            object_id in path_parts,
            "probe path does not contain its object identity",
        )
        _require(
            f"episode_{episode_id:04d}" in path_parts,
            "probe path does not contain its episode identity",
        )
        paths.append(relative_path)
        objects.add(object_id)
    _require(len(paths) == len(set(paths)), "probe cohort paths must be unique")
    _require(
        len(entries) == cohort.get("episode_count"),
        "probe episode count changed",
    )
    _require(
        len(objects) == cohort.get("object_count"),
        "probe object count changed",
    )
    probe = config.get("probe")
    _require(isinstance(probe, dict), "probe contract is missing")
    required = tuple(probe.get("required_members", ()))
    _require(required == _REQUIRED_MEMBERS, "probe required-member contract changed")
    _require(
        probe.get("minimum_point_count_per_frame") == 1,
        "probe minimum point count changed",
    )
    source_inventory = config.get("source_inventory")
    _require(isinstance(source_inventory, dict), "source inventory binding is missing")
    for field in (
        "content_inventory_sha256",
        "inventory_sha256",
        "workflow_artifact_sha256",
    ):
        _require(
            _is_lower_hex_identity(source_inventory.get(field), lengths=(64,)),
            f"source inventory {field} is invalid",
        )
    for field in ("product_head_sha", "evaluated_merge_sha"):
        _require(
            _is_lower_hex_identity(source_inventory.get(field), lengths=(40, 64)),
            f"source inventory {field} is invalid",
        )
    for field in ("workflow_run_id", "workflow_artifact_id"):
        identity = source_inventory.get(field)
        _require(
            isinstance(identity, int)
            and not isinstance(identity, bool)
            and identity > 0,
            f"source inventory {field} is invalid",
        )
    return payload


def _read_npy_header(stream: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NumPy member version: {version}")
    return tuple(int(value) for value in shape), bool(fortran_order), np.dtype(dtype)


def _load_integer_member(
    archive: zipfile.ZipFile,
    name: str,
) -> np.ndarray:
    with archive.open(name, "r") as stream:
        value = np.load(stream, allow_pickle=False)
    _require(
        isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer),
        f"{name} must contain an integer array",
    )
    return np.asarray(value, dtype=np.int64)


def _probe_archive(
    path: Path,
    *,
    object_id: str,
    episode_id: int,
    relative_path: str,
) -> dict[str, Any]:
    _require(path.is_file(), f"locked hull archive is missing: {relative_path}")
    with zipfile.ZipFile(path, "r") as archive:
        raw_members = archive.namelist()
        _require(
            len(raw_members) == len(set(raw_members)),
            f"locked hull archive contains duplicate members: {relative_path}",
        )
        members = tuple(sorted(raw_members))
        _require(
            all(name in members for name in _REQUIRED_MEMBERS),
            f"locked hull archive lacks required members: {relative_path}",
        )
        frame_indices = _load_integer_member(archive, "frame_indices.npy")
        point_offsets = _load_integer_member(archive, "point_offsets.npy")
        with archive.open("points_world_m.npy", "r") as stream:
            points_shape, fortran_order, points_dtype = _read_npy_header(stream)

    _require(frame_indices.ndim == 1, "frame_indices must be one-dimensional")
    _require(len(frame_indices) >= 1, "frame_indices must contain at least one row")
    _require(frame_indices[0] >= 0, "frame_indices must be nonnegative")
    frame_differences = np.diff(frame_indices)
    _require(
        np.all(frame_differences > 0),
        "frame_indices must be strictly increasing",
    )
    _require(point_offsets.ndim == 1, "point_offsets must be one-dimensional")
    _require(
        point_offsets.shape == (len(frame_indices) + 1,),
        "point_offsets length must equal frame count plus one",
    )
    _require(point_offsets[0] == 0, "point_offsets must start at zero")
    _require(
        np.all(np.diff(point_offsets) > 0),
        "point_offsets must be strictly increasing",
    )
    _require(
        len(points_shape) == 2 and points_shape[1] == 3,
        "points_world_m must have shape (N, 3)",
    )
    _require(
        np.issubdtype(points_dtype, np.floating),
        "points_world_m must use a floating dtype",
    )
    _require(
        int(point_offsets[-1]) == points_shape[0],
        "point_offsets final value differs from points_world_m row count",
    )
    point_counts = np.diff(point_offsets)
    _require(
        np.all(point_counts >= 1),
        "every sampled visual hull must contain at least one point",
    )
    unique_strides, stride_counts = np.unique(frame_differences, return_counts=True)
    transition_count = len(frame_differences)
    return {
        "object_id": object_id,
        "episode_id": episode_id,
        "relative_path": relative_path,
        "archive_sha256": _file_sha256(path),
        "archive_size_bytes": int(path.stat().st_size),
        "archive_members": list(members),
        "frame_count": int(len(frame_indices)),
        "transition_count": int(transition_count),
        "frame_start": int(frame_indices[0]),
        "frame_stop_inclusive": int(frame_indices[-1]),
        "frame_indices": frame_indices.astype(int).tolist(),
        "frame_stride_counts": {
            str(int(stride)): int(count)
            for stride, count in zip(unique_strides, stride_counts, strict=True)
        },
        "stride_observable": transition_count > 0,
        "constant_frame_stride": len(unique_strides) == 1,
        "prediction_eligible": len(frame_indices) >= 3,
        "prediction_ineligibility_reason": (
            None
            if len(frame_indices) >= 3
            else "rolling one-step evaluation requires at least three sampled hulls"
        ),
        "points_world_m_header": {
            "shape": list(points_shape),
            "dtype": points_dtype.str,
            "fortran_order": fortran_order,
            "coordinate_values_decoded": False,
        },
        "point_count_minimum": int(np.min(point_counts)),
        "point_count_median": float(np.median(point_counts)),
        "point_count_maximum": int(np.max(point_counts)),
    }


def probe_locked_source_hulls(
    data_root: Path,
    *,
    protocol_path: Path,
    revision: str | None = None,
) -> dict[str, Any]:
    """Probe locked archive contracts and cadence without reading geometry values."""

    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Deform360 data root is missing: {root}")
    protocol_file = protocol_path.expanduser().resolve()
    protocol = load_probe_protocol(protocol_file)
    config = protocol["config"]
    records: list[dict[str, Any]] = []
    for entry in config["cohort"]["entries"]:
        relative_path = str(entry["relative_path"])
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("locked hull path escapes the dataset root") from error
        records.append(
            _probe_archive(
                candidate,
                object_id=str(entry["object_id"]),
                episode_id=int(entry["episode_id"]),
                relative_path=relative_path,
            )
        )

    stride_records = [record for record in records if record["stride_observable"]]
    all_constant = all(record["constant_frame_stride"] for record in stride_records)
    object_counts = Counter(record["object_id"] for record in records)
    constant_stride_counts: Counter[str] = Counter()
    irregular_count = 0
    no_transition_count = 0
    for record in records:
        if not record["stride_observable"]:
            no_transition_count += 1
        elif record["constant_frame_stride"]:
            stride = next(iter(record["frame_stride_counts"]))
            constant_stride_counts[stride] += 1
        else:
            irregular_count += 1
    prediction_ineligible = [
        str(record["relative_path"])
        for record in records
        if not record["prediction_eligible"]
    ]
    result: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": protocol["config_sha256"],
        "repository_revision": revision,
        "dataset_root": str(root),
        "information_boundary": {
            "archive_metadata_opened": True,
            "complete_archive_bytes_hashed": True,
            "points_world_m_coordinate_values_decoded": False,
            "model_prediction_run": False,
            "score_bearing_outcome_computed": False,
            "reserved_target_outcomes_opened": False,
        },
        "object_count": len(object_counts),
        "episode_count": len(records),
        "episodes_per_object": dict(sorted(object_counts.items())),
        "prediction_eligible_episode_count": len(records) - len(prediction_ineligible),
        "prediction_ineligible_episode_count": len(prediction_ineligible),
        "prediction_ineligible_archives": prediction_ineligible,
        "all_stride_observable_archives_constant_frame_stride": all_constant,
        "constant_frame_stride_counts": dict(sorted(constant_stride_counts.items())),
        "irregular_frame_stride_archive_count": irregular_count,
        "no_transition_archive_count": no_transition_count,
        "frame_stride_patterns": dict(
            sorted(
                Counter(
                    json.dumps(record["frame_stride_counts"], sort_keys=True)
                    for record in records
                ).items()
            )
        ),
        "archives": records,
    }
    result["content_probe_sha256"] = _content_sha256(result)
    result["probe_sha256"] = _signed_sha256(result, "probe_sha256")
    return result


def write_probe(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = probe_locked_source_hulls(
        args.data_root,
        protocol_path=args.protocol,
        revision=os.environ.get("GITHUB_SHA"),
    )
    write_probe(args.output, result)
    summary = {
        "content_probe_sha256": result["content_probe_sha256"],
        "probe_sha256": result["probe_sha256"],
        "object_count": result["object_count"],
        "episode_count": result["episode_count"],
        "prediction_eligible_episode_count": result[
            "prediction_eligible_episode_count"
        ],
        "prediction_ineligible_episode_count": result[
            "prediction_ineligible_episode_count"
        ],
        "all_stride_observable_archives_constant_frame_stride": result[
            "all_stride_observable_archives_constant_frame_stride"
        ],
        "constant_frame_stride_counts": result["constant_frame_stride_counts"],
        "irregular_frame_stride_archive_count": (
            result["irregular_frame_stride_archive_count"]
        ),
        "no_transition_archive_count": result["no_transition_archive_count"],
        "frame_stride_patterns": result["frame_stride_patterns"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
