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
AMENDMENT_SCHEMA = "bayesian-phystwin/deform360-source-hull-contract-probe-amendment-v2"
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


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


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
        _require(
            _positive_integer(source_inventory.get(field)),
            f"source inventory {field} is invalid",
        )
    return payload


def load_probe_amendment(
    path: Path,
    *,
    base_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the empty-frame amendment without changing the locked cohort."""

    payload = _load_json(path.resolve())
    _require(payload.get("schema") == AMENDMENT_SCHEMA, "unexpected amendment schema")
    _require(payload.get("schema_version") == 1, "unsupported amendment version")
    config = payload.get("config")
    _require(isinstance(config, dict), "probe amendment lacks config")
    expected = payload.get("config_sha256")
    _require(
        _is_lower_hex_identity(expected, lengths=(64,))
        and hashlib.sha256(_canonical_bytes(config)).hexdigest() == expected,
        "probe amendment config checksum changed",
    )
    _require(
        config.get("amendment_id") == "deform360-source-hull-contract-probe-v2",
        "unexpected probe amendment identity",
    )
    _require(
        config.get("status")
        == "locked-after-v1-structural-failure-before-coordinate-access",
        "probe amendment status changed",
    )
    base = config.get("base_protocol")
    _require(isinstance(base, dict), "probe amendment base binding is missing")
    _require(
        base.get("config_sha256") == base_protocol.get("config_sha256"),
        "probe amendment base protocol checksum changed",
    )
    policy = config.get("policy")
    _require(isinstance(policy, dict), "probe amendment policy is missing")
    _require(
        policy.get("point_offsets_order") == "nondecreasing"
        and policy.get("minimum_points_for_usable_frame") == 1
        and policy.get("minimum_usable_frames_for_rolling_prediction") == 3
        and policy.get("cadence_basis")
        == "strictly increasing frame_indices after empty-frame exclusion"
        and policy.get("object_balancing_for_future_scoring") is True,
        "probe amendment policy changed",
    )
    boundary = config.get("information_boundary")
    _require(isinstance(boundary, dict), "amendment information boundary is missing")
    _require(
        boundary.get("points_world_m_coordinate_values_decoded") is False
        and boundary.get("model_prediction_run") is False
        and boundary.get("score_bearing_outcome_computed") is False
        and boundary.get("reserved_target_outcomes_opened") is False,
        "probe amendment information boundary changed",
    )
    trigger = config.get("trigger_evidence")
    _require(isinstance(trigger, dict), "amendment trigger evidence is missing")
    _require(
        _is_lower_hex_identity(trigger.get("artifact_sha256"), lengths=(64,))
        and _is_lower_hex_identity(trigger.get("evaluated_merge_sha"), lengths=(40, 64))
        and _positive_integer(trigger.get("workflow_run_id"))
        and _positive_integer(trigger.get("workflow_job_id"))
        and _positive_integer(trigger.get("artifact_id")),
        "probe amendment trigger evidence is invalid",
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
    allow_empty_frames: bool,
    minimum_points_for_usable_frame: int,
    minimum_usable_frames: int,
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
    _require(
        np.all(np.diff(frame_indices) > 0),
        "frame_indices must be strictly increasing",
    )
    _require(point_offsets.ndim == 1, "point_offsets must be one-dimensional")
    _require(
        point_offsets.shape == (len(frame_indices) + 1,),
        "point_offsets length must equal frame count plus one",
    )
    _require(point_offsets[0] == 0, "point_offsets must start at zero")
    offset_differences = np.diff(point_offsets)
    if allow_empty_frames:
        _require(
            np.all(offset_differences >= 0),
            "point_offsets must be nondecreasing",
        )
    else:
        _require(
            np.all(offset_differences > 0),
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
    point_counts = offset_differences
    usable_mask = point_counts >= minimum_points_for_usable_frame
    usable_frame_indices = frame_indices[usable_mask]
    empty_frame_indices = frame_indices[~usable_mask]
    usable_differences = np.diff(usable_frame_indices)
    unique_strides, stride_counts = np.unique(
        usable_differences,
        return_counts=True,
    )
    transition_count = len(usable_differences)
    prediction_eligible = len(usable_frame_indices) >= minimum_usable_frames
    return {
        "object_id": object_id,
        "episode_id": episode_id,
        "relative_path": relative_path,
        "archive_sha256": _file_sha256(path),
        "archive_size_bytes": int(path.stat().st_size),
        "archive_members": list(members),
        "frame_count": int(len(frame_indices)),
        "usable_frame_count": int(len(usable_frame_indices)),
        "empty_frame_count": int(len(empty_frame_indices)),
        "transition_count": int(transition_count),
        "frame_start": int(frame_indices[0]),
        "frame_stop_inclusive": int(frame_indices[-1]),
        "frame_indices": frame_indices.astype(int).tolist(),
        "usable_frame_indices": usable_frame_indices.astype(int).tolist(),
        "empty_frame_indices": empty_frame_indices.astype(int).tolist(),
        "frame_stride_counts": {
            str(int(stride)): int(count)
            for stride, count in zip(unique_strides, stride_counts, strict=True)
        },
        "stride_observable": transition_count > 0,
        "constant_frame_stride": len(unique_strides) == 1,
        "prediction_eligible": prediction_eligible,
        "prediction_ineligibility_reason": (
            None
            if prediction_eligible
            else (
                "rolling one-step evaluation requires at least "
                f"{minimum_usable_frames} nonempty sampled hulls"
            )
        ),
        "empty_frame_policy": (
            "retained-and-excluded-from-prediction"
            if allow_empty_frames
            else "forbidden"
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
    amendment_path: Path | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Probe locked archive contracts and cadence without reading geometry values."""

    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Deform360 data root is missing: {root}")
    protocol_file = protocol_path.expanduser().resolve()
    protocol = load_probe_protocol(protocol_file)
    amendment: dict[str, Any] | None = None
    if amendment_path is not None:
        amendment = load_probe_amendment(
            amendment_path.expanduser().resolve(),
            base_protocol=protocol,
        )
    if amendment is None:
        allow_empty_frames = False
        minimum_points_for_usable_frame = 1
        minimum_usable_frames = 3
    else:
        policy = amendment["config"]["policy"]
        allow_empty_frames = policy["point_offsets_order"] == "nondecreasing"
        minimum_points_for_usable_frame = int(policy["minimum_points_for_usable_frame"])
        minimum_usable_frames = int(
            policy["minimum_usable_frames_for_rolling_prediction"]
        )

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
                allow_empty_frames=allow_empty_frames,
                minimum_points_for_usable_frame=minimum_points_for_usable_frame,
                minimum_usable_frames=minimum_usable_frames,
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
    empty_frame_archives = [
        str(record["relative_path"])
        for record in records
        if int(record["empty_frame_count"]) > 0
    ]
    result: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": protocol["config_sha256"],
        "amendment_id": (
            None if amendment is None else amendment["config"]["amendment_id"]
        ),
        "amendment_config_sha256": (
            None if amendment is None else amendment["config_sha256"]
        ),
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
        "empty_frame_archive_count": len(empty_frame_archives),
        "empty_frame_archives": empty_frame_archives,
        "total_empty_frame_count": int(
            sum(int(record["empty_frame_count"]) for record in records)
        ),
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
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = probe_locked_source_hulls(
        args.data_root,
        protocol_path=args.protocol,
        amendment_path=args.amendment,
        revision=os.environ.get("GITHUB_SHA"),
    )
    write_probe(args.output, result)
    summary = {
        "content_probe_sha256": result["content_probe_sha256"],
        "probe_sha256": result["probe_sha256"],
        "amendment_id": result["amendment_id"],
        "object_count": result["object_count"],
        "episode_count": result["episode_count"],
        "prediction_eligible_episode_count": result[
            "prediction_eligible_episode_count"
        ],
        "prediction_ineligible_episode_count": result[
            "prediction_ineligible_episode_count"
        ],
        "empty_frame_archive_count": result["empty_frame_archive_count"],
        "total_empty_frame_count": result["total_empty_frame_count"],
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
