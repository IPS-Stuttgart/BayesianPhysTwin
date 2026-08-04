#!/usr/bin/env python3
"""Seal and evaluate a leak-safe external Deform360 cohort.

The ``seal`` command inventories names and directory structure only. It selects
exact object/archive paths without opening dataset payloads and writes a
content-addressed selection artifact. The ``evaluate`` command runs in a
separate process and may open only those sealed paths.

The scientific comparison is deliberately narrow: persistence, last supported
residual, the frozen cumulative-evidence endpoint model average, and the same
component bank reweighted by mean log evidence per supported observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointPosteriorV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-normalized-evidence-external-v1"
SEAL_SCHEMA = "bayesian-phystwin/deform360-normalized-evidence-selection-v1"
STRUCTURE_SCHEMA = "bayesian-phystwin/deform360-normalized-evidence-structure-v1"
RESULT_SCHEMA = "bayesian-phystwin/deform360-normalized-evidence-result-v1"
CHI_SQUARE_3D_90 = 6.251388631170325
_OBJECT_PATTERN = re.compile(r"^\d{3}-[A-Za-z0-9][A-Za-z0-9-]*$")
_OBJECT_IN_TEXT = re.compile(
    r"(?<![A-Za-z0-9])(?P<object>\d{3}-[A-Za-z][A-Za-z0-9-]*)(?![A-Za-z0-9-])"
)
_TRAJECTORY_HINTS = (
    "positions_world_m",
    "points_world_m",
    "positions_m",
    "particle_tracks",
    "particles",
    "tracks",
    "trajectory",
    "positions",
    "points",
)
_VALIDITY_HINTS = ("valid_mask", "track_valid", "visibility", "valid")
METHODS = (
    "persistence",
    "last_supported_residual",
    "cumulative_evidence_model_average_v1",
    "per_observation_normalized_evidence_model_average_v1",
)
PREDICTIVE_METHODS = METHODS[-2:]


@dataclass(frozen=True, slots=True)
class EvaluationLimits:
    """Deterministic numerical resource limits."""

    max_frames_per_archive: int = 96
    max_tracks: int = 2048
    chamfer_points: int = 512
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_804

    def __post_init__(self) -> None:
        for name in (
            "max_frames_per_archive",
            "max_tracks",
            "chamfer_points",
            "bootstrap_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(
            self.bootstrap_seed, int
        ):
            raise ValueError("bootstrap_seed must be an integer")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array of strings")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected Deform360 protocol schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Deform360 protocol version")
    if payload.get("status") != "locked-before-numeric-payload-access":
        raise ValueError("protocol is not locked before numeric access")
    selection = _require_mapping(
        payload.get("cohort_selection"), name="cohort_selection"
    )
    requested = selection.get("requested_object_count")
    minimum = selection.get("minimum_selected_object_count_before_numeric_access")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("requested_object_count must be a positive integer")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or not 1 <= minimum <= requested
    ):
        raise ValueError("minimum selected object count is invalid")
    excluded = _require_string_sequence(
        payload.get("explicit_excluded_object_ids"),
        name="explicit_excluded_object_ids",
    )
    if len(set(excluded)) != len(excluded):
        raise ValueError("explicit excluded object IDs must be unique")
    if any(_OBJECT_PATTERN.fullmatch(item) is None for item in excluded):
        raise ValueError("explicit excluded object ID is malformed")
    methods = _require_string_sequence(payload.get("methods"), name="methods")
    if methods != METHODS:
        raise ValueError("protocol method ordering changed")
    return payload, _file_sha256(path)


def _git_revision(repository_root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _tracked_paths(repository_root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return tuple(
        repository_root / os.fsdecode(raw) for raw in result.stdout.split(b"\0") if raw
    )


def _evidence_bearing_path(path: Path, repository_root: Path) -> bool:
    relative = path.relative_to(repository_root).as_posix().lower()
    if relative.startswith("results/"):
        return True
    if relative.startswith("protocols/") and "deform360" in relative:
        return True
    if relative.startswith("configs/sota/") and "deform360" in relative:
        return True
    if relative.startswith("docs/") and "deform360" in relative:
        return True
    if relative.startswith("scripts/") and "deform360" in relative:
        return True
    if relative.startswith("tests/") and "deform360" in relative:
        return True
    return False


def _repository_evidence_mentions(
    repository_root: Path,
) -> dict[str, tuple[str, ...]]:
    mentions: dict[str, set[str]] = defaultdict(set)
    for path in _tracked_paths(repository_root):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(repository_root).as_posix()
        for match in _OBJECT_IN_TEXT.finditer(text):
            object_id = match.group("object")
            if _OBJECT_PATTERN.fullmatch(object_id):
                mentions[object_id].add(relative)
    return {
        object_id: tuple(sorted(paths)) for object_id, paths in sorted(mentions.items())
    }


def _path_object_id(path: Path, root: Path) -> str | None:
    return next(
        (
            part
            for part in path.relative_to(root).parts
            if _OBJECT_PATTERN.fullmatch(part)
        ),
        None,
    )


def _rank(seed: str, *parts: str) -> str:
    value = "\0".join((seed, *parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _matches_prefix(relative: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        relative == prefix.rstrip("/") or relative.startswith(prefix)
        for prefix in prefixes
    )


def _dataset_name_inventory(
    root: Path,
    *,
    suffixes: tuple[str, ...],
    hints: tuple[str, ...],
    allowed_prefixes: tuple[str, ...],
    forbidden_prefixes: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    candidates: dict[str, list[str]] = defaultdict(list)
    scoped_names: list[str] = []
    total_files = 0
    total_directories = 0
    admitted_files = 0
    admitted_directories = 0
    forbidden_name_count = 0
    lowered_suffixes = tuple(value.lower() for value in suffixes)
    lowered_hints = tuple(value.lower() for value in hints)
    for directory, names, files in os.walk(root):
        names[:] = sorted(
            name
            for name in names
            if name not in {".git", "__pycache__", "node_modules"}
        )
        total_directories += 1
        relative_directory = Path(directory).relative_to(root).as_posix()
        if relative_directory == ".":
            relative_directory = ""
        if relative_directory and _matches_prefix(
            relative_directory, forbidden_prefixes
        ):
            forbidden_name_count += 1
            names[:] = []
            continue
        if relative_directory and _matches_prefix(relative_directory, allowed_prefixes):
            admitted_directories += 1
            scoped_names.append(f"D\0{relative_directory}")
        for name in sorted(files):
            total_files += 1
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if _matches_prefix(relative, forbidden_prefixes):
                forbidden_name_count += 1
                continue
            if not _matches_prefix(relative, allowed_prefixes):
                continue
            admitted_files += 1
            scoped_names.append(f"F\0{relative}")
            object_id = _path_object_id(path, root)
            if object_id is None:
                continue
            lowered = relative.lower()
            if not lowered.endswith(lowered_suffixes):
                continue
            if not any(hint in lowered for hint in lowered_hints):
                continue
            candidates[object_id].append(relative)
    names_inventory_sha256 = hashlib.sha256(
        "\n".join(sorted(scoped_names)).encode("utf-8")
    ).hexdigest()
    return (
        {
            object_id: tuple(sorted(paths))
            for object_id, paths in sorted(candidates.items())
        },
        {
            "total_files": total_files,
            "total_directories": total_directories,
            "admitted_files": admitted_files,
            "admitted_directories": admitted_directories,
            "forbidden_name_count": forbidden_name_count,
            "names_inventory_sha256": names_inventory_sha256,
        },
    )


def seal_selection(
    data_root: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Select exact archives from names only and write a hash-bound seal."""

    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Deform360 root is missing: {root}")
    repository = repository_root.expanduser().resolve()
    protocol, protocol_sha256 = _load_protocol(protocol_path.resolve())
    selection = _require_mapping(protocol["cohort_selection"], name="cohort_selection")
    suffixes = _require_string_sequence(
        selection["archive_suffixes"], name="archive_suffixes"
    )
    hints = _require_string_sequence(
        selection["archive_name_hints"], name="archive_name_hints"
    )
    allowed_prefixes = _require_string_sequence(
        selection["allowed_archive_path_prefixes"],
        name="allowed_archive_path_prefixes",
    )
    forbidden_prefixes = _require_string_sequence(
        selection["forbidden_archive_path_prefixes"],
        name="forbidden_archive_path_prefixes",
    )
    inventory, counts = _dataset_name_inventory(
        root,
        suffixes=suffixes,
        hints=hints,
        allowed_prefixes=allowed_prefixes,
        forbidden_prefixes=forbidden_prefixes,
    )
    evidence_mentions = _repository_evidence_mentions(repository)
    explicit = set(
        _require_string_sequence(
            protocol["explicit_excluded_object_ids"],
            name="explicit_excluded_object_ids",
        )
    )
    excluded = explicit | set(evidence_mentions)
    seed = str(selection["selection_seed"])
    per_object: list[dict[str, Any]] = []
    for object_id, paths in inventory.items():
        if object_id in excluded:
            continue
        chosen = min(paths, key=lambda path: _rank(seed, object_id, path))
        per_object.append(
            {
                "object_id": object_id,
                "archive_path": chosen,
                "object_rank_sha256": _rank(seed, object_id, chosen),
                "candidate_archive_count": len(paths),
            }
        )
    per_object.sort(key=lambda item: (item["object_rank_sha256"], item["object_id"]))
    requested = int(selection["requested_object_count"])
    selected = per_object[:requested]
    minimum = int(selection["minimum_selected_object_count_before_numeric_access"])
    body: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_file": str(protocol_path.resolve()),
        "protocol_file_sha256": protocol_sha256,
        "repository_revision": _git_revision(repository),
        "dataset_root": str(root),
        "information_boundary": {
            "dataset_payload_opened": False,
            "file_contents_hashed": False,
            "names_and_directory_structure_only": True,
            "selection_uses_target_outcomes": False,
        },
        "inventory": {
            **counts,
            "candidate_object_count": len(inventory),
            "eligible_object_count": len(per_object),
            "evidence_mentioned_object_count": len(evidence_mentions),
            "explicit_excluded_object_count": len(explicit),
        },
        "selection_rule": selection["selection_rule"],
        "selection_seed": seed,
        "allowed_archive_path_prefixes": list(allowed_prefixes),
        "forbidden_archive_path_prefixes": list(forbidden_prefixes),
        "requested_object_count": requested,
        "minimum_object_count": minimum,
        "support_passed": len(selected) >= minimum,
        "selected": selected,
        "excluded_object_ids": sorted(excluded),
        "evidence_mention_paths": {
            object_id: list(paths) for object_id, paths in evidence_mentions.items()
        },
    }
    body["selection_sha256"] = _canonical_sha256(body)
    _write_json(output_path.resolve(), body)
    return body


def _verify_seal(
    seal_path: Path,
    protocol_path: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema") != SEAL_SCHEMA or seal.get("schema_version") != 1:
        raise ValueError("unsupported selection seal")
    expected = seal.get("selection_sha256")
    body = dict(seal)
    body.pop("selection_sha256", None)
    if expected != _canonical_sha256(body):
        raise ValueError("selection seal digest changed")
    protocol, protocol_sha256 = _load_protocol(protocol_path)
    if seal.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("selection seal protocol ID changed")
    if seal.get("protocol_file_sha256") != protocol_sha256:
        raise ValueError("selection seal protocol digest changed")
    if not seal.get("support_passed"):
        raise ValueError("selection support gate did not pass")
    if seal.get("repository_revision") != _git_revision(repository_root):
        raise ValueError("selection seal repository revision changed")
    selection = _require_mapping(protocol["cohort_selection"], name="cohort_selection")
    inventory, counts = _dataset_name_inventory(
        Path(str(seal["dataset_root"])).resolve(),
        suffixes=_require_string_sequence(
            selection["archive_suffixes"], name="archive_suffixes"
        ),
        hints=_require_string_sequence(
            selection["archive_name_hints"], name="archive_name_hints"
        ),
        allowed_prefixes=_require_string_sequence(
            selection["allowed_archive_path_prefixes"],
            name="allowed_archive_path_prefixes",
        ),
        forbidden_prefixes=_require_string_sequence(
            selection["forbidden_archive_path_prefixes"],
            name="forbidden_archive_path_prefixes",
        ),
    )
    del inventory
    recorded_inventory = _require_mapping(seal.get("inventory"), name="inventory")
    if (
        recorded_inventory.get("names_inventory_sha256")
        != counts["names_inventory_sha256"]
    ):
        raise ValueError("mounted upstream names changed after selection sealing")
    selected = seal.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection seal contains no selected archives")
    return seal, protocol


def _read_npy_header(
    archive: zipfile.ZipFile,
    member: str,
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    with archive.open(member, "r") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NumPy member version: {version}")
    return tuple(int(value) for value in shape), bool(fortran_order), np.dtype(dtype)


def _load_integer_member(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with archive.open(member, "r") as stream:
        value = np.load(stream, allow_pickle=False)
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{member} must contain an integer array")
    return np.asarray(value, dtype=np.int64)


def _constant_stride_segments(
    frame_indices: np.ndarray,
    usable_positions: np.ndarray,
    *,
    minimum_frames: int,
) -> list[dict[str, Any]]:
    positions = np.asarray(usable_positions, dtype=np.int64)
    if len(positions) < minimum_frames:
        return []
    values = frame_indices[positions]
    if np.any(np.diff(values) <= 0):
        raise ValueError("usable frame indices must be strictly increasing")
    if len(values) == 1:
        return []
    segments: list[dict[str, Any]] = []
    start = 0
    stride = int(values[1] - values[0])
    for index in range(2, len(values)):
        candidate = int(values[index] - values[index - 1])
        if candidate == stride:
            continue
        stop = index
        if stop - start >= minimum_frames:
            selected = positions[start:stop]
            segments.append(
                {
                    "positions": selected.astype(int).tolist(),
                    "frame_indices": frame_indices[selected].astype(int).tolist(),
                    "frame_stride": stride,
                }
            )
        start = index - 1
        stride = candidate
    if len(values) - start >= minimum_frames:
        selected = positions[start:]
        segments.append(
            {
                "positions": selected.astype(int).tolist(),
                "frame_indices": frame_indices[selected].astype(int).tolist(),
                "frame_stride": stride,
            }
        )
    return segments


def _trajectory_header(
    archive: zipfile.ZipFile,
) -> tuple[str, tuple[int, ...], str] | None:
    candidates: list[tuple[int, str, tuple[int, ...], str]] = []
    for member in archive.namelist():
        if not member.endswith(".npy"):
            continue
        key = member[:-4]
        lowered = key.lower()
        ranks = [
            index for index, hint in enumerate(_TRAJECTORY_HINTS) if hint in lowered
        ]
        if not ranks:
            continue
        shape, _, dtype = _read_npy_header(archive, member)
        if len(shape) == 3 and shape[-1] == 3 and shape[0] >= 4:
            _strict_scale_to_meters(key)
            candidates.append((min(ranks), key, shape, dtype.str))
    if not candidates:
        return None
    _, key, shape, dtype = min(candidates, key=lambda item: (item[0], item[1]))
    return key, shape, dtype


def inspect_structure(
    seal_path: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Bind headers, frame cadence, empty frames, and usable segments."""

    repository = repository_root.resolve()
    seal, protocol = _verify_seal(
        seal_path.resolve(), protocol_path.resolve(), repository
    )
    structure_policy = _require_mapping(
        protocol.get("structure_preflight"), name="structure_preflight"
    )
    minimum_frames = int(structure_policy["minimum_usable_frames_per_segment"])
    root = Path(str(seal["dataset_root"])).resolve()
    records: list[dict[str, Any]] = []
    supported = 0
    selected = seal["selected"]
    assert isinstance(selected, list)
    for entry in selected:
        if not isinstance(entry, Mapping):
            raise ValueError("selected archive entry is malformed")
        relative = str(entry["archive_path"])
        object_id = str(entry["object_id"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("selected archive escapes the mounted root") from error
        record: dict[str, Any] = {
            "object_id": object_id,
            "archive_path": relative,
            "archive_sha256": _file_sha256(path),
        }
        try:
            with zipfile.ZipFile(path, "r") as archive:
                members = set(archive.namelist())
                packed_members = {
                    "frame_indices.npy",
                    "point_offsets.npy",
                    "points_world_m.npy",
                }
                if packed_members.issubset(members):
                    frames = _load_integer_member(archive, "frame_indices.npy")
                    offsets = _load_integer_member(archive, "point_offsets.npy")
                    points_shape, _, points_dtype = _read_npy_header(
                        archive, "points_world_m.npy"
                    )
                    valid = (
                        frames.ndim == 1
                        and offsets.shape == (len(frames) + 1,)
                        and len(frames) >= 1
                        and offsets[0] == 0
                        and np.all(np.diff(offsets) >= 0)
                        and len(points_shape) == 2
                        and points_shape[1] == 3
                        and int(offsets[-1]) == points_shape[0]
                        and np.issubdtype(points_dtype, np.floating)
                    )
                    if not valid:
                        raise ValueError("packed visual-hull structure is malformed")
                    point_counts = np.diff(offsets)
                    usable = np.flatnonzero(point_counts > 0)
                    segments = _constant_stride_segments(
                        frames,
                        usable,
                        minimum_frames=minimum_frames,
                    )
                    record.update(
                        {
                            "representation": "packed_visual_hulls",
                            "frame_count": len(frames),
                            "usable_frame_count": len(usable),
                            "empty_frame_positions": np.flatnonzero(point_counts == 0)
                            .astype(int)
                            .tolist(),
                            "segments": segments,
                            "points_world_m_header": {
                                "shape": list(points_shape),
                                "dtype": points_dtype.str,
                                "coordinate_values_decoded": False,
                            },
                        }
                    )
                else:
                    candidate = _trajectory_header(archive)
                    if candidate is None:
                        raise ValueError(
                            "no declared trajectory or packed-hull structure"
                        )
                    key, shape, dtype = candidate
                    positions = np.arange(shape[0], dtype=np.int64)
                    segments = _constant_stride_segments(
                        positions,
                        positions,
                        minimum_frames=minimum_frames,
                    )
                    record.update(
                        {
                            "representation": "fixed_identity_trajectory",
                            "array_key": key,
                            "array_shape": list(shape),
                            "array_dtype": dtype,
                            "frame_count": shape[0],
                            "usable_frame_count": shape[0],
                            "empty_frame_positions": [],
                            "segments": segments,
                        }
                    )
            if not record["segments"]:
                record["status"] = "prediction_ineligible"
                record["error"] = (
                    "no constant-stride segment reaches the minimum length"
                )
            else:
                record["status"] = "supported"
                supported += 1
        except (OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
            record["status"] = "invalid_or_unsupported"
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    minimum_supported = int(
        structure_policy["minimum_supported_object_count_before_coordinate_access"]
    )
    body: dict[str, Any] = {
        "schema": STRUCTURE_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_file_sha256": seal["protocol_file_sha256"],
        "selection_sha256": seal["selection_sha256"],
        "repository_revision": seal["repository_revision"],
        "dataset_names_inventory_sha256": seal["inventory"]["names_inventory_sha256"],
        "information_boundary": {
            "coordinate_values_decoded": False,
            "opened_integer_metadata": ["frame_indices", "point_offsets"],
            "opened_array_headers": True,
            "model_prediction_run": False,
            "score_bearing_outcome_computed": False,
        },
        "selected_object_count": len(selected),
        "supported_object_count": supported,
        "minimum_supported_object_count": minimum_supported,
        "support_passed": supported >= minimum_supported,
        "archives": records,
    }
    body["structure_sha256"] = _canonical_sha256(body)
    _write_json(output_path.resolve(), body)
    return body


def _verify_structure(
    structure_path: Path,
    seal: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    if (
        structure.get("schema") != STRUCTURE_SCHEMA
        or structure.get("schema_version") != 1
    ):
        raise ValueError("unsupported structure seal")
    expected = structure.get("structure_sha256")
    body = dict(structure)
    body.pop("structure_sha256", None)
    if expected != _canonical_sha256(body):
        raise ValueError("structure seal digest changed")
    if structure.get("selection_sha256") != seal.get("selection_sha256"):
        raise ValueError("structure seal selection identity changed")
    if structure.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("structure seal protocol identity changed")
    if structure.get("repository_revision") != seal.get("repository_revision"):
        raise ValueError("structure seal repository revision changed")
    if (
        structure.get("dataset_names_inventory_sha256")
        != seal["inventory"]["names_inventory_sha256"]
    ):
        raise ValueError("structure seal mounted-name identity changed")
    if not structure.get("support_passed"):
        raise ValueError("structure support gate did not pass")
    return structure


def _strict_scale_to_meters(key: str) -> tuple[float, str]:
    lowered = key.lower()
    if lowered.endswith("_mm") or "millimet" in lowered:
        return 1e-3, "declared_mm"
    if lowered.endswith("_m") or "world_m" in lowered:
        return 1.0, "declared_m"
    raise ValueError(
        f"trajectory key {key!r} has no declared metric or millimetric unit"
    )


def _trajectory(stored: Any) -> tuple[str, np.ndarray] | None:
    candidates: list[tuple[int, str, np.ndarray]] = []
    for key in stored.files:
        lowered = key.lower()
        ranks = [
            index for index, hint in enumerate(_TRAJECTORY_HINTS) if hint in lowered
        ]
        if not ranks:
            continue
        value = np.asarray(stored[key])
        if value.ndim == 3 and value.shape[-1] == 3 and value.shape[0] >= 4:
            candidates.append((min(ranks), key, value))
    if not candidates:
        return None
    _, key, value = min(candidates, key=lambda item: (item[0], item[1]))
    return key, value


def _validity(stored: Any, shape: tuple[int, int]) -> np.ndarray:
    for key in stored.files:
        if not any(hint in key.lower() for hint in _VALIDITY_HINTS):
            continue
        value = np.asarray(stored[key])
        if value.shape == shape:
            return np.asarray(value, dtype=bool)
    return np.ones(shape, dtype=bool)


def _packed_hulls(stored: Any) -> tuple[np.ndarray, tuple[np.ndarray, ...]] | None:
    keys = set(stored.files)
    required = {"frame_indices", "point_offsets", "points_world_m"}
    if not required.issubset(keys):
        return None
    frames = np.asarray(stored["frame_indices"])
    offsets = np.asarray(stored["point_offsets"])
    points = np.asarray(stored["points_world_m"], dtype=np.float64)
    if not np.issubdtype(frames.dtype, np.integer) or not np.issubdtype(
        offsets.dtype, np.integer
    ):
        raise ValueError("packed hull indices and offsets must be integer typed")
    frames = np.asarray(frames, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    valid = (
        frames.ndim == 1
        and len(frames) >= 4
        and offsets.shape == (len(frames) + 1,)
        and offsets[0] == 0
        and offsets[-1] == len(points)
        and np.all(np.diff(offsets) >= 0)
        and points.ndim == 2
        and points.shape[1] == 3
        and np.all(np.isfinite(points))
    )
    if not valid:
        raise ValueError("packed visual-hull contract is malformed")
    hulls = tuple(
        points[offsets[index] : offsets[index + 1]] for index in range(len(frames))
    )
    return frames, hulls


def _indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if len(array) == 0 else float(np.mean(array))


def _rmse(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    error = prediction[valid] - target[valid]
    if len(error) == 0:
        raise ValueError("RMSE requires at least one valid point")
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def _chamfer_rmse(
    first: np.ndarray,
    second: np.ndarray,
    *,
    maximum_points: int,
) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1:] != (3,)
        or right.shape[1:] != (3,)
    ):
        raise ValueError("Chamfer inputs must have shape (N, 3)")
    if len(left) == 0 or len(right) == 0:
        raise ValueError("Chamfer inputs must be nonempty")
    left = left[_indices(len(left), maximum_points)]
    right = right[_indices(len(right), maximum_points)]

    def directed(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        minimum = np.full(len(source), np.inf, dtype=np.float64)
        for start in range(0, len(target), 128):
            block = target[start : start + 128]
            squared = np.sum(np.square(source[:, None, :] - block[None, :, :]), axis=2)
            minimum = np.minimum(minimum, np.min(squared, axis=1))
        return minimum

    mean_squared = 0.5 * (
        float(np.mean(directed(left, right))) + float(np.mean(directed(right, left)))
    )
    return float(np.sqrt(max(mean_squared, 0.0)))


def _predictive_metrics(
    error: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float]:
    residual = np.asarray(error, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if residual.ndim != 2 or residual.shape[1] != 3:
        raise ValueError("predictive error must have shape (N, 3)")
    if cov.shape != (len(residual), 3, 3):
        raise ValueError("predictive covariance shape changed")
    regularized = cov + np.eye(3)[None] * 1e-12
    sign, logdet = np.linalg.slogdet(regularized)
    if np.any(sign <= 0):
        raise ValueError("predictive covariance is not positive definite")
    solved = np.linalg.solve(regularized, residual[..., None])[..., 0]
    quadratic = np.sum(residual * solved, axis=1)
    nll = 0.5 * (3.0 * math.log(2.0 * math.pi) + logdet + quadratic)
    return {
        "coverage_90": float(np.mean(quadratic <= CHI_SQUARE_3D_90)),
        "mean_nll": float(np.mean(nll)),
        "mean_nees_over_dimension": float(np.mean(quadratic) / 3.0),
        "mean_predictive_std_m": float(
            np.mean(np.sqrt(np.trace(cov, axis1=1, axis2=2) / 3.0))
        ),
    }


def _weight_diagnostics(
    weights: np.ndarray,
    component_variance: np.ndarray,
    component_mean: np.ndarray,
    mixture_mean: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float]:
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1)
    effective = np.exp(entropy)
    within_trace = 3.0 * np.einsum("nk,kn->n", weights, component_variance)
    total_trace = np.trace(covariance, axis1=1, axis2=2)
    centered = component_mean - mixture_mean[None, :, :]
    direct_between = np.einsum("nk,knc,knc->n", weights, centered, centered)
    between_fraction = direct_between / np.maximum(total_trace, 1e-30)
    return {
        "mean_effective_component_count": float(np.mean(effective)),
        "mean_component_entropy_nats": float(np.mean(entropy)),
        "median_between_model_covariance_fraction": float(
            np.median(np.maximum(between_fraction, 0.0))
        ),
        "mean_within_trace_m2": float(np.mean(within_trace)),
    }


def _normalized_weights(
    posterior: ModelAveragedEndpointPosteriorV1,
) -> np.ndarray:
    prior = np.asarray(posterior.config.component_prior_probability, dtype=np.float64)
    denominator = np.maximum(posterior.update_count, 1)[:, None]
    logits = np.log(prior)[None, :] + posterior.component_log_evidence / denominator
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights


def _moments_for_weights(
    posterior: ModelAveragedEndpointPosteriorV1,
    weights: np.ndarray,
    *,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if weights.shape != posterior.component_weights.shape:
        raise ValueError("reweighted component shape changed")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("reweighted components are invalid")
    if not np.allclose(np.sum(weights, axis=1), 1.0):
        raise ValueError("reweighted components are not normalized")
    mean = np.einsum("nk,knc->nc", weights, posterior.component_mean_m)
    component_variance = (
        posterior.component_variance_m2
        + int(horizon_steps) * posterior.component_process_variance_m2[:, None]
    )
    centered = posterior.component_mean_m - mean[None, :, :]
    outer = centered[:, :, :, None] * centered[:, :, None, :]
    within = component_variance[:, :, None, None] * np.eye(3)
    covariance = np.einsum("nk,knij->nij", weights, within + outer)
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    diagnostics = _weight_diagnostics(
        weights,
        component_variance,
        posterior.component_mean_m,
        mean,
        covariance,
    )
    return mean, covariance, diagnostics


def _last_supported_residual(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
) -> np.ndarray:
    history = np.asarray(residual[:end_frame], dtype=np.float64)
    support = np.asarray(valid[:end_frame], dtype=bool)
    if (
        history.ndim != 3
        or history.shape[-1] != 3
        or support.shape != history.shape[:2]
    ):
        raise ValueError("residual history and support shapes are inconsistent")
    result = np.zeros(history.shape[1:], dtype=np.float64)
    found = np.zeros(history.shape[1], dtype=bool)
    for index in range(len(history) - 1, -1, -1):
        use = support[index] & ~found
        result[use] = history[index, use]
        found |= use
        if np.all(found):
            break
    return result


def _rolling_prediction(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]]:
    posterior = infer_model_averaged_endpoint(
        residual[:end_frame], valid[:end_frame], end_frame=end_frame
    )
    raw_prediction = predict_model_averaged_endpoint(posterior, horizon_steps=1)
    raw_diagnostics = _weight_diagnostics(
        raw_prediction.component_weights,
        posterior.component_variance_m2
        + posterior.component_process_variance_m2[:, None],
        posterior.component_mean_m,
        raw_prediction.mean_m,
        raw_prediction.covariance_m2,
    )
    normalized_mean, normalized_covariance, normalized_diagnostics = (
        _moments_for_weights(
            posterior,
            _normalized_weights(posterior),
            horizon_steps=1,
        )
    )
    return {
        METHODS[2]: (
            raw_prediction.mean_m,
            raw_prediction.covariance_m2,
            raw_diagnostics,
        ),
        METHODS[3]: (
            normalized_mean,
            normalized_covariance,
            normalized_diagnostics,
        ),
    }


def _evaluate_fixed(
    stored: Any,
    *,
    path: str,
    object_id: str,
    segments: Sequence[Mapping[str, Any]],
    limits: EvaluationLimits,
) -> dict[str, Any] | None:
    candidate = _trajectory(stored)
    if candidate is None:
        return None
    key, raw = candidate
    scale, unit_source = _strict_scale_to_meters(key)
    points = np.asarray(raw, dtype=np.float64) * scale
    if not np.all(np.isfinite(points)):
        raise ValueError("trajectory contains non-finite values")
    validity = _validity(stored, points.shape[:2])
    tracks = _indices(points.shape[1], limits.max_tracks)
    points = points[:, tracks]
    validity = validity[:, tracks]
    steps: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        positions = np.asarray(segment["positions"], dtype=np.int64)
        positions = positions[positions < len(points)][: limits.max_frames_per_archive]
        if len(positions) < 4:
            continue
        selected_points = points[positions]
        selected_validity = validity[positions]
        residual = np.diff(selected_points, axis=0)
        residual_valid = selected_validity[:-1] & selected_validity[1:]
        for current in range(2, len(selected_points) - 1):
            target_valid = selected_validity[current] & selected_validity[current + 1]
            if not np.any(target_valid):
                continue
            predictions = _rolling_prediction(
                residual, residual_valid, end_frame=current
            )
            last_supported = _last_supported_residual(
                residual,
                residual_valid,
                end_frame=current,
            )
            target = selected_points[current + 1]
            predicted_positions = {
                METHODS[0]: selected_points[current],
                METHODS[1]: selected_points[current] + last_supported,
                METHODS[2]: selected_points[current] + predictions[METHODS[2]][0],
                METHODS[3]: selected_points[current] + predictions[METHODS[3]][0],
            }
            predictive = {}
            for method in PREDICTIVE_METHODS:
                mean, covariance, diagnostics = predictions[method]
                metrics = _predictive_metrics(
                    residual[current, target_valid] - mean[target_valid],
                    covariance[target_valid],
                )
                predictive[method] = {**metrics, **diagnostics}
            steps.append(
                {
                    "segment_index": segment_index,
                    "frame": int(positions[current + 1]),
                    "frame_stride": int(segment["frame_stride"]),
                    "identity_rmse_m": {
                        method: _rmse(value, target, target_valid)
                        for method, value in predicted_positions.items()
                    },
                    "chamfer_rmse_m": {
                        method: _chamfer_rmse(
                            value[target_valid],
                            target[target_valid],
                            maximum_points=limits.chamfer_points,
                        )
                        for method, value in predicted_positions.items()
                    },
                    "predictive": predictive,
                }
            )
    if not steps:
        return None
    return {
        "object_id": object_id,
        "archive_path": path,
        "representation": "fixed_identity_trajectory",
        "array_key": key,
        "unit_source": unit_source,
        "frame_count": len(points),
        "track_count": points.shape[1],
        "segment_count": len(segments),
        "steps": steps,
    }


def _evaluate_hulls(
    packed: tuple[np.ndarray, tuple[np.ndarray, ...]],
    *,
    path: str,
    object_id: str,
    segments: Sequence[Mapping[str, Any]],
    limits: EvaluationLimits,
) -> dict[str, Any] | None:
    frames, hulls = packed
    steps: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        positions = np.asarray(segment["positions"], dtype=np.int64)
        positions = positions[positions < len(hulls)][: limits.max_frames_per_archive]
        if len(positions) < 4:
            continue
        selected = tuple(
            hulls[position][_indices(len(hulls[position]), limits.max_tracks)]
            for position in positions
        )
        if any(len(hull) == 0 for hull in selected):
            raise ValueError("structure seal admitted an empty visual hull")
        centroids = np.asarray([np.mean(hull, axis=0) for hull in selected])
        residual = np.diff(centroids, axis=0)[:, None, :]
        valid = np.ones(residual.shape[:2], dtype=bool)
        for current in range(2, len(selected) - 1):
            predictions = _rolling_prediction(residual, valid, end_frame=current)
            target = selected[current + 1]
            target_translation = centroids[current + 1] - centroids[current]
            last_supported = _last_supported_residual(
                residual,
                valid,
                end_frame=current,
            )[0]
            translations = {
                METHODS[0]: np.zeros(3),
                METHODS[1]: last_supported,
                METHODS[2]: predictions[METHODS[2]][0][0],
                METHODS[3]: predictions[METHODS[3]][0][0],
            }
            predictive = {}
            for method in PREDICTIVE_METHODS:
                mean, covariance, diagnostics = predictions[method]
                metrics = _predictive_metrics(
                    (target_translation - mean[0])[None], covariance
                )
                predictive[method] = {**metrics, **diagnostics}
            steps.append(
                {
                    "segment_index": segment_index,
                    "frame_index": int(frames[positions[current + 1]]),
                    "frame_stride": int(segment["frame_stride"]),
                    "centroid_error_m": {
                        method: float(np.linalg.norm(value - target_translation))
                        for method, value in translations.items()
                    },
                    "chamfer_rmse_m": {
                        method: _chamfer_rmse(
                            selected[current] + value,
                            target,
                            maximum_points=limits.chamfer_points,
                        )
                        for method, value in translations.items()
                    },
                    "predictive": predictive,
                }
            )
    if not steps:
        return None
    return {
        "object_id": object_id,
        "archive_path": path,
        "representation": "packed_visual_hulls",
        "array_key": "points_world_m",
        "unit_source": "declared_m",
        "frame_count": len(hulls),
        "track_count": None,
        "segment_count": len(segments),
        "steps": steps,
    }


def _case_metric(case: Mapping[str, Any], metric: str, method: str) -> float | None:
    steps = case.get("steps")
    if not isinstance(steps, list):
        raise ValueError("case steps are malformed")
    return _mean(float(step[metric][method]) for step in steps if metric in step)


def _case_predictive(case: Mapping[str, Any], metric: str, method: str) -> float | None:
    steps = case.get("steps")
    if not isinstance(steps, list):
        raise ValueError("case steps are malformed")
    return _mean(float(step["predictive"][method][metric]) for step in steps)


def _paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 1:
        raise ValueError("paired bootstrap inputs are invalid")
    delta = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    draws = np.mean(delta[indices], axis=1)
    return {
        "object_count": len(delta),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "lower_95_delta": float(np.quantile(draws, 0.025)),
        "upper_95_delta": float(np.quantile(draws, 0.975)),
        "probability_mean_improvement": float(np.mean(draws < 0.0)),
        "candidate_win_count": int(np.sum(delta < 0.0)),
        "tie_count": int(np.sum(delta == 0.0)),
    }


def _aggregate(cases: list[dict[str, Any]], limits: EvaluationLimits) -> dict[str, Any]:
    by_representation: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    point_non_regression = True
    for representation in sorted({case["representation"] for case in cases}):
        selected = [case for case in cases if case["representation"] == representation]
        report: dict[str, Any] = {
            "object_count": len(selected),
            "step_count": sum(len(case["steps"]) for case in selected),
        }
        metrics = (
            ("identity_rmse_m", "centroid_error_m", "chamfer_rmse_m")
            if representation == "fixed_identity_trajectory"
            else ("centroid_error_m", "chamfer_rmse_m")
        )
        for metric in metrics:
            values_by_method: dict[str, list[float]] = {}
            for method in METHODS:
                values = [
                    value
                    for case in selected
                    if (value := _case_metric(case, metric, method)) is not None
                ]
                if values:
                    values_by_method[method] = values
            if not values_by_method:
                continue
            report[metric] = {
                method: float(np.mean(values))
                for method, values in values_by_method.items()
            }
            raw = np.asarray(values_by_method[METHODS[2]], dtype=np.float64)
            normalized = np.asarray(values_by_method[METHODS[3]], dtype=np.float64)
            comparison = _paired_bootstrap(
                normalized,
                raw,
                samples=limits.bootstrap_samples,
                seed=limits.bootstrap_seed,
            )
            comparisons[f"{representation}/{metric}/normalized_vs_cumulative"] = (
                comparison
            )
            point_non_regression = point_non_regression and (
                comparison["upper_95_delta"] <= 0.0
            )
        by_representation[representation] = report

    predictive: dict[str, Any] = {}
    for method in PREDICTIVE_METHODS:
        predictive[method] = {
            metric: _mean(
                value
                for case in cases
                if (value := _case_predictive(case, metric, method)) is not None
            )
            for metric in (
                "coverage_90",
                "mean_nll",
                "mean_nees_over_dimension",
                "mean_predictive_std_m",
                "mean_effective_component_count",
                "mean_component_entropy_nats",
                "median_between_model_covariance_fraction",
            )
        }
    raw_predictive = predictive[METHODS[2]]
    normalized_predictive = predictive[METHODS[3]]
    uncertainty_gate = (
        normalized_predictive["mean_nll"] is not None
        and raw_predictive["mean_nll"] is not None
        and normalized_predictive["coverage_90"] is not None
        and raw_predictive["coverage_90"] is not None
        and normalized_predictive["mean_nll"] < raw_predictive["mean_nll"]
        and abs(normalized_predictive["coverage_90"] - 0.9)
        < abs(raw_predictive["coverage_90"] - 0.9)
    )
    return {
        "representations": by_representation,
        "predictive": predictive,
        "paired_comparisons": comparisons,
        "gates": {
            "normalized_vs_cumulative_uncertainty_passed": uncertainty_gate,
            "normalized_vs_cumulative_point_non_regression_passed": point_non_regression,
        },
    }


def evaluate_selection(
    seal_path: Path,
    structure_path: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    repository_root: Path,
    limits: EvaluationLimits | None = None,
) -> dict[str, Any]:
    """Open only sealed archives and evaluate frozen rolling predictors."""

    settings = EvaluationLimits() if limits is None else limits
    repository = repository_root.resolve()
    seal, protocol = _verify_seal(
        seal_path.resolve(), protocol_path.resolve(), repository
    )
    structure = _verify_structure(structure_path.resolve(), seal, protocol)
    root = Path(str(seal["dataset_root"])).resolve()
    cases: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    selected = seal["selected"]
    assert isinstance(selected, list)
    admitted = [
        entry for entry in structure["archives"] if entry.get("status") == "supported"
    ]
    for entry in admitted:
        if not isinstance(entry, Mapping):
            raise ValueError("selected archive entry is malformed")
        object_id = str(entry["object_id"])
        relative = str(entry["archive_path"])
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("selected archive escapes the mounted root")
        if _path_object_id(path, root) != object_id:
            raise ValueError("selected archive object identity changed")
        inspection: dict[str, Any] = {
            "object_id": object_id,
            "archive_path": relative,
        }
        try:
            with np.load(path, allow_pickle=False) as stored:
                packed = _packed_hulls(stored)
                if packed is not None:
                    case = _evaluate_hulls(
                        packed,
                        path=relative,
                        object_id=object_id,
                        segments=entry["segments"],
                        limits=settings,
                    )
                else:
                    case = _evaluate_fixed(
                        stored,
                        path=relative,
                        object_id=object_id,
                        segments=entry["segments"],
                        limits=settings,
                    )
        except (OSError, TypeError, ValueError) as error:
            inspection["status"] = "invalid_or_unsupported"
            inspection["error"] = f"{type(error).__name__}: {error}"
            inspections.append(inspection)
            continue
        inspection["status"] = "evaluated" if case is not None else "unsupported"
        inspections.append(inspection)
        if case is not None:
            cases.append(case)
    minimum_supported = int(
        _require_mapping(protocol["cohort_selection"], name="cohort_selection")[
            "minimum_supported_object_count_for_scientific_readout"
        ]
    )
    support_passed = len(cases) >= minimum_supported
    summary = (
        _aggregate(cases, settings)
        if cases
        else {
            "representations": {},
            "predictive": {},
            "paired_comparisons": {},
            "gates": {
                "normalized_vs_cumulative_uncertainty_passed": False,
                "normalized_vs_cumulative_point_non_regression_passed": False,
            },
        }
    )
    summary["gates"]["support_passed"] = support_passed
    summary["gates"]["overall_passed"] = bool(
        support_passed
        and summary["gates"]["normalized_vs_cumulative_uncertainty_passed"]
        and summary["gates"]["normalized_vs_cumulative_point_non_regression_passed"]
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_file_sha256": seal["protocol_file_sha256"],
        "selection_sha256": seal["selection_sha256"],
        "structure_sha256": structure["structure_sha256"],
        "repository_revision": seal["repository_revision"],
        "information_boundary": {
            "opened_paths_restricted_to_selection_seal": True,
            "selection_uses_numeric_payload": False,
            "selection_uses_target_outcomes": False,
            "structure_seal_precedes_coordinate_access": True,
            "constant_stride_segments_only": True,
            "method_parameters_refit": False,
            "official_deform360_benchmark_parity": False,
        },
        "limits": {
            "max_frames_per_archive": settings.max_frames_per_archive,
            "max_tracks": settings.max_tracks,
            "chamfer_points": settings.chamfer_points,
            "bootstrap_samples": settings.bootstrap_samples,
            "bootstrap_seed": settings.bootstrap_seed,
        },
        "selected_object_count": len(selected),
        "supported_object_count": len(cases),
        "minimum_supported_object_count": minimum_supported,
        "inspection": inspections,
        "summary": summary,
        "cases": cases,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_sha256"] = _canonical_sha256(result)
    _write_json(output_path.resolve(), result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal", help="select archives from names only")
    seal.add_argument("--data-root", required=True, type=Path)
    seal.add_argument("--protocol", required=True, type=Path)
    seal.add_argument("--output", required=True, type=Path)
    seal.add_argument("--repository-root", default=Path.cwd(), type=Path)

    structure = subparsers.add_parser(
        "structure", help="seal headers, cadence, and usable segments"
    )
    structure.add_argument("--seal", required=True, type=Path)
    structure.add_argument("--protocol", required=True, type=Path)
    structure.add_argument("--output", required=True, type=Path)
    structure.add_argument("--repository-root", default=Path.cwd(), type=Path)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate only the structure-sealed archive segments"
    )
    evaluate.add_argument("--seal", required=True, type=Path)
    evaluate.add_argument("--structure", required=True, type=Path)
    evaluate.add_argument("--protocol", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--repository-root", default=Path.cwd(), type=Path)
    evaluate.add_argument("--max-frames-per-archive", type=int, default=96)
    evaluate.add_argument("--max-tracks", type=int, default=2048)
    evaluate.add_argument("--chamfer-points", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "seal":
        result = seal_selection(
            args.data_root,
            args.protocol,
            args.output,
            repository_root=args.repository_root,
        )
        print(
            json.dumps(
                {
                    "selection_sha256": result["selection_sha256"],
                    "selected_object_count": len(result["selected"]),
                    "support_passed": result["support_passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result["support_passed"] else 3
    if args.command == "structure":
        result = inspect_structure(
            args.seal,
            args.protocol,
            args.output,
            repository_root=args.repository_root,
        )
        print(
            json.dumps(
                {
                    "structure_sha256": result["structure_sha256"],
                    "supported_object_count": result["supported_object_count"],
                    "support_passed": result["support_passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result["support_passed"] else 3
    if args.command == "evaluate":
        result = evaluate_selection(
            args.seal,
            args.structure,
            args.protocol,
            args.output,
            repository_root=args.repository_root,
            limits=EvaluationLimits(
                max_frames_per_archive=args.max_frames_per_archive,
                max_tracks=args.max_tracks,
                chamfer_points=args.chamfer_points,
            ),
        )
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
