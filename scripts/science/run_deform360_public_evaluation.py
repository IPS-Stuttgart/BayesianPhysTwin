#!/usr/bin/env python3
"""Locked independent-object Deform360 endpoint-evidence evaluation.

The workflow is deliberately split into three irreversible stages:

``select``
    consumes a names-only mounted-dataset inventory and committed historical
    exclusion contracts.  It selects fresh physical objects and exact archive
    paths without opening any numerical dataset payload.

``calibrate``
    opens only the selected calibration-object archives, evaluates cumulative
    and per-observation-normalized component evidence, and serializes
    object-group covariance scales.  Confirmation remains unauthorized when
    support is too small.

``confirm``
    opens only the separately selected confirmation-object archives after a
    hash-bound calibration artifact explicitly authorizes that stage.

This is an external trajectory-displacement transfer experiment.  It is not
Deform360 Table-4 parity, a physical simulator-state correction, or permission
to open the historically reserved Deform360 target cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointPosteriorV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

SELECTION_SCHEMA = "bayesian-phystwin/deform360-independent-selection-v1"
CALIBRATION_SCHEMA = "bayesian-phystwin/deform360-independent-calibration-v1"
RESULT_SCHEMA = "bayesian-phystwin/deform360-independent-result-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-independent-protocol-v1"
INVENTORY_SCHEMA = "bayesian-phystwin/deform360-name-inventory-v1"
CHI_SQUARE_3D_90 = 6.251388631170325
HORIZON_LABELS = ("h1", "h2", "h4", "h8")
METHODS = (
    "persistence",
    "last_displacement",
    "cumulative_evidence",
    "normalized_evidence",
)
POSTERIORS = ("cumulative_evidence", "normalized_evidence")
REPRESENTATIONS = ("fixed_identity_trajectory", "packed_visual_hulls")
_OBJECT_PATTERN = re.compile(r"^\d{3}-.+")
_EPISODE_PATTERN = re.compile(r"(?:episode[_-]?|ep)(\d{1,4})", re.IGNORECASE)
_PATH_HINTS = (
    "control",
    "track",
    "trajectory",
    "hull",
    "position",
    "particle",
)
_TRAJECTORY_HINTS = (
    "positions_world_m",
    "points_world_m",
    "control_points_world_m",
    "particle_tracks_world_m",
    "tracks_world_m",
    "trajectory_world_m",
    "positions_m",
    "points_m",
    "positions_mm",
    "points_mm",
)
_VALID_HINTS = ("valid_mask", "track_valid", "visibility", "valid")

Partition = Literal["calibration", "confirmation"]


@dataclass(frozen=True, slots=True)
class Protocol:
    """Validated low-dimensional external-evaluation protocol."""

    payload: Mapping[str, Any]
    sha256: str
    horizons: tuple[int, ...]
    strata: tuple[str, ...]
    max_frames: int
    max_tracks: int
    max_events_per_horizon: int
    minimum_prefix_displacements: int
    calibration_per_stratum: int
    confirmation_per_stratum: int
    minimum_calibration_groups: int
    minimum_calibration_groups_per_stratum: int
    minimum_confirmation_groups: int
    minimum_confirmation_groups_per_stratum: int
    group_coverage: float
    within_object_quantile: float
    bootstrap_samples: int
    bootstrap_seed: int


@dataclass(frozen=True, slots=True)
class SelectedArchive:
    """One metadata-selected physical object and its locked candidate archives."""

    object_id: str
    stratum: str
    episode_id: int
    archive_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrajectoryCase:
    """A supported Deform360 numerical representation from one locked archive."""

    representation: str
    points: np.ndarray | None
    valid: np.ndarray | None
    hulls: tuple[np.ndarray, ...] | None
    archive_path: str
    archive_sha256: str
    array_key: str
    unit_source: str

    @property
    def frame_count(self) -> int:
        if self.points is not None:
            return len(self.points)
        assert self.hulls is not None
        return len(self.hulls)


@dataclass(frozen=True, slots=True)
class PredictionMoments:
    """Endpoint mean, covariance, and mixture diagnostics."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    component_weights: np.ndarray


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """JSON record plus in-memory posterior events used for group calibration."""

    record: dict[str, Any]
    posterior_events: Mapping[str, Mapping[str, tuple[np.ndarray, ...]]]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
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


def _identity_payload(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(field, None)
    return result


def _require_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _require_probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _load_protocol(path: Path) -> Protocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROTOCOL_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError("unsupported Deform360 independent protocol")
    if payload.get("status") != "locked-before-numerical-payload-access":
        raise ValueError("protocol must be locked before numerical payload access")
    boundary = payload.get("information_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("protocol information boundary is missing")
    if boundary.get("historical_reserved_targets_must_remain_unopened") is not True:
        raise ValueError("protocol must preserve historical reserved targets")
    if boundary.get("selection_uses_names_only") is not True:
        raise ValueError("selection must use names only")
    if tuple(payload.get("methods", ())) != METHODS:
        raise ValueError("protocol method ordering changed")
    raw_strata = tuple(payload.get("strata", ()))
    if raw_strata != ("sheet", "volumetric"):
        raise ValueError("protocol strata must be exactly sheet and volumetric")
    raw_horizons = tuple(payload.get("horizons_frames", ()))
    horizons = tuple(
        _require_integer(value, name="horizons_frames", minimum=1)
        for value in raw_horizons
    )
    if (
        horizons != tuple(sorted(set(horizons)))
        or tuple(f"h{value}" for value in horizons) != HORIZON_LABELS
    ):
        raise ValueError("protocol horizons must be exactly 1, 2, 4, and 8")
    limits = payload.get("limits")
    selection = payload.get("selection")
    support = payload.get("support_gate")
    calibration = payload.get("group_calibration")
    bootstrap = payload.get("bootstrap")
    if not all(
        isinstance(value, Mapping)
        for value in (limits, selection, support, calibration, bootstrap)
    ):
        raise ValueError("protocol sections are malformed")
    assert isinstance(limits, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(support, Mapping)
    assert isinstance(calibration, Mapping)
    assert isinstance(bootstrap, Mapping)
    return Protocol(
        payload=payload,
        sha256=_canonical_sha256(payload),
        horizons=horizons,
        strata=raw_strata,
        max_frames=_require_integer(
            limits.get("max_frames_per_archive"),
            name="max_frames_per_archive",
            minimum=max(horizons) + 3,
        ),
        max_tracks=_require_integer(
            limits.get("max_tracks_per_archive"),
            name="max_tracks_per_archive",
            minimum=1,
        ),
        max_events_per_horizon=_require_integer(
            limits.get("max_evaluation_prefixes_per_horizon"),
            name="max_evaluation_prefixes_per_horizon",
            minimum=1,
        ),
        minimum_prefix_displacements=_require_integer(
            limits.get("minimum_prefix_displacements"),
            name="minimum_prefix_displacements",
            minimum=2,
        ),
        calibration_per_stratum=_require_integer(
            selection.get("calibration_objects_per_stratum"),
            name="calibration_objects_per_stratum",
            minimum=1,
        ),
        confirmation_per_stratum=_require_integer(
            selection.get("confirmation_objects_per_stratum"),
            name="confirmation_objects_per_stratum",
            minimum=1,
        ),
        minimum_calibration_groups=_require_integer(
            support.get("minimum_supported_calibration_objects"),
            name="minimum_supported_calibration_objects",
            minimum=3,
        ),
        minimum_calibration_groups_per_stratum=_require_integer(
            support.get("minimum_supported_calibration_objects_per_stratum"),
            name="minimum_supported_calibration_objects_per_stratum",
            minimum=1,
        ),
        minimum_confirmation_groups=_require_integer(
            support.get("minimum_supported_confirmation_objects"),
            name="minimum_supported_confirmation_objects",
            minimum=1,
        ),
        minimum_confirmation_groups_per_stratum=_require_integer(
            support.get("minimum_supported_confirmation_objects_per_stratum"),
            name="minimum_supported_confirmation_objects_per_stratum",
            minimum=1,
        ),
        group_coverage=_require_probability(
            calibration.get("across_object_coverage"),
            name="across_object_coverage",
        ),
        within_object_quantile=_require_probability(
            calibration.get("within_object_event_quantile"),
            name="within_object_event_quantile",
        ),
        bootstrap_samples=_require_integer(
            bootstrap.get("samples"), name="bootstrap.samples", minimum=100
        ),
        bootstrap_seed=_require_integer(
            bootstrap.get("seed"), name="bootstrap.seed", minimum=0
        ),
    )


def _load_inventory(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != INVENTORY_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError("unsupported Deform360 names-only inventory")
    boundary = payload.get("information_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("inventory information boundary is missing")
    required = {
        "dataset_payload_opened": False,
        "file_contents_hashed": False,
        "names_and_directory_structure_only": True,
        "reserved_target_outcomes_opened": False,
    }
    if any(boundary.get(key) is not expected for key, expected in required.items()):
        raise ValueError("inventory does not preserve the names-only boundary")
    declared = payload.get("inventory_sha256")
    actual = _canonical_sha256(_identity_payload(payload, "inventory_sha256"))
    if declared != actual:
        raise ValueError("inventory SHA-256 does not match its canonical payload")
    return payload, actual


def _historical_exclusions(
    v1_path: Path,
    v2_path: Path,
) -> tuple[set[str], dict[str, tuple[str, ...]]]:
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))["config"]
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))["config"]
    pools = {
        stratum: tuple(str(value) for value in values)
        for stratum, values in v1["candidate_pools"].items()
    }
    exclusions = set(map(str, v1["open_or_reserved_objects"]))
    for cohort_name in ("calibration_cohort", "target_cohort"):
        for records in v1[cohort_name].values():
            exclusions.update(str(record["object_id"]) for record in records)
        for records in v2[cohort_name].values():
            exclusions.update(str(record["object_id"]) for record in records)
    repair = v2.get("repair", {})
    exclusions.update(map(str, repair.get("prior_path_open_objects_excluded", ())))
    # V2's three fresh calibration objects are included in calibration_cohort, but
    # retaining this explicit assertion makes accidental config drift visible.
    exclusions.update(
        str(record["object_id"]) for record in repair.get("fresh_calibration", ())
    )
    return exclusions, pools


def _episode_from_path(path: str) -> int | None:
    return next(
        (
            int(match.group(1))
            for part in PurePosixPath(path).parts
            if (match := _EPISODE_PATTERN.search(part)) is not None
        ),
        None,
    )


def _path_rank(seed: str, object_id: str, episode: int, path: str) -> tuple[int, str]:
    lowered = path.lower()
    hint_rank = next(
        (index for index, hint in enumerate(_PATH_HINTS) if hint in lowered),
        len(_PATH_HINTS),
    )
    digest = hashlib.sha256(
        f"{seed}|archive|{object_id}|{episode}|{path}".encode()
    ).hexdigest()
    return hint_rank, digest


def _object_rank(seed: str, stratum: str, object_id: str) -> str:
    return hashlib.sha256(f"{seed}|object|{stratum}|{object_id}".encode()).hexdigest()


def _episode_rank(seed: str, object_id: str, episode: int) -> str:
    return hashlib.sha256(f"{seed}|episode|{object_id}|{episode}".encode()).hexdigest()


def build_selection(
    inventory_path: Path,
    protocol_path: Path,
    v1_path: Path,
    v2_path: Path,
) -> dict[str, Any]:
    """Select exact fresh-object archive paths from names-only metadata."""

    protocol = _load_protocol(protocol_path)
    inventory, inventory_sha = _load_inventory(inventory_path)
    exclusions, pools = _historical_exclusions(v1_path, v2_path)
    seed = str(protocol.payload["selection"]["seed"])
    max_paths = _require_integer(
        protocol.payload["selection"]["maximum_locked_archive_paths_per_object"],
        name="maximum_locked_archive_paths_per_object",
        minimum=1,
    )
    objects = inventory.get("objects")
    if not isinstance(objects, Sequence):
        raise ValueError("inventory objects must be an array")
    inventory_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in objects:
        if not isinstance(raw, Mapping):
            raise ValueError("inventory object entry is malformed")
        object_id = raw.get("object_id")
        if not isinstance(object_id, str) or not _OBJECT_PATTERN.fullmatch(object_id):
            raise ValueError("inventory object ID is malformed")
        if object_id in inventory_by_id:
            raise ValueError(f"inventory repeats object {object_id}")
        inventory_by_id[object_id] = raw

    partitions: dict[str, list[dict[str, Any]]] = {
        "calibration": [],
        "confirmation": [],
    }
    insufficiency: dict[str, dict[str, int]] = {}
    for stratum in protocol.strata:
        candidates: list[tuple[str, str, int, tuple[str, ...]]] = []
        for object_id in pools[stratum]:
            if object_id in exclusions or object_id not in inventory_by_id:
                continue
            record = inventory_by_id[object_id]
            raw_episodes = record.get("episode_ids_from_names", ())
            if not isinstance(raw_episodes, Sequence):
                continue
            episodes = sorted(
                {
                    _require_integer(value, name="episode ID", minimum=0)
                    for value in raw_episodes
                },
                key=lambda value: _episode_rank(seed, object_id, value),
            )
            raw_paths = record.get("numeric_paths", record.get("sample_paths", ()))
            if not isinstance(raw_paths, Sequence):
                continue
            paths = tuple(
                str(value)
                for value in raw_paths
                if isinstance(value, str) and value.lower().endswith(".npz")
            )
            for episode in episodes:
                episode_paths = tuple(
                    path for path in paths if _episode_from_path(path) == episode
                )
                if not episode_paths:
                    continue
                ranked_paths = tuple(
                    sorted(
                        episode_paths,
                        key=lambda path: _path_rank(seed, object_id, episode, path),
                    )[:max_paths]
                )
                candidates.append(
                    (
                        _object_rank(seed, stratum, object_id),
                        object_id,
                        episode,
                        ranked_paths,
                    )
                )
                break
        candidates.sort(key=lambda value: (value[0], value[1]))
        required = protocol.calibration_per_stratum + protocol.confirmation_per_stratum
        insufficiency[stratum] = {
            "eligible": len(candidates),
            "required": required,
        }
        selected = candidates[:required]
        calibration = selected[: protocol.calibration_per_stratum]
        confirmation = selected[protocol.calibration_per_stratum : required]
        for partition, records in (
            ("calibration", calibration),
            ("confirmation", confirmation),
        ):
            for _, object_id, episode, paths in records:
                partitions[partition].append(
                    {
                        "object_id": object_id,
                        "stratum": stratum,
                        "episode_id": episode,
                        "archive_paths": list(paths),
                    }
                )

    complete = all(
        item["eligible"] >= item["required"] for item in insufficiency.values()
    )
    payload: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        "inventory_sha256": inventory_sha,
        "selection_seed": seed,
        "historical_exclusion_sha256": _canonical_sha256(sorted(exclusions)),
        "historical_exclusion_count": len(exclusions),
        "selection_complete": complete,
        "insufficiency": insufficiency,
        "information_boundary": {
            "selection_used_dataset_payload": False,
            "selection_used_numerical_outcome": False,
            "selection_used_names_only": True,
            "historical_reserved_targets_selected": False,
            "replacement_after_payload_access_allowed": False,
        },
        "calibration": sorted(
            partitions["calibration"],
            key=lambda item: (item["stratum"], item["object_id"]),
        ),
        "confirmation": sorted(
            partitions["confirmation"],
            key=lambda item: (item["stratum"], item["object_id"]),
        ),
    }
    selected_ids = {
        item["object_id"]
        for partition in ("calibration", "confirmation")
        for item in payload[partition]
    }
    if selected_ids & exclusions:
        raise AssertionError("historically open or reserved object was selected")
    if len(selected_ids) != len(payload["calibration"]) + len(payload["confirmation"]):
        raise AssertionError("selection repeats a physical object")
    payload["selection_sha256"] = _canonical_sha256(payload)
    return payload


def _load_selection(path: Path, protocol: Protocol) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SELECTION_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError("unsupported Deform360 selection")
    if payload.get("protocol_sha256") != protocol.sha256:
        raise ValueError("selection is bound to a different protocol")
    declared = payload.get("selection_sha256")
    actual = _canonical_sha256(_identity_payload(payload, "selection_sha256"))
    if declared != actual:
        raise ValueError("selection SHA-256 changed")
    if payload.get("selection_complete") is not True:
        raise ValueError("names-only selection is incomplete")
    boundary = payload.get("information_boundary")
    if not isinstance(boundary, Mapping) or any(
        (
            boundary.get("selection_used_dataset_payload") is not False,
            boundary.get("selection_used_numerical_outcome") is not False,
            boundary.get("selection_used_names_only") is not True,
            boundary.get("historical_reserved_targets_selected") is not False,
            boundary.get("replacement_after_payload_access_allowed") is not False,
        )
    ):
        raise ValueError("selection information boundary changed")
    seen: set[str] = set()
    for partition, expected_count in (
        (
            "calibration",
            len(protocol.strata) * protocol.calibration_per_stratum,
        ),
        (
            "confirmation",
            len(protocol.strata) * protocol.confirmation_per_stratum,
        ),
    ):
        records = payload.get(partition)
        if not isinstance(records, Sequence) or len(records) != expected_count:
            raise ValueError(f"selection {partition} count changed")
        for raw in records:
            selected = _parse_selected_archive(raw)
            if selected.object_id in seen:
                raise ValueError("selection repeats a physical object")
            seen.add(selected.object_id)
    return payload, actual


def _parse_selected_archive(raw: object) -> SelectedArchive:
    if not isinstance(raw, Mapping):
        raise ValueError("selected archive entry must be an object")
    object_id = raw.get("object_id")
    stratum = raw.get("stratum")
    episode = raw.get("episode_id")
    raw_paths = raw.get("archive_paths")
    if not isinstance(object_id, str) or not _OBJECT_PATTERN.fullmatch(object_id):
        raise ValueError("selected object ID is malformed")
    if stratum not in {"sheet", "volumetric"}:
        raise ValueError("selected stratum is malformed")
    episode_id = _require_integer(episode, name="selected episode ID", minimum=0)
    if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes)):
        raise ValueError("selected archive paths must be an array")
    paths = tuple(raw_paths)
    if not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("selected archive paths must be nonempty strings")
    if len(paths) != len(set(paths)):
        raise ValueError("selected archive paths contain duplicates")
    for path in paths:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError("selected archive path escapes the data root")
        if object_id not in pure.parts:
            raise ValueError("selected archive path is not object scoped")
        if not path.lower().endswith(".npz"):
            raise ValueError("selected archive path must be an NPZ file")
        path_episode = _episode_from_path(path)
        if path_episode != episode_id:
            raise ValueError("selected archive path episode changed")
    return SelectedArchive(
        object_id=object_id,
        stratum=str(stratum),
        episode_id=episode_id,
        archive_paths=paths,
    )


def _safe_path(root: Path, relative: str) -> Path:
    candidate = (root / PurePosixPath(relative)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("selected archive path escapes the data root")
    return candidate


def _indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def _validity(stored: Any, shape: tuple[int, int]) -> np.ndarray:
    for key in stored.files:
        if not any(hint in key.lower() for hint in _VALID_HINTS):
            continue
        value = np.asarray(stored[key])
        if value.shape == shape:
            return np.asarray(value, dtype=bool)
    return np.ones(shape, dtype=bool)


def _packed_hulls(
    stored: Any, *, max_frames: int, max_tracks: int
) -> TrajectoryCase | None:
    keys = set(stored.files)
    required = {"frame_indices", "point_offsets", "points_world_m"}
    if not required.issubset(keys):
        return None
    frames = np.asarray(stored["frame_indices"])
    offsets = np.asarray(stored["point_offsets"])
    points = np.asarray(stored["points_world_m"], dtype=np.float64)
    if (
        frames.ndim != 1
        or len(frames) < 4
        or offsets.shape != (len(frames) + 1,)
        or not np.issubdtype(offsets.dtype, np.integer)
        or offsets[0] != 0
        or offsets[-1] != len(points)
        or np.any(np.diff(offsets) <= 0)
        or points.ndim != 2
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        return None
    count = min(len(frames), max_frames)
    hulls = tuple(
        points[int(offsets[index]) : int(offsets[index + 1])][
            _indices(int(offsets[index + 1] - offsets[index]), max_tracks)
        ]
        for index in range(count)
    )
    if any(len(hull) == 0 for hull in hulls):
        return None
    return TrajectoryCase(
        representation="packed_visual_hulls",
        points=None,
        valid=None,
        hulls=hulls,
        archive_path="",
        archive_sha256="",
        array_key="points_world_m",
        unit_source="declared_m",
    )


def _fixed_trajectory(
    stored: Any, *, max_frames: int, max_tracks: int
) -> TrajectoryCase | None:
    candidates: list[tuple[int, str, np.ndarray, float, str]] = []
    for key in stored.files:
        lowered = key.lower()
        rank = next(
            (index for index, hint in enumerate(_TRAJECTORY_HINTS) if hint == lowered),
            None,
        )
        if rank is None:
            continue
        value = np.asarray(stored[key])
        if value.ndim != 3 or value.shape[-1] != 3 or value.shape[0] < 4:
            continue
        if lowered.endswith("_mm"):
            scale, unit = 1e-3, "declared_mm"
        elif lowered.endswith("_m") or "world_m" in lowered:
            scale, unit = 1.0, "declared_m"
        else:  # pragma: no cover - all exact hints declare a unit
            continue
        candidates.append((rank, key, value, scale, unit))
    if not candidates:
        return None
    _, key, raw, scale, unit = min(candidates, key=lambda item: (item[0], item[1]))
    raw = np.asarray(raw[:max_frames], dtype=np.float64)
    tracks = _indices(raw.shape[1], max_tracks)
    raw = raw[:, tracks]
    valid = _validity(stored, np.asarray(stored[key]).shape[:2])[: len(raw), tracks]
    finite = np.all(np.isfinite(raw), axis=2)
    valid &= finite
    if not np.any(valid):
        return None
    points = np.where(finite[:, :, None], raw, 0.0) * scale
    return TrajectoryCase(
        representation="fixed_identity_trajectory",
        points=points,
        valid=valid,
        hulls=None,
        archive_path="",
        archive_sha256="",
        array_key=key,
        unit_source=unit,
    )


def _open_selected_case(
    root: Path,
    selected: SelectedArchive,
    protocol: Protocol,
) -> tuple[TrajectoryCase | None, list[dict[str, str]]]:
    attempts: list[dict[str, str]] = []
    for relative in selected.archive_paths:
        path = _safe_path(root, relative)
        if not path.is_file():
            attempts.append({"path": relative, "status": "missing"})
            continue
        digest = _sha256(path)
        try:
            with np.load(path, allow_pickle=False) as stored:
                case = _packed_hulls(
                    stored,
                    max_frames=protocol.max_frames,
                    max_tracks=protocol.max_tracks,
                )
                if case is None:
                    case = _fixed_trajectory(
                        stored,
                        max_frames=protocol.max_frames,
                        max_tracks=protocol.max_tracks,
                    )
        except (OSError, TypeError, ValueError) as error:
            attempts.append(
                {
                    "path": relative,
                    "status": "invalid",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if case is None:
            attempts.append({"path": relative, "status": "unsupported"})
            continue
        attempts.append({"path": relative, "status": "selected"})
        return (
            TrajectoryCase(
                representation=case.representation,
                points=case.points,
                valid=case.valid,
                hulls=case.hulls,
                archive_path=relative,
                archive_sha256=digest,
                array_key=case.array_key,
                unit_source=case.unit_source,
            ),
            attempts,
        )
    return None, attempts


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(centered)
    return weights / np.sum(weights, axis=1, keepdims=True)


def _moments_from_weights(
    posterior: ModelAveragedEndpointPosteriorV1,
    weights: np.ndarray,
    *,
    horizon_steps: int,
) -> PredictionMoments:
    component_variance = (
        posterior.component_variance_m2
        + horizon_steps * posterior.component_process_variance_m2[:, None]
    )
    mean = np.einsum("nk,knc->nc", weights, posterior.component_mean_m)
    centered = posterior.component_mean_m - mean[None, :, :]
    outer = centered[:, :, :, None] * centered[:, :, None, :]
    within = component_variance[:, :, None, None] * np.eye(3)
    covariance = np.einsum("nk,knij->nij", weights, within + outer)
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    return PredictionMoments(
        mean_m=mean,
        covariance_m2=covariance,
        component_weights=weights,
    )


def normalized_evidence_prediction(
    posterior: ModelAveragedEndpointPosteriorV1,
    *,
    horizon_steps: int = 1,
) -> PredictionMoments:
    """Average components by mean log evidence per supported observation."""

    if not isinstance(posterior, ModelAveragedEndpointPosteriorV1):
        raise TypeError("posterior must be a ModelAveragedEndpointPosteriorV1")
    horizon = _require_integer(horizon_steps, name="horizon_steps", minimum=0)
    prior = np.asarray(posterior.config.component_prior_probability, dtype=np.float64)
    counts = np.maximum(posterior.update_count, 1).astype(np.float64)
    logits = np.log(prior)[None, :] + posterior.component_log_evidence / counts[:, None]
    weights = _softmax(logits)
    return _moments_from_weights(posterior, weights, horizon_steps=horizon)


def _cumulative_prediction(
    posterior: ModelAveragedEndpointPosteriorV1,
) -> PredictionMoments:
    prediction = predict_model_averaged_endpoint(posterior, horizon_steps=1)
    return PredictionMoments(
        mean_m=prediction.mean_m,
        covariance_m2=prediction.covariance_m2,
        component_weights=prediction.component_weights,
    )


def _effective_component_count(weights: np.ndarray) -> float:
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1)
    return float(np.mean(np.exp(entropy)))


def _chamfer_rmse(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left = left[_indices(len(left), 512)]
    right = right[_indices(len(right), 512)]
    if len(left) == 0 or len(right) == 0:
        raise ValueError("Chamfer inputs must be nonempty")

    def directed(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        minimum = np.full(len(source), np.inf, dtype=np.float64)
        for start in range(0, len(target), 128):
            block = target[start : start + 128]
            squared = np.sum(np.square(source[:, None, :] - block[None, :, :]), axis=2)
            minimum = np.minimum(minimum, np.min(squared, axis=1))
        return minimum

    value = 0.5 * (
        float(np.mean(directed(left, right))) + float(np.mean(directed(right, left)))
    )
    return float(np.sqrt(max(value, 0.0)))


def _last_valid_displacement(
    displacement: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    result = np.zeros(displacement.shape[1:], dtype=np.float64)
    for track in range(displacement.shape[1]):
        support = np.flatnonzero(valid[:, track])
        if len(support):
            result[track] = displacement[support[-1], track]
    return result


def _evaluation_currents(
    frame_count: int,
    horizon: int,
    minimum_prefix: int,
    maximum_events: int,
) -> np.ndarray:
    earliest = horizon + minimum_prefix - 1
    latest_exclusive = frame_count - horizon
    if earliest >= latest_exclusive:
        return np.empty(0, dtype=np.int64)
    values = np.arange(earliest, latest_exclusive, dtype=np.int64)
    if len(values) <= maximum_events:
        return values
    return np.unique(np.linspace(values[0], values[-1], maximum_events, dtype=np.int64))


def _predictive_events(
    error: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(error, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("predictive error must have shape (N, 3)")
    if cov.shape != (len(values), 3, 3):
        raise ValueError("predictive covariance shape changed")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    projected = np.einsum("nji,nj->ni", eigenvectors, values)
    nees = np.sum(np.square(projected) / eigenvalues, axis=1)
    logdet = np.sum(np.log(eigenvalues), axis=1)
    nll = 0.5 * (3.0 * math.log(2.0 * math.pi) + logdet + nees)
    return {
        "nees": nees,
        "nll": nll,
        "error_norm_m": np.linalg.norm(values, axis=1),
        "predictive_std_m": np.sqrt(np.trace(cov, axis1=1, axis2=2) / 3.0),
    }


def _empty_event_collector() -> dict[str, list[np.ndarray]]:
    return {
        "nees": [],
        "nll": [],
        "error_norm_m": [],
        "predictive_std_m": [],
    }


def _append_events(
    collector: dict[str, list[np.ndarray]],
    events: Mapping[str, np.ndarray],
) -> None:
    for name, values in events.items():
        collector[name].append(np.asarray(values, dtype=np.float64))


def _event_summary(
    events: Mapping[str, Sequence[np.ndarray]], scale: float = 1.0
) -> dict[str, Any]:
    if not events["nees"]:
        return {"count": 0}
    nees = np.concatenate(events["nees"]) / scale
    nll_values = np.concatenate(events["nll"])
    error = np.concatenate(events["error_norm_m"])
    std = np.concatenate(events["predictive_std_m"]) * math.sqrt(scale)
    if scale != 1.0:
        # Recompute NLL under an isotropic covariance multiplier.  The exact
        # Mahalanobis term scales by 1/scale and log determinant by 3 log(scale).
        nll_values = nll_values + 0.5 * (
            3.0 * math.log(scale) + np.concatenate(events["nees"]) * (1.0 / scale - 1.0)
        )
    return {
        "count": int(len(nees)),
        "coverage_90": float(np.mean(nees <= CHI_SQUARE_3D_90)),
        "mean_nees_over_dimension": float(np.mean(nees) / 3.0),
        "mean_nll": float(np.mean(nll_values)),
        "mean_error_norm_m": float(np.mean(error)),
        "mean_predictive_std_m": float(np.mean(std)),
    }


def _case_arrays(
    case: TrajectoryCase,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...] | None]:
    if case.points is not None:
        assert case.valid is not None
        return case.points, case.valid, None
    assert case.hulls is not None
    centroids = np.asarray([np.mean(hull, axis=0) for hull in case.hulls])[:, None, :]
    return centroids, np.ones(centroids.shape[:2], dtype=bool), case.hulls


def _mean_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    return None if not len(array) else float(np.mean(array))


def _evaluate_case(
    root: Path,
    selected: SelectedArchive,
    protocol: Protocol,
) -> CaseEvaluation:
    case, attempts = _open_selected_case(root, selected, protocol)
    if case is None:
        return CaseEvaluation(
            record={
                "object_id": selected.object_id,
                "stratum": selected.stratum,
                "episode_id": selected.episode_id,
                "status": "unsupported",
                "archive_attempts": attempts,
            },
            posterior_events={},
        )
    points, validity, hulls = _case_arrays(case)
    metrics: dict[str, dict[str, list[float]]] = {
        label: {f"kinematic_rmse_m/{method}": [] for method in METHODS}
        for label in HORIZON_LABELS
    }
    for label in HORIZON_LABELS:
        for method in METHODS:
            metrics[label][f"chamfer_rmse_m/{method}"] = []
    event_collectors: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        posterior: {label: _empty_event_collector() for label in HORIZON_LABELS}
        for posterior in POSTERIORS
    }
    effective: dict[str, dict[str, list[float]]] = {
        posterior: {label: [] for label in HORIZON_LABELS} for posterior in POSTERIORS
    }

    for horizon in protocol.horizons:
        label = f"h{horizon}"
        currents = _evaluation_currents(
            len(points),
            horizon,
            protocol.minimum_prefix_displacements,
            protocol.max_events_per_horizon,
        )
        for current in currents:
            starts = np.arange(0, current - horizon + 1, dtype=np.int64)
            historical = points[starts + horizon] - points[starts]
            historical_valid = validity[starts + horizon] & validity[starts]
            historical = np.where(historical_valid[:, :, None], historical, 0.0)
            target_valid = validity[current] & validity[current + horizon]
            if not np.any(target_valid):
                continue
            posterior = infer_model_averaged_endpoint(
                historical,
                historical_valid,
                end_frame=len(historical),
            )
            cumulative = _cumulative_prediction(posterior)
            normalized = normalized_evidence_prediction(posterior)
            last = _last_valid_displacement(historical, historical_valid)
            displacement = {
                "persistence": np.zeros_like(last),
                "last_displacement": last,
                "cumulative_evidence": cumulative.mean_m,
                "normalized_evidence": normalized.mean_m,
            }
            target_position = points[current + horizon]
            current_position = points[current]
            for method, delta in displacement.items():
                prediction = current_position + delta
                error = prediction[target_valid] - target_position[target_valid]
                metrics[label][f"kinematic_rmse_m/{method}"].append(
                    float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))
                )
                if hulls is None:
                    metrics[label][f"chamfer_rmse_m/{method}"].append(
                        _chamfer_rmse(
                            prediction[target_valid], target_position[target_valid]
                        )
                    )
                else:
                    translation = delta[0]
                    metrics[label][f"chamfer_rmse_m/{method}"].append(
                        _chamfer_rmse(
                            hulls[current] + translation,
                            hulls[current + horizon],
                        )
                    )
            target_displacement = target_position - current_position
            for posterior_name, moments in (
                ("cumulative_evidence", cumulative),
                ("normalized_evidence", normalized),
            ):
                events = _predictive_events(
                    target_displacement[target_valid] - moments.mean_m[target_valid],
                    moments.covariance_m2[target_valid],
                )
                _append_events(event_collectors[posterior_name][label], events)
                effective[posterior_name][label].append(
                    _effective_component_count(moments.component_weights[target_valid])
                )

    per_horizon: dict[str, Any] = {}
    for label in HORIZON_LABELS:
        point: dict[str, Any] = {}
        for method in METHODS:
            point[method] = {
                "kinematic_rmse_m": _mean_or_none(
                    metrics[label][f"kinematic_rmse_m/{method}"]
                ),
                "chamfer_rmse_m": _mean_or_none(
                    metrics[label][f"chamfer_rmse_m/{method}"]
                ),
            }
        posterior_summary = {
            name: {
                "raw": _event_summary(event_collectors[name][label]),
                "mean_effective_component_count": _mean_or_none(effective[name][label]),
            }
            for name in POSTERIORS
        }
        per_horizon[label] = {
            "point": point,
            "posterior": posterior_summary,
        }
    supported_horizons = sum(
        per_horizon[label]["posterior"]["normalized_evidence"]["raw"].get("count", 0)
        > 0
        for label in HORIZON_LABELS
    )
    status = "evaluated" if supported_horizons else "insufficient_temporal_support"
    # Keep field-level arrays in a compact nested mapping expected by calibration.
    events_by_posterior: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
    for posterior_name in POSTERIORS:
        events_by_posterior[posterior_name] = {}
        for label in HORIZON_LABELS:
            collector = event_collectors[posterior_name][label]
            events_by_posterior[posterior_name][f"{label}/nees"] = tuple(
                collector["nees"]
            )
            events_by_posterior[posterior_name][f"{label}/nll"] = tuple(
                collector["nll"]
            )
            events_by_posterior[posterior_name][f"{label}/error_norm_m"] = tuple(
                collector["error_norm_m"]
            )
            events_by_posterior[posterior_name][f"{label}/predictive_std_m"] = tuple(
                collector["predictive_std_m"]
            )
    record = {
        "object_id": selected.object_id,
        "stratum": selected.stratum,
        "episode_id": selected.episode_id,
        "status": status,
        "archive_attempts": attempts,
        "selected_archive": {
            "path": case.archive_path,
            "sha256": case.archive_sha256,
            "representation": case.representation,
            "array_key": case.array_key,
            "unit_source": case.unit_source,
            "frame_count": case.frame_count,
            "track_count": None if case.points is None else int(case.points.shape[1]),
        },
        "horizons": per_horizon,
    }
    return CaseEvaluation(record=record, posterior_events=events_by_posterior)


def _event_mapping(
    case: CaseEvaluation,
    posterior: str,
    label: str,
) -> dict[str, tuple[np.ndarray, ...]]:
    values = case.posterior_events.get(posterior, {})
    return {
        field: tuple(values.get(f"{label}/{field}", ()))
        for field in ("nees", "nll", "error_norm_m", "predictive_std_m")
    }


def _finite_group_quantile(
    values: np.ndarray, coverage: float
) -> tuple[float, int, float]:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("group conformal scores must be a finite vector")
    rank = int(math.ceil((len(scores) + 1) * coverage))
    if rank > len(scores):
        return math.inf, rank, rank / (len(scores) + 1)
    value = float(np.partition(scores, rank - 1)[rank - 1])
    return value, rank, rank / (len(scores) + 1)


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("within-object quantile requires events")
    rank = int(math.ceil(probability * len(array))) - 1
    return float(np.sort(array)[min(max(rank, 0), len(array) - 1)])


def _support_summary(
    cases: Sequence[CaseEvaluation],
    strata: Sequence[str],
) -> tuple[list[CaseEvaluation], dict[str, int]]:
    supported = [case for case in cases if case.record["status"] == "evaluated"]
    by_stratum = {
        stratum: sum(case.record["stratum"] == stratum for case in supported)
        for stratum in strata
    }
    return supported, by_stratum


def calibrate(
    data_root: Path,
    protocol_path: Path,
    selection_path: Path,
    *,
    revision: str | None,
) -> dict[str, Any]:
    """Open only calibration objects and freeze normalized-evidence scales."""

    protocol = _load_protocol(protocol_path)
    selection, selection_sha = _load_selection(selection_path, protocol)
    root = data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    selected = tuple(_parse_selected_archive(raw) for raw in selection["calibration"])
    cases = tuple(_evaluate_case(root, item, protocol) for item in selected)
    supported, by_stratum = _support_summary(cases, protocol.strata)
    support_passed = len(supported) >= protocol.minimum_calibration_groups and all(
        count >= protocol.minimum_calibration_groups_per_stratum
        for count in by_stratum.values()
    )
    scales: dict[str, Any] = {}
    previous_object_scale = 1.0
    previous_simultaneous_scale = 1.0
    finite_rank_passed = True
    if support_passed:
        for label in HORIZON_LABELS:
            object_scores: list[float] = []
            simultaneous_scores: list[float] = []
            contributing: list[str] = []
            for case in supported:
                arrays = _event_mapping(case, "normalized_evidence", label)
                if not arrays["nees"]:
                    continue
                nees = np.concatenate(arrays["nees"])
                ratios = nees / CHI_SQUARE_3D_90
                object_scores.append(
                    _higher_quantile(ratios, protocol.within_object_quantile)
                )
                simultaneous_scores.append(float(np.max(ratios)))
                contributing.append(str(case.record["object_id"]))
            if len(object_scores) < protocol.minimum_calibration_groups:
                finite_rank_passed = False
                scales[label] = {
                    "status": "too_few_groups",
                    "group_count": len(object_scores),
                }
                continue
            object_scale, rank, finite_coverage = _finite_group_quantile(
                np.asarray(object_scores), protocol.group_coverage
            )
            simultaneous_scale, simultaneous_rank, simultaneous_coverage = (
                _finite_group_quantile(
                    np.asarray(simultaneous_scores), protocol.group_coverage
                )
            )
            if not math.isfinite(object_scale) or not math.isfinite(simultaneous_scale):
                finite_rank_passed = False
            object_scale = max(previous_object_scale, 1.0, object_scale)
            simultaneous_scale = max(
                previous_simultaneous_scale, 1.0, simultaneous_scale
            )
            previous_object_scale = object_scale
            previous_simultaneous_scale = simultaneous_scale
            scales[label] = {
                "status": "calibrated",
                "group_count": len(object_scores),
                "contributing_object_ids": contributing,
                "object_q90_scale": object_scale,
                "object_q90_group_scores": object_scores,
                "object_q90_finite_sample_rank": rank,
                "object_q90_finite_sample_coverage": finite_coverage,
                "simultaneous_scale": simultaneous_scale,
                "simultaneous_group_scores": simultaneous_scores,
                "simultaneous_finite_sample_rank": simultaneous_rank,
                "simultaneous_finite_sample_coverage": simultaneous_coverage,
            }
    authorized = support_passed and finite_rank_passed
    payload: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "schema_version": 1,
        "repository_revision": revision,
        "protocol_sha256": protocol.sha256,
        "selection_sha256": selection_sha,
        "partition": "calibration",
        "information_boundary": {
            "calibration_payloads_opened": True,
            "confirmation_payloads_opened": False,
            "historical_reserved_targets_opened": False,
            "replacement_after_payload_access": False,
        },
        "support_gate": {
            "passed": support_passed,
            "finite_group_rank_passed": finite_rank_passed,
            "supported_object_count": len(supported),
            "supported_by_stratum": by_stratum,
            "required_object_count": protocol.minimum_calibration_groups,
            "required_per_stratum": protocol.minimum_calibration_groups_per_stratum,
        },
        "confirmation_authorized": authorized,
        "scales": scales,
        "cases": [case.record for case in cases],
    }
    payload["calibration_sha256"] = _canonical_sha256(payload)
    return payload


def _load_calibration(
    path: Path,
    protocol: Protocol,
    selection_sha: str,
) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != CALIBRATION_SCHEMA
        or payload.get("schema_version") != 1
    ):
        raise ValueError("unsupported calibration artifact")
    if payload.get("protocol_sha256") != protocol.sha256:
        raise ValueError("calibration protocol identity changed")
    if payload.get("selection_sha256") != selection_sha:
        raise ValueError("calibration selection identity changed")
    declared = payload.get("calibration_sha256")
    actual = _canonical_sha256(_identity_payload(payload, "calibration_sha256"))
    if declared != actual:
        raise ValueError("calibration SHA-256 changed")
    if payload.get("confirmation_authorized") is not True:
        raise PermissionError(
            "calibration did not authorize confirmation payload access"
        )
    boundary = payload.get("information_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("confirmation_payloads_opened") is not False
    ):
        raise ValueError("calibration artifact already opened confirmation payloads")
    return payload, actual


def _aggregate_point(cases: Sequence[CaseEvaluation]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in HORIZON_LABELS:
        methods: dict[str, Any] = {}
        for method in METHODS:
            methods[method] = {}
            for metric in ("kinematic_rmse_m", "chamfer_rmse_m"):
                values = [
                    case.record["horizons"][label]["point"][method][metric]
                    for case in cases
                    if case.record["horizons"][label]["point"][method][metric]
                    is not None
                ]
                methods[method][metric] = _mean_or_none(values)
        result[label] = methods
    return result


def _paired_bootstrap(
    cases: Sequence[CaseEvaluation],
    *,
    label: str,
    metric: str,
    candidate: str,
    reference: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    deltas: list[float] = []
    for case in cases:
        point = case.record["horizons"][label]["point"]
        left = point[candidate][metric]
        right = point[reference][metric]
        if left is not None and right is not None:
            deltas.append(float(left) - float(right))
    if not deltas:
        return {"object_count": 0}
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = np.mean(values[indices], axis=1)
    return {
        "object_count": len(values),
        "mean_delta_m": float(np.mean(values)),
        "median_delta_m": float(np.median(values)),
        "lower_95_delta_m": float(np.quantile(means, 0.025)),
        "upper_95_delta_m": float(np.quantile(means, 0.975)),
        "bootstrap_probability_improvement": float(np.mean(means < 0.0)),
        "candidate_win_count": int(np.sum(values < 0.0)),
        "tie_count": int(np.sum(values == 0.0)),
    }


def _aggregate_posterior(
    cases: Sequence[CaseEvaluation],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for posterior in POSTERIORS:
        result[posterior] = {}
        for label in HORIZON_LABELS:
            combined = _empty_event_collector()
            object_summaries = []
            for case in cases:
                arrays = _event_mapping(case, posterior, label)
                if not arrays["nees"]:
                    continue
                _append_events(
                    combined,
                    {name: np.concatenate(values) for name, values in arrays.items()},
                )
                object_summaries.append(_event_summary(arrays))
            raw = _event_summary(combined)
            horizon_result: dict[str, Any] = {
                "raw": raw,
                "object_balanced": {
                    key: _mean_or_none(
                        float(summary[key])
                        for summary in object_summaries
                        if key in summary
                    )
                    for key in (
                        "coverage_90",
                        "mean_nees_over_dimension",
                        "mean_nll",
                        "mean_error_norm_m",
                        "mean_predictive_std_m",
                    )
                },
            }
            if posterior == "normalized_evidence":
                scale_record = calibration["scales"][label]
                for scale_name in ("object_q90_scale", "simultaneous_scale"):
                    scale = float(scale_record[scale_name])
                    horizon_result[scale_name] = {
                        "scale": scale,
                        "event_weighted": _event_summary(combined, scale=scale),
                        "object_balanced": {
                            key: _mean_or_none(
                                float(_event_summary(arrays, scale=scale)[key])
                                for case in cases
                                if (arrays := _event_mapping(case, posterior, label))[
                                    "nees"
                                ]
                            )
                            for key in (
                                "coverage_90",
                                "mean_nees_over_dimension",
                                "mean_nll",
                                "mean_error_norm_m",
                                "mean_predictive_std_m",
                            )
                        },
                    }
            result[posterior][label] = horizon_result
    return result


def confirm(
    data_root: Path,
    protocol_path: Path,
    selection_path: Path,
    calibration_path: Path,
    *,
    revision: str | None,
) -> dict[str, Any]:
    """Open only confirmation objects after calibration authorization."""

    protocol = _load_protocol(protocol_path)
    selection, selection_sha = _load_selection(selection_path, protocol)
    calibration, calibration_sha = _load_calibration(
        calibration_path, protocol, selection_sha
    )
    root = data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    selected = tuple(_parse_selected_archive(raw) for raw in selection["confirmation"])
    cases = tuple(_evaluate_case(root, item, protocol) for item in selected)
    supported, by_stratum = _support_summary(cases, protocol.strata)
    support_passed = len(supported) >= protocol.minimum_confirmation_groups and all(
        count >= protocol.minimum_confirmation_groups_per_stratum
        for count in by_stratum.values()
    )
    point = _aggregate_point(supported)
    posterior = _aggregate_posterior(supported, calibration)
    comparisons: dict[str, Any] = {}
    for label_index, label in enumerate(HORIZON_LABELS):
        comparisons[label] = {}
        for metric_index, metric in enumerate(("kinematic_rmse_m", "chamfer_rmse_m")):
            for reference in ("last_displacement", "cumulative_evidence"):
                key = f"normalized_evidence_vs_{reference}/{metric}"
                comparisons[label][key] = _paired_bootstrap(
                    supported,
                    label=label,
                    metric=metric,
                    candidate="normalized_evidence",
                    reference=reference,
                    samples=protocol.bootstrap_samples,
                    seed=protocol.bootstrap_seed
                    + 100 * label_index
                    + 10 * metric_index
                    + (0 if reference == "last_displacement" else 1),
                )
    success: dict[str, Any] = {
        "support_gate_passed": support_passed,
        "normalized_effective_components_exceed_cumulative": {},
        "normalized_raw_nll_better_than_cumulative": {},
        "object_q90_calibrated_coverage_in_registered_range": {},
    }
    coverage_gate = protocol.payload["success_gates"][
        "object_q90_calibrated_coverage_range"
    ]
    lower, upper = map(float, coverage_gate)
    for label in HORIZON_LABELS:
        cumulative_summary = posterior["cumulative_evidence"][label]["raw"]
        normalized_summary = posterior["normalized_evidence"][label]["raw"]
        normalized_effective = _mean_or_none(
            float(
                case.record["horizons"][label]["posterior"]["normalized_evidence"][
                    "mean_effective_component_count"
                ]
            )
            for case in supported
            if case.record["horizons"][label]["posterior"]["normalized_evidence"][
                "mean_effective_component_count"
            ]
            is not None
        )
        cumulative_effective = _mean_or_none(
            float(
                case.record["horizons"][label]["posterior"]["cumulative_evidence"][
                    "mean_effective_component_count"
                ]
            )
            for case in supported
            if case.record["horizons"][label]["posterior"]["cumulative_evidence"][
                "mean_effective_component_count"
            ]
            is not None
        )
        success["normalized_effective_components_exceed_cumulative"][label] = (
            normalized_effective is not None
            and cumulative_effective is not None
            and normalized_effective > cumulative_effective
        )
        success["normalized_raw_nll_better_than_cumulative"][label] = (
            normalized_summary.get("count", 0) > 0
            and cumulative_summary.get("count", 0) > 0
            and normalized_summary["mean_nll"] < cumulative_summary["mean_nll"]
        )
        calibrated = posterior["normalized_evidence"][label]["object_q90_scale"][
            "object_balanced"
        ]["coverage_90"]
        success["object_q90_calibrated_coverage_in_registered_range"][label] = (
            calibrated is not None and lower <= calibrated <= upper
        )
    success["overall_passed"] = bool(
        support_passed
        and all(
            all(values.values())
            for key, values in success.items()
            if key not in {"support_gate_passed", "overall_passed"}
        )
    )
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "repository_revision": revision,
        "protocol_sha256": protocol.sha256,
        "selection_sha256": selection_sha,
        "calibration_sha256": calibration_sha,
        "partition": "confirmation",
        "information_boundary": {
            "selection_used_names_only": True,
            "calibration_opened_before_confirmation": True,
            "calibration_serialized_before_confirmation": True,
            "confirmation_payloads_opened": True,
            "historical_reserved_targets_opened": False,
            "method_parameters_refit_on_confirmation": False,
            "official_deform360_table_parity": False,
            "physical_state_correction_claim": False,
        },
        "support_gate": {
            "passed": support_passed,
            "supported_object_count": len(supported),
            "supported_by_stratum": by_stratum,
            "required_object_count": protocol.minimum_confirmation_groups,
            "required_per_stratum": protocol.minimum_confirmation_groups_per_stratum,
        },
        "point": point,
        "posterior": posterior,
        "paired_object_bootstrap": comparisons,
        "success_gates": success,
        "cases": [case.record for case in cases],
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def _command_select(args: argparse.Namespace) -> int:
    result = build_selection(
        args.inventory,
        args.protocol,
        args.v1_config,
        args.v2_config,
    )
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "selection_complete": result["selection_complete"],
                "selection_sha256": result["selection_sha256"],
                "calibration_count": len(result["calibration"]),
                "confirmation_count": len(result["confirmation"]),
                "insufficiency": result["insufficiency"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["selection_complete"] else 3


def _command_calibrate(args: argparse.Namespace) -> int:
    result = calibrate(
        args.data_root,
        args.protocol,
        args.selection,
        revision=args.revision,
    )
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "calibration_sha256": result["calibration_sha256"],
                "confirmation_authorized": result["confirmation_authorized"],
                "support_gate": result["support_gate"],
                "scales": result["scales"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["confirmation_authorized"] else 4


def _command_confirm(args: argparse.Namespace) -> int:
    result = confirm(
        args.data_root,
        args.protocol,
        args.selection,
        args.calibration,
        revision=args.revision,
    )
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "result_sha256": result["result_sha256"],
                "support_gate": result["support_gate"],
                "success_gates": result["success_gates"],
                "point": result["point"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="select from names-only inventory")
    select.add_argument("--inventory", type=Path, required=True)
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--v1-config", type=Path, required=True)
    select.add_argument("--v2-config", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(handler=_command_select)

    calibrate_parser = subparsers.add_parser(
        "calibrate", help="open only calibration objects"
    )
    calibrate_parser.add_argument("--data-root", type=Path, required=True)
    calibrate_parser.add_argument("--protocol", type=Path, required=True)
    calibrate_parser.add_argument("--selection", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    calibrate_parser.add_argument("--revision")
    calibrate_parser.set_defaults(handler=_command_calibrate)

    confirm_parser = subparsers.add_parser(
        "confirm", help="open confirmation after calibration authorization"
    )
    confirm_parser.add_argument("--data-root", type=Path, required=True)
    confirm_parser.add_argument("--protocol", type=Path, required=True)
    confirm_parser.add_argument("--selection", type=Path, required=True)
    confirm_parser.add_argument("--calibration", type=Path, required=True)
    confirm_parser.add_argument("--output", type=Path, required=True)
    confirm_parser.add_argument("--revision")
    confirm_parser.set_defaults(handler=_command_confirm)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
