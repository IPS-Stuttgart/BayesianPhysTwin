"""Outcome-safe download boundary for the selective Deform360 study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .deform360_selective_virtual_sensing_protocol import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    EXPECTED_STRATA,
    PROTOCOL_ID,
    load_selective_virtual_sensing_protocol,
)


SnapshotDownload = Callable[..., str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SelectiveVirtualSensingDownloadPlan:
    """The immutable dataset request derived from the prospective lock."""

    repository: str
    revision: str
    object_ids: tuple[str, ...]
    episode_ids_by_object: tuple[tuple[str, tuple[int, ...]], ...]
    allow_patterns: tuple[str, ...]
    ignore_patterns: tuple[str, ...]
    protocol_config_sha256: str


def selective_virtual_sensing_download_plan(
    protocol_path: str | Path,
) -> SelectiveVirtualSensingDownloadPlan:
    """Derive the exact object-only request from the canonical protocol."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    cohort = protocol["normalized_cohort"]
    rows = tuple(
        (object_id, tuple(episode_ids))
        for stratum in EXPECTED_STRATA
        for object_id, episode_ids in cohort[stratum].items()
    )
    object_ids = tuple(object_id for object_id, _ in rows)
    _require(len(object_ids) == 12, "prospective download cohort is incomplete")
    _require(len(set(object_ids)) == len(object_ids), "download object repeated")
    return SelectiveVirtualSensingDownloadPlan(
        repository=DATASET_REPOSITORY,
        revision=DATASET_REVISION,
        object_ids=object_ids,
        episode_ids_by_object=rows,
        allow_patterns=tuple(f"raw/{object_id}/*" for object_id in object_ids),
        ignore_patterns=("*.flac", "*.wav"),
        protocol_config_sha256=str(protocol["config_sha256"]),
    )


def validate_selective_download_root(
    output_root: str | Path,
    *,
    plan: SelectiveVirtualSensingDownloadPlan,
    require_complete: bool,
) -> None:
    """Reject contamination by any object outside the locked cohort."""

    raw_root = Path(output_root).resolve() / "raw"
    if not raw_root.exists():
        _require(not require_complete, "prospective raw download root is missing")
        return
    _require(raw_root.is_dir(), "prospective raw path is not a directory")
    present = {path.name for path in raw_root.iterdir() if path.is_dir()}
    expected = set(plan.object_ids)
    unexpected = sorted(present - expected)
    missing = sorted(expected - present)
    _require(not unexpected, f"unlocked objects exist in download root: {unexpected}")
    if require_complete:
        _require(not missing, f"locked objects are missing from download root: {missing}")


def _validate_object_metadata(
    metadata_path: Path,
    *,
    object_id: str,
    selected_episode_ids: tuple[int, ...],
) -> None:
    _require(metadata_path.is_file(), f"object metadata is missing: {object_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(isinstance(metadata, Mapping), f"object metadata is invalid: {object_id}")
    _require(metadata.get("object") == object_id, f"metadata object changed: {object_id}")
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping), f"metadata sequences are missing: {object_id}")
    _require(
        set(sequences) == {str(index) for index in range(10)},
        f"metadata episode inventory changed: {object_id}",
    )
    _require(
        all(str(index) in sequences for index in selected_episode_ids),
        f"selected metadata episode is missing: {object_id}",
    )


def build_selective_download_manifest(
    output_root: str | Path,
    *,
    plan: SelectiveVirtualSensingDownloadPlan,
) -> dict[str, Any]:
    """Hash metadata and inventory the exact downloaded prospective panel."""

    root = Path(output_root).resolve()
    validate_selective_download_root(root, plan=plan, require_complete=True)
    episode_map = dict(plan.episode_ids_by_object)
    rows = []
    for object_id in plan.object_ids:
        object_root = root / "raw" / object_id
        metadata_path = object_root / "metadata.json"
        _validate_object_metadata(
            metadata_path,
            object_id=object_id,
            selected_episode_ids=episode_map[object_id],
        )
        files = sorted(path for path in object_root.rglob("*") if path.is_file())
        rows.append(
            {
                "object_id": object_id,
                "selected_episode_ids": list(episode_map[object_id]),
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
                "metadata_sha256": _file_sha256(metadata_path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SelectiveVirtualSensingDownload",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": plan.protocol_config_sha256,
        "repository": plan.repository,
        "revision": plan.revision,
        "audio_included": False,
        "object_count": len(rows),
        "objects": rows,
        "information_boundary": {
            "selected_objects_only": True,
            "future_dense_reconstruction_opened": False,
            "future_particle_tracks_opened": False,
            "target_metrics_opened": False,
        },
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def download_selective_virtual_sensing_panel(
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    max_workers: int,
    snapshot_download: SnapshotDownload,
) -> dict[str, Any]:
    """Download exactly the locked objects and return a sealed manifest."""

    _require(max_workers >= 1, "download workers must be positive")
    plan = selective_virtual_sensing_download_plan(protocol_path)
    validate_selective_download_root(output_root, plan=plan, require_complete=False)
    snapshot_download(
        repo_id=plan.repository,
        repo_type="dataset",
        revision=plan.revision,
        local_dir=str(Path(output_root).resolve()),
        allow_patterns=list(plan.allow_patterns),
        ignore_patterns=list(plan.ignore_patterns),
        max_workers=max_workers,
    )
    return build_selective_download_manifest(output_root, plan=plan)


def write_selective_download_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "SelectiveVirtualSensingDownloadPlan",
    "build_selective_download_manifest",
    "download_selective_virtual_sensing_panel",
    "selective_virtual_sensing_download_plan",
    "validate_selective_download_root",
    "write_selective_download_manifest",
]
