"""Outcome-safe download boundary for the bias-aware prospective panel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from .deform360_bias_aware_prospective_protocol import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    EXPECTED_STRATA,
    PROTOCOL_ID,
    load_bias_aware_prospective_protocol,
)


SnapshotDownload = Callable[..., str]
ListRepoTree = Callable[..., Any]
HubDownload = Callable[..., str]


# A few released directory identifiers intentionally differ from the human-readable
# ``object`` field in metadata.json. Keep this exception list explicit so a new or
# misspelled identity still fails closed at the pinned dataset revision.
RELEASED_METADATA_OBJECT_ALIASES = {
    "112-wristband-cloth": "112-wristband",
    "163-bear": "teddy bear",
    "164-sheep": "white sheep",
}


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
class BiasAwareProspectiveDownloadPlan:
    """Exact object-only request derived from the prospective lock."""

    repository: str
    revision: str
    calibration_objects: tuple[str, ...]
    target_objects: tuple[str, ...]
    episodes_by_object: tuple[tuple[str, str, tuple[int, ...]], ...]
    allow_patterns: tuple[str, ...]
    ignore_patterns: tuple[str, ...]
    protocol_config_sha256: str

    @property
    def object_ids(self) -> tuple[str, ...]:
        return self.calibration_objects + self.target_objects


def bias_aware_prospective_download_plan(
    protocol_path: str | Path,
) -> BiasAwareProspectiveDownloadPlan:
    """Derive the immutable 21-object request from the canonical protocol."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    rows: list[tuple[str, str, tuple[int, ...]]] = []
    role_objects: dict[str, list[str]] = {"calibration": [], "target": []}
    for role in ("calibration", "target"):
        cohort = protocol[f"{role}_cohort"]
        for stratum in EXPECTED_STRATA:
            for object_id, episode_ids in cohort[stratum].items():
                rows.append((role, object_id, tuple(episode_ids)))
                role_objects[role].append(object_id)
    object_ids = role_objects["calibration"] + role_objects["target"]
    _require(len(object_ids) == 21, "prospective download cohort is incomplete")
    _require(len(set(object_ids)) == len(object_ids), "download object repeated")
    return BiasAwareProspectiveDownloadPlan(
        repository=DATASET_REPOSITORY,
        revision=DATASET_REVISION,
        calibration_objects=tuple(role_objects["calibration"]),
        target_objects=tuple(role_objects["target"]),
        episodes_by_object=tuple(rows),
        allow_patterns=tuple(f"raw/{object_id}/*" for object_id in object_ids),
        ignore_patterns=("*.flac", "*.wav"),
        protocol_config_sha256=str(protocol["config_sha256"]),
    )


def validate_bias_aware_download_root(
    output_root: str | Path,
    *,
    plan: BiasAwareProspectiveDownloadPlan,
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
    _require(
        not present - expected,
        f"unlocked objects exist in download root: {sorted(present - expected)}",
    )
    if require_complete:
        _require(
            not expected - present,
            f"locked objects are missing from download root: {sorted(expected - present)}",
        )


def _validate_object_metadata(
    metadata_path: Path,
    *,
    object_id: str,
    selected_episode_ids: tuple[int, ...],
) -> str:
    _require(metadata_path.is_file(), f"object metadata is missing: {object_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(isinstance(metadata, Mapping), f"object metadata is invalid: {object_id}")
    released_object = metadata.get("object")
    expected_object = RELEASED_METADATA_OBJECT_ALIASES.get(object_id, object_id)
    _require(
        released_object == expected_object,
        f"metadata object changed: {object_id}",
    )
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping), f"sequences are missing: {object_id}")
    _require(
        set(sequences) == {str(index) for index in range(10)},
        f"episode inventory changed: {object_id}",
    )
    _require(
        all(str(index) in sequences for index in selected_episode_ids),
        f"selected episode is missing: {object_id}",
    )
    return str(released_object)


def build_bias_aware_download_manifest(
    output_root: str | Path,
    *,
    plan: BiasAwareProspectiveDownloadPlan,
) -> dict[str, Any]:
    """Hash metadata and inventory the exact downloaded panel."""

    root = Path(output_root).resolve()
    validate_bias_aware_download_root(root, plan=plan, require_complete=True)
    rows = []
    for role, object_id, episode_ids in plan.episodes_by_object:
        object_root = root / "raw" / object_id
        metadata_path = object_root / "metadata.json"
        released_object = _validate_object_metadata(
            metadata_path,
            object_id=object_id,
            selected_episode_ids=episode_ids,
        )
        files = sorted(path for path in object_root.rglob("*") if path.is_file())
        rows.append(
            {
                "role": role,
                "object_id": object_id,
                "selected_episode_ids": list(episode_ids),
                "released_metadata_object": released_object,
                "metadata_identity_alias": released_object != object_id,
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
                "metadata_sha256": _file_sha256(metadata_path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareProspectiveDownload",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": plan.protocol_config_sha256,
        "repository": plan.repository,
        "revision": plan.revision,
        "audio_included": False,
        "object_count": len(rows),
        "objects": rows,
        "information_boundary": {
            "locked_objects_only": True,
            "calibration_and_target_roles_preserved": True,
            "prediction_process_reads_prefix_staging_only": True,
            "target_future_opened": False,
            "target_metrics_opened": False,
        },
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def download_bias_aware_prospective_panel(
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    max_workers: int,
    snapshot_download: SnapshotDownload,
) -> dict[str, Any]:
    """Download exactly the locked objects and return a sealed manifest."""

    _require(max_workers >= 1, "download workers must be positive")
    plan = bias_aware_prospective_download_plan(protocol_path)
    validate_bias_aware_download_root(
        output_root, plan=plan, require_complete=False
    )
    snapshot_download(
        repo_id=plan.repository,
        repo_type="dataset",
        revision=plan.revision,
        local_dir=str(Path(output_root).resolve()),
        allow_patterns=list(plan.allow_patterns),
        ignore_patterns=list(plan.ignore_patterns),
        max_workers=max_workers,
    )
    return build_bias_aware_download_manifest(output_root, plan=plan)


def download_bias_aware_prospective_panel_by_object(
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    max_workers: int,
    object_delay_seconds: float,
    list_repo_tree: ListRepoTree,
    hub_download: HubDownload,
) -> dict[str, Any]:
    """Download locked object subtrees without enumerating the whole repository."""

    _require(max_workers >= 1, "download workers must be positive")
    _require(object_delay_seconds >= 0.0, "object delay must be non-negative")
    plan = bias_aware_prospective_download_plan(protocol_path)
    root = Path(output_root).resolve()
    validate_bias_aware_download_root(root, plan=plan, require_complete=False)
    for object_index, object_id in enumerate(plan.object_ids):
        prefix = f"raw/{object_id}/"
        entries = list(
            list_repo_tree(
                repo_id=plan.repository,
                path_in_repo=f"raw/{object_id}",
                recursive=True,
                expand=False,
                revision=plan.revision,
                repo_type="dataset",
            )
        )
        files = sorted(
            str(entry.path)
            for entry in entries
            if getattr(entry, "blob_id", None) is not None
            and str(entry.path).startswith(prefix)
            and Path(str(entry.path)).suffix.lower() not in {".flac", ".wav"}
        )
        _require(files, f"locked object subtree is empty: {object_id}")
        _require(
            f"raw/{object_id}/metadata.json" in files,
            f"locked object metadata is absent: {object_id}",
        )
        _require(
            all(path.startswith(prefix) for path in files),
            f"object listing escaped its locked subtree: {object_id}",
        )

        def download_one(filename: str) -> str:
            return hub_download(
                repo_id=plan.repository,
                filename=filename,
                repo_type="dataset",
                revision=plan.revision,
                local_dir=str(root),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tuple(executor.map(download_one, files))
        if object_index + 1 < len(plan.object_ids) and object_delay_seconds:
            time.sleep(object_delay_seconds)
    return build_bias_aware_download_manifest(root, plan=plan)


def write_bias_aware_download_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BiasAwareProspectiveDownloadPlan",
    "RELEASED_METADATA_OBJECT_ALIASES",
    "bias_aware_prospective_download_plan",
    "build_bias_aware_download_manifest",
    "download_bias_aware_prospective_panel",
    "download_bias_aware_prospective_panel_by_object",
    "validate_bias_aware_download_root",
    "write_bias_aware_download_manifest",
]
