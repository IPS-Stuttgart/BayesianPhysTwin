"""Outcome-safe download planning for the reusable Deform360 SOTA protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .deform360_reusable_sota_protocol import load_reusable_sota_config


SnapshotDownload = Callable[..., str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ReusableSotaDownloadPlan:
    repository: str
    revision: str
    object_ids: tuple[str, ...]
    allow_patterns: tuple[str, ...]
    ignore_patterns: tuple[str, ...]
    config_sha256: str


def development_download_plan(
    protocol_path: str | Path,
) -> ReusableSotaDownloadPlan:
    """Return the immutable development-only download plan."""

    payload = load_reusable_sota_config(protocol_path)
    config = payload["config"]
    development = config["development_objects"]
    object_ids = tuple(
        object_id
        for category in ("1d", "2d", "3d")
        for object_id in development[category]
    )
    _require(len(object_ids) == 12, "development download panel is incomplete")
    return ReusableSotaDownloadPlan(
        repository=str(config["dataset"]["repository"]),
        revision=str(config["dataset"]["revision"]),
        object_ids=object_ids,
        allow_patterns=tuple(f"raw/{object_id}/*" for object_id in object_ids),
        ignore_patterns=("*.flac", "*.wav"),
        config_sha256=str(payload["config_sha256"]),
    )


def confirmatory_object_ids(protocol_path: str | Path) -> tuple[str, ...]:
    payload = load_reusable_sota_config(protocol_path)
    panel = payload["config"]["confirmatory_objects"]
    return tuple(
        object_id
        for category in ("1d", "2d", "3d")
        for object_id in panel[category]
    )


def validate_development_root(
    output_root: str | Path,
    *,
    protocol_path: str | Path,
) -> None:
    """Fail closed if confirmatory raw objects entered the development root."""

    raw_root = Path(output_root).resolve() / "raw"
    leaked = [
        object_id
        for object_id in confirmatory_object_ids(protocol_path)
        if (raw_root / object_id).exists()
    ]
    _require(not leaked, f"confirmatory objects exist in development root: {leaked}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_download_manifest(
    output_root: str | Path,
    *,
    plan: ReusableSotaDownloadPlan,
) -> dict[str, Any]:
    """Describe staged files without hashing every large video."""

    root = Path(output_root).resolve()
    rows = []
    for object_id in plan.object_ids:
        object_root = root / "raw" / object_id
        _require(object_root.is_dir(), f"downloaded object is missing: {object_id}")
        files = sorted(path for path in object_root.rglob("*") if path.is_file())
        metadata = object_root / "metadata.json"
        _require(metadata.is_file(), f"object metadata is missing: {object_id}")
        rows.append(
            {
                "object_id": object_id,
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
                "metadata_sha256": _file_sha256(metadata),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableSotaDevelopmentDownload",
        "repository": plan.repository,
        "revision": plan.revision,
        "protocol_config_sha256": plan.config_sha256,
        "audio_included": False,
        "objects": rows,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def download_development_panel(
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    max_workers: int,
    snapshot_download: SnapshotDownload,
) -> dict[str, Any]:
    """Download exactly the locked development panel and return its manifest."""

    _require(max_workers >= 1, "download workers must be positive")
    plan = development_download_plan(protocol_path)
    validate_development_root(output_root, protocol_path=protocol_path)
    snapshot_download(
        repo_id=plan.repository,
        repo_type="dataset",
        revision=plan.revision,
        local_dir=str(Path(output_root).resolve()),
        allow_patterns=list(plan.allow_patterns),
        ignore_patterns=list(plan.ignore_patterns),
        max_workers=max_workers,
    )
    validate_development_root(output_root, protocol_path=protocol_path)
    return build_download_manifest(output_root, plan=plan)


def write_download_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ReusableSotaDownloadPlan",
    "build_download_manifest",
    "confirmatory_object_ids",
    "development_download_plan",
    "download_development_panel",
    "validate_development_root",
    "write_download_manifest",
]
