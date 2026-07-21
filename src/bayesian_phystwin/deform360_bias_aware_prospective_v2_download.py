"""Fresh-calibration-only download boundary for prospective v2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any, Mapping

from .deform360_bias_aware_prospective_download import (
    BiasAwareProspectiveDownloadPlan,
    HubDownload,
    ListRepoTree,
    _canonical_sha256,
    _file_sha256,
    _validate_object_metadata,
    validate_bias_aware_download_root,
)
from .deform360_bias_aware_prospective_v2_protocol import (
    PROTOCOL_ID,
    load_bias_aware_prospective_v2_protocol,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bias_aware_prospective_v2_fresh_download_plan(
    protocol_path: str | Path,
) -> BiasAwareProspectiveDownloadPlan:
    """Return the exact three-object request authorized by the v2 repair lock."""

    protocol = load_bias_aware_prospective_v2_protocol(protocol_path)
    config = protocol["config"]
    rows = tuple(
        (
            "calibration",
            str(record["object_id"]),
            tuple(int(value) for value in record["episode_ids"]),
        )
        for record in config["repair"]["fresh_calibration"]
    )
    object_ids = tuple(row[1] for row in rows)
    _require(len(object_ids) == 3, "v2 fresh download cohort is incomplete")
    _require(len(set(object_ids)) == len(object_ids), "v2 download object repeated")
    return BiasAwareProspectiveDownloadPlan(
        repository=str(config["dataset"]["repository"]),
        revision=str(config["dataset"]["revision"]),
        calibration_objects=object_ids,
        target_objects=(),
        episodes_by_object=rows,
        allow_patterns=tuple(f"raw/{object_id}/*" for object_id in object_ids),
        ignore_patterns=("*.flac", "*.wav"),
        protocol_config_sha256=str(protocol["config_sha256"]),
    )


def build_bias_aware_prospective_v2_download_manifest(
    output_root: str | Path,
    *,
    plan: BiasAwareProspectiveDownloadPlan,
) -> dict[str, Any]:
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
        "artifact_kind": "Deform360BiasAwareProspectiveV2FreshDownload",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": plan.protocol_config_sha256,
        "repository": plan.repository,
        "revision": plan.revision,
        "audio_included": False,
        "object_count": len(rows),
        "objects": rows,
        "information_boundary": {
            "fresh_calibration_objects_only": True,
            "base_calibration_future_opened": False,
            "fresh_calibration_future_opened": False,
            "reserved_target_downloaded": False,
            "reserved_target_media_read": False,
            "target_metrics_opened": False,
        },
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def download_bias_aware_prospective_v2_fresh_by_object(
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    max_workers: int,
    object_delay_seconds: float,
    list_repo_tree: ListRepoTree,
    hub_download: HubDownload,
) -> dict[str, Any]:
    _require(max_workers >= 1, "download workers must be positive")
    _require(object_delay_seconds >= 0.0, "object delay must be non-negative")
    plan = bias_aware_prospective_v2_fresh_download_plan(protocol_path)
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
    return build_bias_aware_prospective_v2_download_manifest(root, plan=plan)


def write_bias_aware_prospective_v2_download_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "bias_aware_prospective_v2_fresh_download_plan",
    "build_bias_aware_prospective_v2_download_manifest",
    "download_bias_aware_prospective_v2_fresh_by_object",
    "write_bias_aware_prospective_v2_download_manifest",
]
