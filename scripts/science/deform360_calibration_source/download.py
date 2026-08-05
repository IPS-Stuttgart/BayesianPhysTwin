"""Exact-file official-Hub calibration download and byte verification."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .contracts import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DOWNLOAD_SCHEMA,
    PROTOCOL_ID,
    canonical_sha256,
    file_sha256,
    load_json,
    load_units,
    require,
    write_json,
)
from .planning import verify_plan


def download_one(
    *,
    record: Mapping[str, Any],
    root: Path,
    hub_download: Any,
) -> dict[str, Any]:
    relative = record.get("path")
    require(isinstance(relative, str), "download path is malformed")
    destination = Path(
        hub_download(
            repo_id=DATASET_REPOSITORY,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=relative,
            local_dir=str(root),
        )
    ).resolve()
    expected = (root / relative).resolve()
    require(destination == expected, f"download path changed: {relative}")
    require(
        destination.is_file() and not destination.is_symlink(),
        f"download is not a regular file: {relative}",
    )
    size = destination.stat().st_size
    declared_size = record.get("size")
    if isinstance(declared_size, int):
        require(size == declared_size, f"download size changed: {relative}")
    digest = file_sha256(destination)
    lfs_sha256 = record.get("lfs_sha256")
    if isinstance(lfs_sha256, str):
        require(
            digest == lfs_sha256,
            f"download LFS digest changed: {relative}",
        )
    return {
        **dict(record),
        "downloaded_size": size,
        "downloaded_sha256": digest,
    }


def download_plan(
    *,
    plan_path: Path,
    protocol_path: Path,
    selection_path: Path,
    provider_path: Path,
    data_root: Path,
    output_path: Path,
    max_workers: int,
    hub_download: Any,
) -> dict[str, Any]:
    require(max_workers >= 1, "max_workers must be positive")
    plan, confirmations = verify_plan(
        plan_path,
        protocol_path=protocol_path,
        selection_path=selection_path,
        provider_path=provider_path,
    )
    require(
        plan["gate"]["support_passed"] is True,
        "names-only support gate failed",
    )
    root = data_root.resolve()
    raw_root = root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    present = {path.name for path in raw_root.iterdir() if path.is_dir()}
    require(
        not present & set(confirmations),
        "confirmation payload exists in calibration root",
    )
    planned_objects = {
        row["object_id"] for row in plan["objects"] if row.get("status") == "planned"
    }
    require(
        not (present - planned_objects),
        "unregistered objects exist in calibration root: "
        f"{sorted(present - planned_objects)}",
    )
    records = [
        file
        for row in plan["objects"]
        if row.get("status") == "planned"
        for file in row["selected_files"]
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        downloaded = tuple(
            executor.map(
                lambda record: download_one(
                    record=record,
                    root=root,
                    hub_download=hub_download,
                ),
                records,
            )
        )
    by_path = {record["path"]: record for record in downloaded}
    units, _ = load_units(selection_path)
    for unit in units:
        if unit.object_id not in planned_objects:
            continue
        metadata = by_path.get(unit.metadata_path)
        require(
            metadata is not None,
            f"download omitted metadata: {unit.object_id}",
        )
        require(
            metadata["downloaded_sha256"] == unit.metadata_sha256,
            f"metadata digest changed: {unit.object_id}",
        )
    payload: dict[str, Any] = {
        "schema": DOWNLOAD_SCHEMA,
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": plan["plan_sha256"],
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "data_root": str(root),
        "files": list(sorted(downloaded, key=lambda item: item["path"])),
        "object_ids": sorted(planned_objects),
        "information_boundary": {
            "calibration_payloads_opened": True,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
    }
    payload["download_sha256"] = canonical_sha256(
        payload,
        digest_key="download_sha256",
    )
    write_json(output_path, payload)
    return payload


def verify_download(
    path: Path,
    *,
    plan_path: Path,
    protocol_path: Path,
    selection_path: Path,
    provider_path: Path,
    data_root: Path,
) -> dict[str, Any]:
    plan, confirmations = verify_plan(
        plan_path,
        protocol_path=protocol_path,
        selection_path=selection_path,
        provider_path=provider_path,
    )
    value = load_json(path)
    require(value.get("schema") == DOWNLOAD_SCHEMA, "download schema changed")
    require(
        value.get("plan_sha256") == plan["plan_sha256"],
        "download plan changed",
    )
    require(
        value.get("dataset_revision") == DATASET_REVISION,
        "download revision changed",
    )
    require(
        Path(str(value.get("data_root"))).resolve() == data_root.resolve(),
        "download root changed",
    )
    supplied = value.get("download_sha256")
    require(isinstance(supplied, str), "download digest is missing")
    require(
        supplied == canonical_sha256(value, digest_key="download_sha256"),
        "download digest changed",
    )
    for record in value.get("files", []):
        require(
            isinstance(record, Mapping),
            "download file record is malformed",
        )
        relative = record.get("path")
        digest = record.get("downloaded_sha256")
        require(
            isinstance(relative, str) and isinstance(digest, str),
            "download file identity is malformed",
        )
        require(
            not any(
                relative.startswith(f"raw/{object_id}/") for object_id in confirmations
            ),
            "download contains confirmation payload",
        )
        local = (data_root / relative).resolve()
        require(
            local.is_file() and not local.is_symlink(),
            f"download file is missing: {relative}",
        )
        require(
            file_sha256(local) == digest,
            f"download bytes changed: {relative}",
        )
    return value
