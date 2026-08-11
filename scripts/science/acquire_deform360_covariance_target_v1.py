#!/usr/bin/env python3
"""Plan and download the exact locked Deform360 covariance target bytes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.science.deform360_calibration_source.contracts import (
    CalibrationUnit,
    RepositoryFile,
    canonical_sha256,
    file_sha256,
    load_json,
    require,
    write_json,
)
from scripts.science.deform360_calibration_source.download import download_one
from scripts.science.deform360_calibration_source.planning import (
    repository_files,
    select_object_files,
)

LOCK_SCHEMA = "bayesian-phystwin/deform360-covariance-target-acquisition-lock-v1"
PLAN_SCHEMA = "bayesian-phystwin/deform360-covariance-target-file-plan-v1"
DOWNLOAD_SCHEMA = "bayesian-phystwin/deform360-covariance-target-download-v1"
ROSTER_SCHEMA = "bayesian-phystwin/deform360-covariance-only-target-selection-v1"
EXPECTED_TARGET_COUNT = 24


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_context(
    *,
    repository: Path,
    lock_path: Path,
    selection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = repository.resolve()
    lock_path = lock_path.resolve()
    selection_path = selection_path.resolve()
    lock = load_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA, "acquisition lock schema changed")
    require(lock.get("schema_version") == 1, "acquisition lock version changed")
    require(
        lock.get("status") == "locked-before-full-support-audit-and-payload-download",
        "acquisition lock is not prospective",
    )
    supplied_lock = lock.get("lock_sha256")
    require(isinstance(supplied_lock, str), "acquisition lock digest is missing")
    require(
        supplied_lock == canonical_sha256(lock, digest_key="lock_sha256"),
        "acquisition lock digest changed",
    )
    source_record = lock.get("source_protocol")
    require(isinstance(source_record, Mapping), "source-protocol binding is missing")
    source_path = (repository / str(source_record.get("artifact_path", ""))).resolve()
    require(
        source_path.is_relative_to(repository), "source protocol escaped repository"
    )
    require(
        file_sha256(source_path) == source_record.get("file_sha256"),
        "source protocol bytes changed",
    )
    source_protocol = load_json(source_path)
    require(
        source_protocol.get("protocol_id") == source_record.get("protocol_id")
        and source_protocol.get("protocol_sha256")
        == source_record.get("protocol_sha256"),
        "source protocol identity changed",
    )
    require(
        source_protocol.get("protocol_sha256")
        == canonical_sha256(source_protocol, digest_key="protocol_sha256"),
        "source protocol canonical digest changed",
    )
    selection_record = lock.get("selection")
    require(isinstance(selection_record, Mapping), "selection binding is missing")
    expected_selection = (
        repository / str(selection_record.get("artifact_path", ""))
    ).resolve()
    require(
        expected_selection.is_relative_to(repository),
        "selection binding escaped repository",
    )
    require(selection_path == expected_selection, "selection artifact path changed")
    require(
        file_sha256(selection_path) == selection_record.get("file_sha256"),
        "selection artifact bytes changed",
    )
    selection = load_json(selection_path)
    require(selection.get("schema") == ROSTER_SCHEMA, "selection schema changed")
    supplied_selection = selection.get("selection_sha256")
    require(isinstance(supplied_selection, str), "selection digest is missing")
    require(
        supplied_selection
        == canonical_sha256(selection, digest_key="selection_sha256")
        == selection_record.get("selection_sha256"),
        "selection canonical digest changed",
    )
    roster = selection.get("target_roster")
    require(
        isinstance(roster, list) and len(roster) == EXPECTED_TARGET_COUNT,
        "target roster count changed",
    )
    require(
        len({str(row.get("object_id")) for row in roster}) == len(roster),
        "target roster repeats a physical object",
    )
    require(
        selection.get("roster_sha256") == selection_record.get("roster_sha256"),
        "target roster digest changed",
    )
    boundary = selection.get("information_boundary")
    require(isinstance(boundary, Mapping), "selection boundary is missing")
    require(boundary.get("camera_media_decoded") is False, "media was decoded")
    require(
        boundary.get("target_outcomes_opened") is False,
        "target outcomes were opened",
    )
    return lock, selection


def _records_or_empty(
    entries: Sequence[object],
    *,
    prefix: str,
) -> tuple[RepositoryFile, ...]:
    if not entries:
        return ()
    return repository_files(entries, prefix=prefix)


def _list_tree(
    api: object,
    *,
    repository: str,
    revision: str,
    path: str,
    allow_absent: bool,
    recursive: bool = True,
) -> tuple[object, ...]:
    try:
        return tuple(
            api.list_repo_tree(
                repo_id=repository,
                repo_type="dataset",
                revision=revision,
                path_in_repo=path,
                recursive=recursive,
                expand=False,
            )
        )
    except Exception as error:
        if allow_absent and type(error).__name__ in {
            "EntryNotFoundError",
            "RemoteEntryNotFoundError",
        }:
            return ()
        raise


def build_file_plan(
    *,
    repository: Path,
    lock_path: Path,
    selection_path: Path,
    output_path: Path,
    api: object,
    implementation_revision: str | None = None,
) -> dict[str, Any]:
    """Resolve names and immutable blob metadata without opening payload bytes."""

    lock, selection = _load_context(
        repository=repository,
        lock_path=lock_path,
        selection_path=selection_path,
    )
    dataset = lock["dataset"]
    info = api.repo_info(
        repo_id=str(dataset["repository"]),
        repo_type="dataset",
        revision=str(dataset["revision"]),
        files_metadata=False,
    )
    require(
        getattr(info, "sha", None) == dataset["revision"],
        "official dataset revision changed",
    )
    processed_root_entries = _list_tree(
        api,
        repository=str(dataset["repository"]),
        revision=str(dataset["revision"]),
        path=str(dataset["processed_prefix"]),
        allow_absent=True,
        recursive=False,
    )
    processed_object_paths = {
        str(entry.path)
        for entry in processed_root_entries
        if isinstance(getattr(entry, "path", None), str)
    }
    rows: list[dict[str, Any]] = []
    for target in selection["target_roster"]:
        object_id = str(target["object_id"])
        episode_id = int(target["episode_id"])
        raw_prefix = f"raw/{object_id}/"
        raw_entries = _list_tree(
            api,
            repository=str(dataset["repository"]),
            revision=str(dataset["revision"]),
            path=f"raw/{object_id}",
            allow_absent=False,
        )
        raw_records = repository_files(raw_entries, prefix=raw_prefix)
        unit = CalibrationUnit(
            object_id=object_id,
            episode_id=episode_id,
            stratum=str(target["stratum"]),
            metadata_path=str(target["metadata_path"]),
            metadata_sha256=str(target["metadata_sha256"]),
        )
        try:
            raw = select_object_files(raw_records, unit=unit)
        except ValueError as error:
            raw = {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": unit.stratum,
                "metadata_path": unit.metadata_path,
                "metadata_sha256": unit.metadata_sha256,
                "status": "retained_raw_plan_failure",
                "errors": [str(error)],
                "camera_streams": [],
                "tactile_streams": [],
                "selected_files": [],
            }

        processed_prefix = f"processed/{object_id}/episode_{episode_id}/"
        if f"processed/{object_id}" in processed_object_paths:
            processed_entries = _list_tree(
                api,
                repository=str(dataset["repository"]),
                revision=str(dataset["revision"]),
                path=processed_prefix.rstrip("/"),
                allow_absent=True,
            )
        else:
            processed_entries = ()
        processed_records = _records_or_empty(
            processed_entries,
            prefix=processed_prefix,
        )
        selected_by_path = {
            str(record["path"]): dict(record) for record in raw["selected_files"]
        }
        selected_by_path.update(
            {record.path: record.to_record() for record in processed_records}
        )
        rows.append(
            {
                **raw,
                "object_hash": target["object_hash"],
                "factorial_cell": target["factorial_cell"],
                "official_processed_annotation_status": (
                    "exact_episode_available"
                    if processed_records
                    else "absent_at_locked_revision"
                ),
                "official_processed_file_count": len(processed_records),
                "selected_files": [
                    selected_by_path[path] for path in sorted(selected_by_path)
                ],
            }
        )

    require(len(rows) == EXPECTED_TARGET_COUNT, "file plan lost target rows")
    require(
        len({row["object_id"] for row in rows}) == EXPECTED_TARGET_COUNT,
        "file plan repeated a physical object",
    )
    all_paths = [
        str(record["path"]) for row in rows for record in row["selected_files"]
    ]
    require(len(all_paths) == len(set(all_paths)), "file plan repeats a path")
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "lock_sha256": lock["lock_sha256"],
        "selection_file_sha256": file_sha256(selection_path),
        "selection_sha256": selection["selection_sha256"],
        "roster_sha256": selection["roster_sha256"],
        "dataset": dataset,
        "implementation_revision": (
            implementation_revision or _git_revision(repository.resolve())
        ),
        "objects": rows,
        "summary": {
            "locked_target_count": len(rows),
            "ordinary_raw_plan_count": sum(row["status"] == "planned" for row in rows),
            "retained_raw_plan_failure_count": sum(
                row["status"] != "planned" for row in rows
            ),
            "exact_processed_annotation_count": sum(
                row["official_processed_annotation_status"] == "exact_episode_available"
                for row in rows
            ),
            "selected_file_count": len(all_paths),
            "selected_declared_bytes": sum(
                int(record["size"])
                for row in rows
                for record in row["selected_files"]
                if isinstance(record.get("size"), int)
            ),
        },
        "information_boundary": {
            "repository_names_and_blob_metadata_opened": True,
            "payload_bytes_opened": False,
            "camera_media_decoded": False,
            "sensor_arrays_opened": False,
            "geometry_or_tracks_opened": False,
            "target_outcomes_opened": False,
            "replacement_allowed": False,
        },
        "next_gate": "commit this exact file plan before payload download",
    }
    payload["plan_sha256"] = canonical_sha256(payload, digest_key="plan_sha256")
    write_json(output_path, payload)
    return payload


def verify_file_plan(
    *,
    repository: Path,
    lock_path: Path,
    selection_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    lock, selection = _load_context(
        repository=repository,
        lock_path=lock_path,
        selection_path=selection_path,
    )
    plan = load_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA, "file-plan schema changed")
    require(plan.get("schema_version") == 1, "file-plan version changed")
    require(plan.get("lock_sha256") == lock["lock_sha256"], "file-plan lock changed")
    require(
        plan.get("selection_sha256") == selection["selection_sha256"],
        "file-plan selection changed",
    )
    supplied = plan.get("plan_sha256")
    require(isinstance(supplied, str), "file-plan digest is missing")
    require(
        supplied == canonical_sha256(plan, digest_key="plan_sha256"),
        "file-plan digest changed",
    )
    rows = plan.get("objects")
    require(
        isinstance(rows, list) and len(rows) == EXPECTED_TARGET_COUNT,
        "file-plan denominator changed",
    )
    expected = {
        (str(row["object_id"]), int(row["episode_id"]))
        for row in selection["target_roster"]
    }
    observed = {(str(row["object_id"]), int(row["episode_id"])) for row in rows}
    require(observed == expected, "file-plan roster changed")
    return plan


def download_file_plan(
    *,
    repository: Path,
    lock_path: Path,
    selection_path: Path,
    plan_path: Path,
    data_root: Path,
    output_path: Path,
    max_workers: int,
    hub_download: Any,
) -> dict[str, Any]:
    """Download, byte-check, and inventory every exact path in the sealed plan."""

    require(max_workers >= 1, "download workers must be positive")
    plan = verify_file_plan(
        repository=repository,
        lock_path=lock_path,
        selection_path=selection_path,
        plan_path=plan_path,
    )
    records = [record for row in plan["objects"] for record in row["selected_files"]]
    planned_paths = {str(record["path"]) for record in records}
    require(len(planned_paths) == len(records), "download plan repeats a path")
    root = data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for candidate in root.rglob("*"):
        if not candidate.is_file() or ".cache" in candidate.relative_to(root).parts:
            continue
        relative = candidate.relative_to(root).as_posix()
        require(relative in planned_paths, f"unplanned payload exists: {relative}")
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=min(max_workers, 2)) as executor:
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
    completed = datetime.now(timezone.utc).isoformat()
    downloaded_by_path = {str(record["path"]): record for record in downloaded}
    require(len(downloaded_by_path) == len(records), "download lost an exact file")
    selection = load_json(selection_path)
    for target in selection["target_roster"]:
        metadata = downloaded_by_path.get(str(target["metadata_path"]))
        require(metadata is not None, "download omitted selected metadata")
        require(
            metadata["downloaded_sha256"] == target["metadata_sha256"],
            "selected metadata bytes changed",
        )
    payload: dict[str, Any] = {
        "schema": DOWNLOAD_SCHEMA,
        "schema_version": 1,
        "protocol_id": plan["protocol_id"],
        "plan_sha256": plan["plan_sha256"],
        "selection_sha256": plan["selection_sha256"],
        "roster_sha256": plan["roster_sha256"],
        "dataset": plan["dataset"],
        "implementation_revision": _git_revision(repository.resolve()),
        "data_root": str(root),
        "download_started_at_utc": started,
        "download_completed_at_utc": completed,
        "downloader_invocation": " ".join(shlex.quote(value) for value in sys.argv),
        "files": sorted(downloaded, key=lambda item: str(item["path"])),
        "objects": [
            {
                "object_id": row["object_id"],
                "object_hash": row["object_hash"],
                "episode_id": row["episode_id"],
                "status": row["status"],
                "errors": row["errors"],
                "official_processed_annotation_status": row[
                    "official_processed_annotation_status"
                ],
            }
            for row in plan["objects"]
        ],
        "summary": {
            "locked_target_count": len(plan["objects"]),
            "downloaded_file_count": len(downloaded),
            "downloaded_bytes": sum(int(row["downloaded_size"]) for row in downloaded),
            "retained_raw_plan_failure_count": plan["summary"][
                "retained_raw_plan_failure_count"
            ],
            "exact_processed_annotation_count": plan["summary"][
                "exact_processed_annotation_count"
            ],
        },
        "information_boundary": {
            "exact_locked_payload_bytes_downloaded": True,
            "camera_media_decoded": False,
            "sensor_arrays_loaded": False,
            "geometry_or_tracks_loaded": False,
            "predictions_run": False,
            "target_outcomes_opened": False,
            "replacement_allowed": False,
        },
    }
    payload["download_sha256"] = canonical_sha256(
        payload,
        digest_key="download_sha256",
    )
    write_json(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--lock", type=Path, required=True)
        command.add_argument("--selection", type=Path, required=True)

    plan = subparsers.add_parser("plan")
    common(plan)
    plan.add_argument("--output", type=Path, required=True)

    download = subparsers.add_parser("download")
    common(download)
    download.add_argument("--plan", type=Path, required=True)
    download.add_argument("--data-root", type=Path, required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--workers", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        from huggingface_hub import HfApi

        result = build_file_plan(
            repository=args.repository.resolve(),
            lock_path=args.lock.resolve(),
            selection_path=args.selection.resolve(),
            output_path=args.output.resolve(),
            api=HfApi(),
        )
        print(json.dumps({"plan_sha256": result["plan_sha256"], **result["summary"]}))
        return 0

    from huggingface_hub import hf_hub_download

    result = download_file_plan(
        repository=args.repository.resolve(),
        lock_path=args.lock.resolve(),
        selection_path=args.selection.resolve(),
        plan_path=args.plan.resolve(),
        data_root=args.data_root.resolve(),
        output_path=args.output.resolve(),
        max_workers=args.workers,
        hub_download=hf_hub_download,
    )
    print(
        json.dumps({"download_sha256": result["download_sha256"], **result["summary"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
