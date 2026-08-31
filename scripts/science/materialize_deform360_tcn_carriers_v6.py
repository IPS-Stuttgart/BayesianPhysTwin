#!/usr/bin/env python3
"""Materialize only the Deform360 carriers required by the v6 TCN audit.

The official raw and processed Hugging Face repositories are pinned by commit.
Only metadata, raw tactile arrays/medians, and processed robot action arrays are
downloaded. Camera, point-cloud, Gaussian-splat, and video payloads are excluded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RAW_REPOSITORY = "brownu/deform360"
RAW_REVISION = "5ea8c5d3fc7b4a7b4f9f921f2ceb1de24610f6a4"
PROCESSED_REPOSITORY = "brownu/deform360_processed"
PROCESSED_REVISION = "e92deaf7e437e7e51ad464706ae647f522a279d9"
USER_AGENT = "BayesianPhysTwin-Deform360-TCN-carrier-materializer-v6"
ALLOWED_SUFFIXES = (".npy", ".npz", ".json")
FORBIDDEN_TOKENS = (
    "camera",
    "pcd",
    "splat",
    ".mp4",
    ".avi",
    ".mov",
    ".tar",
)


class MaterializationError(RuntimeError):
    """Raised when a pinned selective materialization cannot be completed."""


@dataclass(frozen=True)
class DownloadPlan:
    repository: str
    revision: str
    source_candidates: tuple[str, ...]
    destination: Path
    kind: str
    expected_size: int | None = None


@dataclass(frozen=True)
class DownloadRecord:
    repository: str
    revision: str
    source_path: str
    destination: str
    kind: str
    size_bytes: int
    sha256: str
    cached: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "source_path": self.source_path,
            "destination": self.destination,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "cached": self.cached,
        }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def request_bytes(url: str, *, attempts: int = 8) -> bytes:
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            retryable = error.code in {408, 425, 429, 500, 502, 503, 504}
            if not retryable or attempt + 1 == attempts:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
        time.sleep(min(2**attempt + random.random(), 30.0))
    raise AssertionError("unreachable")


def request_json(url: str) -> Any:
    return json.loads(request_bytes(url).decode("utf-8"))


def tree_url(repository: str, revision: str, path: str) -> str:
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(path, safe="/")
    suffix = f"/{encoded_path}" if encoded_path else ""
    return (
        f"https://huggingface.co/api/datasets/{repository}/tree/"
        f"{encoded_revision}{suffix}?expand=true"
    )


def list_tree(repository: str, revision: str, path: str) -> list[dict[str, Any]]:
    value = request_json(tree_url(repository, revision, path))
    if not isinstance(value, list):
        raise MaterializationError(
            f"unexpected tree response for {repository}@{revision}:{path}"
        )
    return [dict(item) for item in value if isinstance(item, Mapping)]


def resolve_url(repository: str, revision: str, path: str) -> str:
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(path, safe="/")
    return (
        f"https://huggingface.co/datasets/{repository}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )


def ensure_destination(root: Path, destination: Path) -> None:
    root_resolved = root.resolve()
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    parent_resolved = destination_parent.resolve()
    if root_resolved != parent_resolved and root_resolved not in parent_resolved.parents:
        raise MaterializationError(f"destination escapes materialization root: {destination}")


def validate_source_path(plan: DownloadPlan, source_path: str) -> None:
    lowered = source_path.lower()
    if any(token in lowered for token in FORBIDDEN_TOKENS):
        raise MaterializationError(f"forbidden carrier entered plan: {source_path}")
    if not lowered.endswith(ALLOWED_SUFFIXES):
        raise MaterializationError(f"unsupported carrier suffix: {source_path}")
    if plan.kind == "metadata" and not lowered.endswith("metadata.json"):
        raise MaterializationError(f"metadata plan resolved to non-metadata file: {source_path}")
    if plan.kind == "tactile" and not lowered.endswith(".npy"):
        raise MaterializationError(f"tactile plan resolved to non-npy file: {source_path}")
    if plan.kind == "robot" and not lowered.endswith(("robot.npy", "robot.npz")):
        raise MaterializationError(f"robot plan resolved to non-robot file: {source_path}")


def download_one(plan: DownloadPlan, root: Path) -> DownloadRecord:
    ensure_destination(root, plan.destination)
    if plan.destination.is_file() and plan.destination.stat().st_size > 0:
        size = plan.destination.stat().st_size
        if plan.expected_size is None or size == plan.expected_size:
            source_path = plan.source_candidates[0]
            validate_source_path(plan, source_path)
            return DownloadRecord(
                repository=plan.repository,
                revision=plan.revision,
                source_path=source_path,
                destination=plan.destination.relative_to(root).as_posix(),
                kind=plan.kind,
                size_bytes=size,
                sha256=sha256_file(plan.destination),
                cached=True,
            )

    errors: list[str] = []
    for source_path in plan.source_candidates:
        validate_source_path(plan, source_path)
        temporary = plan.destination.with_name(
            plan.destination.name + f".part-{os.getpid()}-{random.randrange(1 << 30)}"
        )
        try:
            request = urllib.request.Request(
                resolve_url(plan.repository, plan.revision, source_path),
                headers={"User-Agent": USER_AGENT},
            )
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=300) as response:
                with temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if size <= 0:
                raise MaterializationError(f"empty carrier: {source_path}")
            if plan.expected_size is not None and size != plan.expected_size:
                raise MaterializationError(
                    f"size mismatch for {source_path}: {size} != {plan.expected_size}"
                )
            os.replace(temporary, plan.destination)
            return DownloadRecord(
                repository=plan.repository,
                revision=plan.revision,
                source_path=source_path,
                destination=plan.destination.relative_to(root).as_posix(),
                kind=plan.kind,
                size_bytes=size,
                sha256=digest.hexdigest(),
                cached=False,
            )
        except urllib.error.HTTPError as error:
            errors.append(f"{source_path}: HTTP {error.code}")
            temporary.unlink(missing_ok=True)
            if error.code != 404:
                try:
                    payload = request_bytes(
                        resolve_url(plan.repository, plan.revision, source_path)
                    )
                    if not payload:
                        raise MaterializationError(f"empty carrier: {source_path}")
                    if plan.expected_size is not None and len(payload) != plan.expected_size:
                        raise MaterializationError(
                            f"size mismatch for {source_path}: "
                            f"{len(payload)} != {plan.expected_size}"
                        )
                    temporary.write_bytes(payload)
                    os.replace(temporary, plan.destination)
                    return DownloadRecord(
                        repository=plan.repository,
                        revision=plan.revision,
                        source_path=source_path,
                        destination=plan.destination.relative_to(root).as_posix(),
                        kind=plan.kind,
                        size_bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(),
                        cached=False,
                    )
                except Exception as retry_error:  # noqa: BLE001
                    errors.append(f"{source_path}: retry {retry_error!r}")
                    temporary.unlink(missing_ok=True)
        except Exception as error:  # noqa: BLE001
            errors.append(f"{source_path}: {error!r}")
            temporary.unlink(missing_ok=True)
    raise MaterializationError(
        "all carrier candidates failed for "
        f"{plan.destination}: {'; '.join(errors)}"
    )


def episode_records(metadata: Mapping[str, Any]) -> list[int]:
    raw = metadata.get("sequences", metadata.get("episodes", metadata.get("takes")))
    if isinstance(raw, Mapping):
        items: Iterable[tuple[Any, Any]] = sorted(
            raw.items(),
            key=lambda item: (
                0 if str(item[0]).isdigit() else 1,
                int(item[0]) if str(item[0]).isdigit() else str(item[0]),
            ),
        )
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = enumerate(raw)
    else:
        return []
    result: list[int] = []
    for raw_id, record in items:
        if not isinstance(record, Mapping):
            continue
        result.append(int(raw_id) if str(raw_id).isdigit() else len(result))
    return result


def object_ids(protocol: Mapping[str, Any]) -> list[str]:
    development = [str(item) for item in protocol["development_object_ids"]]
    evaluation = [str(item) for item in protocol["evaluation_object_ids"]]
    combined = development + evaluation
    if len(combined) != len(set(combined)):
        raise MaterializationError("development and evaluation object rosters overlap")
    if len(development) != 14 or len(evaluation) != 92:
        raise MaterializationError(
            f"unexpected roster sizes: development={len(development)}, "
            f"evaluation={len(evaluation)}"
        )
    return combined


def metadata_plan(root: Path, object_id: str, expected_size: int | None) -> DownloadPlan:
    source = f"raw/{object_id}/metadata.json"
    destination = root / "raw-repository" / "raw" / object_id / "metadata.json"
    return DownloadPlan(
        repository=RAW_REPOSITORY,
        revision=RAW_REVISION,
        source_candidates=(source,),
        destination=destination,
        kind="metadata",
        expected_size=expected_size,
    )


def discover_object_plans(
    root: Path,
    object_id: str,
) -> tuple[DownloadRecord, list[DownloadPlan], dict[str, Any]]:
    raw_root = f"raw/{object_id}"
    entries = list_tree(RAW_REPOSITORY, RAW_REVISION, raw_root)
    metadata_entry = next(
        (
            item
            for item in entries
            if str(item.get("path", "")).lower().endswith("/metadata.json")
            and item.get("type") == "file"
        ),
        None,
    )
    if metadata_entry is None:
        raise MaterializationError(f"metadata missing for {object_id}")
    metadata_record = download_one(
        metadata_plan(root, object_id, int(metadata_entry.get("size") or 0) or None),
        root,
    )
    metadata = read_json(root / "raw-repository" / "raw" / object_id / "metadata.json")
    episodes = episode_records(metadata)
    if len(episodes) < 4:
        raise MaterializationError(f"too few episodes for {object_id}: {episodes}")

    tactile_directories = sorted(
        str(item.get("path"))
        for item in entries
        if item.get("type") == "directory"
        and "tactile" in str(item.get("path", "")).lower()
    )
    if len(tactile_directories) < 2:
        raise MaterializationError(
            f"too few tactile directories for {object_id}: {tactile_directories}"
        )

    plans: list[DownloadPlan] = []
    tactile_counts: dict[str, int] = {}
    for directory in tactile_directories:
        children = list_tree(RAW_REPOSITORY, RAW_REVISION, directory)
        selected = sorted(
            (
                item
                for item in children
                if item.get("type") == "file"
                and str(item.get("path", "")).lower().endswith(".npy")
            ),
            key=lambda item: str(item.get("path")),
        )
        nonmedian = [
            item
            for item in selected
            if not Path(str(item.get("path"))).name.lower().startswith("median_")
        ]
        if len(nonmedian) != len(episodes):
            raise MaterializationError(
                f"tactile episode count mismatch for {object_id}:{directory}: "
                f"{len(nonmedian)} != {len(episodes)}"
            )
        tactile_counts[Path(directory).name] = len(nonmedian)
        for item in selected:
            source = str(item["path"])
            destination = root / "raw-repository" / source
            plans.append(
                DownloadPlan(
                    repository=RAW_REPOSITORY,
                    revision=RAW_REVISION,
                    source_candidates=(source,),
                    destination=destination,
                    kind="tactile",
                    expected_size=int(item.get("size") or 0) or None,
                )
            )

    for episode_id in episodes:
        source_candidates = tuple(
            f"processed/{object_id}/{directory}/robot/{filename}"
            for directory in (
                f"episode_{episode_id}",
                f"episode_{episode_id:04d}",
                f"episode-{episode_id}",
            )
            for filename in ("robot.npy", "robot.npz")
        )
        destination = (
            root
            / "processed-repository"
            / "processed"
            / object_id
            / f"episode_{episode_id}"
            / "robot"
            / "robot.npy"
        )
        plans.append(
            DownloadPlan(
                repository=PROCESSED_REPOSITORY,
                revision=PROCESSED_REVISION,
                source_candidates=source_candidates,
                destination=destination,
                kind="robot",
            )
        )
    inventory = {
        "object_id": object_id,
        "episode_ids": episodes,
        "tactile_directories": tactile_directories,
        "tactile_episode_counts": tactile_counts,
        "planned_tactile_files": sum(plan.kind == "tactile" for plan in plans),
        "planned_robot_files": sum(plan.kind == "robot" for plan in plans),
    }
    return metadata_record, plans, inventory


def execute_downloads(
    plans: Sequence[DownloadPlan],
    root: Path,
    workers: int,
) -> list[DownloadRecord]:
    records: list[DownloadRecord] = []
    failures: list[str] = []
    completed = 0
    total = len(plans)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, plan, root): plan for plan in plans}
        for future in concurrent.futures.as_completed(futures):
            plan = futures[future]
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001
                failures.append(f"{plan.destination}: {error!r}")
            completed += 1
            if completed % 250 == 0 or completed == total:
                print(
                    f"download_progress={completed}/{total} "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise MaterializationError(
            f"{len(failures)} carrier downloads failed:\n" + "\n".join(failures[:50])
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    objects = object_ids(protocol)
    root = args.output_root.expanduser().resolve()
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    metadata_records: list[DownloadRecord] = []
    plans: list[DownloadPlan] = []
    inventories: list[dict[str, Any]] = []
    print(f"discovering_objects={len(objects)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max(args.workers, 1), 12)
    ) as executor:
        futures = {
            executor.submit(discover_object_plans, root, object_id): object_id
            for object_id in objects
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            object_id = futures[future]
            try:
                metadata_record, object_plans, inventory = future.result()
            except Exception as error:  # noqa: BLE001
                raise MaterializationError(
                    f"carrier discovery failed for {object_id}: {error!r}"
                ) from error
            metadata_records.append(metadata_record)
            plans.extend(object_plans)
            inventories.append(inventory)
            if index % 10 == 0 or index == len(objects):
                print(
                    f"discovery_progress={index}/{len(objects)} "
                    f"planned_files={len(plans) + len(metadata_records)}",
                    flush=True,
                )

    source_paths = [
        candidate
        for plan in plans
        for candidate in plan.source_candidates[:1]
    ]
    if any(
        any(token in source.lower() for token in FORBIDDEN_TOKENS)
        for source in source_paths
    ):
        raise MaterializationError("forbidden payload found in materialization plan")

    downloaded_records = execute_downloads(plans, root, max(args.workers, 1))
    records = metadata_records + downloaded_records
    records.sort(key=lambda item: item.destination)
    inventories.sort(key=lambda item: item["object_id"])
    total_bytes = sum(item.size_bytes for item in records)
    manifest = {
        "schema": "bayesian-phystwin/deform360-selective-tcn-carriers-v6",
        "schema_version": 6,
        "status": "complete",
        "raw_repository": RAW_REPOSITORY,
        "raw_revision": RAW_REVISION,
        "processed_repository": PROCESSED_REPOSITORY,
        "processed_revision": PROCESSED_REVISION,
        "output_root": str(root),
        "object_count": len(objects),
        "development_object_count": len(protocol["development_object_ids"]),
        "evaluation_object_count": len(protocol["evaluation_object_ids"]),
        "file_count": len(records),
        "total_size_bytes": total_bytes,
        "camera_files_downloaded": 0,
        "geometry_or_point_cloud_files_downloaded": 0,
        "video_files_downloaded": 0,
        "object_inventory": inventories,
        "files": [record.as_dict() for record in records],
    }
    write_json(args.manifest, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "object_count": len(objects),
                "file_count": len(records),
                "total_size_bytes": total_bytes,
                "manifest": str(args.manifest),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
