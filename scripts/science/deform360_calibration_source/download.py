"""Exact-file official-Hub calibration download and byte verification."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep as _sleep
from typing import Any, Final

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

DOWNLOAD_MAX_WORKERS: Final = 2
DOWNLOAD_MAX_ATTEMPTS: Final = 6
DOWNLOAD_INITIAL_BACKOFF_SECONDS: Final = 15.0
DOWNLOAD_MAX_BACKOFF_SECONDS: Final = 240.0

_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectionError",
        "ConnectTimeout",
        "NetworkError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "TimeoutException",
        "TransportError",
    }
)
_HTTP_STATUS_PATTERN = re.compile(
    r"(?:http\s+status|status(?:\s+code)?)\D{0,40}(\d{3})",
    flags=re.IGNORECASE,
)


def _error_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _http_status_code(error: BaseException) -> int | None:
    for item in _error_chain(error):
        response = getattr(item, "response", None)
        for candidate in (response, item):
            status = getattr(candidate, "status_code", None)
            if type(status) is int and 100 <= status <= 599:
                return status
        match = _HTTP_STATUS_PATTERN.search(str(item))
        if match is not None:
            return int(match.group(1))
    return None


def _retry_after_seconds(error: BaseException) -> float | None:
    for item in _error_chain(error):
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", None)
        if not isinstance(headers, Mapping):
            headers = getattr(item, "headers", None)
        if not isinstance(headers, Mapping):
            continue
        raw = headers.get("Retry-After", headers.get("retry-after"))
        if raw is None:
            continue
        try:
            delay = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(delay) and delay >= 0.0:
            return delay
    return None


def _retryable_download_error(error: BaseException) -> bool:
    status = _http_status_code(error)
    if status is not None:
        return status in _RETRYABLE_HTTP_STATUS
    names = {
        base.__name__ for item in _error_chain(error) for base in type(item).__mro__
    }
    return bool(names & _RETRYABLE_EXCEPTION_NAMES)


def _validated_download(
    *,
    record: Mapping[str, Any],
    root: Path,
    destination: Path,
) -> dict[str, Any]:
    relative = record.get("path")
    require(isinstance(relative, str), "download path is malformed")
    expected = (root / relative).resolve()
    require(expected.is_relative_to(root), f"download path escaped root: {relative}")
    require(destination.resolve() == expected, f"download path changed: {relative}")
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


def _completed_download(
    *,
    record: Mapping[str, Any],
    root: Path,
) -> dict[str, Any] | None:
    relative = record.get("path")
    require(isinstance(relative, str), "download path is malformed")
    candidate = root / relative
    require(not candidate.is_symlink(), f"download is a symlink: {relative}")
    if not candidate.exists():
        return None
    return _validated_download(
        record=record,
        root=root,
        destination=candidate,
    )


def _download_once(
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
    )
    return _validated_download(
        record=record,
        root=root,
        destination=destination,
    )


def _retry_configuration(
    *,
    max_attempts: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
) -> None:
    require(max_attempts >= 1, "download max_attempts must be positive")
    require(
        math.isfinite(initial_backoff_seconds) and initial_backoff_seconds >= 0.0,
        "download initial backoff must be finite and nonnegative",
    )
    require(
        math.isfinite(max_backoff_seconds)
        and max_backoff_seconds >= initial_backoff_seconds,
        "download maximum backoff must be finite and no smaller than initial",
    )


def download_one(
    *,
    record: Mapping[str, Any],
    root: Path,
    hub_download: Any,
    max_attempts: int = DOWNLOAD_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DOWNLOAD_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DOWNLOAD_MAX_BACKOFF_SECONDS,
    sleeper: Callable[[float], None] = _sleep,
) -> dict[str, Any]:
    """Download one exact file, reusing verified bytes and retrying transport only."""

    _retry_configuration(
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
    )
    root = root.resolve()
    completed = _completed_download(record=record, root=root)
    if completed is not None:
        return completed

    for attempt in range(max_attempts):
        try:
            return _download_once(
                record=record,
                root=root,
                hub_download=hub_download,
            )
        except Exception as error:
            completed = _completed_download(record=record, root=root)
            if completed is not None:
                return completed
            if attempt + 1 >= max_attempts or not _retryable_download_error(error):
                raise
            exponential = initial_backoff_seconds * (2.0**attempt)
            retry_after = _retry_after_seconds(error) or 0.0
            sleeper(min(max(exponential, retry_after), max_backoff_seconds))
    raise AssertionError("download retry loop did not return or raise")


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
    max_attempts: int = DOWNLOAD_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DOWNLOAD_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DOWNLOAD_MAX_BACKOFF_SECONDS,
    sleeper: Callable[[float], None] = _sleep,
) -> dict[str, Any]:
    require(max_workers >= 1, "max_workers must be positive")
    _retry_configuration(
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
    )
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
    worker_count = min(max_workers, DOWNLOAD_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        downloaded = tuple(
            executor.map(
                lambda record: download_one(
                    record=record,
                    root=root,
                    hub_download=hub_download,
                    max_attempts=max_attempts,
                    initial_backoff_seconds=initial_backoff_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                    sleeper=sleeper,
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
