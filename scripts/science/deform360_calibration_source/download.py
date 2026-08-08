"""Exact-file official-Hub calibration download and byte verification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import time
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

MAXIMUM_DOWNLOAD_WORKERS = 2
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_INITIAL_BACKOFF_SECONDS = 5.0
DEFAULT_MAXIMUM_BACKOFF_SECONDS = 120.0
_TRANSIENT_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "TimeoutError",
    }
)


class DownloadRetryExhaustedError(RuntimeError):
    """Raised when one planned public-Hub file exhausts its retry budget."""


def _planned_destination(root: Path, relative: str) -> Path:
    logical = PurePosixPath(relative)
    require(
        not logical.is_absolute()
        and bool(logical.parts)
        and all(part not in {"", ".", ".."} for part in logical.parts),
        "download path is unsafe",
    )
    resolved_root = root.resolve()
    destination = resolved_root.joinpath(*logical.parts).resolve()
    require(
        destination.is_relative_to(resolved_root),
        "download path escaped the calibration root",
    )
    return destination


def _verified_file_record(
    *,
    record: Mapping[str, Any],
    destination: Path,
    source: str,
    attempt_count: int,
) -> dict[str, Any] | None:
    if not destination.exists():
        return None
    require(not destination.is_symlink(), "download path is a symbolic link")
    require(destination.is_file(), "download path is not a regular file")
    declared_size = record.get("size")
    lfs_sha256 = record.get("lfs_sha256")
    if not isinstance(declared_size, int) or not isinstance(lfs_sha256, str):
        return None
    size = destination.stat().st_size
    if size != declared_size:
        return None
    digest = file_sha256(destination)
    if digest != lfs_sha256:
        return None
    return {
        **dict(record),
        "downloaded_size": size,
        "downloaded_sha256": digest,
        "download_source": source,
        "download_attempt_count": attempt_count,
    }


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    values: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        values.append(current)
        cause = current.__cause__
        current = cause if cause is not None else current.__context__
    return tuple(values)


def _exception_status_code(error: BaseException) -> int | None:
    for candidate in _exception_chain(error):
        response = getattr(candidate, "response", None)
        for value in (
            getattr(candidate, "status_code", None),
            getattr(response, "status_code", None),
        ):
            if isinstance(value, int):
                return value
        match = re.search(r"\b(429|500|502|503|504)\b", str(candidate))
        if match is not None:
            return int(match.group(1))
    return None


def _is_transient_download_error(error: BaseException) -> bool:
    status_code = _exception_status_code(error)
    if status_code is not None:
        return status_code in _TRANSIENT_HTTP_STATUS_CODES
    return any(
        type(candidate).__name__ in _TRANSIENT_EXCEPTION_NAMES
        for candidate in _exception_chain(error)
    )


def _retry_delay(
    *,
    failed_attempt: int,
    initial_backoff_seconds: float,
    maximum_backoff_seconds: float,
) -> float:
    return min(
        initial_backoff_seconds * (2 ** (failed_attempt - 1)),
        maximum_backoff_seconds,
    )


def _planned_file_label(relative: str) -> str:
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]


def download_one(
    *,
    record: Mapping[str, Any],
    root: Path,
    hub_download: Any,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    maximum_backoff_seconds: float = DEFAULT_MAXIMUM_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    relative = record.get("path")
    require(isinstance(relative, str), "download path is malformed")
    require(max_attempts >= 1, "download attempts must be positive")
    require(
        initial_backoff_seconds >= 0.0,
        "initial download backoff must be non-negative",
    )
    require(
        maximum_backoff_seconds >= initial_backoff_seconds,
        "maximum download backoff must cover the initial backoff",
    )
    expected = _planned_destination(root, relative)
    reusable = _verified_file_record(
        record=record,
        destination=expected,
        source="verified_local",
        attempt_count=0,
    )
    if reusable is not None:
        return reusable
    if expected.exists():
        require(expected.is_file() and not expected.is_symlink(), "download path is unsafe")
        expected.unlink()

    for attempt in range(1, max_attempts + 1):
        try:
            destination = Path(
                hub_download(
                    repo_id=DATASET_REPOSITORY,
                    repo_type="dataset",
                    revision=DATASET_REVISION,
                    filename=relative,
                    local_dir=str(root),
                )
            ).resolve()
        except Exception as error:
            recovered = _verified_file_record(
                record=record,
                destination=expected,
                source="verified_local_after_transient_error",
                attempt_count=attempt,
            )
            if recovered is not None:
                return recovered
            if not _is_transient_download_error(error):
                raise
            if attempt == max_attempts:
                label = _planned_file_label(relative)
                raise DownloadRetryExhaustedError(
                    "public-Hub retry budget exhausted for planned file "
                    f"{label} after {max_attempts} attempts"
                ) from error
            sleep(
                _retry_delay(
                    failed_attempt=attempt,
                    initial_backoff_seconds=initial_backoff_seconds,
                    maximum_backoff_seconds=maximum_backoff_seconds,
                )
            )
            continue

        require(destination == expected, "download path changed")
        require(
            destination.is_file() and not destination.is_symlink(),
            "download is not a regular file",
        )
        size = destination.stat().st_size
        declared_size = record.get("size")
        if isinstance(declared_size, int):
            require(size == declared_size, "download size changed")
        digest = file_sha256(destination)
        lfs_sha256 = record.get("lfs_sha256")
        if isinstance(lfs_sha256, str):
            require(digest == lfs_sha256, "download LFS digest changed")
        return {
            **dict(record),
            "downloaded_size": size,
            "downloaded_sha256": digest,
            "download_source": "hub_download",
            "download_attempt_count": attempt,
        }
    raise AssertionError("unreachable download retry state")


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
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    maximum_backoff_seconds: float = DEFAULT_MAXIMUM_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    require(max_workers >= 1, "max_workers must be positive")
    require(max_attempts >= 1, "download attempts must be positive")
    effective_workers = min(max_workers, MAXIMUM_DOWNLOAD_WORKERS)
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
    records = sorted(
        (
            file
            for row in plan["objects"]
            if row.get("status") == "planned"
            for file in row["selected_files"]
        ),
        key=lambda item: item["path"],
    )
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        downloaded = tuple(
            executor.map(
                lambda record: download_one(
                    record=record,
                    root=root,
                    hub_download=hub_download,
                    max_attempts=max_attempts,
                    initial_backoff_seconds=initial_backoff_seconds,
                    maximum_backoff_seconds=maximum_backoff_seconds,
                    sleep=sleep,
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
        "download_policy": {
            "requested_max_workers": max_workers,
            "effective_max_workers": effective_workers,
            "maximum_allowed_workers": MAXIMUM_DOWNLOAD_WORKERS,
            "max_attempts": max_attempts,
            "initial_backoff_seconds": initial_backoff_seconds,
            "maximum_backoff_seconds": maximum_backoff_seconds,
            "verified_local_reuse": True,
            "xet_disabled": os.environ.get("HF_HUB_DISABLE_XET") == "1",
            "implicit_token_disabled": (
                os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1"
            ),
        },
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
    policy = value.get("download_policy")
    require(isinstance(policy, Mapping), "download policy is missing")
    require(
        policy.get("maximum_allowed_workers") == MAXIMUM_DOWNLOAD_WORKERS,
        "download worker ceiling changed",
    )
    effective_workers = policy.get("effective_max_workers")
    require(
        isinstance(effective_workers, int)
        and 1 <= effective_workers <= MAXIMUM_DOWNLOAD_WORKERS,
        "effective download workers are invalid",
    )
    require(
        policy.get("verified_local_reuse") is True,
        "verified local download reuse is disabled",
    )
    require(policy.get("xet_disabled") is True, "Xet download was not disabled")
    require(
        policy.get("implicit_token_disabled") is True,
        "implicit Hub credentials were not disabled",
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
        local = _planned_destination(data_root, relative)
        require(
            local.is_file() and not local.is_symlink(),
            "download file is missing",
        )
        require(
            file_sha256(local) == digest,
            "download bytes changed",
        )
    return value
