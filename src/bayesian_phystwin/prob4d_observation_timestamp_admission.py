"""Claim-bearing admission for Prob4D observation timestamp lineages.

The portable timestamp loader proves that a sidecar is internally
content-addressed. Claim-bearing use additionally needs an independently known
identity for the raw timestamp source. Otherwise a forged sidecar and a forged
content ID could remain mutually self-consistent.

This module snapshots the timestamp sidecar and factor-bundle manifest through
regular-file descriptors, parses private byte-for-byte copies, checks the raw
source identity against a separate verification artifact, and rejects concurrent
replacement or in-place mutation of either original file.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._canonical_contracts import plain_json
from ._portable_contracts import sha256_digest
from .observation_belief import ObservationBeliefV1
from .prob4d_observation_timestamps import (
    Prob4DObservationTimestampBindingV1,
    load_prob4d_observation_timestamp_binding,
    load_prob4d_observation_timestamp_lineage,
)

_CLAIM_METADATA_FIELDS = frozenset(
    {
        "prob4d_timestamp_source_sha256",
        "prob4d_timestamp_source_independently_verified",
        "prob4d_timestamp_source_verification_artifact_id",
        "prob4d_timestamp_lineage_artifact_id",
        "prob4d_timestamp_lineage_file_sha256",
    }
)
_MAXIMUM_SNAPSHOT_BYTES = 64 * 1024 * 1024


def _file_identity(information: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(information.st_dev),
        int(information.st_ino),
        int(information.st_size),
        int(information.st_mtime_ns),
        int(information.st_ctime_ns),
    )


def _ordinary_snapshot(
    path: str | Path,
    *,
    name: str,
) -> tuple[Path, bytes, str]:
    """Read one stable regular-file snapshot without following a symlink."""

    artifact_path = Path(path)
    try:
        path_information = os.lstat(artifact_path)
    except OSError as error:
        raise ValueError(f"{name} is unreadable") from error
    if stat.S_ISLNK(path_information.st_mode):
        raise ValueError(f"{name} path must not be a symlink")
    if not stat.S_ISREG(path_information.st_mode):
        raise ValueError(f"{name} must be an ordinary file")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(artifact_path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"{name} path must not be a symlink") from error
        raise ValueError(f"{name} is unreadable") from error

    payload = bytearray()
    try:
        descriptor_information = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_information.st_mode):
            raise ValueError(f"{name} must be an ordinary file")
        if (
            descriptor_information.st_dev,
            descriptor_information.st_ino,
        ) != (path_information.st_dev, path_information.st_ino):
            raise ValueError(f"{name} changed while being opened")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
            if len(payload) > _MAXIMUM_SNAPSHOT_BYTES:
                raise ValueError(f"{name} exceeds the snapshot size limit")
        final_descriptor_information = os.fstat(descriptor)
        if _file_identity(final_descriptor_information) != _file_identity(
            descriptor_information
        ):
            raise ValueError(f"{name} changed while being read")
        if len(payload) != descriptor_information.st_size:
            raise ValueError(f"{name} changed while being read")
        try:
            final_path_information = os.lstat(artifact_path)
        except OSError as error:
            raise ValueError(f"{name} changed while being read") from error
        if stat.S_ISLNK(final_path_information.st_mode) or _file_identity(
            final_path_information
        ) != _file_identity(final_descriptor_information):
            raise ValueError(f"{name} changed while being read")
    except OSError as error:
        raise ValueError(f"{name} is unreadable") from error
    finally:
        os.close(descriptor)

    snapshot = bytes(payload)
    return artifact_path, snapshot, hashlib.sha256(snapshot).hexdigest()


def _write_private_snapshot(
    directory: Path,
    filename: str,
    payload: bytes,
) -> Path:
    """Create one owner-only, non-replacing snapshot file."""

    path = directory / filename
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("private snapshot write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def load_claim_bearing_prob4d_observation_timestamp_binding(
    observation: ObservationBeliefV1,
    *,
    timestamp_lineage_path: str | Path,
    expected_timestamp_source_sha256: str,
    timestamp_source_verification_artifact_id: str,
    bundle_manifest_path: str | Path,
    expected_bundle_manifest_sha256: str,
    row_factor_ids: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> Prob4DObservationTimestampBindingV1:
    """Admit timestamp evidence through private exact-byte snapshots.

    ``expected_timestamp_source_sha256`` and
    ``timestamp_source_verification_artifact_id`` must come from an independently
    frozen source/calibration manifest, not from the timestamp sidecar being
    admitted. The verification artifact must be distinct from the sidecar itself.
    Both the timestamp sidecar and the observation-factor bundle are read once
    through regular-file descriptors and parsed only from private snapshots.
    Their original paths are re-snapshotted afterward to reject replacement or
    in-place mutation.
    """

    expected_source = sha256_digest(
        expected_timestamp_source_sha256,
        name="expected_timestamp_source_sha256",
    )
    verification_id = sha256_digest(
        timestamp_source_verification_artifact_id,
        name="timestamp_source_verification_artifact_id",
    )
    expected_bundle = sha256_digest(
        expected_bundle_manifest_sha256,
        name="expected_bundle_manifest_sha256",
    )
    timestamp_path, timestamp_bytes, timestamp_sha_before = _ordinary_snapshot(
        timestamp_lineage_path,
        name="Prob4D timestamp lineage",
    )
    bundle_path, bundle_bytes, bundle_sha_before = _ordinary_snapshot(
        bundle_manifest_path,
        name="Prob4D observation-factor bundle",
    )
    if bundle_sha_before != expected_bundle:
        raise ValueError("Prob4D bundle manifest checksum mismatch")

    with tempfile.TemporaryDirectory(
        prefix="bpt-prob4d-timestamp-admission-"
    ) as temporary:
        snapshot_root = Path(temporary)
        timestamp_snapshot_path = _write_private_snapshot(
            snapshot_root,
            "timestamp-lineage.json",
            timestamp_bytes,
        )
        bundle_snapshot_path = _write_private_snapshot(
            snapshot_root,
            "observation-factor-bundle.json",
            bundle_bytes,
        )
        lineage = load_prob4d_observation_timestamp_lineage(
            timestamp_snapshot_path
        )
        if lineage.source_artifact_sha256 != expected_source:
            raise ValueError(
                "Prob4D timestamp source artifact differs from independent evidence"
            )
        lineage_id = lineage.artifact_id
        if lineage_id is None:
            raise AssertionError(
                "validated Prob4D timestamp lineage lacks an artifact ID"
            )
        if verification_id == lineage_id:
            raise ValueError(
                "Prob4D timestamp lineage cannot serve as its own verification artifact"
            )

        caller_metadata = {} if metadata is None else plain_json(metadata)
        if not isinstance(caller_metadata, dict):
            raise ValueError("claim-bearing timestamp metadata must be a mapping")
        overlap = _CLAIM_METADATA_FIELDS.intersection(caller_metadata)
        if overlap:
            raise ValueError(
                "claim-bearing timestamp metadata reserves fields "
                f"{sorted(overlap)}"
            )
        admitted_metadata = {
            **caller_metadata,
            "prob4d_timestamp_source_sha256": expected_source,
            "prob4d_timestamp_source_independently_verified": True,
            "prob4d_timestamp_source_verification_artifact_id": verification_id,
            "prob4d_timestamp_lineage_artifact_id": lineage_id,
            "prob4d_timestamp_lineage_file_sha256": timestamp_sha_before,
        }
        binding = load_prob4d_observation_timestamp_binding(
            observation,
            timestamp_lineage_path=timestamp_snapshot_path,
            bundle_manifest_path=bundle_snapshot_path,
            expected_bundle_manifest_sha256=bundle_sha_before,
            row_factor_ids=row_factor_ids,
            metadata=admitted_metadata,
        )

    _, _, timestamp_sha_after = _ordinary_snapshot(
        timestamp_path,
        name="Prob4D timestamp lineage",
    )
    if timestamp_sha_after != timestamp_sha_before:
        raise ValueError("Prob4D timestamp lineage changed during admission")
    _, _, bundle_sha_after = _ordinary_snapshot(
        bundle_path,
        name="Prob4D observation-factor bundle",
    )
    if bundle_sha_after != bundle_sha_before:
        raise ValueError(
            "Prob4D observation-factor bundle changed during admission"
        )
    if binding.timestamp_lineage_artifact_id != lineage_id:
        raise ValueError("Prob4D timestamp lineage identity changed during admission")
    if dict(binding.metadata) != admitted_metadata:
        raise ValueError("claim-bearing timestamp evidence was not retained exactly")
    return binding


__all__ = ["load_claim_bearing_prob4d_observation_timestamp_binding"]
