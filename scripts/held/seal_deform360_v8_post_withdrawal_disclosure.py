#!/usr/bin/env python3
"""Seal the cumulative v8.1 disclosure before attempt 4.

This operator reads no array, image, mask, metric, or JSON payload.  It only
checks already-sealed byte identities and archive metadata, then writes one
fixed report for the prospective lock.  It bars reuse of every v7 execution
artifact and every attempt-3 product.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


PROTOCOL_ID = "deform360-held-online-belief-v8.1"
ARTIFACT_KIND = "Deform360HeldV81PostWithdrawalDevelopmentUseDisclosure"

_BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
_V7_ROOT = _BASE / "held-v7"
_V8_ROOT = _BASE / "held-v8"
_OUTPUT = _V8_ROOT / "post-withdrawal-development-use-disclosure.json"
_ATTEMPT3_ARCHIVE = _BASE / "held-v8-attempt-3-withdrawn-postbarrier"
_ATTEMPT3_REPORT = (
    _ATTEMPT3_ARCHIVE / "execution-withdrawal-postbarrier-attempt3.json"
)
_ATTEMPT3_POINTER = _BASE / "held-v8-attempt-3-withdrawal-pointer.json"
_ATTEMPT3_COMPLETION = (
    _BASE / "held-v8-attempt-3-withdrawal-integrity-completion.json"
)

_V7_FILE_NAMES = frozenset(
    {
        "v7_outcome_withdrawal_report",
        "retired_case_official_target",
        "retired_case_online_prediction",
        "retired_case_online_prediction_seal",
    }
)
_ATTEMPT3_FILE_NAMES = frozenset(
    {
        "v8_attempt3_withdrawal_report",
        "v8_attempt3_withdrawal_pointer",
        "v8_attempt3_withdrawal_integrity_completion",
    }
)
_EXPECTED_FILES: Mapping[str, tuple[Path, int | None, str]] = {
    "v7_outcome_withdrawal_report": (
        _V7_ROOT / "v7-outcome-withdrawal-report.json",
        10_295,
        "7bcab7169fc2addad8e56b7bb5ca9086b5249e9a744e18b9d51a7f395098c1a3",
    ),
    "retired_case_official_target": (
        _V7_ROOT / "calibration/outcomes/002-rope-silk-ep0003/official_target.npz",
        536_992,
        "850a894f1e1eb447fddbb877ac2fbf38225e97514a1218cc7ea1182212f471a8",
    ),
    "retired_case_online_prediction": (
        _V7_ROOT
        / "calibration/cases/002-rope-silk-ep0003/online/online_prediction.npz",
        994_650,
        "ecae2a595b50c91bf842c3e86eb38559eec0ad43aeeba40da2dd8a9098a31f8d",
    ),
    "retired_case_online_prediction_seal": (
        _V7_ROOT
        / "calibration/cases/002-rope-silk-ep0003/online/online_prediction_seal.json",
        3_684,
        "afac640547cf4f0de1f168dd4642b841ee96cc274b5e47401aadd4e361255814",
    ),
    "v8_attempt3_withdrawal_report": (
        _ATTEMPT3_REPORT,
        None,
        "6d9c62606d18744d275df51fd08e041205bf15b38175d74c69690eafd511054b",
    ),
    "v8_attempt3_withdrawal_pointer": (
        _ATTEMPT3_POINTER,
        None,
        "75acc7e9535f41528d22739ae8eeb5a0a2247c0fe63c097ad1da2859d7b33246",
    ),
    "v8_attempt3_withdrawal_integrity_completion": (
        _ATTEMPT3_COMPLETION,
        None,
        "f3d1e8a6670484c81ac04743bcdb020cdee3fba02229a64844a8a9c9f4b8b989",
    ),
}

_ATTEMPT3_ARCHIVE_INTEGRITY = {
    "path": str(_ATTEMPT3_ARCHIVE),
    "root_mode_octal": "0500",
    "fully_nonwritable": True,
    "postseal_noncode_inventory_sha256": (
        "5d398e998e2b738db545ffefd254712c6822017cfc5be6e7de435d5883c8c4c8"
    ),
    "postseal_noncode_entry_count": 1466,
}

_POST_WITHDRAWAL_DEVELOPMENT_HASHES = {
    "scratch_frozen_field_source_sha256": (
        "e106611d9f5e9c6125b5c4c1704db06703108f1ce635d55e6e15d8c8b3a32822"
    ),
    "scratch_query_development_source_sha256": (
        "3f008ef9f9b6fe52c6a36e1939a56ec35e160912efae44ba5a12d11a59a572ae"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _artifact(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    unsigned = dict(value)
    digest = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    signed = {**unsigned, "artifact_sha256": digest}
    return signed, _canonical_json(signed)


def _bind_expected_regular_file(
    path: Path,
    *,
    expected_size: int | None,
    expected_sha256: str,
    role: str,
) -> dict[str, Any]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(absolute)
    _require(not stat.S_ISLNK(before.st_mode), f"{role} is a symlink")
    _require(stat.S_ISREG(before.st_mode), f"{role} is not a regular file")
    _require(absolute.resolve() == absolute, f"{role} path is non-canonical")
    _require(
        stat.S_IMODE(before.st_mode) == 0o400,
        f"{role} mode is not exactly 0400",
    )
    if expected_size is not None:
        _require(before.st_size == expected_size, f"{role} size changed")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (before.st_dev, before.st_ino, before.st_size)
            == (opened.st_dev, opened.st_ino, opened.st_size),
            f"{role} changed while opening",
        )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(absolute)
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        == (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ),
        f"{role} changed while hashing",
    )
    observed_sha256 = digest.hexdigest()
    _require(observed_sha256 == expected_sha256, f"{role} SHA-256 changed")
    return {
        "path": os.fspath(absolute),
        "sha256": observed_sha256,
        "size_bytes": before.st_size,
        "mode_octal": "0400",
    }


def _bind_attempt3_archive_integrity() -> dict[str, Any]:
    archive = Path(os.path.abspath(os.fspath(_ATTEMPT3_ARCHIVE)))
    observed = os.lstat(archive)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o500
        and archive.resolve() == archive,
        "attempt-3 archive is not a canonical mode-0500 directory",
    )
    for current, directories, files in os.walk(archive, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            child = current_path / name
            state = os.lstat(child)
            _require(
                not stat.S_ISLNK(state.st_mode)
                and (stat.S_ISDIR(state.st_mode) or stat.S_ISREG(state.st_mode))
                and state.st_mode & 0o222 == 0,
                f"attempt-3 archive entry is writable or unsafe: {child}",
            )
    return {**_ATTEMPT3_ARCHIVE_INTEGRITY, "path": str(archive)}


def expected_unsigned_report(
    bindings: Mapping[str, Mapping[str, Any]],
    archive_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        set(bindings) == set(_EXPECTED_FILES),
        "disclosure input binding set changed",
    )
    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "disclosed_v7_files": {
            name: dict(bindings[name]) for name in sorted(_V7_FILE_NAMES)
        },
        "disclosed_v8_attempt3_files": {
            name: dict(bindings[name]) for name in sorted(_ATTEMPT3_FILE_NAMES)
        },
        "v8_attempt3_archive_integrity": dict(archive_integrity),
        "v8_attempt3_revision_basis": {
            "official_x0_geometry_used_to_diagnose_exclusion_liveness": True,
            "future_target_coordinates_masks_or_scores_used_for_revision": False,
            "queried_prediction_score_or_gate_existed": False,
            "revision": (
                "replace exact-one-per-center matching with the inclusive 15 mm "
                "x0-only radius union"
            ),
        },
        "post_withdrawal_development": {
            **_POST_WITHDRAWAL_DEVELOPMENT_HASHES,
            "retired_official_target_opened_by_development_process": True,
            "retired_online_prediction_opened_by_development_process": True,
            "future_coordinates_or_masks_may_have_been_read": True,
            "derived_metrics_may_have_been_computed": True,
            "field_hypothesis_was_subsequently_reselected_on_independent_open27": True,
        },
        "retirement": {
            "exact_episode": "002-rope-silk-ep0003",
            "replacement_episode": "072-cotton-clohesline-ep0003",
            "replacement_search_excluded_entire_002_rope_silk_object": True,
            "reason": (
                "the exact held-v7 episode was exposed after formal withdrawal; "
                "the replacement was selected outside that object's episodes"
            ),
        },
        "v8_1_reuse_boundary": {
            "v7_target_or_staging_reused": False,
            "v7_physical_prediction_reused": False,
            "v7_online_prediction_reused": False,
            "v7_query_or_score_reused": False,
            "v7_execution_artifact_reused": False,
            "v7_withdrawal_report_used_only_as_immutable_lineage": True,
            "v8_attempt3_predictions_reused": False,
            "v8_attempt3_source_manifests_reused": False,
            "v8_attempt3_frozen_fields_reused": False,
            "v8_attempt3_target_artifacts_reused": False,
            "v8_attempt3_official_x0_query_artifacts_reused": False,
            "v8_attempt3_queried_prediction_artifacts_reused": False,
            "v8_attempt3_score_or_gate_artifacts_reused": False,
            "v8_attempt3_partial_artifacts_reused": False,
            "all_v8_1_attempt4_predictions_targets_queries_and_scores_fresh": True,
            "full_15_case_fresh_rerun_required": True,
        },
        "claim_boundary": (
            "This disclosure preserves prospective episode-level evaluation; it "
            "does not turn open development or v8.1 into an official Deform360 "
            "state-of-the-art comparison."
        ),
    }


def build_report() -> tuple[dict[str, Any], bytes]:
    bindings = {
        name: _bind_expected_regular_file(
            path,
            expected_size=size,
            expected_sha256=sha256,
            role=name.replace("_", " "),
        )
        for name, (path, size, sha256) in _EXPECTED_FILES.items()
    }
    return _artifact(
        expected_unsigned_report(bindings, _bind_attempt3_archive_integrity())
    )


def _write_once(path: Path, payload: bytes) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require(absolute.parent.is_dir(), "held-v8 root does not exist")
    _require(not absolute.parent.is_symlink(), "held-v8 root is a symlink")
    _require(
        absolute.parent.resolve() == absolute.parent, "held-v8 root is non-canonical"
    )
    if os.path.lexists(absolute):
        before = os.lstat(absolute)
        _require(
            stat.S_ISREG(before.st_mode)
            and not stat.S_ISLNK(before.st_mode)
            and stat.S_IMODE(before.st_mode) == 0o400,
            "existing disclosure is not a sealed regular file",
        )
        descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            observed = b""
            while block := os.read(descriptor, 1024 * 1024):
                observed += block
        finally:
            os.close(descriptor)
        _require(observed == payload, "existing disclosure payload changed")
        return
    descriptor = os.open(
        absolute,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(absolute, 0o400, follow_symlinks=False)
    except BaseException:
        absolute.unlink(missing_ok=True)
        raise


def main() -> None:
    _require(_V8_ROOT == _OUTPUT.parent, "disclosure output root changed")
    _, payload = build_report()
    _write_once(_OUTPUT, payload)
    print(hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
