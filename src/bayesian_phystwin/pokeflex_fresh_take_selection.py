"""Outcome-blind PokeFlex take selection against complete Git history."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git(repository: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        input=input_bytes,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def canonical_payload_sha256(
    payload: Mapping[str, Any], *, digest_field: str
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def take_object(take_id: str) -> str:
    object_name, separator, number = str(take_id).rpartition("_T")
    _require(bool(separator) and object_name and number.isdigit(), "invalid take id")
    return object_name


def take_inventory_sha256(take_ids: Sequence[str]) -> str:
    encoded = "\n".join(sorted(take_ids)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_fresh_takes(
    public_take_ids: Sequence[str],
    referenced_take_ids: Sequence[str],
    *,
    salt_label: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select one never-referenced public take per still-eligible object."""

    takes = tuple(sorted({str(value) for value in public_take_ids}))
    _require(len(takes) == len(public_take_ids), "public take inventory is duplicated")
    _require(bool(salt_label), "selection salt is empty")
    referenced = {str(value) for value in referenced_take_ids}
    by_object: dict[str, list[str]] = {}
    for take_id in takes:
        by_object.setdefault(take_object(take_id), []).append(take_id)
    selected = []
    exhausted = []
    prefix = salt_label.encode("utf-8") + b"\0"
    for object_name, object_takes in sorted(by_object.items()):
        eligible = [take_id for take_id in object_takes if take_id not in referenced]
        if not eligible:
            exhausted.append(object_name)
            continue
        selected.append(
            min(
                eligible,
                key=lambda take_id: hashlib.sha256(
                    prefix + take_id.encode("utf-8")
                ).digest(),
            )
        )
    return tuple(selected), tuple(exhausted)


def _repository_ref_inventory(repository: Path) -> tuple[list[str], list[str]]:
    raw = _git(
        repository,
        "for-each-ref",
        "--format=%(refname)%00%(objectname)",
        "refs/heads",
        "refs/remotes",
        "refs/tags",
    ).decode("utf-8")
    rows = sorted(line for line in raw.splitlines() if line)
    tips = sorted({line.rpartition("\0")[2] for line in rows})
    _require(all(len(value) == 40 for value in tips), "Git ref tip is invalid")
    return rows, tips


def _reachable_blob_inventory(repository: Path) -> tuple[list[str], list[str]]:
    raw = _git(repository, "rev-list", "--objects", "--all").decode(
        "utf-8", errors="surrogateescape"
    )
    object_rows = [line for line in raw.splitlines() if line]
    object_ids = sorted({line.split(" ", 1)[0] for line in object_rows})
    batch_input = ("\n".join(object_ids) + "\n").encode("ascii")
    checks = _git(
        repository,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_bytes=batch_input,
    ).decode("ascii")
    blob_rows = []
    for line in checks.splitlines():
        object_id, object_type, size = line.split(" ")
        if object_type == "blob":
            blob_rows.append(f"{object_id}\0{size}")
    paths = [line.split(" ", 1)[1] for line in object_rows if " " in line]
    return sorted(blob_rows), paths


def scan_repository_take_references(
    repository: Path,
    public_take_ids: Sequence[str],
) -> dict[str, Any]:
    """Find take IDs in every reachable path and blob without reading outcomes."""

    repository = Path(repository).resolve()
    takes = tuple(sorted({str(value) for value in public_take_ids}))
    _require(bool(takes), "public take inventory is empty")
    ref_rows, ref_tips = _repository_ref_inventory(repository)
    blob_rows, paths = _reachable_blob_inventory(repository)
    referenced = {
        take_id for take_id in takes if any(take_id in path for path in paths)
    }
    tokens = {take_id: take_id.encode("utf-8") for take_id in takes}
    process = subprocess.Popen(
        ("git", "-C", str(repository), "cat-file", "--batch"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(process.stdin is not None and process.stdout is not None, "Git pipe failed")
    for blob_row in blob_rows:
        object_id, size_text = blob_row.split("\0")
        process.stdin.write(object_id.encode("ascii") + b"\n")
        process.stdin.flush()
        header = process.stdout.readline().decode("ascii").strip()
        expected = f"{object_id} blob {size_text}"
        _require(header == expected, f"Git blob header changed: {header}")
        size = int(size_text)
        content = process.stdout.read(size)
        _require(len(content) == size, "Git blob was truncated")
        _require(process.stdout.read(1) == b"\n", "Git blob separator changed")
        for take_id, token in tokens.items():
            if take_id not in referenced and token in content:
                referenced.add(take_id)
    process.stdin.close()
    return_code = process.wait()
    stderr = process.stderr.read() if process.stderr is not None else b""
    _require(return_code == 0, f"Git blob scan failed: {stderr.decode('utf-8')}")
    return {
        "git_ref_count": len(ref_rows),
        "git_ref_tip_count": len(ref_tips),
        "git_ref_digest": hashlib.sha256(
            "\n".join(ref_rows).encode("utf-8")
        ).hexdigest(),
        "git_ref_tip_digest": hashlib.sha256(
            "\n".join(ref_tips).encode("ascii")
        ).hexdigest(),
        "reachable_blob_count": len(blob_rows),
        "reachable_blob_digest": hashlib.sha256(
            "\n".join(blob_rows).encode("ascii")
        ).hexdigest(),
        "referenced_take_ids": sorted(referenced),
        "referenced_take_count": len(referenced),
        "referenced_take_digest": take_inventory_sha256(tuple(referenced)),
    }


def build_fresh_take_selection_manifest(
    repository: Path,
    public_take_ids: Sequence[str],
    *,
    salt_label: str,
    selection_id: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Build one checksummed selection manifest from target-free inventories."""

    takes = tuple(sorted({str(value) for value in public_take_ids}))
    scan = scan_repository_take_references(repository, takes)
    selected, exhausted = select_fresh_takes(
        takes,
        scan["referenced_take_ids"],
        salt_label=salt_label,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexFreshTakeSelectionManifest",
        "selection_id": selection_id,
        "created_at_utc": created_at_utc,
        "outcome_access_before_selection": False,
        "repository_head": _git(repository, "rev-parse", "HEAD")
        .decode("ascii")
        .strip(),
        "repository_clean": not bool(
            _git(repository, "status", "--porcelain").strip()
        ),
        "scan_scope": (
            "every path and blob reachable from all local, remote, and tag refs "
            "present after the registered fetch"
        ),
        **{name: value for name, value in scan.items() if name != "referenced_take_ids"},
        "public_take_count": len(takes),
        "public_take_digest": take_inventory_sha256(takes),
        "unreferenced_public_take_count": len(takes) - scan["referenced_take_count"],
        "eligible_object_count": len(selected),
        "exhausted_objects": list(exhausted),
        "salt_label": salt_label,
        "selection_rule": (
            "For each object with an unreferenced public take, select the take "
            "minimizing SHA256(salt_label || NUL || take_id); sort by object; "
            "never replace a selected take."
        ),
        "selected_take_ids": list(selected),
        "selected_take_digest": take_inventory_sha256(selected),
        "replacement_allowed": False,
        "claim_boundary": (
            "Freshness is relative to all Git refs and reachable blobs present at "
            "the recorded repository head; no target content or outcome is read."
        ),
    }
    _require(payload["repository_clean"] is True, "selection checkout is dirty")
    payload["selection_manifest_sha256"] = canonical_payload_sha256(
        payload, digest_field="selection_manifest_sha256"
    )
    return payload
