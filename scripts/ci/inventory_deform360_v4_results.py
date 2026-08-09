#!/usr/bin/env python3
"""Inventory reusable Deform360 result artifacts without opening raw datasets.

The inventory is intentionally structure-first.  It walks one admitted results
root without following symbolic links, records bounded metadata for files that
may contribute to the joint-sparse v4 development manifest, and parses only
small JSON descriptors.  Binary scientific payloads are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

SCHEMA: Final = "bayesian-phystwin.deform360-v4-results-inventory"
SCHEMA_VERSION: Final = 1
CANDIDATE_SUFFIXES: Final = frozenset(
    {".json", ".npz", ".npy", ".yaml", ".yml", ".pt", ".pth"}
)
CANDIDATE_KEYWORDS: Final = (
    "joint",
    "sparse",
    "tree",
    "gauge",
    "factor",
    "linearization",
    "observation",
    "calibration",
    "source-gate",
    "metric",
    "manifest",
    "sample",
    "provider",
    "result",
    "receipt",
    "seal",
    "query",
    "covariance",
    "prediction",
)
EXTRACTED_JSON_FIELDS: Final = (
    "schema",
    "schema_version",
    "semantics",
    "artifact_id",
    "input_id",
    "manifest_id",
    "result_id",
    "bundle_id",
    "observation_artifact_id",
    "linearization_artifact_id",
    "provider_manifest_id",
    "gauge_tree_prior_artifact_id",
    "gauge_tree_prior_id",
    "gauge_prior_id",
    "object_id",
    "episode_id",
    "stratum",
    "case_id",
    "stream_id",
    "status",
    "implementation_revision",
    "source_revision",
    "prob4d_revision",
    "visual_production_result_id",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _load_small_json(path: Path, *, maximum_bytes: int) -> Mapping[str, Any]:
    size = path.stat().st_size
    _require(size <= maximum_bytes, "JSON descriptor exceeds the read bound")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot parse bounded JSON descriptor") from error
    _require(isinstance(value, Mapping), "JSON descriptor is not an object")
    return cast(Mapping[str, Any], value)


def _normal_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    _require(
        not parsed.is_absolute() and ".." not in parsed.parts and relative != ".",
        "inventory path is not a confined relative path",
    )
    return relative


def _is_candidate(relative: str, suffix: str) -> bool:
    lowered = relative.lower()
    return suffix in CANDIDATE_SUFFIXES and any(
        keyword in lowered for keyword in CANDIDATE_KEYWORDS
    )


def _json_summary(
    path: Path,
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    try:
        value = _load_small_json(path, maximum_bytes=maximum_bytes)
    except ValueError as error:
        return {
            "json_status": "unreadable-or-out-of-bound",
            "json_error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        }
    selected: dict[str, Any] = {}
    for key in EXTRACTED_JSON_FIELDS:
        item = value.get(key)
        if item is None or isinstance(item, (str, int, float, bool)):
            selected[key] = item
    return {
        "json_status": "parsed",
        "json_sha256": _sha256_file(path),
        "top_level_keys": sorted(map(str, value))[:128],
        "selected_fields": selected,
    }


def _regular_directory(path: Path, *, name: str) -> Path:
    _require(not path.is_symlink(), f"{name} must not be a symbolic link")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    _require(stat.S_ISDIR(metadata.st_mode), f"{name} is not a directory")
    return resolved


def _outside_forbidden(root: Path, forbidden: Sequence[Path]) -> None:
    for blocked in forbidden:
        resolved = _regular_directory(blocked, name="forbidden root")
        _require(
            root != resolved and resolved not in root.parents and root not in resolved.parents,
            "results root overlaps a forbidden raw-data root",
        )


def inventory_results(
    root: Path,
    *,
    forbidden_roots: Sequence[Path],
    maximum_depth: int,
    maximum_entries: int,
    maximum_candidates: int,
    maximum_json_bytes: int,
) -> dict[str, Any]:
    """Return a bounded, content-addressed inventory of one results tree."""

    admitted_root = _regular_directory(root, name="results root")
    _outside_forbidden(admitted_root, forbidden_roots)
    _require(maximum_depth >= 1, "maximum_depth must be positive")
    _require(maximum_entries >= 1, "maximum_entries must be positive")
    _require(maximum_candidates >= 1, "maximum_candidates must be positive")
    _require(maximum_json_bytes >= 1, "maximum_json_bytes must be positive")

    stack: list[tuple[Path, int]] = [(admitted_root, 0)]
    entry_count = 0
    regular_file_count = 0
    directory_count = 1
    symlink_count = 0
    other_count = 0
    candidates: list[dict[str, Any]] = []
    suffix_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    candidate_directory_counts: Counter[str] = Counter()

    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise ValueError("cannot scan results directory") from error
        for entry in entries:
            entry_count += 1
            _require(entry_count <= maximum_entries, "results inventory entry bound exceeded")
            entry_path = Path(entry.path)
            relative = _normal_relative(entry_path, admitted_root)
            try:
                if entry.is_symlink():
                    symlink_count += 1
                    continue
                if entry.is_dir(follow_symlinks=False):
                    directory_count += 1
                    if depth < maximum_depth:
                        stack.append((entry_path, depth + 1))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    other_count += 1
                    continue
                regular_file_count += 1
                suffix = entry_path.suffix.lower()
                suffix_counts[suffix or "<none>"] += 1
                if not _is_candidate(relative, suffix):
                    continue
                _require(
                    len(candidates) < maximum_candidates,
                    "results inventory candidate bound exceeded",
                )
                metadata = entry.stat(follow_symlinks=False)
                record: dict[str, Any] = {
                    "path": relative,
                    "byte_count": metadata.st_size,
                    "suffix": suffix,
                }
                if suffix == ".json":
                    summary = _json_summary(
                        entry_path,
                        maximum_bytes=maximum_json_bytes,
                    )
                    record.update(summary)
                    fields = summary.get("selected_fields")
                    if isinstance(fields, Mapping):
                        schema = fields.get("schema")
                        if isinstance(schema, str):
                            schema_counts[schema] += 1
                parent = PurePosixPath(relative).parent.as_posix()
                candidate_directory_counts[parent] += 1
                candidates.append(record)
            except OSError as error:
                raise ValueError("cannot inspect results entry") from error

    candidates.sort(key=lambda item: cast(str, item["path"]))
    identity: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "root": str(admitted_root),
        "bounds": {
            "maximum_depth": maximum_depth,
            "maximum_entries": maximum_entries,
            "maximum_candidates": maximum_candidates,
            "maximum_json_bytes": maximum_json_bytes,
        },
        "counts": {
            "entry_count": entry_count,
            "directory_count": directory_count,
            "regular_file_count": regular_file_count,
            "symlink_count": symlink_count,
            "other_count": other_count,
            "candidate_count": len(candidates),
        },
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "candidate_schema_counts": dict(sorted(schema_counts.items())),
        "candidate_directory_counts": dict(
            sorted(candidate_directory_counts.items())
        ),
        "candidates": candidates,
        "information_boundary": {
            "results_tree_only": True,
            "binary_scientific_payloads_loaded": False,
            "raw_dataset_payloads_opened": False,
            "adaptive_confirmation_payloads_opened": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
    }
    return {**identity, "inventory_id": hashlib.sha256(_canonical_bytes(identity)).hexdigest()}


def _write_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not os.path.lexists(path), "inventory output already exists")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--forbidden-root",
        type=Path,
        action="append",
        default=[],
        help="raw-data root that must be disjoint from the admitted results root",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-depth", type=int, default=10)
    parser.add_argument("--maximum-entries", type=int, default=250_000)
    parser.add_argument("--maximum-candidates", type=int, default=50_000)
    parser.add_argument("--maximum-json-bytes", type=int, default=8 * 1024 * 1024)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    inventory = inventory_results(
        arguments.root,
        forbidden_roots=arguments.forbidden_root,
        maximum_depth=arguments.maximum_depth,
        maximum_entries=arguments.maximum_entries,
        maximum_candidates=arguments.maximum_candidates,
        maximum_json_bytes=arguments.maximum_json_bytes,
    )
    _write_no_clobber(arguments.output, inventory)
    print(
        json.dumps(
            {
                "inventory_id": inventory["inventory_id"],
                "counts": inventory["counts"],
                "candidate_schema_counts": inventory["candidate_schema_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
