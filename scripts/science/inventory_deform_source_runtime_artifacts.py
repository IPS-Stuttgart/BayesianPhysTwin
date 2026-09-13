#!/usr/bin/env python3
"""Inventory reusable DLO2/DLO3 runtime artifacts without opening DLO4/DLO5.

The census is intentionally source-only. It searches bounded, explicitly named
roots for prior physical rollouts, predictions, residual coefficients, progress
records, and alternate-backend artifacts. Paths containing DLO4/DLO5 or the
protected DLO45 target run are pruned before their children are enumerated.
Structured source artifacts are inspected only for schema, array shape, hashes,
and narrowly allow-listed metadata needed to build the real residual-panel
adapter. PyTorch/pickle payloads are never deserialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FORBIDDEN_TOKENS = (
    "dlo4",
    "dlo_4",
    "dlo-4",
    "dlo5",
    "dlo_5",
    "dlo-5",
    "33361441865",
    "deform-dlo45",
    "dlo45",
)
SOURCE_TOKENS = ("dlo2", "dlo_2", "dlo-2", "dlo3", "dlo_3", "dlo-3")
RELEVANT_TOKENS = (
    "bayesian",
    "phystwin",
    "deform",
    "dlo",
    "residual",
    "coefficient",
    "prediction",
    "physical",
    "pyelastica",
    "backend",
    "source",
)
DESCENT_TOKENS = (
    "bayesian",
    "phystwin",
    "deform",
    "dlo",
    "github",
    "runner",
    "workflow",
    "_work",
    "cache",
    "dataset",
    "data_set",
    "experiment",
    "output",
    "result",
    "artifact",
    "pyelastica",
)
CANDIDATE_SUFFIXES = {
    ".json",
    ".npz",
    ".npy",
    ".csv",
    ".txt",
    ".dat",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".yaml",
    ".yml",
    ".toml",
}
SAFE_JSON_KEYS = {
    "schema",
    "schema_version",
    "dlo",
    "dataset",
    "dataset_id",
    "source_dlo",
    "target_dlo",
    "backend",
    "backend_id",
    "provider",
    "provider_id",
    "protocol_id",
    "request_id",
    "result_id",
    "artifact_id",
    "model_id",
    "physical_model",
    "residual_model",
    "selected_seed",
    "selected_coefficients",
    "coefficient_names",
    "case_names",
    "trajectory_names",
    "updates",
    "completed_updates",
    "requested_updates",
    "target_eval_read",
    "information_boundary",
    "files",
    "paths",
}
MAX_JSON_BYTES = 20_000_000
MAX_TEXT_SAMPLE_BYTES = 64_000


class CensusError(RuntimeError):
    """Raised when the source-only boundary or bounded-search contract fails."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_path(path: Path) -> str:
    return path.as_posix().lower()


def is_forbidden(path: Path) -> bool:
    value = normalized_path(path)
    return any(token in value for token in FORBIDDEN_TOKENS)


def has_source_identity(path: Path) -> bool:
    value = normalized_path(path)
    return any(token in value for token in SOURCE_TOKENS)


def is_relevant(path: Path) -> bool:
    value = normalized_path(path)
    return any(token in value for token in RELEVANT_TOKENS)


def may_descend(path: Path, *, depth: int) -> bool:
    if is_forbidden(path):
        return False
    if depth <= 1:
        return True
    value = normalized_path(path)
    return any(token in value for token in DESCENT_TOKENS)


def bounded_files(
    roots: Iterable[Path], *, max_depth: int, max_entries: int
) -> tuple[list[Path], dict[str, Any]]:
    queue: deque[tuple[Path, int, Path]] = deque()
    root_records = []
    for supplied in roots:
        root = supplied.expanduser().resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise CensusError(f"root must be a real directory: {root}")
        if is_forbidden(root):
            raise CensusError(f"root crosses protected-target boundary: {root}")
        queue.append((root, 0, root))
        root_records.append(root.as_posix())

    candidates: list[Path] = []
    entries_seen = 0
    directories_seen = 0
    forbidden_directories_pruned = 0
    symlinks_skipped = 0
    permission_errors = []

    while queue:
        directory, depth, root = queue.popleft()
        if depth > max_depth:
            continue
        directories_seen += 1
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(list(iterator), key=lambda item: item.name)
        except (PermissionError, FileNotFoundError, OSError) as error:
            permission_errors.append({"path": directory.as_posix(), "error": repr(error)})
            continue
        for entry in entries:
            entries_seen += 1
            if entries_seen > max_entries:
                raise CensusError(
                    f"bounded search exceeded max_entries={max_entries}; "
                    "narrow the requested roots"
                )
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    symlinks_skipped += 1
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if is_forbidden(path):
                        forbidden_directories_pruned += 1
                        continue
                    if depth < max_depth and may_descend(path, depth=depth + 1):
                        queue.append((path, depth + 1, root))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError as error:
                permission_errors.append({"path": path.as_posix(), "error": repr(error)})
                continue
            if is_forbidden(path):
                continue
            suffix = path.suffix.lower()
            if suffix not in CANDIDATE_SUFFIXES:
                continue
            if not is_relevant(path):
                continue
            # A generic result.json is only eligible when its path carries source
            # identity or an explicit DEFORM/BayesianPhysTwin context.
            if path.name.lower() in {"result.json", "progress.json", "manifest.json"}:
                value = normalized_path(path.parent)
                if not (
                    has_source_identity(path)
                    or "deform" in value
                    or "phystwin" in value
                    or "bayesian" in value
                ):
                    continue
            candidates.append(path.resolve(strict=True))

    candidates = sorted(set(candidates))
    audit = {
        "roots": root_records,
        "max_depth": max_depth,
        "max_entries": max_entries,
        "entries_seen": entries_seen,
        "directories_seen": directories_seen,
        "candidate_files": len(candidates),
        "forbidden_directories_pruned_before_enumeration": forbidden_directories_pruned,
        "symlinks_skipped": symlinks_skipped,
        "permission_errors": permission_errors[:200],
    }
    return candidates, audit


def summarize_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        return {"kind": "json", "inspection": "size-limit"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"kind": "json", "inspection_error": repr(error)}
    summary: dict[str, Any] = {"kind": "json", "type": type(value).__name__}
    if isinstance(value, dict):
        summary["keys"] = sorted(value)[:500]
        allowlisted: dict[str, Any] = {}
        for key in sorted(SAFE_JSON_KEYS & set(value)):
            item = value[key]
            encoded = json.dumps(item, sort_keys=True, default=str)
            if len(encoded) <= 100_000:
                allowlisted[key] = item
            else:
                allowlisted[key] = {
                    "omitted": "allowlisted value exceeds 100 kB",
                    "encoded_bytes": len(encoded),
                }
        summary["allowlisted_metadata"] = allowlisted
        summary["nested_candidate_keys"] = sorted(
            {
                nested_key
                for nested_value in value.values()
                if isinstance(nested_value, dict)
                for nested_key in nested_value
                if any(
                    token in nested_key.lower()
                    for token in (
                        "physical",
                        "residual",
                        "coefficient",
                        "prediction",
                        "trajectory",
                        "source",
                        "backend",
                    )
                )
            }
        )[:500]
    elif isinstance(value, list):
        summary["length"] = len(value)
    return summary


def summarize_npz(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: {
                    "shape": list(archive[name].shape),
                    "dtype": str(archive[name].dtype),
                    "bytes": int(archive[name].nbytes),
                }
                for name in archive.files
            }
    except Exception as error:  # schema census records unreadable source artifacts
        return {"kind": "npz", "inspection_error": repr(error)}
    return {"kind": "npz", "arrays": arrays}


def summarize_npy(path: Path) -> dict[str, Any]:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        return {"kind": "npy", "inspection_error": repr(error)}
    return {
        "kind": "npy",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": int(value.nbytes),
    }


def summarize_text(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_TEXT_SAMPLE_BYTES)
    text = raw.decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()][:20]
    delimiter = None
    if lines:
        try:
            delimiter = csv.Sniffer().sniff("\n".join(lines[:8]), delimiters=",;\t ").delimiter
        except csv.Error:
            delimiter = None
    column_counts = []
    numeric_rows = 0
    for line in lines:
        fields = line.split(delimiter) if delimiter and delimiter != " " else line.split()
        fields = [field for field in fields if field]
        column_counts.append(len(fields))
        try:
            [float(field) for field in fields]
        except ValueError:
            continue
        numeric_rows += 1
    return {
        "kind": "text",
        "sampled_lines": len(lines),
        "delimiter": delimiter,
        "sampled_column_counts": column_counts,
        "sampled_numeric_rows": numeric_rows,
        "first_line": lines[0][:1000] if lines else "",
    }


def summarize(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return summarize_json(path)
    if suffix == ".npz":
        return summarize_npz(path)
    if suffix == ".npy":
        return summarize_npy(path)
    if suffix in {".csv", ".txt", ".dat", ".yaml", ".yml", ".toml"}:
        return summarize_text(path)
    if suffix in {".pt", ".pth", ".pkl", ".pickle"}:
        return {
            "kind": "opaque-pickle-family",
            "deserialized": False,
            "reason": "untrusted/pickle-bearing payloads are hash-only",
        }
    return {"kind": "unknown"}


def candidate_score(path: Path, schema: dict[str, Any]) -> tuple[int, list[str]]:
    value = normalized_path(path)
    score = 0
    reasons = []
    weights = {
        "dlo2": 8,
        "dlo3": 8,
        "source": 4,
        "physical": 6,
        "residual": 8,
        "coefficient": 7,
        "prediction": 6,
        "pyelastica": 9,
        "backend": 5,
        "progress": 2,
        "result": 3,
    }
    for token, weight in weights.items():
        if token in value:
            score += weight
            reasons.append(f"path:{token}")
    schema_text = json.dumps(schema, sort_keys=True).lower()
    for token, weight in (
        ("physical", 4),
        ("residual", 6),
        ("coefficient", 6),
        ("prediction", 4),
        ("trajectory", 3),
        ("backend", 3),
        ("source_result", 4),
    ):
        if token in schema_text:
            score += weight
            reasons.append(f"schema:{token}")
    if schema.get("kind") in {"npz", "npy"}:
        score += 4
        reasons.append("numeric-array-carrier")
    if has_source_identity(path):
        score += 6
        reasons.append("explicit-source-DLO-identity")
    return score, reasons


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-entries", type=int, default=250_000)
    parser.add_argument("--max-candidates", type=int, default=5_000)
    arguments = parser.parse_args()

    files, audit = bounded_files(
        arguments.root,
        max_depth=arguments.max_depth,
        max_entries=arguments.max_entries,
    )
    if len(files) > arguments.max_candidates:
        raise CensusError(
            f"candidate count {len(files)} exceeds {arguments.max_candidates}"
        )
    records = []
    suffix_counts: Counter[str] = Counter()
    for path in files:
        if is_forbidden(path):
            raise CensusError(f"forbidden candidate escaped pruning: {path}")
        schema = summarize(path)
        score, reasons = candidate_score(path, schema)
        suffix_counts[path.suffix.lower()] += 1
        records.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_identity_in_path": has_source_identity(path),
                "score": score,
                "score_reasons": reasons,
                "schema": schema,
            }
        )
    records.sort(key=lambda item: (-item["score"], item["path"]))

    residual_ready = []
    coefficient_candidates = []
    alternate_backend_candidates = []
    for record in records:
        text = (record["path"] + " " + json.dumps(record["schema"], sort_keys=True)).lower()
        if "residual" in text or ("physical" in text and "prediction" in text):
            residual_ready.append(record["path"])
        if "coefficient" in text or "coeff" in text:
            coefficient_candidates.append(record["path"])
        if "pyelastica" in text or "alternate" in text or "backend" in text:
            alternate_backend_candidates.append(record["path"])

    result = {
        "schema": "bayesian-phystwin.deform-dlo23-source-runtime-artifact-census",
        "schema_version": 1,
        "search_audit": audit,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "ranked_candidates": records,
        "adapter_readiness": {
            "residual_or_physical_prediction_candidates": residual_ready,
            "coefficient_candidates": coefficient_candidates,
            "alternate_backend_candidates": alternate_backend_candidates,
            "has_numeric_residual_candidate": any(
                record["path"] in residual_ready
                and record["schema"].get("kind") in {"npz", "npy"}
                for record in records
            ),
            "has_coefficient_candidate": bool(coefficient_candidates),
            "has_alternate_backend_candidate": bool(alternate_backend_candidates),
        },
        "information_boundary": {
            "dlo2_dlo3_source_metadata_and_numeric_schema_read": True,
            "dlo4_directory_children_enumerated": False,
            "dlo4_payload_read": False,
            "dlo5_directory_children_enumerated": False,
            "dlo5_payload_read": False,
            "protected_parent_run_33361441865_artifact_read": False,
            "pickle_bearing_payload_deserialized": False,
            "target_scores_read": False,
            "target_dependent_model_selection": False,
        },
    }
    result["census_id"] = canonical_hash(result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "census_id": result["census_id"],
                "search_audit": audit,
                "candidate_count": len(records),
                "top_candidates": records[:25],
                "adapter_readiness": result["adapter_readiness"],
                "information_boundary": result["information_boundary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
