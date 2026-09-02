#!/usr/bin/env python3
"""Inventory only the already-open DEFORM DLO2/DLO3 source cohort.

The script is deliberately target-blind: it refuses paths whose resolved name is
DLO4 or DLO5 and does not enumerate their children.  It summarizes source file
layouts, representative numeric carrier schemas, and repository-side adapters
needed to build the hierarchical missing-physics residual panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_DLOS = ("DLO2", "DLO3")
FORBIDDEN_DLOS = ("DLO4", "DLO5")
TEXT_SUFFIXES = {".txt", ".csv", ".dat"}
NUMERIC_SUFFIXES = TEXT_SUFFIXES | {".npy", ".npz", ".mat"}


class InventoryError(RuntimeError):
    """Raised when the source-only custody boundary cannot be guaranteed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_directories(root: Path, dlo: str) -> list[Path]:
    names = {dlo, dlo.lower(), dlo.replace("DLO", "DLO_"), dlo.replace("DLO", "dlo_")}
    candidates: list[Path] = []
    for name in sorted(names):
        direct = root / name
        if direct.is_dir() and not direct.is_symlink():
            candidates.append(direct.resolve(strict=True))
    if candidates:
        return sorted(set(candidates))
    # Bounded search: directory names only.  Do not descend into DLO4/DLO5.
    for current, directories, _files in os.walk(root):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if name.upper() not in FORBIDDEN_DLOS and not (current_path / name).is_symlink()
        ]
        for name in directories:
            normalized = re.sub(r"[^A-Z0-9]", "", name.upper())
            if normalized == dlo:
                candidates.append((current_path / name).resolve(strict=True))
        if len(current_path.relative_to(root).parts) >= 4:
            directories[:] = []
    return sorted(set(candidates))


def locate_sources(root: Path) -> dict[str, Path]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise InventoryError("dataset root must be a real directory")
    result: dict[str, Path] = {}
    for dlo in SOURCE_DLOS:
        candidates = _candidate_directories(resolved, dlo)
        if len(candidates) != 1:
            raise InventoryError(f"expected exactly one {dlo} directory, found {candidates}")
        path = candidates[0]
        if any(part.upper() in FORBIDDEN_DLOS for part in path.parts):
            raise InventoryError(f"source path crosses forbidden target boundary: {path}")
        result[dlo] = path
    return result


def _finite_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    summary: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "size": int(values.size),
    }
    if values.dtype.kind in "biufc" and values.size:
        real_values = np.real(values).astype(float, copy=False).reshape(-1)
        finite = np.isfinite(real_values)
        summary.update(
            {
                "finite_fraction": float(np.mean(finite)),
                "minimum": float(np.min(real_values[finite])) if np.any(finite) else None,
                "maximum": float(np.max(real_values[finite])) if np.any(finite) else None,
                "mean": float(np.mean(real_values[finite])) if np.any(finite) else None,
            }
        )
    return summary


def inspect_text(path: Path) -> dict[str, Any]:
    raw_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for _ in range(12):
            line = stream.readline()
            if not line:
                break
            raw_lines.append(line.rstrip("\n\r"))
    nonempty = [line for line in raw_lines if line.strip()]
    delimiter = None
    if nonempty:
        try:
            delimiter = csv.Sniffer().sniff("\n".join(nonempty[:6]), delimiters=",;\t ").delimiter
        except csv.Error:
            delimiter = None
    counts = []
    numeric_rows = 0
    for line in nonempty:
        fields = line.split(delimiter) if delimiter and delimiter != " " else line.split()
        fields = [field for field in fields if field != ""]
        counts.append(len(fields))
        try:
            [float(field) for field in fields]
        except ValueError:
            continue
        numeric_rows += 1
    return {
        "kind": "text",
        "sampled_line_count": len(raw_lines),
        "delimiter": delimiter,
        "sampled_column_counts": counts,
        "sampled_numeric_rows": numeric_rows,
        "first_line": raw_lines[0][:500] if raw_lines else "",
    }


def inspect_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    record: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "suffix": suffix,
        "sha256": sha256(path),
    }
    if suffix in TEXT_SUFFIXES:
        record["schema"] = inspect_text(path)
    elif suffix == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        record["schema"] = {"kind": "npy", **_finite_summary(value)}
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            record["schema"] = {
                "kind": "npz",
                "arrays": {key: _finite_summary(archive[key]) for key in archive.files},
            }
    elif suffix == ".mat":
        try:
            import scipy.io
        except ImportError as error:
            record["schema"] = {"kind": "mat", "inspection_error": str(error)}
        else:
            record["schema"] = {
                "kind": "mat",
                "variables": [
                    {"name": name, "shape": list(shape), "dtype": dtype}
                    for name, shape, dtype in scipy.io.whosmat(path)
                ],
            }
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            schema: Any = {"kind": "json-object", "keys": sorted(value)[:200]}
        elif isinstance(value, list):
            schema = {"kind": "json-list", "length": len(value)}
        else:
            schema = {"kind": "json-scalar", "type": type(value).__name__}
        record["schema"] = schema
    return record


def inventory_dlo(path: Path) -> dict[str, Any]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file() and not candidate.is_symlink()]
    files.sort()
    suffix_counts = Counter(candidate.suffix.lower() or "<none>" for candidate in files)
    directory_counts: Counter[str] = Counter()
    for candidate in files:
        relative = candidate.relative_to(path)
        directory_counts[relative.parent.as_posix()] += 1

    representatives: list[Path] = []
    by_suffix: dict[str, list[Path]] = defaultdict(list)
    for candidate in files:
        by_suffix[candidate.suffix.lower()].append(candidate)
    for suffix in sorted(by_suffix):
        candidates = by_suffix[suffix]
        if suffix in NUMERIC_SUFFIXES or suffix == ".json":
            representatives.extend(candidates[: min(5, len(candidates))])
    representatives = sorted(set(representatives))[:30]

    return {
        "path": path.as_posix(),
        "file_count": len(files),
        "total_bytes": int(sum(candidate.stat().st_size for candidate in files)),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "directory_file_counts": dict(sorted(directory_counts.items())),
        "filenames": [candidate.relative_to(path).as_posix() for candidate in files],
        "representative_schemas": {
            candidate.relative_to(path).as_posix(): inspect_file(candidate)
            for candidate in representatives
        },
    }


def repository_adapter_matches(repository: Path) -> list[dict[str, Any]]:
    patterns = re.compile(
        r"DLO[2345]|deform.*dlo|residual_coeff|physical_model|PyElastica|pyelastica",
        re.IGNORECASE,
    )
    matches: list[dict[str, Any]] = []
    for path in sorted(repository.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(repository).as_posix()
        if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
            continue
        if path.stat().st_size > 2_000_000:
            continue
        if path.suffix.lower() not in {".py", ".json", ".yaml", ".yml", ".md", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        line_numbers = [index for index, line in enumerate(text.splitlines(), start=1) if patterns.search(line)]
        if line_numbers:
            matches.append({"path": relative, "matching_lines": line_numbers[:100]})
    return matches[:300]


def canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    sources = locate_sources(arguments.dataset_root)
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.deform-dlo23-source-schema-inventory",
        "schema_version": 1,
        "source_dlos": {
            dlo: inventory_dlo(path) for dlo, path in sorted(sources.items())
        },
        "repository_adapter_matches": repository_adapter_matches(
            arguments.repository.resolve(strict=True)
        ),
        "information_boundary": {
            "dlo2_payload_read": True,
            "dlo3_payload_read": True,
            "dlo4_path_enumerated": False,
            "dlo4_payload_read": False,
            "dlo5_path_enumerated": False,
            "dlo5_payload_read": False,
            "protected_parent_target_result_read": False,
            "target_dependent_model_choice": False,
        },
    }
    result["inventory_id"] = canonical_hash(result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "inventory_id": result["inventory_id"],
                "source_summary": {
                    dlo: {
                        "path": value["path"],
                        "file_count": value["file_count"],
                        "total_bytes": value["total_bytes"],
                        "suffix_counts": value["suffix_counts"],
                    }
                    for dlo, value in result["source_dlos"].items()
                },
                "repository_adapter_match_count": len(
                    result["repository_adapter_matches"]
                ),
                "information_boundary": result["information_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
