#!/usr/bin/env python3
"""Audit the complete DEFORM DLO4/DLO5 release without fitting a model.

The audit is deliberately format-aware but outcome-neutral.  It hashes the
small release, inventories both DLO trees, inspects safe array/table metadata,
and extracts the public interface of any existing BayesianPhysTwin DEFORM
adapter.  It never unpickles arbitrary objects and never performs score-bearing
model selection.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "bayesian-phystwin/deform-dlo45-realdata-audit-v1"
DLO_RE = re.compile(r"(?:^|[^a-z0-9])dlo[_ -]?([45])(?:$|[^a-z0-9])", re.I)
SAFE_TEXT_SUFFIXES = {".csv", ".txt", ".tsv", ".dat"}
NUMERIC_SUFFIXES = {".mat", ".npy", ".npz"}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_id(value: dict[str, Any], field: str = "audit_id") -> str:
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def locate_dlo_roots(root: Path) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    for path in [root, *sorted(root.rglob("*"))]:
        if not path.is_dir() or path.is_symlink():
            continue
        match = DLO_RE.search(path.name)
        if match:
            candidates[f"DLO{match.group(1)}"].append(path)
    selected: dict[str, Path] = {}
    for dlo in ("DLO4", "DLO5"):
        options = candidates.get(dlo, [])
        require(options, f"could not locate {dlo} below {root}")
        ranked = sorted(
            options,
            key=lambda path: (
                -sum(1 for item in path.rglob("*") if item.is_file()),
                len(path.parts),
                path.as_posix(),
            ),
        )
        selected[dlo] = ranked[0]
    require(selected["DLO4"] != selected["DLO5"], "DLO roots collapsed")
    return selected


def inspect_npy(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:
        return {"reader": "numpy-unavailable", "error": str(error)}
    try:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        if isinstance(value, np.lib.npyio.NpzFile):
            rows = []
            for key in sorted(value.files):
                array = value[key]
                rows.append(
                    {"key": key, "shape": list(array.shape), "dtype": str(array.dtype)}
                )
            value.close()
            return {"reader": "numpy-npz", "arrays": rows}
        return {
            "reader": "numpy-npy",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    except Exception as error:  # format diagnostic, retained verbatim
        return {"reader": "numpy", "error": f"{type(error).__name__}: {error}"}


def inspect_mat(path: Path) -> dict[str, Any]:
    try:
        import scipy.io

        rows = [
            {"name": name, "shape": list(shape), "class": class_name}
            for name, shape, class_name in scipy.io.whosmat(path)
        ]
        return {"reader": "scipy-whosmat", "variables": rows}
    except Exception as scipy_error:
        try:
            import h5py

            rows: list[dict[str, Any]] = []
            with h5py.File(path, "r") as handle:
                def visitor(name: str, value: Any) -> None:
                    if isinstance(value, h5py.Dataset):
                        rows.append(
                            {
                                "name": name,
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                            }
                        )
                handle.visititems(visitor)
            return {"reader": "h5py", "variables": rows}
        except Exception as h5_error:
            return {
                "reader": "mat-unreadable",
                "error": (
                    f"scipy={type(scipy_error).__name__}: {scipy_error}; "
                    f"h5py={type(h5_error).__name__}: {h5_error}"
                ),
            }


def inspect_text(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            lines = []
            for _ in range(8):
                line = stream.readline()
                if line == "":
                    break
                lines.append(line.rstrip("\r\n"))
    except (OSError, UnicodeError) as error:
        return {"reader": "text-unreadable", "error": f"{type(error).__name__}: {error}"}
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    if lines and path.suffix.lower() in {".csv", ".tsv"}:
        try:
            parsed = list(csv.reader(lines, delimiter=delimiter))
            widths = [len(row) for row in parsed]
        except csv.Error:
            widths = []
    else:
        widths = [len(line.split()) for line in lines]
    return {
        "reader": "text-prefix",
        "sample_line_count": len(lines),
        "sample_column_counts": widths,
        "sample_character_counts": [len(line) for line in lines],
    }


def inspect_file(path: Path, root: Path, dlo: str) -> dict[str, Any]:
    stat = path.stat()
    suffix = path.suffix.lower()
    record: dict[str, Any] = {
        "dlo": dlo,
        "path": path.relative_to(root).as_posix(),
        "name": path.name,
        "suffix": suffix,
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }
    if suffix in {".npy", ".npz"}:
        record["format"] = inspect_npy(path)
    elif suffix == ".mat":
        record["format"] = inspect_mat(path)
    elif suffix in SAFE_TEXT_SUFFIXES:
        record["format"] = inspect_text(path)
    elif suffix == ".json" and stat.st_size <= 4 * 1024 * 1024:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            record["format"] = {
                "reader": "json",
                "root_type": type(value).__name__,
                "top_level_keys": sorted(value) if isinstance(value, dict) else [],
                "length": len(value) if isinstance(value, (dict, list)) else None,
            }
        except Exception as error:
            record["format"] = {
                "reader": "json-unreadable",
                "error": f"{type(error).__name__}: {error}",
            }
    else:
        try:
            with path.open("rb") as stream:
                prefix = stream.read(16)
            record["format"] = {"reader": "binary-prefix", "hex": prefix.hex()}
        except OSError as error:
            record["format"] = {
                "reader": "binary-unreadable",
                "error": f"{type(error).__name__}: {error}",
            }
    return record


def adapter_interfaces(repository_root: Path) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in sorted(repository_root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        lowered = path.as_posix().lower()
        if "deform" not in lowered and "dlo" not in lowered:
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeError, SyntaxError) as error:
            matches.append(
                {
                    "path": path.relative_to(repository_root).as_posix(),
                    "parse_error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        functions = sorted(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        classes = sorted(
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        )
        arguments: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument":
                continue
            for value in node.args:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    arguments.add(value.value)
        matches.append(
            {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(path),
                "functions": functions,
                "classes": classes,
                "cli_arguments": sorted(arguments),
            }
        )
    return matches


def summarize_numeric(records: list[dict[str, Any]]) -> dict[str, Any]:
    shapes: Counter[str] = Counter()
    variables: Counter[str] = Counter()
    readable = 0
    unreadable = 0
    for record in records:
        metadata = record.get("format", {})
        reader = str(metadata.get("reader", ""))
        if reader.endswith("unreadable") or "error" in metadata:
            unreadable += 1
        if record["suffix"] in NUMERIC_SUFFIXES and "error" not in metadata:
            readable += 1
        arrays = metadata.get("arrays", [])
        for array in arrays:
            shapes[str(array.get("shape"))] += 1
            variables[str(array.get("key"))] += 1
        for variable in metadata.get("variables", []):
            shapes[str(variable.get("shape"))] += 1
            variables[str(variable.get("name"))] += 1
        if "shape" in metadata:
            shapes[str(metadata["shape"])] += 1
    return {
        "readable_numeric_file_count": readable,
        "unreadable_or_error_file_count": unreadable,
        "common_shapes": shapes.most_common(30),
        "common_variable_names": variables.most_common(50),
    }


def write_outputs(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output / "files.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("dlo", "path", "name", "suffix", "bytes", "sha256", "reader"),
        )
        writer.writeheader()
        for record in result["files"]:
            writer.writerow(
                {
                    "dlo": record["dlo"],
                    "path": record["path"],
                    "name": record["name"],
                    "suffix": record["suffix"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                    "reader": record.get("format", {}).get("reader"),
                }
            )
    lines = [
        "# DEFORM DLO4/DLO5 real-data audit v1",
        "",
        f"- Decision: `{result['summary']['decision']}`",
        f"- Audit ID: `{result['audit_id']}`",
        f"- Dataset root: `{result['dataset_root']}`",
        f"- DLO4 files: `{result['summary']['DLO4_file_count']}`",
        f"- DLO5 files: `{result['summary']['DLO5_file_count']}`",
        f"- Total bytes: `{result['summary']['total_bytes']}`",
        f"- Readable numeric files: `{result['numeric_summary']['readable_numeric_file_count']}`",
        f"- Adapter files found: `{len(result['adapter_interfaces'])}`",
        "",
        "## Suffix counts",
        "",
        "| Suffix | Count |",
        "|---|---:|",
    ]
    for suffix, count in result["summary"]["suffix_counts"]:
        lines.append(f"| `{suffix or '<none>'}` | {count} |")
    lines.extend(
        [
            "",
            "## Most common numeric shapes",
            "",
            "| Shape | Count |",
            "|---|---:|",
        ]
    )
    for shape, count in result["numeric_summary"]["common_shapes"][:20]:
        lines.append(f"| `{shape}` | {count} |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            result["claim_boundary"],
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = args.dataset_root.resolve(strict=True)
    require(root.is_dir(), f"dataset root is not a directory: {root}")
    dlo_roots = locate_dlo_roots(root)
    files: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    bytes_by_dlo: dict[str, int] = {}
    for dlo, dlo_root in dlo_roots.items():
        paths = [
            path
            for path in sorted(dlo_root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ]
        counts[dlo] = len(paths)
        bytes_by_dlo[dlo] = sum(path.stat().st_size for path in paths)
        files.extend(inspect_file(path, root, dlo) for path in paths)
    suffix_counts = Counter(record["suffix"] for record in files)
    numeric_summary = summarize_numeric(files)
    adapters = adapter_interfaces(args.repository_root.resolve(strict=True))
    expected_count = args.expected_files_per_dlo
    exact_counts = all(counts[dlo] == expected_count for dlo in ("DLO4", "DLO5"))
    readable = numeric_summary["readable_numeric_file_count"] > 0
    decision = (
        "complete-dlo45-adapter-audit-ready"
        if exact_counts and readable
        else "dlo45-audit-needs-contract-resolution"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "repository_revision": args.repository_revision,
        "dataset_root": str(root),
        "dlo_roots": {key: str(value) for key, value in dlo_roots.items()},
        "files": files,
        "numeric_summary": numeric_summary,
        "adapter_interfaces": adapters,
        "summary": {
            "DLO4_file_count": counts["DLO4"],
            "DLO5_file_count": counts["DLO5"],
            "DLO4_bytes": bytes_by_dlo["DLO4"],
            "DLO5_bytes": bytes_by_dlo["DLO5"],
            "total_bytes": sum(bytes_by_dlo.values()),
            "exact_70_file_contract": exact_counts,
            "suffix_counts": sorted(suffix_counts.items()),
            "decision": decision,
        },
        "information_boundary": {
            "arbitrary_pickle_loaded": False,
            "model_fitted": False,
            "target_pair_selected": False,
            "score_computed": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "This audit establishes release completeness, safe format readability, "
            "and adapter-interface availability only. It is not a model result, a "
            "transport result, a calibration result, or paper-level evidence."
        ),
    }
    result["audit_id"] = content_id(result)
    write_outputs(args.output, result)
    print(json.dumps({"audit_id": result["audit_id"], **result["summary"]}, sort_keys=True))
    return 0 if decision == "complete-dlo45-adapter-audit-ready" else 3


def self_test() -> None:
    with __import__("tempfile").TemporaryDirectory() as temporary:
        root = Path(temporary) / "data_set"
        repository = Path(temporary) / "repository"
        repository.mkdir()
        (repository / "deform_adapter.py").write_text(
            "import argparse\n"
            "def main():\n"
            "    p = argparse.ArgumentParser()\n"
            "    p.add_argument('--dataset-root')\n",
            encoding="utf-8",
        )
        for dlo in ("DLO4", "DLO5"):
            directory = root / dlo
            directory.mkdir(parents=True)
            for index in range(70):
                (directory / f"trajectory_{index:03d}.txt").write_text(
                    "0 1 2\n3 4 5\n", encoding="utf-8"
                )
        args = argparse.Namespace(
            dataset_root=root,
            repository_root=repository,
            repository_revision="1" * 40,
            output=Path(temporary) / "output",
            expected_files_per_dlo=70,
        )
        code = run(args)
        require(code == 0, "fixture audit failed")
        value = json.loads((args.output / "audit.json").read_text(encoding="utf-8"))
        require(value["summary"]["exact_70_file_contract"] is True, "count gate changed")
        require(value["audit_id"] == content_id(value), "content ID changed")
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--repository-revision", default="0" * 40)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-files-per-dlo", type=int, default=70)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    require(args.dataset_root is not None, "--dataset-root is required")
    require(args.output is not None, "--output is required")
    require(
        len(args.repository_revision) == 40
        and all(character in "0123456789abcdef" for character in args.repository_revision),
        "repository revision must be a full lowercase SHA",
    )
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"DEFORM DLO4/DLO5 audit failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
