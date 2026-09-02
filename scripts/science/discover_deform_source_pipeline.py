#!/usr/bin/env python3
"""Discover reusable DEFORM DLO2/DLO3 source-pipeline entry points.

This is a static repository analysis. It does not open any dataset payload. It
extracts program entry points, CLI arguments, imports, artifact names, and
residual/physics symbols from the checked-out source so the hierarchical
missing-physics adapter can reuse the existing evaluation implementation rather
than creating a second simulator or metric path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ALLOWED_SUFFIXES = {".py", ".yml", ".yaml", ".json", ".toml", ".md"}
IGNORE_PARTS = {".git", ".venv", "node_modules", "__pycache__", "evidence"}
TERM_WEIGHTS = {
    "dlo2": 6,
    "dlo3": 6,
    "deform": 3,
    "physical": 3,
    "residual": 5,
    "coefficient": 4,
    "no_refit": 7,
    "no-refit": 7,
    "pyelastica": 8,
    "backend": 4,
    "trajectory": 2,
    "source_result": 4,
    "prediction": 2,
}
FORBIDDEN_TARGET_PATTERNS = ("dlo4", "dlo5")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe_read(path: Path) -> str:
    if path.stat().st_size > 2_000_000:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def source_score(relative: str, text: str) -> tuple[int, dict[str, int]]:
    haystack = (relative + "\n" + text).lower()
    hits = {term: haystack.count(term) for term in TERM_WEIGHTS}
    score = sum(TERM_WEIGHTS[term] * min(count, 20) for term, count in hits.items())
    return score, {term: count for term, count in hits.items() if count}


def python_details(path: Path, text: str) -> dict[str, Any]:
    details: dict[str, Any] = {
        "functions": [],
        "classes": [],
        "imports": [],
        "argparse_options": [],
        "path_literals": [],
        "artifact_literals": [],
        "symbols": [],
    }
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        details["syntax_error"] = str(error)
        return details

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            details["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            details["classes"].append(node.name)
        elif isinstance(node, ast.Import):
            details["imports"].extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            details["imports"].append(module)
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name == "add_argument":
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        if argument.value.startswith("-"):
                            details["argparse_options"].append(argument.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            lower = value.lower()
            if any(token in lower for token in (".json", ".npz", ".npy", ".pt", ".pth", ".csv")):
                details["artifact_literals"].append(value[:300])
            if "/" in value and any(token in lower for token in ("dlo", "deform", "cache", "result")):
                details["path_literals"].append(value[:300])

    identifier_pattern = re.compile(
        r"\b(?:physical|residual|coefficient|prediction|trajectory|state|force|position|velocity|curvature|strain)[A-Za-z0-9_]*\b",
        re.IGNORECASE,
    )
    details["symbols"] = sorted(set(identifier_pattern.findall(text)))[:250]
    for key in (
        "functions",
        "classes",
        "imports",
        "argparse_options",
        "path_literals",
        "artifact_literals",
    ):
        details[key] = sorted(set(details[key]))[:300]
    return details


def workflow_details(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    return {
        "runner_lines": [line.strip() for line in lines if "runs-on:" in line][:20],
        "python_invocations": [
            line.strip()
            for line in lines
            if re.search(r"(?:python3?|uv run|pytest).*\.(?:py|toml)", line)
        ][:100],
        "artifact_lines": [
            line.strip()
            for line in lines
            if any(term in line.lower() for term in ("upload-artifact", "download-artifact", "artifact", "result.json", "progress.json"))
        ][:150],
        "environment_paths": [
            line.strip()
            for line in lines
            if any(term in line.lower() for term in ("dataset_root", "cache_root", "dlo2", "dlo3", "pyelastica"))
        ][:150],
    }


def discover(repository: Path) -> dict[str, Any]:
    root = repository.resolve(strict=True)
    candidates: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        relative_path = path.relative_to(root)
        if any(part in IGNORE_PARTS for part in relative_path.parts):
            continue
        relative = relative_path.as_posix()
        text = safe_read(path)
        if not text:
            continue
        score, hits = source_score(relative, text)
        if score < 8:
            continue
        lower = (relative + "\n" + text).lower()
        target_mentions = {
            term: lower.count(term) for term in FORBIDDEN_TARGET_PATTERNS if term in lower
        }
        record: dict[str, Any] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "score": score,
            "term_hits": hits,
            "mentions_protected_targets": target_mentions,
            "is_workflow": path.suffix.lower() in {".yml", ".yaml"} and ".github/workflows" in relative,
        }
        if path.suffix.lower() == ".py":
            record["python"] = python_details(path, text)
        if record["is_workflow"]:
            record["workflow"] = workflow_details(text)
        candidates.append(record)

    candidates.sort(key=lambda value: (-value["score"], value["path"]))
    source_safe = [
        candidate
        for candidate in candidates
        if not candidate["mentions_protected_targets"]
        and (
            "dlo2" in candidate["term_hits"]
            or "dlo3" in candidate["term_hits"]
            or "pyelastica" in candidate["term_hits"]
        )
    ]
    adapter_recommendations = []
    for candidate in candidates:
        details = candidate.get("python", {})
        symbols = " ".join(details.get("symbols", [])).lower()
        artifacts = " ".join(details.get("artifact_literals", [])).lower()
        score = 0
        reasons = []
        if "residual" in symbols:
            score += 4
            reasons.append("residual symbols")
        if "physical" in symbols:
            score += 3
            reasons.append("physical-model symbols")
        if "trajectory" in symbols:
            score += 2
            reasons.append("trajectory symbols")
        if any(suffix in artifacts for suffix in (".npz", ".npy", "result.json", "progress.json")):
            score += 3
            reasons.append("structured artifact output")
        if details.get("argparse_options"):
            score += 1
            reasons.append("CLI entry point")
        if "pyelastica" in candidate["term_hits"]:
            score += 5
            reasons.append("alternate backend")
        if score:
            adapter_recommendations.append(
                {
                    "path": candidate["path"],
                    "adapter_score": score,
                    "reasons": reasons,
                    "argparse_options": details.get("argparse_options", []),
                    "functions": details.get("functions", []),
                    "artifact_literals": details.get("artifact_literals", []),
                    "mentions_protected_targets": candidate["mentions_protected_targets"],
                }
            )
    adapter_recommendations.sort(key=lambda value: (-value["adapter_score"], value["path"]))

    result = {
        "schema": "bayesian-phystwin.deform-source-pipeline-discovery",
        "schema_version": 1,
        "repository": root.as_posix(),
        "candidate_count": len(candidates),
        "source_safe_candidate_count": len(source_safe),
        "candidates": candidates[:300],
        "source_safe_candidates": source_safe[:100],
        "adapter_recommendations": adapter_recommendations[:100],
        "information_boundary": {
            "dataset_payload_read": False,
            "dlo4_payload_read": False,
            "dlo5_payload_read": False,
            "target_result_read": False,
            "static_repository_analysis_only": True,
        },
    }
    result["discovery_id"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = discover(arguments.repository)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "discovery_id": result["discovery_id"],
                "candidate_count": result["candidate_count"],
                "source_safe_candidate_count": result["source_safe_candidate_count"],
                "top_adapter_recommendations": result["adapter_recommendations"][:15],
                "information_boundary": result["information_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
