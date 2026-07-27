#!/usr/bin/env python3
"""Enforce stable-core and changed-code coverage ratchets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_TOLERANCE_PERCENTAGE_POINTS = 1e-6
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _percentage(summary: Mapping[str, Any], covered: str, total: str) -> float:
    total_count = int(summary[total])
    if total_count == 0:
        return 100.0
    return 100.0 * int(summary[covered]) / total_count


def _metrics(summary: Mapping[str, Any]) -> tuple[float, float]:
    return (
        _percentage(summary, "covered_lines", "num_statements"),
        _percentage(summary, "covered_branches", "num_branches"),
    )


def _require_floor(
    *,
    label: str,
    actual_lines: float,
    actual_branches: float,
    floor: Mapping[str, Any],
    failures: list[str],
) -> None:
    minimum_lines = float(floor["line_percent"])
    minimum_branches = float(floor["branch_percent"])
    print(
        f"{label}: lines={actual_lines:.2f}% (floor {minimum_lines:.2f}%), "
        f"branches={actual_branches:.2f}% (floor {minimum_branches:.2f}%)"
    )
    if actual_lines + _TOLERANCE_PERCENTAGE_POINTS < minimum_lines:
        failures.append(
            f"{label} line coverage {actual_lines:.4f}% is below "
            f"{minimum_lines:.4f}%"
        )
    if actual_branches + _TOLERANCE_PERCENTAGE_POINTS < minimum_branches:
        failures.append(
            f"{label} branch coverage {actual_branches:.4f}% is below "
            f"{minimum_branches:.4f}%"
        )


def _changed_lines(base: str, head: str) -> dict[str, set[int]]:
    completed = subprocess.run(
        (
            "git",
            "diff",
            "--unified=0",
            "--no-color",
            "--no-ext-diff",
            "--diff-filter=ACMR",
            f"{base}...{head}",
            "--",
            ":(glob)src/bayesian_phystwin/**/*.py",
        ),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    result: dict[str, set[int]] = defaultdict(set)
    path: str | None = None
    new_line = 0
    in_hunk = False
    for line in completed.stdout.splitlines():
        if line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/")
            in_hunk = False
            continue
        match = _HUNK_HEADER.match(line)
        if match:
            new_line = int(match.group(1))
            in_hunk = True
            continue
        if not in_hunk or path is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            result[path].add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        else:
            new_line += 1
    return dict(result)


def _require_changed_code_coverage(
    coverage: Mapping[str, Any],
    *,
    base: str,
    head: str,
    minimum_line_percent: float,
    minimum_branch_percent: float,
    failures: list[str],
) -> None:
    changed = _changed_lines(base, head)
    measured_files = coverage["files"]
    for path, added_lines in sorted(changed.items()):
        file_coverage = measured_files.get(path)
        if file_coverage is None:
            failures.append(f"changed package module is absent from coverage: {path}")
            continue
        executable = set(file_coverage["executed_lines"]) | set(
            file_coverage["missing_lines"]
        )
        changed_executable = added_lines & executable
        if changed_executable:
            covered_lines = changed_executable & set(file_coverage["executed_lines"])
            line_percent = 100.0 * len(covered_lines) / len(changed_executable)
            print(
                f"CHANGED {path}: executable lines={line_percent:.2f}% "
                f"(floor {minimum_line_percent:.2f}%)"
            )
            if line_percent + _TOLERANCE_PERCENTAGE_POINTS < minimum_line_percent:
                missing = sorted(changed_executable - covered_lines)
                failures.append(
                    f"{path} changed-line coverage {line_percent:.2f}% is below "
                    f"{minimum_line_percent:.2f}%; missing lines={missing}"
                )

        executed_branches = {
            tuple(map(int, branch))
            for branch in file_coverage.get("executed_branches", [])
        }
        missing_branches = {
            tuple(map(int, branch))
            for branch in file_coverage.get("missing_branches", [])
        }
        changed_branches = {
            branch
            for branch in executed_branches | missing_branches
            if branch[0] in added_lines
        }
        if changed_branches:
            covered_branches = changed_branches & executed_branches
            branch_percent = 100.0 * len(covered_branches) / len(changed_branches)
            print(
                f"CHANGED {path}: branches={branch_percent:.2f}% "
                f"(floor {minimum_branch_percent:.2f}%)"
            )
            if (
                branch_percent + _TOLERANCE_PERCENTAGE_POINTS
                < minimum_branch_percent
            ):
                missing = sorted(changed_branches - covered_branches)
                failures.append(
                    f"{path} changed-branch coverage {branch_percent:.2f}% is "
                    f"below {minimum_branch_percent:.2f}%; missing arcs={missing}"
                )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare coverage.py JSON output with committed floors."
    )
    parser.add_argument("coverage_json", type=Path)
    parser.add_argument("floors_json", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument(
        "--changed-coverage-json",
        type=Path,
        help="optional package-wide coverage JSON for changed-line enforcement",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    coverage = json.loads(arguments.coverage_json.read_text(encoding="utf-8"))
    floors = json.loads(arguments.floors_json.read_text(encoding="utf-8"))
    if int(floors.get("schema_version", -1)) != 1:
        raise SystemExit("unsupported stable-core coverage-floor schema")

    failures: list[str] = []
    total_lines, total_branches = _metrics(coverage["totals"])
    _require_floor(
        label="TOTAL",
        actual_lines=total_lines,
        actual_branches=total_branches,
        floor=floors["minimum_total"],
        failures=failures,
    )

    measured_files = coverage["files"]
    for path, floor in sorted(floors["files"].items()):
        if path not in measured_files:
            failures.append(
                f"required stable-core file is absent from coverage: {path}"
            )
            continue
        line_percent, branch_percent = _metrics(measured_files[path]["summary"])
        _require_floor(
            label=path,
            actual_lines=line_percent,
            actual_branches=branch_percent,
            floor=floor,
            failures=failures,
        )

    changed_floor = floors.get("changed_code")
    if arguments.base and changed_floor:
        changed_coverage = coverage
        if arguments.changed_coverage_json is not None:
            changed_coverage = json.loads(
                arguments.changed_coverage_json.read_text(encoding="utf-8")
            )
        _require_changed_code_coverage(
            changed_coverage,
            base=arguments.base,
            head=arguments.head,
            minimum_line_percent=float(changed_floor["line_percent"]),
            minimum_branch_percent=float(changed_floor["branch_percent"]),
            failures=failures,
        )

    if failures:
        print("\nCoverage ratchet failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
