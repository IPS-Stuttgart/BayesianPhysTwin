"""Require line and branch coverage for changed scientific-core code."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

_ZERO_SHA = "0" * 40
_HUNK_HEADER = re.compile(
    r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@"
)

SEMANTIC_PATHS = frozenset(
    {
        "src/bayesian_phystwin/_gauge_aware_contracts.py",
        "src/bayesian_phystwin/_gauge_aware_solver.py",
        "src/bayesian_phystwin/causal4d_provider_v1.py",
        "src/bayesian_phystwin/gauge_aware_belief.py",
        "src/bayesian_phystwin/observation_belief.py",
        "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
        "src/bayesian_phystwin/repository_provenance.py",
        "src/bayesian_phystwin/run_manifest.py",
        "src/bayesian_phystwin/run_manifest_v2.py",
        "src/bayesian_phystwin/cli/main.py",
        "src/bayesian_phystwin/cli/run_manifest.py",
    }
)


class CoverageFile(TypedDict, total=False):
    executed_lines: list[int]
    missing_lines: list[int]
    excluded_lines: list[int]
    executed_branches: list[list[int]]
    missing_branches: list[list[int]]


class CoveragePayload(TypedDict):
    files: dict[str, CoverageFile]


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository_root), *arguments],
        text=True,
    )


def _commit_exists(repository_root: Path, revision: str) -> bool:
    if not revision or revision == _ZERO_SHA:
        return False
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "cat-file",
            "-e",
            f"{revision}^{{commit}}",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _comparison_base(
    repository_root: Path,
    *,
    base: str,
    head: str,
) -> str | None:
    if _commit_exists(repository_root, base):
        merge_base = _git_output(
            repository_root,
            "merge-base",
            base,
            head,
        ).strip()
        if merge_base:
            return merge_base
    parent = f"{head}^"
    return parent if _commit_exists(repository_root, parent) else None


def parse_added_lines(diff: str) -> frozenset[int]:
    """Return head-side line numbers introduced or modified by a zero-context diff."""

    lines: set[int] = set()
    for line in diff.splitlines():
        match = _HUNK_HEADER.match(line)
        if match is None:
            continue
        start = int(match.group("start"))
        count_text = match.group("count")
        count = 1 if count_text is None else int(count_text)
        lines.update(range(start, start + count))
    return frozenset(lines)


def _changed_paths(
    repository_root: Path,
    *,
    base: str,
    head: str,
) -> tuple[str, ...]:
    resolved_base = _comparison_base(repository_root, base=base, head=head)
    if resolved_base is None:
        output = _git_output(repository_root, "ls-files")
    else:
        output = _git_output(
            repository_root,
            "diff",
            "--name-only",
            "--diff-filter=ACMRT",
            resolved_base,
            head,
        )
    return tuple(
        sorted(path for path in output.splitlines() if path in SEMANTIC_PATHS)
    )


def changed_semantic_lines(
    repository_root: Path,
    *,
    base: str = "",
    head: str = "HEAD",
) -> dict[str, frozenset[int]]:
    """Return changed head-side lines for every scientific-core path."""

    resolved_root = repository_root.resolve()
    resolved_base = _comparison_base(resolved_root, base=base, head=head)
    changed: dict[str, frozenset[int]] = {}
    for path in _changed_paths(resolved_root, base=base, head=head):
        if resolved_base is None:
            line_count = len((resolved_root / path).read_text().splitlines())
            changed[path] = frozenset(range(1, line_count + 1))
            continue
        diff = _git_output(
            resolved_root,
            "diff",
            "--unified=0",
            "--no-color",
            resolved_base,
            head,
            "--",
            path,
        )
        changed[path] = parse_added_lines(diff)
    return changed


def _relative_coverage_files(
    repository_root: Path,
    files: Mapping[str, CoverageFile],
) -> dict[str, CoverageFile]:
    resolved_root = repository_root.resolve()
    result: dict[str, CoverageFile] = {}
    for raw_path, coverage in files.items():
        candidate = Path(raw_path)
        if candidate.is_absolute():
            try:
                relative = candidate.resolve().relative_to(resolved_root)
            except ValueError:
                continue
        else:
            relative = candidate
        result[relative.as_posix()] = coverage
    return result


def coverage_failures(
    changed_lines: frozenset[int],
    coverage: CoverageFile,
) -> tuple[str, ...]:
    """Describe uncovered changed statements and branch origins for one file."""

    missing_lines = set(coverage.get("missing_lines", []))
    excluded_lines = set(coverage.get("excluded_lines", []))
    executable_lines = set(coverage.get("executed_lines", [])) | missing_lines
    relevant_lines = (set(changed_lines) & executable_lines) - excluded_lines
    uncovered_lines = sorted(relevant_lines & missing_lines)

    uncovered_branches = sorted(
        (int(branch[0]), int(branch[1]))
        for branch in coverage.get("missing_branches", [])
        if len(branch) == 2 and int(branch[0]) in changed_lines
    )

    failures: list[str] = []
    if uncovered_lines:
        failures.append(f"uncovered changed lines: {uncovered_lines}")
    if uncovered_branches:
        failures.append(f"uncovered changed branches: {uncovered_branches}")
    return tuple(failures)


def check_changed_semantic_coverage(
    repository_root: Path,
    coverage_json: Path,
    *,
    base: str = "",
    head: str = "HEAD",
) -> dict[str, frozenset[int]]:
    """Fail when changed semantic statements or branch origins lack coverage."""

    resolved_root = repository_root.resolve()
    payload = cast(
        CoveragePayload,
        json.loads(coverage_json.read_text()),
    )
    files = _relative_coverage_files(resolved_root, payload.get("files", {}))
    changed = changed_semantic_lines(resolved_root, base=base, head=head)

    failures: list[str] = []
    for path, lines in changed.items():
        if not lines:
            continue
        coverage = files.get(path)
        if coverage is None:
            failures.append(f"{path}: no coverage data")
            continue
        for failure in coverage_failures(lines, coverage):
            failures.append(f"{path}: {failure}")

    if failures:
        formatted = "\n".join(f"  - {failure}" for failure in failures)
        raise SystemExit(
            "Changed scientific-core code lacks complete diff coverage:\n"
            f"{formatted}"
        )

    if changed:
        print("Changed scientific-core lines have complete line and branch coverage:")
        for path, lines in changed.items():
            print(f"  {path}: {len(lines)} changed lines")
    else:
        print(
            "No scientific-core source files changed; "
            "diff coverage gate is inactive."
        )
    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Git worktree whose semantic diff should be checked",
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        required=True,
        help="coverage.py JSON report generated with branch coverage enabled",
    )
    parser.add_argument(
        "--base",
        default="",
        help="base revision; falls back to the head parent when unavailable",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="head revision whose semantic diff should be checked",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check_changed_semantic_coverage(
        args.repository_root,
        args.coverage_json,
        base=args.base,
        head=args.head,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
