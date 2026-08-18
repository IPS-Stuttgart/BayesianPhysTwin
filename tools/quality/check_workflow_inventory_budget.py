#!/usr/bin/env python3
"""Ratchet the checked-in GitHub Actions inventory without rewriting history."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
_DEFAULT_CONTRACT: Final = Path(".github/quality/workflow-inventory-budget-v1.json")
_SCHEMA: Final = "bayesian-phystwin.workflow-inventory-budget"
_SCHEMA_VERSION: Final = 1
_WORKFLOW_DIRECTORY: Final = PurePosixPath(".github/workflows")
_WORKFLOW_SUFFIXES: Final = frozenset({".yml", ".yaml"})
_SHA: Final = re.compile(r"^[0-9a-f]{40}$")
_TEMPORARY_NAME_PATTERNS: Final = (
    re.compile(r"^_"),
    re.compile(r"(?:^|[-_])one[-_]?shot(?:[-_]|$)"),
    re.compile(r"(?:^|[-_])once(?:[-_]|$)"),
    re.compile(r"^(?:diagnose|format|fix|patch|rerun)(?:[-_]|$)"),
)
_REQUIRED_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "baseline_revision",
        "maximum_checked_in_workflows",
        "temporary_looking_workflow_allowlist",
        "retirement_target_maximum_checked_in_workflows",
        "retirement_target_maximum_temporary_looking_workflows",
    }
)


class WorkflowInventoryBudgetError(ValueError):
    """Raised when the inventory contract or repository violates the ratchet."""


def _pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowInventoryBudgetError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise WorkflowInventoryBudgetError(f"non-finite JSON constant: {value}")


def _positive_int(value: object, *, name: str, allow_zero: bool = False) -> int:
    lower_bound = 0 if allow_zero else 1
    if type(value) is not int or value < lower_bound:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise WorkflowInventoryBudgetError(f"{name} must be a {qualifier} integer")
    return value


def _canonical_workflow_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise WorkflowInventoryBudgetError(f"{name} must be a nonempty string")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or ".." in pure.parts
        or len(pure.parts) != 3
        or tuple(pure.parts[:2]) != tuple(_WORKFLOW_DIRECTORY.parts)
        or pure.suffix.lower() not in _WORKFLOW_SUFFIXES
    ):
        raise WorkflowInventoryBudgetError(
            f"{name} is not a canonical checked-in workflow path: {value!r}"
        )
    return value


def _temporary_looking(path: str) -> bool:
    stem = PurePosixPath(path).stem.lower()
    return any(pattern.search(stem) for pattern in _TEMPORARY_NAME_PATTERNS)


def load_contract(path: Path) -> dict[str, object]:
    """Load and validate the immutable inventory-budget schema."""

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(
                stream,
                object_pairs_hook=_pairs_hook,
                parse_constant=_reject_constant,
            )
    except WorkflowInventoryBudgetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkflowInventoryBudgetError(
            f"cannot load workflow inventory contract: {path}"
        ) from error

    if type(payload) is not dict:
        raise WorkflowInventoryBudgetError(
            "workflow inventory contract must be an object"
        )
    fields = frozenset(payload)
    if fields != _REQUIRED_FIELDS:
        raise WorkflowInventoryBudgetError(
            "workflow inventory contract fields changed: "
            f"missing={sorted(_REQUIRED_FIELDS - fields)}, "
            f"extra={sorted(fields - _REQUIRED_FIELDS)}"
        )
    if payload["schema"] != _SCHEMA:
        raise WorkflowInventoryBudgetError("workflow inventory contract schema changed")
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise WorkflowInventoryBudgetError(
            "workflow inventory contract schema version changed"
        )

    baseline = payload["baseline_revision"]
    if type(baseline) is not str or _SHA.fullmatch(baseline) is None:
        raise WorkflowInventoryBudgetError(
            "baseline_revision must be a lowercase 40-character Git SHA"
        )

    maximum = _positive_int(
        payload["maximum_checked_in_workflows"],
        name="maximum_checked_in_workflows",
    )
    target = _positive_int(
        payload["retirement_target_maximum_checked_in_workflows"],
        name="retirement_target_maximum_checked_in_workflows",
    )
    target_temporary = _positive_int(
        payload["retirement_target_maximum_temporary_looking_workflows"],
        name="retirement_target_maximum_temporary_looking_workflows",
        allow_zero=True,
    )
    if target > maximum:
        raise WorkflowInventoryBudgetError(
            "retirement workflow target may not exceed the current ceiling"
        )

    raw_allowlist = payload["temporary_looking_workflow_allowlist"]
    if type(raw_allowlist) is not list:
        raise WorkflowInventoryBudgetError(
            "temporary_looking_workflow_allowlist must be an array"
        )
    allowlist = tuple(
        _canonical_workflow_path(value, name=f"allowlist[{index}]")
        for index, value in enumerate(raw_allowlist)
    )
    if tuple(sorted(allowlist)) != allowlist:
        raise WorkflowInventoryBudgetError(
            "temporary-looking workflow allowlist must be sorted"
        )
    if len(set(allowlist)) != len(allowlist):
        raise WorkflowInventoryBudgetError(
            "temporary-looking workflow allowlist contains duplicates"
        )
    not_temporary = [path for path in allowlist if not _temporary_looking(path)]
    if not_temporary:
        raise WorkflowInventoryBudgetError(
            "allowlist contains non-temporary-looking workflow paths: "
            + ", ".join(not_temporary)
        )
    if target_temporary > len(allowlist):
        raise WorkflowInventoryBudgetError(
            "retirement temporary-looking target exceeds the current allowlist"
        )
    return payload


def workflow_paths(root: Path) -> tuple[str, ...]:
    """Return every ordinary checked-in workflow path in canonical order."""

    directory = root / Path(*_WORKFLOW_DIRECTORY.parts)
    if not directory.is_dir():
        raise WorkflowInventoryBudgetError(
            f"workflow directory is missing: {_WORKFLOW_DIRECTORY.as_posix()}"
        )
    root_resolved = root.resolve(strict=True)
    paths: list[str] = []
    for source in sorted(directory.iterdir(), key=lambda path: path.name):
        if source.suffix.lower() not in _WORKFLOW_SUFFIXES:
            continue
        relative = source.relative_to(root).as_posix()
        _canonical_workflow_path(relative, name="workflow path")
        if source.is_symlink():
            raise WorkflowInventoryBudgetError(
                f"workflow path must not be a symlink: {relative}"
            )
        try:
            resolved = source.resolve(strict=True)
        except OSError as error:
            raise WorkflowInventoryBudgetError(
                f"workflow path is not readable: {relative}"
            ) from error
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise WorkflowInventoryBudgetError(
                f"workflow path escapes the repository: {relative}"
            ) from error
        if not source.is_file():
            raise WorkflowInventoryBudgetError(
                f"workflow path must be an ordinary file: {relative}"
            )
        paths.append(relative)
    return tuple(paths)


def validate_repository(
    root: Path,
    contract_path: Path = _DEFAULT_CONTRACT,
) -> dict[str, object]:
    """Validate the exact inventory count and temporary-looking path roster."""

    root = root.resolve(strict=True)
    contract_source = contract_path
    if not contract_source.is_absolute():
        contract_source = root / contract_source
    contract = load_contract(contract_source)
    paths = workflow_paths(root)
    temporary = tuple(path for path in paths if _temporary_looking(path))
    expected_temporary = tuple(contract["temporary_looking_workflow_allowlist"])
    maximum = int(contract["maximum_checked_in_workflows"])

    if len(paths) > maximum:
        raise WorkflowInventoryBudgetError(
            f"checked-in workflow count grew from {maximum} to {len(paths)}"
        )
    if len(paths) < maximum:
        raise WorkflowInventoryBudgetError(
            "workflow inventory shrank without ratcheting the checked-in ceiling: "
            f"contract={maximum}, repository={len(paths)}"
        )
    if temporary != expected_temporary:
        expected = set(expected_temporary)
        actual = set(temporary)
        raise WorkflowInventoryBudgetError(
            "temporary-looking workflow roster changed without an explicit ratchet: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    target = int(contract["retirement_target_maximum_checked_in_workflows"])
    target_temporary = int(
        contract["retirement_target_maximum_temporary_looking_workflows"]
    )
    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "baseline_revision": contract["baseline_revision"],
        "checked_in_workflow_count": len(paths),
        "temporary_looking_workflow_count": len(temporary),
        "temporary_looking_workflows": list(temporary),
        "retirement_target_maximum_checked_in_workflows": target,
        "retirement_target_maximum_temporary_looking_workflows": target_temporary,
        "workflow_retirement_gap": max(0, len(paths) - target),
        "temporary_looking_retirement_gap": max(0, len(temporary) - target_temporary),
        "status": "within-ratcheted-budget",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(_REPOSITORY_ROOT))
    parser.add_argument("--contract", default=str(_DEFAULT_CONTRACT))
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = validate_repository(
            Path(arguments.repository_root),
            Path(arguments.contract),
        )
    except WorkflowInventoryBudgetError as error:
        print(f"workflow inventory budget failed: {error}")
        return 2

    if arguments.as_json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(
            "workflow inventory budget matched: "
            f"workflows={report['checked_in_workflow_count']} "
            f"temporary-looking={report['temporary_looking_workflow_count']} "
            f"target={report['retirement_target_maximum_checked_in_workflows']}/"
            f"{report['retirement_target_maximum_temporary_looking_workflows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
