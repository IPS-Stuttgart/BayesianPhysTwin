#!/usr/bin/env python3
"""Ratchet GitHub Actions workflow lifecycle and supply-chain policy.

The repository contains historical one-shot workflows that can remain necessary
for frozen evidence. This checker therefore does not fail merely because an
untouched legacy workflow exists. It does fail when a pull request adds, copies,
renames, or modifies a workflow without explicit lifecycle metadata, adds
another temporary-looking permanent workflow, or weakens the minimum
permissions and action-pinning contract.

Managed temporary workflows are checked on every invocation and become failures
after their declared expiry date. The inventory output is read-only operational
metadata; it is not scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_DIRECTORY = PurePosixPath(".github/workflows")
_WORKFLOW_SUFFIXES = frozenset({".yml", ".yaml"})
_SCHEMA = "bayesian-phystwin.workflow-lifecycle-inventory"
_SCHEMA_VERSION = 1

_WORKFLOW_METADATA_PATTERN = re.compile(
    r"^#\s*workflow-([a-z0-9-]+):\s*(.*?)\s*$",
    re.IGNORECASE,
)
_ALLOWED_METADATA_KEYS = frozenset({"lifecycle", "owner", "issue", "expiry"})
_ALLOWED_LIFECYCLES = frozenset({"permanent", "temporary"})

_TEMPORARY_NAME_PATTERNS = (
    re.compile(r"^_"),
    re.compile(r"(?:^|[-_])one[-_]?shot(?:[-_]|$)"),
    re.compile(r"(?:^|[-_])once(?:[-_]|$)"),
    re.compile(r"^(?:diagnose|format|fix|patch|rerun)(?:[-_]|$)"),
)

_EXTERNAL_ACTION_PATTERN = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
_FULL_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_TOP_LEVEL_PERMISSIONS = re.compile(r"^permissions:(?:\s|$)", re.MULTILINE)
_TOP_LEVEL_CONCURRENCY = re.compile(r"^concurrency:(?:\s|$)", re.MULTILINE)
_FORBIDDEN_TEMPORARY_EVENT = re.compile(
    r"^\s{2}(?:push|pull_request|pull_request_target|schedule|repository_dispatch):",
    re.MULTILINE,
)
_WORKFLOW_DISPATCH_EVENT = re.compile(r"^\s{2}workflow_dispatch:", re.MULTILINE)
_PULL_REQUEST_TARGET_EVENT = re.compile(r"^\s{2}pull_request_target:", re.MULTILINE)


@dataclass(frozen=True)
class WorkflowRecord:
    """One workflow's operational lifecycle classification."""

    path: str
    lifecycle: str
    owner: str | None
    issue: str | None
    expiry: str | None
    temporary_looking_name: bool
    violations: tuple[str, ...]


def _metadata_header(text: str) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return canonical leading workflow metadata and header violations."""

    metadata: dict[str, str] = {}
    violations: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        match = _WORKFLOW_METADATA_PATTERN.fullmatch(line)
        if match is None:
            continue
        key = match.group(1).lower()
        value = match.group(2).strip()
        if key not in _ALLOWED_METADATA_KEYS:
            violations.append(
                f"line {line_number}: unknown workflow metadata key 'workflow-{key}'"
            )
        elif key in metadata:
            violations.append(
                f"line {line_number}: duplicate workflow metadata key 'workflow-{key}'"
            )
        else:
            metadata[key] = value
    return metadata, tuple(violations)


def _temporary_looking_name(path: Path) -> bool:
    stem = path.stem.lower()
    return any(pattern.search(stem) for pattern in _TEMPORARY_NAME_PATTERNS)


def _unpinned_external_actions(text: str) -> tuple[str, ...]:
    unpinned: list[str] = []
    for match in _EXTERNAL_ACTION_PATTERN.finditer(text):
        action = match.group(1)
        if action.startswith(("./", "docker://")):
            continue
        if "@" not in action:
            unpinned.append(action)
            continue
        _, revision = action.rsplit("@", 1)
        if not _FULL_COMMIT_SHA.fullmatch(revision):
            unpinned.append(action)
    return tuple(unpinned)


def _parse_expiry(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def inspect_workflow(
    path: Path,
    text: str,
    *,
    today: date | None = None,
    require_managed: bool = False,
) -> WorkflowRecord:
    """Classify and validate one workflow without mutating repository state."""

    current_date = today or date.today()
    metadata, metadata_violations = _metadata_header(text)
    lifecycle_value = metadata.get("lifecycle")
    owner = metadata.get("owner")
    issue = metadata.get("issue")
    expiry = metadata.get("expiry")
    temporary_name = _temporary_looking_name(path)
    violations = list(metadata_violations)

    if lifecycle_value is None:
        lifecycle = "legacy"
        if require_managed:
            violations.append(
                "added or modified workflows require "
                "'# workflow-lifecycle: permanent' or "
                "'# workflow-lifecycle: temporary'"
            )
    elif lifecycle_value.lower() not in _ALLOWED_LIFECYCLES:
        lifecycle = "legacy"
        violations.append("workflow-lifecycle must be 'permanent' or 'temporary'")
    else:
        lifecycle = lifecycle_value.lower()

    if lifecycle != "legacy" and owner is None:
        violations.append("managed workflows require '# workflow-owner: ...'")

    if lifecycle == "permanent":
        if temporary_name:
            violations.append(
                "permanent workflow has a temporary-looking filename; move the "
                "operation into a parameterized script or reusable workflow"
            )
        if issue is not None or expiry is not None:
            violations.append(
                "permanent workflows must not declare workflow-issue or "
                "workflow-expiry metadata"
            )
    elif lifecycle == "temporary":
        if issue is None:
            violations.append("temporary workflows require '# workflow-issue: ...'")
        parsed_expiry = _parse_expiry(expiry)
        if expiry is None:
            violations.append(
                "temporary workflows require '# workflow-expiry: YYYY-MM-DD'"
            )
        elif parsed_expiry is None:
            violations.append("workflow-expiry must be a valid ISO date")
        elif parsed_expiry < current_date:
            violations.append(
                f"temporary workflow expired on {parsed_expiry.isoformat()}"
            )
        if not _WORKFLOW_DISPATCH_EVENT.search(text):
            violations.append(
                "temporary workflows must expose workflow_dispatch and no "
                "automatic trigger"
            )
        if _FORBIDDEN_TEMPORARY_EVENT.search(text):
            violations.append(
                "temporary workflows may not use push, pull_request, schedule, "
                "repository_dispatch, or pull_request_target"
            )

    if require_managed or lifecycle != "legacy":
        if not _TOP_LEVEL_PERMISSIONS.search(text):
            violations.append("workflow must declare top-level permissions")
        if not _TOP_LEVEL_CONCURRENCY.search(text):
            violations.append("workflow must declare top-level concurrency")
        if _PULL_REQUEST_TARGET_EVENT.search(text):
            violations.append("pull_request_target is not permitted")
        unpinned = _unpinned_external_actions(text)
        if unpinned:
            violations.append(
                "external actions must be pinned to full commit SHAs: "
                + ", ".join(unpinned)
            )

    return WorkflowRecord(
        path=path.as_posix(),
        lifecycle=lifecycle,
        owner=owner,
        issue=issue,
        expiry=expiry,
        temporary_looking_name=temporary_name,
        violations=tuple(violations),
    )


def _workflow_paths(root: Path) -> tuple[Path, ...]:
    directory = root / Path(*_WORKFLOW_DIRECTORY.parts)
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            path.relative_to(root)
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _WORKFLOW_SUFFIXES
        )
    )


def build_inventory(
    root: Path, *, today: date | None = None
) -> tuple[WorkflowRecord, ...]:
    """Return a stable inventory of every checked-in workflow."""

    records: list[WorkflowRecord] = []
    for relative_path in _workflow_paths(root):
        text = (root / relative_path).read_text(encoding="utf-8")
        records.append(inspect_workflow(relative_path, text, today=today))
    return tuple(records)


def _commit_exists(root: Path, revision: str | None) -> bool:
    if not revision or set(revision) == {"0"}:
        return False
    completed = subprocess.run(
        ("git", "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"),
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _git_text(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_commit(root: Path, revision: str, *, name: str) -> str:
    if not revision or revision.strip() != revision:
        raise ValueError(f"{name} must be a nonempty canonical revision")
    if not _commit_exists(root, revision):
        raise ValueError(f"{name} is not an available commit: {revision}")
    return _git_text(root, "rev-parse", "--verify", f"{revision}^{{commit}}")


def _resolve_base(root: Path, base: str | None, head: str) -> str | None:
    if _commit_exists(root, base):
        assert base is not None
        return _resolve_commit(root, base, name="base revision")
    parent = f"{head}^"
    if _commit_exists(root, parent):
        return _resolve_commit(root, parent, name="head parent")
    return None


def _canonical_workflow_path(raw_path: bytes) -> Path:
    rendered = os.fsdecode(raw_path)
    if not rendered or "\\" in rendered:
        raise ValueError(f"git reported a noncanonical workflow path: {rendered!r}")
    pure_path = PurePosixPath(rendered)
    if (
        pure_path.is_absolute()
        or pure_path.as_posix() != rendered
        or ".." in pure_path.parts
        or len(pure_path.parts) != 3
        or tuple(pure_path.parts[:2]) != tuple(_WORKFLOW_DIRECTORY.parts)
        or pure_path.suffix.lower() not in _WORKFLOW_SUFFIXES
    ):
        raise ValueError(f"git reported a noncanonical workflow path: {rendered!r}")
    return Path(*pure_path.parts)


def _changed_workflows(root: Path, base: str | None, head: str) -> tuple[Path, ...]:
    """Return ordinary added, copied, modified, or renamed workflow files."""

    if base is None:
        return ()
    completed = subprocess.run(
        (
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            f"{base}...{head}",
            "--",
            ":(glob).github/workflows/*.yml",
            ":(glob).github/workflows/*.yaml",
        ),
        cwd=root,
        check=True,
        capture_output=True,
    )
    root_resolved = root.resolve(strict=True)
    paths: set[Path] = set()
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative_path = _canonical_workflow_path(raw_path)
        source = root / relative_path
        if source.is_symlink():
            raise ValueError(
                "changed workflow path must not be a symlink: "
                f"{relative_path.as_posix()}"
            )
        try:
            resolved = source.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"changed workflow path is not readable: {relative_path.as_posix()}"
            ) from error
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError(
                "changed workflow path escapes the repository: "
                f"{relative_path.as_posix()}"
            ) from error
        if not source.is_file():
            raise ValueError(
                "changed workflow path must be an ordinary file: "
                f"{relative_path.as_posix()}"
            )
        paths.add(relative_path)
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def validate_repository(
    root: Path,
    *,
    base: str | None,
    head: str,
    today: date | None = None,
) -> tuple[WorkflowRecord, ...]:
    """Validate changed workflows and every managed temporary workflow."""

    root = root.resolve(strict=True)
    resolved_head = _resolve_commit(root, head, name="head revision")
    checkout_head = _resolve_commit(root, "HEAD", name="repository HEAD")
    if checkout_head != resolved_head:
        raise ValueError(
            "repository HEAD does not match the requested head revision; "
            "refusing to inspect workflow bytes from a different tree"
        )
    resolved_base = _resolve_base(root, base, resolved_head)
    changed = set(_changed_workflows(root, resolved_base, resolved_head))
    records = build_inventory(root, today=today)
    relevant: list[WorkflowRecord] = []

    for record in records:
        path = Path(record.path)
        if path in changed:
            text = (root / path).read_text(encoding="utf-8")
            relevant.append(
                inspect_workflow(path, text, today=today, require_managed=True)
            )
        elif record.lifecycle == "temporary" and record.violations:
            relevant.append(record)

    return tuple(relevant)


def _inventory_payload(records: Iterable[WorkflowRecord]) -> dict[str, object]:
    record_list = list(records)
    lifecycle_counts = {
        lifecycle: sum(record.lifecycle == lifecycle for record in record_list)
        for lifecycle in ("permanent", "temporary", "legacy")
    }
    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_count": len(record_list),
        "lifecycle_counts": lifecycle_counts,
        "legacy_temporary_looking_count": sum(
            record.lifecycle == "legacy" and record.temporary_looking_name
            for record in record_list
        ),
        "managed_violation_count": sum(
            record.lifecycle != "legacy" and bool(record.violations)
            for record in record_list
        ),
        "workflows": [asdict(record) for record in record_list],
    }


def _inventory_markdown(records: Iterable[WorkflowRecord]) -> str:
    record_list = list(records)
    counts = {
        lifecycle: sum(record.lifecycle == lifecycle for record in record_list)
        for lifecycle in ("permanent", "temporary", "legacy")
    }
    legacy_temporary = [
        record
        for record in record_list
        if record.lifecycle == "legacy" and record.temporary_looking_name
    ]
    managed_violations = [
        record
        for record in record_list
        if record.lifecycle != "legacy" and record.violations
    ]

    lines = [
        "# Workflow lifecycle inventory",
        "",
        "This is a read-only operational inventory. It is not scientific evidence",
        "and does not authorize deletion of evidence-bound workflow files.",
        "",
        f"- Total workflows: **{len(record_list)}**",
        f"- Managed permanent: **{counts['permanent']}**",
        f"- Managed temporary: **{counts['temporary']}**",
        f"- Legacy/unclassified: **{counts['legacy']}**",
        f"- Legacy temporary-looking names: **{len(legacy_temporary)}**",
        f"- Managed workflows with violations: **{len(managed_violations)}**",
        "",
    ]

    if managed_violations:
        lines.extend(
            (
                "## Managed violations",
                "",
                "| Workflow | Violations |",
                "| --- | --- |",
            )
        )
        for record in managed_violations:
            lines.append(f"| `{record.path}` | {'; '.join(record.violations)} |")
        lines.append("")

    if legacy_temporary:
        lines.extend(
            (
                "## Legacy temporary-looking workflows",
                "",
                "The complete roster is in the JSON artifact. The table is capped at ",
                "100 paths to keep the Actions summary usable.",
                "",
                "| Workflow |",
                "| --- |",
            )
        )
        for record in legacy_temporary[:100]:
            lines.append(f"| `{record.path}` |")
        if len(legacy_temporary) > 100:
            lines.append(
                f"| … {len(legacy_temporary) - 100} additional paths in JSON |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ratchet workflow lifecycle metadata and action pinning."
    )
    parser.add_argument("--base", default=None, help="comparison commit")
    parser.add_argument("--head", default="HEAD", help="head commit to inspect")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=_REPOSITORY_ROOT,
        help="repository checkout root",
    )
    parser.add_argument(
        "--inventory-json",
        type=Path,
        default=None,
        help="write complete JSON inventory",
    )
    parser.add_argument(
        "--inventory-markdown",
        type=Path,
        default=None,
        help="write human-readable inventory summary",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="publish inventory without failing lifecycle policy",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    root = arguments.repository_root.resolve()
    records = build_inventory(root)

    if arguments.inventory_json is not None:
        payload = _inventory_payload(records)
        _write_text(
            arguments.inventory_json,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
    if arguments.inventory_markdown is not None:
        _write_text(arguments.inventory_markdown, _inventory_markdown(records))

    print(
        "Workflow lifecycle inventory: "
        f"{len(records)} total, "
        f"{sum(record.lifecycle == 'permanent' for record in records)} "
        "managed permanent, "
        f"{sum(record.lifecycle == 'temporary' for record in records)} "
        "managed temporary, "
        f"{sum(record.lifecycle == 'legacy' for record in records)} legacy.",
        flush=True,
    )

    if arguments.inventory_only:
        return 0

    try:
        relevant = validate_repository(
            root,
            base=arguments.base,
            head=arguments.head,
        )
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Workflow lifecycle policy failed: {error}", file=sys.stderr)
        return 1

    violations = [record for record in relevant if record.violations]
    if not violations:
        print("Workflow lifecycle policy passed.", flush=True)
        return 0

    print("Workflow lifecycle policy violations:", file=sys.stderr)
    for record in violations:
        print(f"  {record.path}", file=sys.stderr)
        for violation in record.violations:
            print(f"    - {violation}", file=sys.stderr)
    print(
        "See docs/workflow_lifecycle.md for the permanent and temporary "
        "workflow contracts.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
