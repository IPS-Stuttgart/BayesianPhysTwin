#!/usr/bin/env python3
"""Audit GitHub Actions registry entries against checked-in workflow files.

GitHub retains workflow registry entries and run history after a workflow file is
removed from the default branch. Consequently, the Actions API's ``total_count``
is not the number of workflow YAML files currently checked in. This tool reports
both surfaces without mutating workflow state or repository contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_SCHEMA = "bayesian-phystwin.workflow-registry-inventory"
_SCHEMA_VERSION = 1
_API_VERSION = "2022-11-28"
_DEFAULT_API_BASE_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
_WORKFLOW_DIRECTORY = PurePosixPath(".github/workflows")
_WORKFLOW_SUFFIXES = {".yml", ".yaml"}
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RegisteredWorkflow:
    """One normalized entry from GitHub's Actions workflow registry."""

    workflow_id: int
    name: str
    path: str
    state: str
    html_url: str
    created_at: str | None
    updated_at: str | None


@dataclass(frozen=True)
class WorkflowRegistryRecord:
    """One registry entry classified against the default-branch checkout."""

    workflow_id: int
    name: str
    path: str
    state: str
    html_url: str
    created_at: str | None
    updated_at: str | None
    classification: str
    checked_in_sha256: str | None


def _canonical_repository(value: str) -> str:
    if not isinstance(value, str) or not _REPOSITORY_PATTERN.fullmatch(value):
        raise ValueError("repository must use canonical owner/name syntax")
    owner, name = value.split("/", 1)
    if owner in {".", ".."} or name in {".", ".."}:
        raise ValueError("repository must use canonical owner/name syntax")
    return value


def _canonical_workflow_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("workflow path must be a nonempty string")
    if "\\" in value:
        raise ValueError("workflow path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"workflow path is not canonical: {value!r}")
    if len(path.parts) != 3 or tuple(path.parts[:2]) != tuple(
        _WORKFLOW_DIRECTORY.parts
    ):
        raise ValueError(
            "workflow path must name a file directly below .github/workflows"
        )
    if path.suffix.lower() not in _WORKFLOW_SUFFIXES:
        raise ValueError("workflow path must end in .yml or .yaml")
    return value


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be null or a nonempty string")
    return value


def parse_registered_workflow(raw: Mapping[str, object]) -> RegisteredWorkflow:
    """Normalize one workflow API object and reject coercive identifiers."""

    workflow_id = raw.get("id")
    if isinstance(workflow_id, bool) or not isinstance(workflow_id, int):
        raise ValueError("workflow id must be an integer")
    if workflow_id <= 0:
        raise ValueError("workflow id must be positive")
    name = raw.get("name")
    state = raw.get("state")
    html_url = raw.get("html_url")
    if not isinstance(name, str) or not name:
        raise ValueError("workflow name must be a nonempty string")
    if not isinstance(state, str) or not state:
        raise ValueError("workflow state must be a nonempty string")
    if not isinstance(html_url, str) or not html_url.startswith("https://"):
        raise ValueError("workflow html_url must be an HTTPS URL")
    return RegisteredWorkflow(
        workflow_id=workflow_id,
        name=name,
        path=_canonical_workflow_path(raw.get("path")),
        state=state,
        html_url=html_url,
        created_at=_optional_string(raw.get("created_at"), name="created_at"),
        updated_at=_optional_string(raw.get("updated_at"), name="updated_at"),
    )


def collect_workflow_pages(
    fetch_page: Callable[[int, int], Mapping[str, object]],
    *,
    per_page: int = 100,
) -> tuple[RegisteredWorkflow, ...]:
    """Collect a stable, complete registry snapshot from paginated API pages."""

    if (
        isinstance(per_page, bool)
        or not isinstance(per_page, int)
        or not 1 <= per_page <= 100
    ):
        raise ValueError("per_page must be an integer in [1, 100]")
    expected_total: int | None = None
    workflows: list[RegisteredWorkflow] = []
    seen_ids: set[int] = set()
    page = 1
    while True:
        payload = fetch_page(page, per_page)
        if not isinstance(payload, Mapping):
            raise ValueError("workflow API page must be a JSON object")
        total_count = payload.get("total_count")
        raw_workflows = payload.get("workflows")
        if (
            isinstance(total_count, bool)
            or not isinstance(total_count, int)
            or total_count < 0
        ):
            raise ValueError("workflow API total_count must be nonnegative")
        if not isinstance(raw_workflows, list):
            raise ValueError("workflow API workflows must be a list")
        if expected_total is None:
            expected_total = total_count
        elif total_count != expected_total:
            raise ValueError("workflow registry changed while pages were collected")
        for raw in raw_workflows:
            if not isinstance(raw, Mapping):
                raise ValueError("workflow API entry must be an object")
            workflow = parse_registered_workflow(raw)
            if workflow.workflow_id in seen_ids:
                raise ValueError(f"duplicate workflow id: {workflow.workflow_id}")
            seen_ids.add(workflow.workflow_id)
            workflows.append(workflow)
        if len(raw_workflows) < per_page:
            break
        page += 1
        if page > 10000:
            raise ValueError("workflow pagination exceeded the safety limit")
    if expected_total != len(workflows):
        raise ValueError(
            "workflow registry total_count disagrees with collected entries: "
            f"expected {expected_total}, collected {len(workflows)}"
        )
    return tuple(workflows)


def _request_json(url: str, *, token: str | None) -> Mapping[str, object]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "bayesian-phystwin-workflow-registry-audit",
        "X-GitHub-Api-Version": _API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read GitHub workflow registry: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("GitHub workflow registry response must be a JSON object")
    return payload


def fetch_registered_workflows(
    repository: str,
    *,
    token: str | None,
    api_base_url: str = _DEFAULT_API_BASE_URL,
) -> tuple[RegisteredWorkflow, ...]:
    """Fetch all Actions workflow registry entries for one repository."""

    canonical_repository = _canonical_repository(repository)
    if not isinstance(api_base_url, str) or not api_base_url.startswith("https://"):
        raise ValueError("api_base_url must be an HTTPS URL")
    owner, name = canonical_repository.split("/", 1)
    base = api_base_url.rstrip("/")

    def fetch_page(page: int, per_page: int) -> Mapping[str, object]:
        url = (
            f"{base}/repos/{quote(owner, safe='')}/{quote(name, safe='')}"
            f"/actions/workflows?per_page={per_page}&page={page}"
        )
        return _request_json(url, token=token)

    return collect_workflow_pages(fetch_page)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result = digest.hexdigest()
    if not _SHA256_PATTERN.fullmatch(result):  # pragma: no cover - hashlib contract
        raise AssertionError("hashlib returned a noncanonical SHA-256 digest")
    return result


def checked_in_workflows(repository_root: Path) -> dict[str, str]:
    """Return canonical workflow paths and content hashes from one checkout."""

    root = repository_root.resolve(strict=True)
    directory = root / _WORKFLOW_DIRECTORY.as_posix()
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(".github/workflows must be an ordinary directory")
    result: dict[str, str] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.suffix.lower() not in _WORKFLOW_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        _canonical_workflow_path(relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"workflow path must be an ordinary file: {relative}")
        result[relative] = _file_sha256(path)
    return result


def _classification(*, state: str, checked_in: bool) -> str:
    if checked_in:
        return "checked-in"
    if state == "active":
        return "orphaned-active"
    if state.startswith("disabled"):
        return "orphaned-disabled"
    return "orphaned-other-state"


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def build_inventory(
    repository: str,
    repository_root: Path,
    workflows: Sequence[RegisteredWorkflow],
    *,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Build one deterministic registry/default-branch comparison payload."""

    canonical_repository = _canonical_repository(repository)
    checked_in = checked_in_workflows(repository_root)
    registry_paths = {workflow.path for workflow in workflows}
    records = [
        WorkflowRegistryRecord(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            path=workflow.path,
            state=workflow.state,
            html_url=workflow.html_url,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            classification=_classification(
                state=workflow.state,
                checked_in=workflow.path in checked_in,
            ),
            checked_in_sha256=checked_in.get(workflow.path),
        )
        for workflow in sorted(
            workflows,
            key=lambda item: (item.path, item.workflow_id),
        )
    ]
    classifications = (
        "checked-in",
        "orphaned-active",
        "orphaned-disabled",
        "orphaned-other-state",
    )
    classification_counts = {
        classification: sum(
            record.classification == classification for record in records
        )
        for classification in classifications
    }
    path_counts: dict[str, int] = {}
    for record in records:
        path_counts[record.path] = path_counts.get(record.path, 0) + 1
    unregistered = sorted(set(checked_in) - registry_paths)
    identity_payload: dict[str, object] = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "repository": canonical_repository,
        "registry_workflow_count": len(records),
        "checked_in_workflow_file_count": len(checked_in),
        "classification_counts": classification_counts,
        "duplicate_registry_paths": sorted(
            path for path, count in path_counts.items() if count > 1
        ),
        "checked_in_unregistered_paths": unregistered,
        "records": [asdict(record) for record in records],
    }
    inventory_id = hashlib.sha256(_canonical_json(identity_payload)).hexdigest()
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    return {
        **identity_payload,
        "generated_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "inventory_id": inventory_id,
    }


def inventory_markdown(inventory: Mapping[str, object]) -> str:
    """Render a compact operator summary from one inventory payload."""

    counts = inventory["classification_counts"]
    records = inventory["records"]
    unregistered = inventory["checked_in_unregistered_paths"]
    duplicates = inventory["duplicate_registry_paths"]
    if not isinstance(counts, Mapping) or not isinstance(records, list):
        raise ValueError("inventory payload is malformed")
    if not isinstance(unregistered, list) or not isinstance(duplicates, list):
        raise ValueError("inventory payload is malformed")
    orphaned_active = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("classification") == "orphaned-active"
    ]
    lines = [
        "# GitHub Actions workflow registry inventory",
        "",
        "GitHub retains registry entries and run history after workflow YAML is",
        "removed. Registry totals therefore must not be interpreted as the number",
        "of workflow files on the default branch.",
        "",
        f"- Repository: `{inventory['repository']}`",
        f"- Registry entries: **{inventory['registry_workflow_count']}**",
        "- Checked-in workflow files: "
        f"**{inventory['checked_in_workflow_file_count']}**",
        f"- Checked-in registry entries: **{counts.get('checked-in', 0)}**",
        f"- Orphaned active registry entries: **{counts.get('orphaned-active', 0)}**",
        "- Orphaned disabled registry entries: "
        f"**{counts.get('orphaned-disabled', 0)}**",
        f"- Checked-in files not yet registered: **{len(unregistered)}**",
        f"- Duplicate registry paths: **{len(duplicates)}**",
        f"- Inventory ID: `{inventory['inventory_id']}`",
        "",
    ]
    if orphaned_active:
        lines.extend(
            (
                "## Orphaned active entries",
                "",
                "These entries have no workflow file in the checkout. Disable them",
                "only through an audited maintainer action; historical runs and",
                "artifacts remain separate from current workflow availability.",
                "",
                "| ID | Path | Name |",
                "| ---: | --- | --- |",
            )
        )
        for record in orphaned_active[:100]:
            lines.append(
                f"| `{record['workflow_id']}` | `{record['path']}` | {record['name']} |"
            )
        if len(orphaned_active) > 100:
            lines.append(
                f"|  | … | {len(orphaned_active) - 100} additional entries in JSON |"
            )
        lines.append("")
    if unregistered:
        lines.extend(("## Checked-in files not yet registered", ""))
        lines.extend(f"- `{path}`" for path in unregistered)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_text(path: Path, content: str) -> None:
    target = path.expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError(f"output must be an ordinary file: {path}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"temporary output path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repository in owner/name form; defaults to GITHUB_REPOSITORY",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="default-branch checkout root",
    )
    parser.add_argument(
        "--api-base-url",
        default=_DEFAULT_API_BASE_URL,
        help="GitHub API base URL",
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="environment variable containing an optional read token",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--fail-on-orphaned-active",
        action="store_true",
        help="return a nonzero status when active registry entries lack files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        repository = _canonical_repository(arguments.repository)
        token = os.environ.get(arguments.token_env)
        workflows = fetch_registered_workflows(
            repository,
            token=token,
            api_base_url=arguments.api_base_url,
        )
        inventory = build_inventory(
            repository,
            arguments.repository_root,
            workflows,
        )
        _write_text(
            arguments.output_json,
            json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
        _write_text(arguments.output_markdown, inventory_markdown(inventory))
    except (OSError, ValueError) as error:
        print(f"workflow registry audit failed: {error}", file=sys.stderr)
        return 1
    counts = inventory["classification_counts"]
    if not isinstance(counts, Mapping):  # pragma: no cover - build contract
        raise AssertionError("classification_counts must be a mapping")
    orphaned_active = int(counts.get("orphaned-active", 0))
    print(
        "workflow registry audit: "
        f"{inventory['registry_workflow_count']} registry entries, "
        f"{inventory['checked_in_workflow_file_count']} checked-in files, "
        f"{orphaned_active} orphaned active entries"
    )
    if arguments.fail_on_orphaned_active and orphaned_active:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
