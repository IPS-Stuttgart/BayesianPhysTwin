#!/usr/bin/env python3
"""Fail closed when a source revision reuses another release tag's version."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
_PROJECT_SECTION = re.compile(r"^\s*\[project\]\s*(?:#.*)?$")
_SECTION = re.compile(r"^\s*\[[^]]+\]\s*(?:#.*)?$")
_PROJECT_VERSION = re.compile(r"^\s*version\s*=\s*(['\"])([^'\"]+)\1\s*(?:#.*)?$")
_CITATION_VERSION = re.compile(
    r"^version:\s*(?:\"([^\"]+)\"|'([^']+)'|([^#\n]+?))\s*(?:#.*)?$",
    re.MULTILINE,
)
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?$")
_FINAL_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


class VersionIdentityError(ValueError):
    """Raised when source metadata and Git release identity disagree."""


def _literal_project_version(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise VersionIdentityError("cannot read pyproject.toml") from error

    in_project = False
    versions: list[str] = []
    for line in lines:
        if _PROJECT_SECTION.fullmatch(line):
            in_project = True
            continue
        if _SECTION.fullmatch(line):
            in_project = False
            continue
        if not in_project:
            continue
        match = _PROJECT_VERSION.fullmatch(line)
        if match is not None:
            versions.append(match.group(2))

    if len(versions) != 1:
        raise VersionIdentityError(
            "pyproject.toml must declare exactly one literal project version"
        )
    version = versions[0]
    if _VERSION.fullmatch(version) is None:
        raise VersionIdentityError("project version is not a supported literal version")
    return version


def _literal_citation_version(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise VersionIdentityError("cannot read CITATION.cff") from error

    matches = list(_CITATION_VERSION.finditer(text))
    if len(matches) != 1:
        raise VersionIdentityError(
            "CITATION.cff must declare exactly one literal version"
        )
    version = next(group for group in matches[0].groups() if group is not None).strip()
    if _VERSION.fullmatch(version) is None:
        raise VersionIdentityError("CITATION.cff version is invalid")
    return version


def _git(
    root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def _resolve_commit(root: Path, revision: str, *, name: str) -> str:
    if not revision or revision.strip() != revision:
        raise VersionIdentityError(f"{name} must be a canonical nonempty revision")
    completed = _git(
        root,
        "rev-parse",
        "--verify",
        "--quiet",
        f"{revision}^{{commit}}",
        check=False,
    )
    if completed.returncode != 0:
        raise VersionIdentityError(f"{name} is not an available commit: {revision}")
    resolved = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise VersionIdentityError(f"{name} did not resolve to a full commit SHA")
    return resolved


def _tag_commit(root: Path, tag: str) -> str | None:
    completed = _git(
        root,
        "rev-parse",
        "--verify",
        "--quiet",
        f"refs/tags/{tag}^{{commit}}",
        check=False,
    )
    if completed.returncode != 0:
        return None
    resolved = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise VersionIdentityError("release tag did not resolve to a full commit SHA")
    return resolved


def _is_shallow(root: Path) -> bool:
    completed = _git(root, "rev-parse", "--is-shallow-repository", check=False)
    if completed.returncode != 0:
        raise VersionIdentityError("project root is not a readable Git repository")
    value = completed.stdout.strip()
    if value not in {"true", "false"}:
        raise VersionIdentityError("Git returned an invalid shallow-repository status")
    return value == "true"


def validate_version_identity(
    project_root: str | Path = ROOT,
    *,
    head: str = "HEAD",
    expected_tag: str | None = None,
    require_complete_history: bool = False,
) -> dict[str, object]:
    """Validate project, citation, commit, and canonical release-tag identity."""

    root = Path(project_root).resolve()
    project_version = _literal_project_version(root / "pyproject.toml")
    citation_version = _literal_citation_version(root / "CITATION.cff")
    if citation_version != project_version:
        raise VersionIdentityError("CITATION.cff version does not match pyproject.toml")

    shallow = _is_shallow(root)
    if shallow and require_complete_history:
        raise VersionIdentityError(
            "complete Git history and tags are required for release identity checks"
        )

    head_revision = _resolve_commit(root, head, name="head revision")
    release_tag = f"v{project_version}"
    tagged_revision = _tag_commit(root, release_tag)

    if expected_tag is not None and expected_tag != release_tag:
        raise VersionIdentityError(
            f"expected release tag must be {release_tag!r}, got {expected_tag!r}"
        )
    if expected_tag is not None and tagged_revision is None:
        raise VersionIdentityError(
            f"expected release tag {release_tag!r} is not available"
        )
    if tagged_revision is not None and tagged_revision != head_revision:
        raise VersionIdentityError(
            f"project version {project_version!r} reuses {release_tag!r}: "
            f"tagged revision is {tagged_revision}, head is {head_revision}"
        )

    is_final = _FINAL_VERSION.fullmatch(project_version) is not None
    if tagged_revision is not None:
        status = "exact-tagged-release" if is_final else "exact-tagged-prerelease"
    elif is_final:
        status = "untagged-release-candidate"
    else:
        status = "development-version"

    return {
        "project_version": project_version,
        "citation_version": citation_version,
        "release_tag": release_tag,
        "head_revision": head_revision,
        "tagged_revision": tagged_revision,
        "complete_history": not shallow,
        "final_version": is_final,
        "tag_required": expected_tag is not None,
        "status": status,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--expected-tag")
    parser.add_argument("--require-complete-history", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = validate_version_identity(
            arguments.project_root,
            head=arguments.head,
            expected_tag=arguments.expected_tag,
            require_complete_history=arguments.require_complete_history,
        )
    except VersionIdentityError as error:
        print(f"release version identity failed: {error}")
        return 2

    if arguments.as_json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        tagged = report["tagged_revision"] or "absent"
        print(
            "release version identity matched: "
            f"version={report['project_version']} "
            f"head={report['head_revision']} tag={tagged} "
            f"status={report['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
