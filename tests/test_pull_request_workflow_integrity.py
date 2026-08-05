"""Repository policies that keep pull-request source directly reviewable."""

from __future__ import annotations

import re
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_ROOT = _REPOSITORY_ROOT / ".github" / "workflows"
_PULL_REQUEST_TRIGGER = re.compile(r"(?m)^\s*pull_request(?:_target)?\s*:\s*(?:#.*)?$")
_CONTENTS_WRITE_PERMISSION = re.compile(r"(?m)^\s*contents\s*:\s*write\s*(?:#.*)?$")
_FORBIDDEN_PULL_REQUEST_COMMANDS = (
    "git " + "push",
    "git reset " + "--soft origin/",
    "git reset " + "--hard origin/",
)
_FORBIDDEN_PULL_REQUEST_TRANSPORT = (
    ".agent" + "/",
    "base64 " + "--decode",
    "base64 " + "-d",
)


def _workflow_texts() -> list[tuple[Path, str]]:
    workflows = sorted(
        path
        for path in _WORKFLOW_ROOT.iterdir()
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )
    return [(path, path.read_text(encoding="utf-8")) for path in workflows]


def test_source_transport_scratch_directory_is_not_committed() -> None:
    transport_paths = sorted(
        path.relative_to(_REPOSITORY_ROOT).as_posix()
        for path in _REPOSITORY_ROOT.rglob("*")
        if ".agent" in path.relative_to(_REPOSITORY_ROOT).parts
    )
    assert not transport_paths, (
        "source-transport scratch files must not be committed; publish the final "
        f"reviewable source instead: {transport_paths}"
    )


def test_pull_request_workflows_are_read_only_and_do_not_rewrite_source() -> None:
    violations: list[str] = []

    for path, text in _workflow_texts():
        if _PULL_REQUEST_TRIGGER.search(text) is None:
            continue

        relative_path = path.relative_to(_REPOSITORY_ROOT).as_posix()
        if _CONTENTS_WRITE_PERMISSION.search(text) is not None:
            violations.append(f"{relative_path}: grants contents: write")

        for marker in _FORBIDDEN_PULL_REQUEST_COMMANDS:
            if marker in text:
                violations.append(f"{relative_path}: contains {marker!r}")

        for marker in _FORBIDDEN_PULL_REQUEST_TRANSPORT:
            if marker in text:
                violations.append(
                    f"{relative_path}: transports hidden generated source via {marker!r}"
                )

    assert not violations, (
        "pull-request workflows must validate the exact reviewed commit without "
        "materializing, committing, or force-pushing replacement source:\n- "
        + "\n- ".join(violations)
    )
