"""Repository policies that keep pull-request source directly reviewable."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_ROOT = _REPOSITORY_ROOT / ".github" / "workflows"
_ON_DECLARATION = re.compile(
    r"^(?P<indent>[ \t]*)[\"']?on[\"']?\s*:\s*(?P<value>.*)$"
)
_PULL_REQUEST_BLOCK_EVENT = re.compile(
    r"(?m)^[ \t]+[\"']?pull_request(?:_target)?[\"']?\s*:"
)
_PULL_REQUEST_FLOW_MAPPING_EVENT = re.compile(
    r"(?:^|[,{])\s*[\"']?pull_request(?:_target)?[\"']?\s*:"
)
_PULL_REQUEST_FLOW_SEQUENCE_EVENT = re.compile(
    r"(?:^|[,\[])\s*[\"']?pull_request(?:_target)?[\"']?\s*(?=,|\])"
)
_CONTENTS_WRITE_PERMISSION = re.compile(
    r"(?im)^\s*contents\s*:\s*[\"']?write[\"']?\s*(?:#.*)?$"
)
_INLINE_CONTENTS_WRITE_PERMISSION = re.compile(
    r"(?im)^\s*permissions\s*:\s*\{[^}\n]*"
    r"[\"']?contents[\"']?\s*:\s*[\"']?write[\"']?\b"
)
_WRITE_ALL_PERMISSION = re.compile(
    r"(?im)^\s*permissions\s*:\s*[\"']?write-all[\"']?\s*(?:#.*)?$"
)
_PERSIST_CREDENTIALS_TRUE = re.compile(
    r"(?im)^\s*persist-credentials\s*:\s*[\"']?true[\"']?\s*(?:#.*)?$"
)
_INLINE_PERSIST_CREDENTIALS_TRUE = re.compile(
    r"(?im)^\s*with\s*:\s*\{[^}\n]*"
    r"[\"']?persist-credentials[\"']?\s*:\s*[\"']?true[\"']?\b"
)
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


def _strip_yaml_comment(line: str) -> str:
    """Remove comments while preserving hashes inside quoted scalars."""

    quote: str | None = None
    escaped = False
    for index, character in enumerate(line):
        if quote == '"':
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if quote == "'":
            if character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "#":
            return line[:index]
    return line


def _flow_balance(value: str) -> int:
    """Return unclosed flow-container depth outside quoted scalars."""

    quote: str | None = None
    escaped = False
    balance = 0
    for character in value:
        if quote == '"':
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if quote == "'":
            if character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "[{":
            balance += 1
        elif character in "]}":
            balance -= 1
    return balance


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _iter_on_values(text: str) -> Iterator[tuple[str, str]]:
    """Yield block or inline values of top-level-style ``on`` declarations."""

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        cleaned = _strip_yaml_comment(lines[index])
        match = _ON_DECLARATION.match(cleaned)
        if match is None:
            index += 1
            continue

        declaration_indent = _indent_width(match.group("indent"))
        value = match.group("value").strip()
        if value:
            parts = [value]
            balance = _flow_balance(value)
            index += 1
            while balance > 0 and index < len(lines):
                continuation = _strip_yaml_comment(lines[index]).strip()
                parts.append(continuation)
                balance += _flow_balance(continuation)
                index += 1
            yield "inline", " ".join(part for part in parts if part)
            continue

        block: list[str] = []
        index += 1
        while index < len(lines):
            candidate = _strip_yaml_comment(lines[index])
            if candidate.strip() and _indent_width(candidate) <= declaration_indent:
                break
            if candidate.strip():
                block.append(candidate)
            index += 1
        yield "block", "\n".join(block)


def _inline_on_value_has_pull_request(value: str) -> bool:
    normalized = value.strip()
    if normalized.startswith("["):
        return _PULL_REQUEST_FLOW_SEQUENCE_EVENT.search(normalized) is not None
    if normalized.startswith("{"):
        return _PULL_REQUEST_FLOW_MAPPING_EVENT.search(normalized) is not None
    return normalized.strip("\"'") in {"pull_request", "pull_request_target"}


def _has_pull_request_trigger(text: str) -> bool:
    for style, value in _iter_on_values(text):
        if style == "block":
            if _PULL_REQUEST_BLOCK_EVENT.search(value) is not None:
                return True
        elif _inline_on_value_has_pull_request(value):
            return True
    return False


def _pull_request_workflow_violations(
    relative_path: str,
    text: str,
) -> list[str]:
    if not _has_pull_request_trigger(text):
        return []

    violations: list[str] = []
    if _CONTENTS_WRITE_PERMISSION.search(text) is not None:
        violations.append(f"{relative_path}: grants contents: write")
    if _INLINE_CONTENTS_WRITE_PERMISSION.search(text) is not None:
        violations.append(f"{relative_path}: grants inline contents: write")
    if _WRITE_ALL_PERMISSION.search(text) is not None:
        violations.append(f"{relative_path}: grants permissions: write-all")
    if _PERSIST_CREDENTIALS_TRUE.search(text) is not None:
        violations.append(f"{relative_path}: persists checkout credentials")
    if _INLINE_PERSIST_CREDENTIALS_TRUE.search(text) is not None:
        violations.append(f"{relative_path}: persists inline checkout credentials")

    for marker in _FORBIDDEN_PULL_REQUEST_COMMANDS:
        if marker in text:
            violations.append(f"{relative_path}: contains {marker!r}")

    for marker in _FORBIDDEN_PULL_REQUEST_TRANSPORT:
        if marker in text:
            violations.append(
                f"{relative_path}: transports hidden generated source via {marker!r}"
            )
    return violations


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


def test_pull_request_trigger_detection_covers_yaml_forms() -> None:
    triggering = (
        "on:\n  pull_request:\n",
        "on:\n  'pull_request_target': {types: [opened]}\n",
        "on: pull_request\n",
        'on: [push, "pull_request"]\n',
        "on: {push: null, pull_request_target: {types: [opened]}}\n",
        "on: [\n  push,\n  'pull_request_target',\n]\n",
        "on: {\n  push: null,\n  pull_request: {types: [opened]},\n}\n",
        '"on": [pull_request] # quoted key\n',
    )
    non_triggering = (
        "on: push\n",
        "on: [push, workflow_dispatch]\n",
        "name: pull_request documentation\n",
        "on: workflow_dispatch\nsteps:\n  - run: |\n      pull_request:\n",
    )

    assert all(_has_pull_request_trigger(text) for text in triggering)
    assert all(not _has_pull_request_trigger(text) for text in non_triggering)


def test_inline_pull_request_workflows_cannot_bypass_write_checks() -> None:
    cases = (
        (
            "on: [pull_request]\npermissions: write-all\n",
            "grants permissions: write-all",
        ),
        (
            "on: pull_request_target\npermissions: {'contents': 'write'}\n",
            "grants inline contents: write",
        ),
        (
            "on: {pull_request: {types: [opened]}}\n"
            "steps:\n  - uses: actions/checkout@v7\n"
            "    with:\n      persist-credentials: TRUE\n",
            "persists checkout credentials",
        ),
        (
            'on: ["pull_request"]\nsteps:\n'
            "  - uses: actions/checkout@v7\n"
            "    with: {'persist-credentials': 'true'}\n",
            "persists inline checkout credentials",
        ),
    )

    for text, expected in cases:
        violations = _pull_request_workflow_violations("fixture.yml", text)
        assert any(expected in violation for violation in violations)


def test_non_pull_request_workflows_are_outside_this_policy() -> None:
    text = (
        "on: workflow_dispatch\n"
        "permissions: write-all\n"
        "steps:\n  - run: git push\n"
    )

    assert _pull_request_workflow_violations("manual.yml", text) == []


def test_pull_request_workflows_are_read_only_and_do_not_rewrite_source() -> None:
    violations: list[str] = []

    for path, text in _workflow_texts():
        relative_path = path.relative_to(_REPOSITORY_ROOT).as_posix()
        violations.extend(_pull_request_workflow_violations(relative_path, text))

    assert not violations, (
        "pull-request workflows must validate the exact reviewed commit without "
        "materializing, committing, or force-pushing replacement source:\n- "
        + "\n- ".join(violations)
    )
