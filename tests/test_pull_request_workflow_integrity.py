"""Repository policies that keep pull-request source directly reviewable."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_ROOT = _REPOSITORY_ROOT / ".github" / "workflows"
_FORBIDDEN_PULL_REQUEST_COMMANDS = (
    "git push",
    "git reset --soft origin/",
    "git reset --hard origin/",
)
_FORBIDDEN_PULL_REQUEST_TRANSPORT = (
    ".agent/",
    "base64 --decode",
    "base64 -d",
)


class _WorkflowLoader(yaml.SafeLoader):
    """Safe YAML loader with YAML-1.2-like booleans and unique mapping keys."""


_WorkflowLoader.yaml_implicit_resolvers = copy.deepcopy(
    yaml.SafeLoader.yaml_implicit_resolvers
)
for resolver_key, resolvers in tuple(_WorkflowLoader.yaml_implicit_resolvers.items()):
    retained = [
        (tag, pattern)
        for tag, pattern in resolvers
        if tag != "tag:yaml.org,2002:bool"
    ]
    if retained:
        _WorkflowLoader.yaml_implicit_resolvers[resolver_key] = retained
    else:
        del _WorkflowLoader.yaml_implicit_resolvers[resolver_key]


def _construct_unique_mapping(
    loader: _WorkflowLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_WorkflowLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _workflow_texts() -> list[tuple[Path, str]]:
    workflows = sorted(
        path
        for path in _WORKFLOW_ROOT.iterdir()
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )
    return [(path, path.read_text(encoding="utf-8")) for path in workflows]


def _load_workflow(text: str) -> Mapping[str, object]:
    try:
        loaded = yaml.load(text, Loader=_WorkflowLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"workflow YAML is invalid or ambiguous: {error}") from error
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError("workflow YAML root must be a mapping")
    return {str(key): value for key, value in loaded.items()}


def _iter_mappings(
    node: object,
    *,
    seen: set[int] | None = None,
) -> Iterator[Mapping[object, object]]:
    visited = set() if seen is None else seen
    if isinstance(node, (Mapping, list)):
        identity = id(node)
        if identity in visited:
            return
        visited.add(identity)
    if isinstance(node, Mapping):
        yield node
        for value in node.values():
            yield from _iter_mappings(value, seen=visited)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_mappings(value, seen=visited)


def _normalized_scalar(value: object) -> str:
    return str(value).strip().lower()


def _has_pull_request_trigger(workflow: Mapping[str, object]) -> bool:
    events = workflow.get("on")
    if isinstance(events, str):
        return _normalized_scalar(events) in {"pull_request", "pull_request_target"}
    if isinstance(events, list):
        return any(
            _normalized_scalar(event) in {"pull_request", "pull_request_target"}
            for event in events
        )
    if isinstance(events, Mapping):
        return any(
            _normalized_scalar(event) in {"pull_request", "pull_request_target"}
            for event in events
        )
    return False


def _permission_violations(
    relative_path: str,
    workflow: Mapping[str, object],
) -> list[str]:
    violations: list[str] = []
    for mapping in _iter_mappings(workflow):
        if "permissions" not in mapping:
            continue
        permissions = mapping["permissions"]
        if isinstance(permissions, str):
            if _normalized_scalar(permissions) == "write-all":
                violations.append(f"{relative_path}: grants permissions: write-all")
            continue
        if not isinstance(permissions, Mapping):
            violations.append(
                f"{relative_path}: permissions must be a mapping or scalar"
            )
            continue
        for name, level in permissions.items():
            if (
                _normalized_scalar(name) == "contents"
                and _normalized_scalar(level) == "write"
            ):
                violations.append(f"{relative_path}: grants contents: write")
    return violations


def _persists_checkout_credentials(workflow: Mapping[str, object]) -> bool:
    for mapping in _iter_mappings(workflow):
        uses = mapping.get("uses")
        if (
            not isinstance(uses, str)
            or not uses.lower().startswith("actions/checkout@")
        ):
            continue
        options = mapping.get("with")
        if options is None:
            continue
        if not isinstance(options, Mapping):
            return True
        persist_values = [
            value
            for name, value in options.items()
            if _normalized_scalar(name) == "persist-credentials"
        ]
        if persist_values and (
            len(persist_values) != 1
            or _normalized_scalar(persist_values[0]) != "false"
        ):
            return True
    return False


def _normalized_run_commands(workflow: Mapping[str, object]) -> Iterator[str]:
    for mapping in _iter_mappings(workflow):
        command = mapping.get("run")
        if not isinstance(command, str):
            continue
        without_continuations = re.sub(r"\\\s*", "", command)
        yield re.sub(r"\s+", " ", without_continuations).strip().lower()


def _pull_request_workflow_violations(
    relative_path: str,
    text: str,
) -> list[str]:
    try:
        workflow = _load_workflow(text)
    except ValueError as error:
        return [f"{relative_path}: {error}"]
    if not _has_pull_request_trigger(workflow):
        return []

    violations = _permission_violations(relative_path, workflow)
    if _persists_checkout_credentials(workflow):
        violations.append(f"{relative_path}: persists checkout credentials")

    for command in _normalized_run_commands(workflow):
        for marker in _FORBIDDEN_PULL_REQUEST_COMMANDS:
            if marker in command:
                violations.append(f"{relative_path}: contains {marker!r}")
        for marker in _FORBIDDEN_PULL_REQUEST_TRANSPORT:
            if marker in command:
                violations.append(
                    f"{relative_path}: transports hidden generated source "
                    f"via {marker!r}"
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


def test_pull_request_trigger_detection_covers_yaml_forms_and_aliases() -> None:
    triggering = (
        "on:\n  pull_request:\n",
        "on:\n  'pull_request_target': {types: [opened]}\n",
        "on: pull_request\n",
        'on: [push, "pull_request"]\n',
        "on: {push: null, pull_request_target: {types: [opened]}}\n",
        "events: &events [push, pull_request]\non: *events\n",
        '"on": [pull_request] # quoted key\n',
    )
    non_triggering = (
        "on: push\n",
        "on: [push, workflow_dispatch]\n",
        "name: pull_request documentation\non: workflow_dispatch\n",
        "on: workflow_dispatch\njobs:\n  task:\n    env:\n      on: pull_request\n",
    )

    assert all(_has_pull_request_trigger(_load_workflow(text)) for text in triggering)
    assert all(
        not _has_pull_request_trigger(_load_workflow(text)) for text in non_triggering
    )


def test_pull_request_workflows_cannot_bypass_write_checks() -> None:
    cases = (
        (
            "on: [pull_request]\npermissions: write-all\n",
            "grants permissions: write-all",
        ),
        (
            "on: pull_request_target\npermissions: {'contents': 'write'}\n",
            "grants contents: write",
        ),
        (
            "danger: &danger\n  contents: write\n"
            "on: pull_request\njobs:\n  test:\n    permissions: *danger\n",
            "grants contents: write",
        ),
        (
            "on: {pull_request: {types: [opened]}}\n"
            "steps:\n  - uses: actions/checkout@v7\n"
            "    with:\n      persist-credentials: TRUE\n",
            "persists checkout credentials",
        ),
        (
            'on: ["pull_request"]\nsteps:\n'
            "  - &checkout\n"
            "    uses: actions/checkout@v7\n"
            "    with: {'persist-credentials': 'true'}\n",
            "persists checkout credentials",
        ),
        (
            "on: pull_request\nsteps:\n"
            "  - run: >-\n      git\n      push origin HEAD\n",
            "contains 'git push'",
        ),
        (
            "on: pull_request\nsteps:\n"
            "  - run: |\n      base64 \\\n        --decode payload\n",
            "transports hidden generated source",
        ),
    )

    for text, expected in cases:
        violations = _pull_request_workflow_violations("fixture.yml", text)
        assert any(expected in violation for violation in violations)


def test_duplicate_yaml_keys_fail_closed() -> None:
    text = "on: pull_request\npermissions: {}\npermissions: {contents: write}\n"
    violations = _pull_request_workflow_violations("fixture.yml", text)
    assert len(violations) == 1
    assert "duplicate key 'permissions'" in violations[0]


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
