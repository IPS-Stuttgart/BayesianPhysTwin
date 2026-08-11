"""Keep global packaging validation out of specialized workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_CENTRAL_MANIFEST_OWNERS = frozenset({"release-candidate.yml"})


def _event_paths(path: Path, event: str) -> tuple[str, ...]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, Mapping)
    events = payload.get("on")
    if not isinstance(events, Mapping):
        return ()
    configuration = events.get(event)
    if not isinstance(configuration, Mapping):
        return ()
    paths = configuration.get("paths")
    if paths is None:
        return ()
    if isinstance(paths, str):
        return (paths,)
    assert isinstance(paths, list)
    return tuple(str(item) for item in paths)


def test_only_central_release_validation_owns_manifest_triggers() -> None:
    offenders: list[str] = []
    for workflow in sorted(_WORKFLOWS.glob("*.y*ml")):
        if workflow.name in _CENTRAL_MANIFEST_OWNERS:
            continue
        for event in ("pull_request", "push"):
            if "MANIFEST.in" in _event_paths(workflow, event):
                offenders.append(f"{workflow.name}:{event}")

    assert not offenders, (
        "MANIFEST.in is repository-wide packaging state and must not launch "
        f"specialized workflows: {offenders}"
    )


def test_release_candidate_retains_manifest_ownership() -> None:
    workflow = _WORKFLOWS / "release-candidate.yml"
    assert "MANIFEST.in" in _event_paths(workflow, "pull_request")
