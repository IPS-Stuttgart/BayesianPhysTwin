"""Run pull-request workflow policy tests after one-shot retirement."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tools.quality.retired_workflow_contract_tests import (
    expose_tests,
    load_retired_contract_test,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/github-actions/retired-one-shot-v1"
_ARCHIVED = load_retired_contract_test(
    archived_test=ARCHIVE / "contract-tests" / Path(__file__).name,
    original_test=Path(__file__).resolve(),
    replacements={},
)
expose_tests(
    globals(),
    _ARCHIVED,
    exclude=frozenset({"test_global_manifest_changes_use_central_validation_only"}),
)


def test_global_manifest_changes_use_central_validation_only() -> None:
    specialized = (
        "deform360-calibration-observability-batch.yml",
        "deform360-calibration-prepared-inventory.yml",
        "deform360-calibration-visual-execution-admission.yml",
        "deform360-calibration-visual-production.yml",
        "deform360-joint-sparse-geometric-v4-contracts.yml",
        "deform360-joint-sparse-observability-v4.yml",
        "deform360-joint-sparse-prospective-v5-contracts.yml",
        "recursive-prob4d-stream-self-hosted.yml",
    )

    def event_paths(relative: str, event: str) -> tuple[str, ...]:
        workflow_root = _ARCHIVED._WORKFLOW_ROOT  # type: ignore[attr-defined]
        workflow = _ARCHIVED._load_workflow(  # type: ignore[attr-defined]
            (workflow_root / relative).read_text(encoding="utf-8")
        )
        events = workflow.get("on")
        assert isinstance(events, Mapping)
        configuration = events.get(event)
        if configuration is None:
            return ()
        assert isinstance(configuration, Mapping)
        paths = configuration.get("paths")
        if paths is None:
            return ()
        if isinstance(paths, str):
            return (paths,)
        assert isinstance(paths, list)
        return tuple(str(path) for path in paths)

    for relative in specialized:
        assert "MANIFEST.in" not in event_paths(relative, "pull_request")
        assert "MANIFEST.in" not in event_paths(relative, "push")

    assert "MANIFEST.in" in event_paths("release-candidate.yml", "pull_request")
    assert event_paths("tests.yml", "pull_request") == ()
