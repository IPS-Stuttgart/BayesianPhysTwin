"""Run the workflow-budget unit tests against the retired exact inventory."""

from __future__ import annotations

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
    exclude=frozenset({"test_checked_in_budget_is_frozen_and_repository_matches"}),
)


def test_checked_in_budget_is_frozen_and_repository_matches() -> None:
    contract = _ARCHIVED.tool.load_contract(  # type: ignore[attr-defined]
        _ARCHIVED.ROOT / _ARCHIVED.CONTRACT_PATH  # type: ignore[attr-defined]
    )

    assert contract["maximum_checked_in_workflows"] == 84
    assert contract["temporary_looking_workflow_allowlist"] == []
    assert contract["retirement_target_maximum_checked_in_workflows"] == 81
    assert contract["retirement_target_maximum_temporary_looking_workflows"] == 0

    report = _ARCHIVED.tool.validate_repository(  # type: ignore[attr-defined]
        _ARCHIVED.ROOT  # type: ignore[attr-defined]
    )
    assert report["checked_in_workflow_count"] == 84
    assert report["temporary_looking_workflow_count"] == 0
    assert report["workflow_retirement_gap"] == 3
    assert report["temporary_looking_retirement_gap"] == 0
