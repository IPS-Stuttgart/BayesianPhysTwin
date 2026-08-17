"""Tests for the checked-in GitHub Actions inventory ratchet."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/quality/check_workflow_inventory_budget.py"
CONTRACT_PATH = Path(".github/quality/workflow-inventory-budget-v1.json")


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_workflow_inventory_budget",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _contract(
    *,
    maximum: int,
    allowlist: list[str] | None = None,
    target: int = 1,
    target_temporary: int = 0,
) -> dict[str, object]:
    return {
        "schema": "bayesian-phystwin.workflow-inventory-budget",
        "schema_version": 1,
        "baseline_revision": "a" * 40,
        "maximum_checked_in_workflows": maximum,
        "temporary_looking_workflow_allowlist": allowlist or [],
        "retirement_target_maximum_checked_in_workflows": target,
        "retirement_target_maximum_temporary_looking_workflows": (
            target_temporary
        ),
    }


def _repository(
    tmp_path: Path,
    names: list[str],
    *,
    contract: dict[str, object] | None = None,
) -> Path:
    root = tmp_path / "repository"
    workflow_dir = root / ".github/workflows"
    quality_dir = root / ".github/quality"
    workflow_dir.mkdir(parents=True)
    quality_dir.mkdir(parents=True)
    for name in names:
        (workflow_dir / name).write_text("name: test\n", encoding="utf-8")
    payload = contract or _contract(maximum=len(names))
    (root / CONTRACT_PATH).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_repository_with_exact_inventory_matches(tmp_path: Path) -> None:
    names = ["ci.yml", "release.yml"]
    root = _repository(tmp_path, names)

    report = tool.validate_repository(root)

    assert report["checked_in_workflow_count"] == 2
    assert report["temporary_looking_workflow_count"] == 0
    assert report["status"] == "within-ratcheted-budget"


def test_workflow_growth_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path, ["ci.yml"], contract=_contract(maximum=1))
    (root / ".github/workflows/new.yml").write_text(
        "name: new\n",
        encoding="utf-8",
    )

    with pytest.raises(tool.WorkflowInventoryBudgetError, match="grew from 1 to 2"):
        tool.validate_repository(root)


def test_cleanup_requires_lowering_the_ceiling(tmp_path: Path) -> None:
    root = _repository(tmp_path, ["ci.yml"], contract=_contract(maximum=2))

    with pytest.raises(tool.WorkflowInventoryBudgetError, match="shrank"):
        tool.validate_repository(root)


def test_temporary_looking_roster_is_exact(tmp_path: Path) -> None:
    allowed = ".github/workflows/launch-source-once.yml"
    root = _repository(
        tmp_path,
        ["ci.yml", "launch-source-once.yml"],
        contract=_contract(maximum=2, allowlist=[allowed]),
    )
    tool.validate_repository(root)
    (root / ".github/workflows/launch-source-once.yml").unlink()
    (root / ".github/workflows/launch-target-once.yml").write_text(
        "name: replacement\n",
        encoding="utf-8",
    )

    with pytest.raises(tool.WorkflowInventoryBudgetError, match="roster changed"):
        tool.validate_repository(root)


def test_contract_rejects_unknown_fields_and_boolean_counts(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    payload = _contract(maximum=1)
    payload["unknown"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(tool.WorkflowInventoryBudgetError, match="fields changed"):
        tool.load_contract(path)

    payload.pop("unknown")
    payload["maximum_checked_in_workflows"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(tool.WorkflowInventoryBudgetError, match="positive integer"):
        tool.load_contract(path)


def test_contract_requires_sorted_temporary_allowlist(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    payload = _contract(
        maximum=2,
        allowlist=[
            ".github/workflows/z-once.yml",
            ".github/workflows/a-once.yml",
        ],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(tool.WorkflowInventoryBudgetError, match="must be sorted"):
        tool.load_contract(path)


def test_symlinked_workflow_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path, ["ci.yml"], contract=_contract(maximum=2))
    outside = root / "outside.yml"
    outside.write_text("name: outside\n", encoding="utf-8")
    (root / ".github/workflows/link.yml").symlink_to(outside)

    with pytest.raises(
        tool.WorkflowInventoryBudgetError, match="must not be a symlink"
    ):
        tool.validate_repository(root)


def test_checked_in_budget_is_frozen_and_repository_matches() -> None:
    contract = tool.load_contract(ROOT / CONTRACT_PATH)

    assert contract["maximum_checked_in_workflows"] == 95
    assert len(contract["temporary_looking_workflow_allowlist"]) == 12
    assert contract["retirement_target_maximum_checked_in_workflows"] == 84
    assert (
        contract["retirement_target_maximum_temporary_looking_workflows"]
        == 0
    )

    report = tool.validate_repository(ROOT)
    assert report["checked_in_workflow_count"] == 95
    assert report["temporary_looking_workflow_count"] == 12
    assert report["workflow_retirement_gap"] == 11
    assert report["temporary_looking_retirement_gap"] == 12
