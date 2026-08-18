"""Tests for the fail-closed ecosystem current-action registry."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/quality/check_ecosystem_current_actions.py"
REGISTRY_PATH = ROOT / "api/ecosystem-current-actions-v1.json"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_ecosystem_current_actions",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _payload() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _action(payload: dict[str, object], action_id: str) -> dict[str, object]:
    actions = payload["actions"]
    assert isinstance(actions, list)
    for candidate in actions:
        assert isinstance(candidate, dict)
        if candidate.get("action_id") == action_id:
            return candidate
    raise AssertionError(f"missing action {action_id}")


def test_checked_in_registry_is_valid_and_target_closed() -> None:
    report = tool.validate_registry(REGISTRY_PATH)

    assert report == {
        "status": "valid",
        "snapshot_date": "2026-08-18",
        "action_count": 4,
        "highest_priority_action": (
            "covariance-only-independent-confirmation"
        ),
        "target_open_action_count": 0,
    }


def test_priority_reordering_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    actions = payload["actions"]
    assert isinstance(actions, list)
    actions[0]["priority"] = 2
    actions[1]["priority"] = 1

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="ordered and contiguous",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_causal4d_blocker_is_required(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(payload, "causal4d-real-36-execution")
    action["blocked_by"] = []

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="blocked but has no blocker",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_confirmation_cannot_be_opened_early(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(
        payload,
        "covariance-only-independent-confirmation",
    )
    action["target_access"] = "not-applicable"

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="target_access changed",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_backend_candidates_are_frozen(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(payload, "material-backend-qualification")
    action["active_candidates"] = ["genesis-mpm-v1", "new-backend-v1"]

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="active candidate roster changed",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_fail_closed_prohibition_cannot_be_removed(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(payload, "real-prob4d-provider-competence")
    forbidden = action["forbidden_actions"]
    assert isinstance(forbidden, list)
    forbidden.remove("relax-support-after-outcomes")

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="lost a fail-closed prohibition",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_unknown_action_field_is_rejected(tmp_path: Path) -> None:
    payload = deepcopy(_payload())
    action = _action(payload, "material-backend-qualification")
    action["target_outcome"] = "unknown"

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="action fields changed",
    ):
        tool.validate_registry(_write(tmp_path, payload))
