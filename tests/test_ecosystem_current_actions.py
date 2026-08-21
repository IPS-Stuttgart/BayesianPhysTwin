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
        "snapshot_date": "2026-08-21",
        "action_count": 4,
        "highest_priority_action": ("covariance-only-independent-confirmation"),
        "active_action_count": 2,
        "blocked_action_count": 1,
        "completed_action_count": 1,
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
        match="blocked but has no blocker|lost a required blocker",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_confirmation_cannot_be_opened_early(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(
        payload,
        "covariance-only-independent-confirmation",
    )
    action["target_access"] = "open"

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="target_access is inconsistent with status",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_allowed_prob4d_lifecycle_transition_does_not_require_policy_rewrite(
    tmp_path: Path,
) -> None:
    payload = _payload()
    action = _action(payload, "real-prob4d-provider-competence")
    action["status"] = "cut3r-source-gates-active"
    action["next_gate"] = "execute-ordered-source-gates"

    report = tool.validate_registry(_write(tmp_path, payload))

    assert report["active_action_count"] == 2
    assert report["target_open_action_count"] == 0


def test_unknown_status_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(payload, "real-prob4d-provider-competence")
    action["status"] = "architecture-added"

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="status is not an allowed lifecycle state",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_completed_backend_action_cannot_retain_active_candidates(
    tmp_path: Path,
) -> None:
    payload = _payload()
    action = _action(payload, "material-backend-qualification")
    action["active_candidates"] = ["genesis-mpm-v1", "jax-fem-v1"]

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="cannot retain active candidates after completion",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_completed_backend_action_cannot_be_relabelled_active(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(payload, "material-backend-qualification")
    action["status"] = "source-qualification-active"
    action["next_gate"] = "qualify-one-candidate"

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="status is not an allowed lifecycle state",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_completed_action_requires_retention_next_gate(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(payload, "material-backend-qualification")
    action["next_gate"] = "run-another-backend"

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="must have a retention next_gate",
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


def test_terminal_backend_prohibition_cannot_be_removed(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(payload, "material-backend-qualification")
    forbidden = action["forbidden_actions"]
    assert isinstance(forbidden, list)
    forbidden.remove("reinterpret-source-physics-as-source-value")

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
