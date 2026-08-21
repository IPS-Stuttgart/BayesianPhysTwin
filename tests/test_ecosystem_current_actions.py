"""Tests for the fail-closed ecosystem current-action registry."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
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


def _open_issue_loader() -> Callable[[str, int], dict[str, object]]:
    def load(repository: str, issue_number: int) -> dict[str, object]:
        return {
            "state": "open",
            "html_url": f"https://github.com/{repository}/issues/{issue_number}",
        }

    return load


def test_checked_in_registry_is_valid_and_target_closed() -> None:
    report = tool.validate_registry(REGISTRY_PATH)

    assert report == {
        "status": "valid",
        "snapshot_date": "2026-08-21",
        "action_count": 3,
        "highest_priority_action": ("covariance-only-independent-confirmation"),
        "target_open_action_count": 0,
    }


def test_checked_in_registry_references_open_github_issues() -> None:
    report = tool.validate_registry(
        REGISTRY_PATH,
        check_github=True,
        issue_loader=_open_issue_loader(),
    )

    assert report["github_status"] == "valid"
    assert report["github_reference_count"] == 4
    assert report["github_action_issue_count"] == 3
    assert report["github_blocker_issue_count"] == 1


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


def test_cut3r_candidate_is_frozen(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(payload, "real-prob4d-provider-competence")
    action["active_candidates"] = ["another-provider-v1"]

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="active candidate roster changed",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_terminal_material_backend_action_cannot_be_reintroduced(
    tmp_path: Path,
) -> None:
    payload = _payload()
    actions = payload["actions"]
    assert isinstance(actions, list)
    actions.append(
        {
            "priority": 4,
            "action_id": "material-backend-qualification",
            "domain": "bayesian-phystwin",
            "owning_repository": "IPS-Stuttgart/BayesianPhysTwin",
            "owning_issue": 664,
            "status": "source-qualification-active",
            "next_gate": "qualify-one-candidate-through-source-value",
            "target_access": "closed",
            "active_candidates": ["genesis-mpm-v1", "jax-fem-v1"],
            "blocked_by": [],
            "forbidden_actions": [
                "admit-new-backend-family",
                "open-fresh-target-before-source-value",
            ],
        }
    )

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="required ecosystem action roster changed",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_fail_closed_prohibition_cannot_be_removed(tmp_path: Path) -> None:
    payload = _payload()
    action = _action(payload, "real-prob4d-provider-competence")
    forbidden = action["forbidden_actions"]
    assert isinstance(forbidden, list)
    forbidden.remove("add-provider-architecture-before-source-localization")

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="lost a fail-closed prohibition",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_unknown_action_field_is_rejected(tmp_path: Path) -> None:
    payload = deepcopy(_payload())
    action = _action(payload, "real-prob4d-provider-competence")
    action["target_outcome"] = "unknown"

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="action fields changed",
    ):
        tool.validate_registry(_write(tmp_path, payload))


def test_closed_owning_issue_is_rejected() -> None:
    def load(repository: str, issue_number: int) -> dict[str, object]:
        state = (
            "closed"
            if (repository, issue_number) == ("IPS-Stuttgart/BayesianPhysTwin", 461)
            else "open"
        )
        return {"state": state}

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="owning issue IPS-Stuttgart/BayesianPhysTwin#461 is closed",
    ):
        tool.validate_registry(
            REGISTRY_PATH,
            check_github=True,
            issue_loader=load,
        )


def test_closed_blocker_is_rejected() -> None:
    def load(repository: str, issue_number: int) -> dict[str, object]:
        state = (
            "closed"
            if (repository, issue_number) == ("IPS-Stuttgart/Causal4D", 377)
            else "open"
        )
        return {"state": state}

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="blocker IPS-Stuttgart/Causal4D#377 is closed",
    ):
        tool.validate_registry(
            REGISTRY_PATH,
            check_github=True,
            issue_loader=load,
        )


def test_pull_request_cannot_own_a_current_action() -> None:
    def load(repository: str, issue_number: int) -> dict[str, object]:
        del repository, issue_number
        return {"state": "open", "pull_request": {}}

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="resolves to a pull request, not an issue",
    ):
        tool.validate_registry(
            REGISTRY_PATH,
            check_github=True,
            issue_loader=load,
        )


def test_issue_loader_failure_is_bounded() -> None:
    def load(repository: str, issue_number: int) -> dict[str, object]:
        raise RuntimeError(f"unavailable: {repository}#{issue_number}")

    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="cannot read GitHub issue IPS-Stuttgart/BayesianPhysTwin#461",
    ):
        tool.validate_registry(
            REGISTRY_PATH,
            check_github=True,
            issue_loader=load,
        )


class _JsonResponse:
    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return b'{"state":"open"}'


def test_github_token_is_not_sent_cross_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[object] = []

    def fake_urlopen(request: object, *, timeout: float) -> _JsonResponse:
        assert timeout == 2.0
        requests.append(request)
        return _JsonResponse()

    monkeypatch.setattr(tool, "urlopen", fake_urlopen)
    token_repository = "IPS-Stuttgart/BayesianPhysTwin"
    tool.fetch_github_issue(
        token_repository,
        461,
        token="secret",
        token_repository=token_repository,
        timeout_seconds=2.0,
    )
    tool.fetch_github_issue(
        "IPS-Stuttgart/Causal4D",
        25,
        token="secret",
        token_repository=token_repository,
        timeout_seconds=2.0,
    )

    assert len(requests) == 2
    assert requests[0].get_header("Authorization") == "Bearer secret"
    assert requests[1].get_header("Authorization") is None


def test_boolean_timeout_is_rejected() -> None:
    with pytest.raises(
        tool.EcosystemCurrentActionsError,
        match="GitHub timeout must be positive",
    ):
        tool.fetch_github_issue(
            "IPS-Stuttgart/BayesianPhysTwin",
            461,
            timeout_seconds=True,
        )
