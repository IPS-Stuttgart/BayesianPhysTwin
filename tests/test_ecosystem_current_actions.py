"""Tests for generated ecosystem actions and source lifecycle records."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "tools/quality/check_ecosystem_current_actions.py"
ISSUE_CHECKER_PATH = ROOT / "tools/quality/check_ecosystem_action_issue_states.py"
REGISTRY_PATH = ROOT / "api/ecosystem-current-actions-v1.json"
RECORDS_PATH = ROOT / "api/ecosystem-action-records-v1.json"


def _tool(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _tool("check_ecosystem_current_actions", CHECKER_PATH)
issue_checker = _tool(
    "check_ecosystem_action_issue_states",
    ISSUE_CHECKER_PATH,
)


def _registry_payload() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _records_payload() -> dict[str, object]:
    return json.loads(RECORDS_PATH.read_text(encoding="utf-8"))


def _write(
    tmp_path: Path,
    name: str,
    payload: dict[str, object],
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(
    payload: dict[str, object],
    action_id: str,
) -> dict[str, object]:
    records = payload["records"]
    assert isinstance(records, list)
    for candidate in records:
        assert isinstance(candidate, dict)
        if candidate.get("action_id") == action_id:
            return candidate
    raise AssertionError(f"missing action record {action_id}")


def _action(
    payload: dict[str, object],
    action_id: str,
) -> dict[str, object]:
    actions = payload["actions"]
    assert isinstance(actions, list)
    for candidate in actions:
        assert isinstance(candidate, dict)
        if candidate.get("action_id") == action_id:
            return candidate
    raise AssertionError(f"missing current action {action_id}")


def _fake_issue_fetcher(
    states: dict[tuple[str, int], str] | None = None,
):
    overrides = {} if states is None else states
    source = _records_payload()
    records = cast(list[dict[str, object]], source["records"])
    expected = {
        (
            cast(str, record["owning_repository"]),
            cast(int, record["owning_issue"]),
        ): cast(str, record["issue_state"])
        for record in records
    }
    expected.update(overrides)

    def fetch(repository: str, issue_number: int) -> dict[str, object]:
        return {
            "number": issue_number,
            "html_url": (f"https://github.com/{repository}/issues/{issue_number}"),
            "state": expected[(repository, issue_number)],
        }

    return fetch


def test_checked_in_registry_is_generated_and_target_closed() -> None:
    report = checker.validate_registry(
        REGISTRY_PATH,
        records_path=RECORDS_PATH,
    )

    assert report == {
        "status": "valid",
        "snapshot_date": "2026-08-21",
        "action_count": 3,
        "terminal_action_count": 1,
        "highest_priority_action": ("covariance-only-independent-confirmation"),
        "target_open_action_count": 0,
    }
    assert checker.render_registry_text(RECORDS_PATH) == REGISTRY_PATH.read_text(
        encoding="utf-8"
    )


def test_priority_reordering_is_rejected(tmp_path: Path) -> None:
    payload = _records_payload()
    first = _record(
        payload,
        "covariance-only-independent-confirmation",
    )
    second = _record(payload, "causal4d-real-36-execution")
    first["priority"] = 2
    second["priority"] = 2

    with pytest.raises(
        checker.EcosystemCurrentActionsError,
        match="ordered and contiguous",
    ):
        checker.validate_registry(
            REGISTRY_PATH,
            records_path=_write(tmp_path, "records.json", payload),
        )


def test_causal4d_blocker_is_required(tmp_path: Path) -> None:
    payload = _records_payload()
    record = _record(payload, "causal4d-real-36-execution")
    record["blocked_by"] = []

    with pytest.raises(
        checker.EcosystemCurrentActionsError,
        match="blocked but has no blocker",
    ):
        checker.validate_registry(
            REGISTRY_PATH,
            records_path=_write(tmp_path, "records.json", payload),
        )


def test_confirmation_cannot_be_opened_early(tmp_path: Path) -> None:
    payload = _records_payload()
    record = _record(
        payload,
        "covariance-only-independent-confirmation",
    )
    record["target_access"] = "not-applicable"

    with pytest.raises(
        checker.EcosystemCurrentActionsError,
        match="target_access changed",
    ):
        checker.validate_registry(
            REGISTRY_PATH,
            records_path=_write(tmp_path, "records.json", payload),
        )


def test_terminal_action_cannot_reenter_current_snapshot(
    tmp_path: Path,
) -> None:
    payload = deepcopy(_registry_payload())
    actions = cast(list[dict[str, object]], payload["actions"])
    terminal = _record(
        _records_payload(),
        "material-backend-qualification",
    )
    actions.append(
        {
            "priority": 4,
            "action_id": terminal["action_id"],
            "domain": terminal["domain"],
            "owning_repository": terminal["owning_repository"],
            "owning_issue": terminal["owning_issue"],
            "status": terminal["status"],
            "next_gate": terminal["next_gate"],
            "target_access": terminal["target_access"],
            "blocked_by": terminal["blocked_by"],
            "forbidden_actions": terminal["forbidden_actions"],
        }
    )

    with pytest.raises(
        checker.EcosystemCurrentActionsError,
        match="not generated from",
    ):
        checker.validate_registry(
            _write(tmp_path, "registry.json", payload),
            records_path=RECORDS_PATH,
        )


def test_terminal_records_cannot_be_removed(tmp_path: Path) -> None:
    payload = _records_payload()
    record = _record(payload, "material-backend-qualification")
    terminal_records = cast(list[str], record["terminal_records"])
    terminal_records.remove("docs/jax_fem_zebra_source_value_v1_result.md")

    with pytest.raises(
        checker.EcosystemCurrentActionsError,
        match="lost a required terminal record",
    ):
        checker.validate_registry(
            REGISTRY_PATH,
            records_path=_write(tmp_path, "records.json", payload),
        )


def test_source_snapshot_cannot_predate_terminal_result(
    tmp_path: Path,
) -> None:
    payload = _records_payload()
    payload["snapshot_date"] = "2026-08-19"

    with pytest.raises(
        checker.EcosystemCurrentActionsError,
        match="completed after the source snapshot",
    ):
        checker.validate_registry(
            REGISTRY_PATH,
            records_path=_write(tmp_path, "records.json", payload),
        )


def test_online_issue_state_audit_accepts_current_lifecycle() -> None:
    report = issue_checker.validate_issue_states(
        RECORDS_PATH,
        fetch_issue=_fake_issue_fetcher(),
    )

    assert report == {
        "status": "valid",
        "checked_issue_count": 4,
        "current_issue_count": 3,
        "terminal_issue_count": 1,
        "snapshot_date": "2026-08-21",
    }


def test_online_issue_state_audit_rejects_closed_current_issue() -> None:
    fetch = _fake_issue_fetcher({("IPS-Stuttgart/Prob4D", 49): "closed"})

    with pytest.raises(
        issue_checker.EcosystemIssueStateError,
        match="GitHub reports closed",
    ):
        issue_checker.validate_issue_states(
            RECORDS_PATH,
            fetch_issue=fetch,
        )


def test_online_issue_state_audit_rejects_open_terminal_issue() -> None:
    fetch = _fake_issue_fetcher({("IPS-Stuttgart/BayesianPhysTwin", 664): "open"})

    with pytest.raises(
        issue_checker.EcosystemIssueStateError,
        match="GitHub reports open",
    ):
        issue_checker.validate_issue_states(
            RECORDS_PATH,
            fetch_issue=fetch,
        )


def test_unknown_source_field_is_rejected(tmp_path: Path) -> None:
    payload = deepcopy(_records_payload())
    record = _record(payload, "material-backend-qualification")
    record["target_outcome"] = "unknown"

    with pytest.raises(
        checker.EcosystemCurrentActionsError,
        match="record fields changed",
    ):
        checker.validate_registry(
            REGISTRY_PATH,
            records_path=_write(tmp_path, "records.json", payload),
        )
