#!/usr/bin/env python3
"""Validate generated ecosystem actions and their lifecycle source records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Final, NoReturn, cast

ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY: Final = ROOT / "api/ecosystem-current-actions-v1.json"
DEFAULT_RECORDS: Final = ROOT / "api/ecosystem-action-records-v1.json"

REGISTRY_SCHEMA: Final = "bayesian-phystwin.ecosystem-current-actions"
RECORDS_SCHEMA: Final = "bayesian-phystwin.ecosystem-action-records"
SCHEMA_VERSION: Final = 1

REGISTRY_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "snapshot_date",
        "claim_boundary",
        "actions",
    }
)
RECORDS_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "snapshot_date",
        "claim_boundary",
        "records",
    }
)
PUBLIC_ACTION_FIELDS: Final = frozenset(
    {
        "priority",
        "action_id",
        "domain",
        "owning_repository",
        "owning_issue",
        "status",
        "next_gate",
        "target_access",
        "blocked_by",
        "forbidden_actions",
    }
)
PUBLIC_OPTIONAL_FIELDS: Final = frozenset({"active_candidates"})
SOURCE_COMMON_FIELDS: Final = frozenset(
    {
        "lifecycle",
        "issue_state",
        "action_id",
        "domain",
        "owning_repository",
        "owning_issue",
        "status",
        "next_gate",
        "target_access",
        "blocked_by",
        "forbidden_actions",
    }
)
SOURCE_CURRENT_FIELDS: Final = SOURCE_COMMON_FIELDS | {"priority"}
SOURCE_TERMINAL_FIELDS: Final = SOURCE_COMMON_FIELDS | {
    "completed_date",
    "terminal_records",
}

ALLOWED_DOMAINS: Final = frozenset(
    {"bayesian-phystwin", "prob4d", "causal4d"}
)
ALLOWED_TARGET_ACCESS: Final = frozenset(
    {"closed", "forbidden", "not-applicable"}
)
REPOSITORY_PATTERN: Final = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
)
BLOCKER_PATTERN: Final = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*$"
)
DATE_PATTERN: Final = re.compile(r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$")
RECORD_PATH_PATTERN: Final = re.compile(
    r"^docs/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.md$"
)

REQUIRED_CURRENT_ACTIONS: Final = {
    "covariance-only-independent-confirmation": {
        "domain": "bayesian-phystwin",
        "owning_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "owning_issue": 461,
        "status": "source-gate-pending",
        "target_access": "closed",
        "required_forbidden": {
            "open-confirmation-before-source-pass",
            "retune-on-opened-target",
        },
    },
    "causal4d-real-36-execution": {
        "domain": "causal4d",
        "owning_repository": "IPS-Stuttgart/Causal4D",
        "owning_issue": 25,
        "status": "blocked",
        "target_access": "forbidden",
        "required_blockers": {"IPS-Stuttgart/Causal4D#377"},
        "required_forbidden": {
            "begin-confirmatory-acquisition-before-readiness",
            "replace-primary-experiment-with-optional-branch",
        },
    },
    "real-prob4d-provider-competence": {
        "domain": "prob4d",
        "owning_repository": "IPS-Stuttgart/Prob4D",
        "owning_issue": 49,
        "status": "separately-versioned-provider-required",
        "target_access": "closed",
        "required_forbidden": {
            "relax-support-after-outcomes",
            "reuse-opened-motioncrafter-or-deform360-targets",
        },
    },
}
REQUIRED_TERMINAL_ACTIONS: Final = {
    "material-backend-qualification": {
        "domain": "bayesian-phystwin",
        "owning_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "owning_issue": 664,
        "status": "terminal-negative-source-value",
        "completed_date": "2026-08-20",
        "target_access": "closed",
        "required_terminal_records": {
            "docs/genesis_mpm_zebra_source_value_v1_result.md",
            "docs/jax_fem_zebra_source_value_v1_result.md",
        },
        "required_forbidden": {
            "admit-closed-candidates-as-source-value-qualified",
            "retune-opened-source-actions",
            "reuse-terminal-action-as-current-priority",
        },
    }
}


class EcosystemCurrentActionsError(ValueError):
    """Raised when the operational action registry is not fail-closed."""


def _fail(message: str) -> NoReturn:
    raise EcosystemCurrentActionsError(message)


def _canonical_strings(value: object, *, name: str) -> list[str]:
    if type(value) is not list:
        _fail(f"{name} must be a list")
    items = cast(list[object], value)
    if any(type(item) is not str or not item for item in items):
        _fail(f"{name} must contain nonempty strings")
    strings = cast(list[str], items)
    if len(strings) != len(set(strings)):
        _fail(f"{name} must contain unique strings")
    if strings != sorted(strings):
        _fail(f"{name} must be sorted")
    return strings


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{name} must be a nonempty string")
    return cast(str, value)


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{name} must be a positive integer")
    return cast(int, value)


def _date(value: object, *, name: str) -> str:
    result = _nonempty_string(value, name=name)
    if DATE_PATTERN.fullmatch(result) is None:
        _fail(f"{name} must be YYYY-MM-DD")
    return result


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EcosystemCurrentActionsError(f"cannot load {name}: {exc}") from exc
    if type(raw) is not dict:
        _fail(f"{name} must be an object")
    return cast(dict[str, Any], raw)


def _validate_action_core(action: dict[str, Any], *, action_id: str) -> None:
    domain = _nonempty_string(action["domain"], name=f"{action_id} domain")
    if domain not in ALLOWED_DOMAINS:
        _fail(f"{action_id} domain is not recognized")

    repository = _nonempty_string(
        action["owning_repository"],
        name=f"{action_id} owning_repository",
    )
    if REPOSITORY_PATTERN.fullmatch(repository) is None:
        _fail(f"{action_id} owning_repository is not canonical")
    _positive_integer(action["owning_issue"], name=f"{action_id} owning_issue")
    _nonempty_string(action["status"], name=f"{action_id} status")
    _nonempty_string(action["next_gate"], name=f"{action_id} next_gate")

    target_access = _nonempty_string(
        action["target_access"],
        name=f"{action_id} target_access",
    )
    if target_access not in ALLOWED_TARGET_ACCESS:
        _fail(f"{action_id} target_access is not recognized")

    blockers = _canonical_strings(
        action["blocked_by"],
        name=f"{action_id} blocked_by",
    )
    if any(BLOCKER_PATTERN.fullmatch(item) is None for item in blockers):
        _fail(f"{action_id} blocked_by contains a noncanonical issue")
    forbidden = _canonical_strings(
        action["forbidden_actions"],
        name=f"{action_id} forbidden_actions",
    )
    if not forbidden:
        _fail(f"{action_id} must fail closed with forbidden_actions")
    if action["status"] == "blocked" and not blockers:
        _fail(f"{action_id} is blocked but has no blocker")


def _validate_public_action(raw: object) -> dict[str, Any]:
    if type(raw) is not dict:
        _fail("every current action must be an object")
    action = cast(dict[str, Any], raw)
    fields = frozenset(action)
    missing = PUBLIC_ACTION_FIELDS - fields
    unknown = fields - PUBLIC_ACTION_FIELDS - PUBLIC_OPTIONAL_FIELDS
    if missing or unknown:
        _fail(
            "current-action fields changed: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    _positive_integer(action["priority"], name="action priority")
    action_id = _nonempty_string(action["action_id"], name="action_id")
    _validate_action_core(action, action_id=action_id)

    if "active_candidates" in action:
        candidates = _canonical_strings(
            action["active_candidates"],
            name=f"{action_id} active_candidates",
        )
        if not candidates:
            _fail(f"{action_id} active_candidates must not be empty")
    return action


def _validate_source_record(raw: object) -> dict[str, Any]:
    if type(raw) is not dict:
        _fail("every action record must be an object")
    record = cast(dict[str, Any], raw)
    lifecycle = _nonempty_string(record.get("lifecycle"), name="lifecycle")
    if lifecycle == "current":
        required = SOURCE_CURRENT_FIELDS
        allowed = required | PUBLIC_OPTIONAL_FIELDS
    elif lifecycle == "terminal":
        required = SOURCE_TERMINAL_FIELDS
        allowed = required
    else:
        _fail("lifecycle must be current or terminal")

    fields = frozenset(record)
    missing = required - fields
    unknown = fields - allowed
    if missing or unknown:
        _fail(
            f"{lifecycle} record fields changed: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    action_id = _nonempty_string(record["action_id"], name="action_id")
    issue_state = _nonempty_string(
        record["issue_state"],
        name=f"{action_id} issue_state",
    )
    expected_issue_state = "open" if lifecycle == "current" else "closed"
    if issue_state != expected_issue_state:
        _fail(
            f"{action_id} {lifecycle} record must bind "
            f"issue_state={expected_issue_state}"
        )

    _validate_action_core(record, action_id=action_id)

    if lifecycle == "current":
        _positive_integer(record["priority"], name=f"{action_id} priority")
        if "active_candidates" in record:
            candidates = _canonical_strings(
                record["active_candidates"],
                name=f"{action_id} active_candidates",
            )
            if not candidates:
                _fail(f"{action_id} active_candidates must not be empty")
    else:
        _date(record["completed_date"], name=f"{action_id} completed_date")
        terminal_records = _canonical_strings(
            record["terminal_records"],
            name=f"{action_id} terminal_records",
        )
        if not terminal_records:
            _fail(f"{action_id} terminal_records must not be empty")
        if any(
            RECORD_PATH_PATTERN.fullmatch(item) is None
            for item in terminal_records
        ):
            _fail(f"{action_id} terminal_records contains a noncanonical path")
    return record


def load_action_records(path: Path = DEFAULT_RECORDS) -> dict[str, Any]:
    """Load and structurally validate the lifecycle source records."""

    payload = _load_object(path, name="ecosystem action records")
    fields = frozenset(payload)
    if fields != RECORDS_TOP_LEVEL_FIELDS:
        _fail(
            "action-record fields changed: "
            f"missing={sorted(RECORDS_TOP_LEVEL_FIELDS - fields)}, "
            f"unknown={sorted(fields - RECORDS_TOP_LEVEL_FIELDS)}"
        )
    if payload["schema"] != RECORDS_SCHEMA:
        _fail("action-record schema changed")
    if payload["schema_version"] != SCHEMA_VERSION:
        _fail("action-record schema_version changed")
    _date(payload["snapshot_date"], name="snapshot_date")
    _nonempty_string(payload["claim_boundary"], name="claim_boundary")
    if type(payload["records"]) is not list or not payload["records"]:
        _fail("records must be a nonempty list")
    records = [_validate_source_record(item) for item in payload["records"]]
    action_ids = [cast(str, item["action_id"]) for item in records]
    if len(action_ids) != len(set(action_ids)):
        _fail("action-record action_id values must be unique")
    payload["records"] = records
    return payload


def _public_action(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "priority": record["priority"],
        "action_id": record["action_id"],
        "domain": record["domain"],
        "owning_repository": record["owning_repository"],
        "owning_issue": record["owning_issue"],
        "status": record["status"],
        "next_gate": record["next_gate"],
        "target_access": record["target_access"],
    }
    if "active_candidates" in record:
        result["active_candidates"] = record["active_candidates"]
    result["blocked_by"] = record["blocked_by"]
    result["forbidden_actions"] = record["forbidden_actions"]
    return result


def derive_current_registry(
    action_records: dict[str, Any],
) -> dict[str, Any]:
    """Derive the public current-action snapshot from lifecycle records."""

    records = cast(list[dict[str, Any]], action_records["records"])
    current = [
        record for record in records if record["lifecycle"] == "current"
    ]
    current.sort(key=lambda item: cast(int, item["priority"]))
    priorities = [cast(int, item["priority"]) for item in current]
    if priorities != list(range(1, len(current) + 1)):
        _fail("current action priorities must be ordered and contiguous")
    return {
        "schema": REGISTRY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "snapshot_date": action_records["snapshot_date"],
        "claim_boundary": action_records["claim_boundary"],
        "actions": [_public_action(record) for record in current],
    }


def render_registry_text(path: Path = DEFAULT_RECORDS) -> str:
    """Render canonical JSON for the generated current-action snapshot."""

    payload = derive_current_registry(load_action_records(path))
    return json.dumps(payload, indent=2) + "\n"


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load and structurally validate a generated current-action snapshot."""

    payload = _load_object(path, name="ecosystem current actions")
    fields = frozenset(payload)
    if fields != REGISTRY_TOP_LEVEL_FIELDS:
        _fail(
            "registry fields changed: "
            f"missing={sorted(REGISTRY_TOP_LEVEL_FIELDS - fields)}, "
            f"unknown={sorted(fields - REGISTRY_TOP_LEVEL_FIELDS)}"
        )
    if payload["schema"] != REGISTRY_SCHEMA:
        _fail("registry schema changed")
    if payload["schema_version"] != SCHEMA_VERSION:
        _fail("registry schema_version changed")
    _date(payload["snapshot_date"], name="snapshot_date")
    _nonempty_string(payload["claim_boundary"], name="claim_boundary")
    if type(payload["actions"]) is not list or not payload["actions"]:
        _fail("actions must be a nonempty list")
    actions = [_validate_public_action(item) for item in payload["actions"]]
    payload["actions"] = actions
    return payload


def _check_expected(
    record: dict[str, Any],
    expected: dict[str, object],
    *,
    action_id: str,
) -> None:
    for field in (
        "domain",
        "owning_repository",
        "owning_issue",
        "status",
        "target_access",
    ):
        if record[field] != expected[field]:
            _fail(f"{action_id} {field} changed")
    required_blockers = set(
        cast(set[str], expected.get("required_blockers", set()))
    )
    if not required_blockers.issubset(set(record["blocked_by"])):
        _fail(f"{action_id} lost a required blocker")
    required_forbidden = cast(set[str], expected["required_forbidden"])
    if not required_forbidden.issubset(set(record["forbidden_actions"])):
        _fail(f"{action_id} lost a fail-closed prohibition")


def _validate_required_lifecycle(
    action_records: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = cast(list[dict[str, Any]], action_records["records"])
    current = [
        record for record in records if record["lifecycle"] == "current"
    ]
    terminal = [
        record for record in records if record["lifecycle"] == "terminal"
    ]
    current_ids = {cast(str, record["action_id"]) for record in current}
    terminal_ids = {cast(str, record["action_id"]) for record in terminal}
    if current_ids != set(REQUIRED_CURRENT_ACTIONS):
        _fail("required current ecosystem action roster changed")
    if terminal_ids != set(REQUIRED_TERMINAL_ACTIONS):
        _fail("required terminal ecosystem action roster changed")

    current_by_id = {
        cast(str, record["action_id"]): record for record in current
    }
    for action_id, expected in REQUIRED_CURRENT_ACTIONS.items():
        record = current_by_id[action_id]
        _check_expected(record, expected, action_id=action_id)
        if "active_candidates" in record:
            _fail(f"{action_id} cannot carry active_candidates")

    terminal_by_id = {
        cast(str, record["action_id"]): record for record in terminal
    }
    for action_id, expected in REQUIRED_TERMINAL_ACTIONS.items():
        record = terminal_by_id[action_id]
        _check_expected(record, expected, action_id=action_id)
        if record["completed_date"] != expected["completed_date"]:
            _fail(f"{action_id} completed_date changed")
        required_records = cast(
            set[str],
            expected["required_terminal_records"],
        )
        if not required_records.issubset(set(record["terminal_records"])):
            _fail(f"{action_id} lost a required terminal record")
        if action_records["snapshot_date"] < record["completed_date"]:
            _fail(f"{action_id} completed after the source snapshot")
    return current, terminal


def validate_registry(
    path: Path = DEFAULT_REGISTRY,
    *,
    records_path: Path = DEFAULT_RECORDS,
) -> dict[str, object]:
    """Validate source lifecycle and byte-equivalent generated output."""

    action_records = load_action_records(records_path)
    current, terminal = _validate_required_lifecycle(action_records)
    expected = derive_current_registry(action_records)
    registry = load_registry(path)
    if registry != expected:
        _fail(
            "checked-in current-action registry is not generated from "
            "api/ecosystem-action-records-v1.json"
        )

    action_ids = [
        cast(str, action["action_id"])
        for action in cast(list[dict[str, Any]], registry["actions"])
    ]
    return {
        "status": "valid",
        "snapshot_date": registry["snapshot_date"],
        "action_count": len(current),
        "terminal_action_count": len(terminal),
        "highest_priority_action": action_ids[0],
        "target_open_action_count": sum(
            action["target_access"] not in {"closed", "forbidden"}
            for action in current
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_REGISTRY,
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS,
    )
    args = parser.parse_args()
    try:
        report = validate_registry(args.path, records_path=args.records)
    except EcosystemCurrentActionsError as exc:
        parser.exit(1, f"ecosystem current-action validation failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
