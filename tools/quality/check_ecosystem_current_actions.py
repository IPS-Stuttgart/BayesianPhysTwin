#!/usr/bin/env python3
"""Validate the ecosystem's current-action and information-boundary registry."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Final, NoReturn, cast

ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY: Final = ROOT / "api/ecosystem-current-actions-v1.json"
SCHEMA: Final = "bayesian-phystwin.ecosystem-current-actions"
SCHEMA_VERSION: Final = 1
TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "snapshot_date",
        "claim_boundary",
        "actions",
    }
)
COMMON_ACTION_FIELDS: Final = frozenset(
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
OPTIONAL_ACTION_FIELDS: Final = frozenset({"active_candidates"})
ALLOWED_DOMAINS: Final = frozenset({"bayesian-phystwin", "prob4d", "causal4d"})
ALLOWED_TARGET_ACCESS: Final = frozenset(
    {"closed", "forbidden", "not-applicable", "open"}
)
ALLOWED_LIFECYCLES: Final = frozenset({"active", "blocked", "completed"})
REPOSITORY_PATTERN: Final = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
BLOCKER_PATTERN: Final = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*$")
DATE_PATTERN: Final = re.compile(r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$")

# Immutable ownership, information-boundary, and lifecycle rules.  Status values
# are deliberately represented as an allowed state machine rather than one exact
# mutable snapshot value.  This lets an action advance without weakening its
# ownership, target-access, blocker, or fail-closed invariants.
ACTION_POLICIES: Final = {
    "covariance-only-independent-confirmation": {
        "domain": "bayesian-phystwin",
        "owning_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "owning_issue": 461,
        "status_rules": {
            "source-gate-pending": ("active", "closed"),
            "source-negative-complete": ("completed", "closed"),
            "source-positive-confirmation-authorized": ("active", "closed"),
            "confirmation-complete": ("completed", "not-applicable"),
        },
        "required_forbidden": {
            "open-confirmation-before-source-pass",
            "retune-on-opened-target",
        },
    },
    "causal4d-real-36-execution": {
        "domain": "causal4d",
        "owning_repository": "IPS-Stuttgart/Causal4D",
        "owning_issue": 25,
        "status_rules": {
            "blocked": ("blocked", "forbidden"),
            "preacquisition-readiness-active": ("active", "forbidden"),
            "confirmatory-acquisition-authorized": ("active", "forbidden"),
            "real-experiment-complete": ("completed", "not-applicable"),
        },
        "required_blockers_by_status": {
            "blocked": {"IPS-Stuttgart/Causal4D#377"},
        },
        "required_forbidden": {
            "begin-confirmatory-acquisition-before-readiness",
            "replace-primary-experiment-with-optional-branch",
        },
    },
    "real-prob4d-provider-competence": {
        "domain": "prob4d",
        "owning_repository": "IPS-Stuttgart/Prob4D",
        "owning_issue": 49,
        "status_rules": {
            "cut3r-source-bundle-required": ("active", "closed"),
            "cut3r-source-gates-active": ("active", "closed"),
            "source-negative-complete": ("completed", "closed"),
            "ready-for-one-target-evaluation": ("active", "closed"),
            "target-evaluation-complete": ("completed", "not-applicable"),
        },
        "required_forbidden": {
            "reuse-opened-motioncrafter-or-deform360-targets",
            "relax-support-after-outcomes",
        },
    },
    "material-backend-qualification": {
        "domain": "bayesian-phystwin",
        "owning_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "owning_issue": 664,
        "status_rules": {
            "completed-bounded-negative": ("completed", "closed"),
        },
        "required_forbidden": {
            "admit-new-backend-family-without-new-protocol",
            "open-fresh-target-from-rejected-candidates",
            "reinterpret-source-physics-as-source-value",
            "rerun-terminal-candidate-under-same-identity",
        },
        "active_candidates_forbidden": True,
    },
}


class EcosystemCurrentActionsError(ValueError):
    """Raised when the operational priority registry is not fail-closed."""


def _fail(message: str) -> NoReturn:
    raise EcosystemCurrentActionsError(message)


def _canonical_strings(value: object, *, name: str) -> list[str]:
    if type(value) is not list:
        _fail(f"{name} must be a list")
    items = cast(list[object], value)
    if any(type(item) is not str or not item for item in items):
        _fail(f"{name} must contain nonempty strings")
    strings = cast(list[str], items)
    if len(set(strings)) != len(strings):
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


def _validate_action(raw: object) -> dict[str, Any]:
    if type(raw) is not dict:
        _fail("every action must be an object")
    action = cast(dict[str, Any], raw)
    fields = frozenset(action)
    missing = COMMON_ACTION_FIELDS - fields
    unknown = fields - COMMON_ACTION_FIELDS - OPTIONAL_ACTION_FIELDS
    if missing or unknown:
        _fail(
            "action fields changed: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    _positive_integer(action["priority"], name="action priority")
    action_id = _nonempty_string(action["action_id"], name="action_id")
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
    if any(BLOCKER_PATTERN.fullmatch(blocker) is None for blocker in blockers):
        _fail(f"{action_id} blocked_by contains a noncanonical issue")
    forbidden = _canonical_strings(
        action["forbidden_actions"],
        name=f"{action_id} forbidden_actions",
    )
    if not forbidden:
        _fail(f"{action_id} must fail closed with forbidden_actions")

    if "active_candidates" in action:
        candidates = _canonical_strings(
            action["active_candidates"],
            name=f"{action_id} active_candidates",
        )
        if not candidates:
            _fail(f"{action_id} active_candidates must not be empty")

    return action


def load_registry(path: Path) -> dict[str, Any]:
    """Load and validate one registry payload."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EcosystemCurrentActionsError(
            f"cannot load ecosystem current actions: {exc}"
        ) from exc
    if type(raw) is not dict:
        _fail("ecosystem current actions must be an object")
    payload = cast(dict[str, Any], raw)
    fields = frozenset(payload)
    if fields != TOP_LEVEL_FIELDS:
        _fail(
            "registry fields changed: "
            f"missing={sorted(TOP_LEVEL_FIELDS - fields)}, "
            f"unknown={sorted(fields - TOP_LEVEL_FIELDS)}"
        )
    if payload["schema"] != SCHEMA:
        _fail("registry schema changed")
    if payload["schema_version"] != SCHEMA_VERSION:
        _fail("registry schema_version changed")
    snapshot_date = _nonempty_string(
        payload["snapshot_date"],
        name="snapshot_date",
    )
    if DATE_PATTERN.fullmatch(snapshot_date) is None:
        _fail("snapshot_date must be YYYY-MM-DD")
    _nonempty_string(payload["claim_boundary"], name="claim_boundary")
    if type(payload["actions"]) is not list or not payload["actions"]:
        _fail("actions must be a nonempty list")
    actions = [_validate_action(item) for item in payload["actions"]]
    payload["actions"] = actions
    return payload


def _status_lifecycle(action: dict[str, Any], policy: dict[str, Any]) -> str:
    action_id = cast(str, action["action_id"])
    status = cast(str, action["status"])
    status_rules = cast(dict[str, tuple[str, str]], policy["status_rules"])
    if status not in status_rules:
        _fail(f"{action_id} status is not an allowed lifecycle state")
    lifecycle, required_target_access = status_rules[status]
    if lifecycle not in ALLOWED_LIFECYCLES:
        _fail(f"{action_id} lifecycle policy is not recognized")
    if action["target_access"] != required_target_access:
        _fail(f"{action_id} target_access is inconsistent with status")
    return lifecycle


def _validate_policy(action: dict[str, Any], policy: dict[str, Any]) -> str:
    action_id = cast(str, action["action_id"])
    for field in ("domain", "owning_repository", "owning_issue"):
        if action[field] != policy[field]:
            _fail(f"{action_id} {field} changed")

    lifecycle = _status_lifecycle(action, policy)
    blockers = set(cast(list[str], action["blocked_by"]))
    required_by_status = cast(
        dict[str, set[str]],
        policy.get("required_blockers_by_status", {}),
    )
    required_blockers = required_by_status.get(cast(str, action["status"]), set())
    if not required_blockers.issubset(blockers):
        _fail(f"{action_id} lost a required blocker")
    if lifecycle == "blocked" and not blockers:
        _fail(f"{action_id} is blocked but has no blocker")
    if lifecycle != "blocked" and blockers:
        _fail(f"{action_id} has blockers outside a blocked state")

    required_forbidden = set(cast(set[str], policy["required_forbidden"]))
    forbidden = set(cast(list[str], action["forbidden_actions"]))
    if not required_forbidden.issubset(forbidden):
        _fail(f"{action_id} lost a fail-closed prohibition")

    if cast(bool, policy.get("active_candidates_forbidden", False)):
        if "active_candidates" in action:
            _fail(f"{action_id} cannot retain active candidates after completion")
    elif "active_candidates" in action:
        _fail(f"{action_id} cannot carry active_candidates")

    if lifecycle == "completed":
        next_gate = cast(str, action["next_gate"])
        if not next_gate.startswith(("retain-", "archive-", "report-")):
            _fail(f"{action_id} completed action must have a retention next_gate")
    return lifecycle


def validate_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, object]:
    """Validate ordering and all non-negotiable current-action boundaries."""

    payload = load_registry(path)
    actions = cast(list[dict[str, Any]], payload["actions"])
    priorities = [cast(int, action["priority"]) for action in actions]
    expected_priorities = list(range(1, len(actions) + 1))
    if priorities != expected_priorities:
        _fail("action priorities must be ordered and contiguous")

    action_ids = [cast(str, action["action_id"]) for action in actions]
    if len(set(action_ids)) != len(action_ids):
        _fail("action_id values must be unique")
    if set(action_ids) != set(ACTION_POLICIES):
        _fail("required ecosystem action roster changed")

    lifecycles = [
        _validate_policy(action, cast(dict[str, Any], ACTION_POLICIES[action_id]))
        for action_id, action in zip(action_ids, actions, strict=True)
    ]
    return {
        "status": "valid",
        "snapshot_date": payload["snapshot_date"],
        "action_count": len(actions),
        "highest_priority_action": action_ids[0],
        "active_action_count": lifecycles.count("active"),
        "blocked_action_count": lifecycles.count("blocked"),
        "completed_action_count": lifecycles.count("completed"),
        "target_open_action_count": sum(
            action["target_access"] == "open" for action in actions
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
    args = parser.parse_args()
    try:
        report = validate_registry(args.path)
    except EcosystemCurrentActionsError as exc:
        parser.exit(1, f"ecosystem current-action validation failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
