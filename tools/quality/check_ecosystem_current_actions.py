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
ALLOWED_TARGET_ACCESS: Final = frozenset({"closed", "forbidden", "not-applicable"})
REPOSITORY_PATTERN: Final = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
BLOCKER_PATTERN: Final = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*$")
DATE_PATTERN: Final = re.compile(r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$")
REQUIRED_ACTIONS: Final = {
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
            "reuse-opened-motioncrafter-or-deform360-targets",
            "relax-support-after-outcomes",
        },
    },
    "material-backend-qualification": {
        "domain": "bayesian-phystwin",
        "owning_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "owning_issue": 664,
        "status": "source-qualification-active",
        "target_access": "closed",
        "required_candidates": ["genesis-mpm-v1", "jax-fem-v1"],
        "required_forbidden": {
            "admit-new-backend-family",
            "open-fresh-target-before-source-value",
        },
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
    if action["status"] == "blocked" and not blockers:
        _fail(f"{action_id} is blocked but has no blocker")

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
    if set(action_ids) != set(REQUIRED_ACTIONS):
        _fail("required ecosystem action roster changed")

    by_id = {cast(str, action["action_id"]): action for action in actions}
    for action_id, expected in REQUIRED_ACTIONS.items():
        action = by_id[action_id]
        for field in (
            "domain",
            "owning_repository",
            "owning_issue",
            "status",
            "target_access",
        ):
            if action[field] != expected[field]:
                _fail(f"{action_id} {field} changed")
        required_blockers = set(expected.get("required_blockers", set()))
        if not required_blockers.issubset(set(action["blocked_by"])):
            _fail(f"{action_id} lost a required blocker")
        required_forbidden = set(expected["required_forbidden"])
        if not required_forbidden.issubset(set(action["forbidden_actions"])):
            _fail(f"{action_id} lost a fail-closed prohibition")
        if "required_candidates" in expected:
            if action.get("active_candidates") != expected["required_candidates"]:
                _fail(f"{action_id} active candidate roster changed")
        elif "active_candidates" in action:
            _fail(f"{action_id} cannot carry active_candidates")

    return {
        "status": "valid",
        "snapshot_date": payload["snapshot_date"],
        "action_count": len(actions),
        "highest_priority_action": action_ids[0],
        "target_open_action_count": sum(
            action["target_access"] not in {"closed", "forbidden"} for action in actions
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
