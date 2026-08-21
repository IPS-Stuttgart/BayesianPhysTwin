#!/usr/bin/env python3
"""Validate the ecosystem's current-action and information-boundary registry."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Final, NoReturn, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
BLOCKER_PATTERN: Final = re.compile(
    r"^(?P<repository>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"#(?P<issue>[1-9][0-9]*)$"
)
DATE_PATTERN: Final = re.compile(r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$")
GITHUB_API_VERSION: Final = "2022-11-28"
GITHUB_USER_AGENT: Final = "BayesianPhysTwin-current-action-audit/1"
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
        "status": "cut3r-source-bundle-pending",
        "target_access": "closed",
        "required_candidates": ["cut3r-recurrent-online-v1"],
        "required_forbidden": {
            "add-provider-architecture-before-source-localization",
            "relax-support-after-outcomes",
            "reuse-opened-motioncrafter-or-deform360-targets",
        },
    },
}

IssueLoader = Callable[[str, int], Mapping[str, object]]


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


def fetch_github_issue(
    repository: str,
    issue_number: int,
    *,
    token: str | None = None,
    token_repository: str | None = None,
    timeout_seconds: float = 10.0,
) -> Mapping[str, object]:
    """Fetch one GitHub issue without using any repository payload."""

    if REPOSITORY_PATTERN.fullmatch(repository) is None:
        _fail("GitHub issue repository is not canonical")
    _positive_integer(issue_number, name="GitHub issue number")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or timeout_seconds <= 0.0
    ):
        _fail("GitHub timeout must be positive")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": GITHUB_USER_AGENT,
    }
    if token and (token_repository is None or repository == token_repository):
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{repository}/issues/{issue_number}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=float(timeout_seconds)) as response:
            raw_bytes = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise EcosystemCurrentActionsError(
            f"cannot read GitHub issue {repository}#{issue_number}: {exc}"
        ) from exc
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EcosystemCurrentActionsError(
            f"GitHub issue {repository}#{issue_number} returned invalid JSON"
        ) from exc
    if type(raw) is not dict:
        _fail(f"GitHub issue {repository}#{issue_number} returned a non-object")
    return cast(dict[str, object], raw)


def _parse_blocker(reference: str) -> tuple[str, int]:
    match = BLOCKER_PATTERN.fullmatch(reference)
    if match is None:  # pragma: no cover - load_registry validates this first.
        _fail(f"blocker {reference!r} is not canonical")
    return match.group("repository"), int(match.group("issue"))


def _require_open_issue(
    payload: Mapping[str, object],
    *,
    reference: str,
    role: str,
) -> None:
    if "pull_request" in payload:
        _fail(f"{role} {reference} resolves to a pull request, not an issue")
    state = payload.get("state")
    if state != "open":
        rendered_state = state if isinstance(state, str) else "invalid-state"
        _fail(
            f"{role} {reference} is {rendered_state}; "
            "current-action references must remain open"
        )


def validate_github_issue_states(
    payload: Mapping[str, object],
    *,
    issue_loader: IssueLoader,
) -> dict[str, object]:
    """Verify that every owning issue and live blocker is still an open issue."""

    actions = cast(list[dict[str, Any]], payload["actions"])
    cache: dict[tuple[str, int], Mapping[str, object]] = {}
    action_references: set[tuple[str, int]] = set()
    blocker_references: set[tuple[str, int]] = set()

    def load(reference: tuple[str, int]) -> Mapping[str, object]:
        if reference not in cache:
            try:
                loaded = issue_loader(*reference)
            except EcosystemCurrentActionsError:
                raise
            except Exception as exc:
                repository, issue_number = reference
                raise EcosystemCurrentActionsError(
                    f"cannot read GitHub issue {repository}#{issue_number}: {exc}"
                ) from exc
            if not isinstance(loaded, Mapping):
                repository, issue_number = reference
                _fail(
                    f"GitHub issue {repository}#{issue_number} returned a non-mapping"
                )
            cache[reference] = loaded
        return cache[reference]

    for action in actions:
        action_id = cast(str, action["action_id"])
        repository = cast(str, action["owning_repository"])
        issue_number = cast(int, action["owning_issue"])
        reference = (repository, issue_number)
        action_references.add(reference)
        _require_open_issue(
            load(reference),
            reference=f"{repository}#{issue_number}",
            role=f"{action_id} owning issue",
        )

        for blocker in cast(list[str], action["blocked_by"]):
            blocker_reference = _parse_blocker(blocker)
            blocker_references.add(blocker_reference)
            _require_open_issue(
                load(blocker_reference),
                reference=blocker,
                role=f"{action_id} blocker",
            )

    return {
        "github_status": "valid",
        "github_reference_count": len(cache),
        "github_action_issue_count": len(action_references),
        "github_blocker_issue_count": len(blocker_references),
    }


def validate_registry(
    path: Path = DEFAULT_REGISTRY,
    *,
    check_github: bool = False,
    issue_loader: IssueLoader | None = None,
) -> dict[str, object]:
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

    report: dict[str, object] = {
        "status": "valid",
        "snapshot_date": payload["snapshot_date"],
        "action_count": len(actions),
        "highest_priority_action": action_ids[0],
        "target_open_action_count": sum(
            action["target_access"] not in {"closed", "forbidden"} for action in actions
        ),
    }
    if check_github:
        loader = fetch_github_issue if issue_loader is None else issue_loader
        report.update(validate_github_issue_states(payload, issue_loader=loader))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_REGISTRY,
    )
    parser.add_argument(
        "--check-github",
        action="store_true",
        help="require every owning issue and blocker to remain an open issue",
    )
    parser.add_argument(
        "--github-token-env",
        default="GITHUB_TOKEN",
        help="environment variable holding a token for the current repository",
    )
    parser.add_argument(
        "--github-token-repository-env",
        default="GITHUB_REPOSITORY",
        help="environment variable naming the repository scoped by that token",
    )
    args = parser.parse_args()

    issue_loader: IssueLoader | None = None
    if args.check_github:
        issue_loader = partial(
            fetch_github_issue,
            token=os.environ.get(args.github_token_env),
            token_repository=os.environ.get(args.github_token_repository_env),
        )
    try:
        report = validate_registry(
            args.path,
            check_github=args.check_github,
            issue_loader=issue_loader,
        )
    except EcosystemCurrentActionsError as exc:
        parser.exit(1, f"ecosystem current-action validation failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
