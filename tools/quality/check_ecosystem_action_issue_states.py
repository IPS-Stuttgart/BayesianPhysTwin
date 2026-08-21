#!/usr/bin/env python3
"""Verify lifecycle records against the owning GitHub issue states."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, NoReturn, cast

from check_ecosystem_current_actions import (
    DEFAULT_RECORDS,
    EcosystemCurrentActionsError,
    load_action_records,
)

DEFAULT_API_ROOT: Final = "https://api.github.com"
IssueFetcher = Callable[[str, int], Mapping[str, object]]


class EcosystemIssueStateError(ValueError):
    """Raised when a lifecycle record contradicts its owning GitHub issue."""


def _fail(message: str) -> NoReturn:
    raise EcosystemIssueStateError(message)


def _github_fetcher(
    *,
    api_root: str,
    token: str | None,
) -> IssueFetcher:
    root = api_root.rstrip("/")

    def fetch(repository: str, issue_number: int) -> Mapping[str, object]:
        url = f"{root}/repos/{repository}/issues/{issue_number}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "BayesianPhysTwin-ecosystem-action-audit",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = json.load(response)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exc:
            raise EcosystemIssueStateError(
                f"cannot fetch {repository}#{issue_number}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            _fail(f"{repository}#{issue_number} returned a non-object")
        return cast(Mapping[str, object], raw)

    return fetch


def validate_issue_states(
    records_path: Path = DEFAULT_RECORDS,
    *,
    fetch_issue: IssueFetcher,
) -> dict[str, object]:
    """Compare every source lifecycle declaration with its owning issue."""

    try:
        payload = load_action_records(records_path)
    except EcosystemCurrentActionsError as exc:
        raise EcosystemIssueStateError(str(exc)) from exc

    records = cast(list[dict[str, object]], payload["records"])
    current_count = 0
    terminal_count = 0
    for record in records:
        repository = cast(str, record["owning_repository"])
        issue_number = cast(int, record["owning_issue"])
        action_id = cast(str, record["action_id"])
        expected_state = cast(str, record["issue_state"])
        issue = fetch_issue(repository, issue_number)

        actual_number = issue.get("number")
        if actual_number != issue_number:
            _fail(f"{action_id} owning issue number does not match GitHub")
        expected_url = (
            f"https://github.com/{repository}/issues/{issue_number}"
        )
        if issue.get("html_url") != expected_url:
            _fail(f"{action_id} owning issue URL does not match GitHub")
        actual_state = issue.get("state")
        if actual_state != expected_state:
            _fail(
                f"{action_id} records issue_state={expected_state}, "
                f"but GitHub reports {actual_state}"
            )

        if record["lifecycle"] == "current":
            current_count += 1
        else:
            terminal_count += 1

    return {
        "status": "valid",
        "checked_issue_count": len(records),
        "current_issue_count": current_count,
        "terminal_issue_count": terminal_count,
        "snapshot_date": payload["snapshot_date"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS,
    )
    parser.add_argument(
        "--api-root",
        default=DEFAULT_API_ROOT,
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable containing an optional GitHub token.",
    )
    args = parser.parse_args()
    token = os.environ.get(args.token_env)
    fetch_issue = _github_fetcher(api_root=args.api_root, token=token)
    try:
        report = validate_issue_states(
            args.records,
            fetch_issue=fetch_issue,
        )
    except EcosystemIssueStateError as exc:
        parser.exit(1, f"ecosystem issue-state validation failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
