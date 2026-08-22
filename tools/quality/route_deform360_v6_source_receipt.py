#!/usr/bin/env python3
"""Verify and route one frozen Deform360 v6 source receipt.

The tool is intentionally target blind. It accepts only a compact artifact from
the registered protected-main dual-runtime source workflow, verifies the closed
information boundary and content identities, and idempotently mirrors the
bounded receipt to the current owning issue.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final, Never, cast

EXPECTED_REPOSITORY: Final = "IPS-Stuttgart/BayesianPhysTwin"
EXPECTED_SCHEMA: Final = (
    "bayesian-phystwin.deform360-v6-source-prediction-execution-receipt"
)
EXPECTED_SCHEMA_VERSION: Final = 1
EXPECTED_AMENDMENT_ID: Final = (
    "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
)
EXPECTED_RUNNER_NAME: Final = "workstation2"
SEALED_STATUS: Final = "source-prediction-evidence-sealed"
ALLOWED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        SEALED_STATUS,
        "source-inputs-incomplete",
        "source-technical-failure-retained",
        "invalid",
    }
)
CLOSED_INFORMATION_FIELDS: Final[tuple[str, ...]] = (
    "development_suffix_opened",
    "future_object_observations_used_for_prediction",
    "v5_confirmation_payloads_opened",
    "v5_confirmation_outcomes_used",
    "v6_fresh_target_selected",
    "v6_target_payloads_opened",
    "v6_target_outcomes_used",
    "replacement_allowed",
)
CLOSED_AUTHORIZATION_FIELDS: Final[tuple[str, ...]] = (
    "claim_authorized",
    "fresh_target_selection_authorized",
    "fresh_target_payload_access_authorized",
)
_SHA256_LINE = re.compile(r"^([0-9a-f]{64}) ([ *])(.+)$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class ReceiptRoutingError(ValueError):
    """Raised when a compact source artifact fails closed verification."""


def _reject(message: str) -> Never:
    raise ReceiptRoutingError(message)


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        _reject(f"required environment variable is missing: {name}")
    return value


def _genuine_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(f"{name} is not a genuine integer")
    result = int(value)
    if minimum is not None and result < minimum:
        _reject(f"{name} is below the registered bound")
    if maximum is not None and result > maximum:
        _reject(f"{name} is above the registered bound")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_manifest_path(raw: str) -> str:
    if not raw or "\\" in raw:
        _reject("SHA256SUMS contains an unsafe path")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        _reject("SHA256SUMS contains an unsafe path")
    normalized = candidate.as_posix()
    if normalized in {"", ".", "SHA256SUMS"}:
        _reject("SHA256SUMS contains an invalid member")
    return normalized


def verify_manifest_closure(root: Path) -> tuple[str, ...]:
    """Verify every regular artifact file against a closed SHA-256 manifest."""

    if not root.is_dir() or root.is_symlink():
        _reject("compact receipt root is not a real directory")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            _reject("compact receipt bundle contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            _reject("compact receipt bundle contains a non-regular member")
        actual_files.add(path.relative_to(root).as_posix())

    manifest_path = root / "SHA256SUMS"
    if "SHA256SUMS" not in actual_files or manifest_path.is_symlink():
        _reject("compact receipt bundle has no real SHA256SUMS file")
    listed: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        match = _SHA256_LINE.fullmatch(line)
        if match is None:
            _reject(f"SHA256SUMS line {line_number} is not canonical")
        expected_digest, _mode, raw_path = match.groups()
        relative = _safe_relative_manifest_path(raw_path)
        if relative in listed:
            _reject("SHA256SUMS contains a duplicate member")
        member = root / relative
        if not member.is_file() or member.is_symlink():
            _reject("SHA256SUMS references a missing or unsafe member")
        observed_digest = _sha256_file(member)
        if observed_digest != expected_digest:
            _reject(f"SHA-256 mismatch for compact member: {relative}")
        listed[relative] = expected_digest

    expected_files = actual_files - {"SHA256SUMS"}
    if set(listed) != expected_files:
        missing = sorted(expected_files - set(listed))
        extra = sorted(set(listed) - expected_files)
        _reject(
            "SHA256SUMS does not close the compact artifact: "
            f"unlisted={missing!r}, missing={extra!r}"
        )
    return tuple(sorted(listed))


def receipt_content_id(receipt: Mapping[str, Any]) -> str:
    canonical_receipt = dict(receipt)
    canonical_receipt.pop("receipt_id", None)
    canonical = json.dumps(
        canonical_receipt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_source_receipt_bundle(
    root: Path,
    *,
    source_run_id: str,
    source_run_attempt: str,
    source_head_sha: str,
    source_workflow_conclusion: str,
) -> dict[str, Any]:
    """Verify manifest closure, source identity, and terminal semantics."""

    verify_manifest_closure(root)
    if not source_run_id.isdecimal() or int(source_run_id) < 1:
        _reject("triggering source run ID is invalid")
    if not source_run_attempt.isdecimal() or int(source_run_attempt) < 1:
        _reject("triggering source run attempt is invalid")
    if _SHA1.fullmatch(source_head_sha) is None:
        _reject("triggering source revision is not a canonical commit SHA")
    if source_workflow_conclusion not in {"success", "failure"}:
        _reject("triggering source workflow has an unsupported conclusion")

    receipt_path = root / "execution-receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        _reject("compact source bundle has no real execution receipt")
    raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(raw_receipt, dict):
        _reject("execution receipt must be a JSON object")
    receipt = cast(dict[str, Any], raw_receipt)

    if receipt.get("schema") != EXPECTED_SCHEMA:
        _reject("unexpected source receipt schema")
    if receipt.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        _reject("unexpected source receipt schema version")
    if receipt.get("amendment_id") != EXPECTED_AMENDMENT_ID:
        _reject("source receipt amendment identity changed")
    if receipt.get("runner_name") != EXPECTED_RUNNER_NAME:
        _reject("source receipt runner identity changed")
    if receipt.get("source_revision") != source_head_sha:
        _reject("source receipt revision does not match triggering run")
    if str(receipt.get("workflow_run_id")) != source_run_id:
        _reject("source receipt run ID does not match triggering run")
    if str(receipt.get("workflow_run_attempt")) != source_run_attempt:
        _reject("source receipt attempt does not match triggering run")

    status = receipt.get("status")
    if not isinstance(status, str) or status not in ALLOWED_STATUSES:
        _reject("unexpected source receipt terminal status")
    manifest_count = _genuine_integer(
        receipt.get("physical_manifest_count"),
        name="physical_manifest_count",
        minimum=0,
        maximum=10,
    )
    seal_count = _genuine_integer(
        receipt.get("source_prediction_seal_count"),
        name="source_prediction_seal_count",
        minimum=0,
        maximum=100,
    )
    exit_code = _genuine_integer(receipt.get("exit_code"), name="exit_code")
    if status == SEALED_STATUS:
        if manifest_count != 10 or seal_count != 100:
            _reject("sealed source receipt does not contain the complete 10/100 panel")
        if source_workflow_conclusion != "success" or exit_code != 0:
            _reject("sealed source receipt has inconsistent workflow termination")
    else:
        if source_workflow_conclusion != "failure" or exit_code == 0:
            _reject("nonsealed source receipt has inconsistent workflow termination")

    boundary = receipt.get("information_boundary")
    if not isinstance(boundary, dict):
        _reject("source receipt has no information boundary")
    for name in CLOSED_INFORMATION_FIELDS:
        if boundary.get(name) is not False:
            _reject(f"source receipt violates closed boundary: {name}")
    for name in CLOSED_AUTHORIZATION_FIELDS:
        if receipt.get(name) is not False:
            _reject(f"source receipt violates authorization boundary: {name}")

    declared_receipt_id = receipt.get("receipt_id")
    observed_receipt_id = receipt_content_id(receipt)
    if declared_receipt_id != observed_receipt_id:
        _reject("source receipt content identity does not verify")
    return receipt


def receipt_marker(receipt_id: str, issue_number: str) -> str:
    return f"<!-- deform360-v6-source-receipt:{receipt_id}:issue-{issue_number} -->"


def receipt_comment_body(
    receipt: Mapping[str, Any],
    *,
    repository: str,
    issue_number: str,
    source_run_id: str,
    source_run_attempt: str,
    source_head_sha: str,
    source_workflow_conclusion: str,
) -> str:
    receipt_id = cast(str, receipt["receipt_id"])
    status = cast(str, receipt["status"])
    marker = receipt_marker(receipt_id, issue_number)
    run_url = f"https://github.com/{repository}/actions/runs/{source_run_id}"
    return f"""{marker}
## Deform360 v6 source receipt routed to the current gate: `{status}`

- source workflow run: [`{source_run_id}`]({run_url}), attempt `{source_run_attempt}`
- source workflow conclusion: `{source_workflow_conclusion}`
- exact protected-main revision: `{source_head_sha}`
- receipt ID: `{receipt_id}`
- terminal stage: `{receipt.get("terminal_stage")}`
- exit code: `{receipt.get("exit_code")}`
- physical manifests: `{receipt["physical_manifest_count"]}/10`
- sealed source predictions: `{receipt["source_prediction_seal_count"]}/100`
- development suffix opened: `false`
- v5 confirmation payloads opened: `false`
- v6 fresh target selected: `false`
- v6 target payloads opened: `false`
- v6 target outcomes used: `false`
- replacement allowed: `false`

This is an independently verified mirror of the compact source receipt produced by the frozen dual-runtime workflow. The artifact manifest, receipt content identity, amendment, runner, triggering revision, run identity, terminal semantics, and closed information/authorization boundary all verified before routing. It does not score the development suffix, authorize confirmation or target access, alter the frozen candidate, or establish a scientific claim.
"""


class GitHubIssueComments:
    """Minimal issue-comment client for one fixed trusted repository."""

    def __init__(self, *, repository: str, issue_number: str, token: str) -> None:
        if repository != EXPECTED_REPOSITORY:
            _reject("receipt router repository identity changed")
        if not issue_number.isdecimal() or int(issue_number) < 1:
            _reject("receipt router issue number is invalid")
        if not token:
            _reject("receipt router has no GitHub token")
        self._comments_url = (
            f"https://api.github.com/repos/{repository}/issues/{issue_number}/comments"
        )
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def contains(self, marker: str) -> bool:
        page = 1
        while True:
            query = urllib.parse.urlencode({"per_page": 100, "page": page})
            request = urllib.request.Request(
                f"{self._comments_url}?{query}",
                headers=self._headers,
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                raw_comments = json.load(response)
            if not isinstance(raw_comments, list):
                _reject("GitHub issue comments response is not a list")
            comments = cast(list[dict[str, Any]], raw_comments)
            if any(marker in str(comment.get("body", "")) for comment in comments):
                return True
            if len(comments) < 100:
                return False
            page += 1

    def post(self, body: str) -> None:
        request = urllib.request.Request(
            self._comments_url,
            data=json.dumps({"body": body}).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 201:
                _reject(f"unexpected issue API status: {response.status}")


def main() -> int:
    repository = _required_environment("GITHUB_REPOSITORY")
    issue_number = _required_environment("ISSUE_NUMBER")
    source_run_id = _required_environment("SOURCE_RUN_ID")
    source_run_attempt = _required_environment("SOURCE_RUN_ATTEMPT")
    source_head_sha = _required_environment("SOURCE_HEAD_SHA")
    source_workflow_conclusion = _required_environment("SOURCE_WORKFLOW_CONCLUSION")
    receipt_root = Path(_required_environment("RECEIPT_ROOT"))
    token = _required_environment("GITHUB_TOKEN")

    receipt = verify_source_receipt_bundle(
        receipt_root,
        source_run_id=source_run_id,
        source_run_attempt=source_run_attempt,
        source_head_sha=source_head_sha,
        source_workflow_conclusion=source_workflow_conclusion,
    )
    receipt_id = cast(str, receipt["receipt_id"])
    marker = receipt_marker(receipt_id, issue_number)
    comments = GitHubIssueComments(
        repository=repository,
        issue_number=issue_number,
        token=token,
    )
    if comments.contains(marker):
        print("bounded source receipt already routed")
        return 0
    comments.post(
        receipt_comment_body(
            receipt,
            repository=repository,
            issue_number=issue_number,
            source_run_id=source_run_id,
            source_run_attempt=source_run_attempt,
            source_head_sha=source_head_sha,
            source_workflow_conclusion=source_workflow_conclusion,
        )
    )
    print(
        json.dumps(
            {
                "issue_number": issue_number,
                "receipt_id": receipt_id,
                "routed": True,
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
