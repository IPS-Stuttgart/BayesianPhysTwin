"""Static safety contract for trusted self-hosted exact-head validation."""

from __future__ import annotations

import re
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = (
    _REPOSITORY_ROOT / ".github" / "workflows" / "trusted-exact-head-validation.yml"
)
_TRIGGER = re.compile(r"(?m)^\s{2}workflow_dispatch\s*:\s*$")
_PULL_REQUEST_TRIGGER = re.compile(r"(?m)^\s{2}pull_request(?:_target)?\s*:\s*$")


def test_trusted_exact_head_workflow_is_dispatch_only_and_read_only() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")

    assert _TRIGGER.search(text) is not None
    assert _PULL_REQUEST_TRIGGER.search(text) is None
    assert "contents: read" in text
    assert "pull-requests: read" in text
    assert "contents: write" not in text
    assert "persist-credentials: false" in text
    assert "environment: trusted-self-hosted-validation" in text


def test_trusted_exact_head_workflow_verifies_and_checks_out_same_sha() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")

    assert "only same-repository pull requests are admitted" in text
    assert "pull request base must be main" in text
    assert "actual_head_sha != expected_head_sha" in text
    assert "ref: ${{ steps.pr.outputs.head_sha }}" in text
    assert 'test "$(git rev-parse HEAD)" = "${EXPECTED_HEAD_SHA}"' in text
    assert "git push" not in text
