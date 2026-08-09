"""Security contracts for the hosted Deform360 bootstrap checkout."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/deform360-runner-local-contracts.yml")


def test_reusable_checkout_is_event_bound_and_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    event_expression = "${{ github.event.pull_request.head.sha || github.sha }}"
    assert f"ref: {event_expression}" in text
    assert "ref: ${{ inputs.source_sha }}" not in text
    assert f"EVENT_SOURCE_SHA: {event_expression}" in text
    assert "DECLARED_SOURCE_SHA: ${{ inputs.source_sha }}" in text
    assert 'test "${DECLARED_SOURCE_SHA}" = "${EVENT_SOURCE_SHA}"' in text
    assert "permissions:\n  contents: read" in text
    assert "persist-credentials: false" in text
    assert "contents: write" not in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
