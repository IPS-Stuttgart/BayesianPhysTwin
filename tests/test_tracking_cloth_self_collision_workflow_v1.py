from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(
    ".github/workflows/tracking-cloth-self-collision-stable-bank-source-v1.yml"
)


def test_stable_bank_workflow_is_source_only_and_runner_bound() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in text
    assert "--stage source" in text
    assert "--stage predict" not in text
    assert "--stage score" not in text
    assert '"source_only": True' in text
    assert '"automatic_target_follow_on": False' in text
    assert '"paper_claim_authorized": False' in text
    assert "rep3_numeric_outcomes_read" in text
    assert "if: always()" in text


def test_request_is_exactly_add_only_and_source_revision_bound() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'test "${#changed[@]}" -eq 1' in text
    assert "$'A\\t.github/requests/" in text
    assert 'git", "rev-parse", f"{os.environ[\'GITHUB_SHA\']}^"' in text
    assert "request is not bound to its exact source revision" in text


def test_stable_bank_protocol_identity_is_bound_in_both_jobs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    expected = "3a24f07b8303a209cf9edc0858bfb460f72ebe4e"
    assert text.count(expected) >= 2
    assert text.count('git hash-object "${PROTOCOL_PATH}"') >= 2
