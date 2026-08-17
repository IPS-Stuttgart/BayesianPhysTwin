from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/revalidate-deform360-prob4d-visible-source-v2.yml")


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_is_hosted_one_shot_and_exact_source_bound() -> None:
    text = _text()
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(document, dict)
    assert "pull_request:" in text
    assert "push:" in text
    assert "workflow_dispatch:" not in text
    assert 'SOURCE_RUN_ID: "31301431579"' in text
    assert 'SOURCE_RUN_ATTEMPT: "1"' in text
    assert "SOURCE_RUN_CONCLUSION: failure" in text
    assert "SOURCE_HEAD_SHA: 136f72b996e9" in text
    assert 'SOURCE_ARTIFACT_ID: "9034737368"' in text
    assert "SOURCE_ARTIFACT_DIGEST: sha256:caa8d5ea887e" in text
    assert "target-free-visible source gate v2 once" in text
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: self-hosted" not in text
    assert "issues: write" not in text
    assert "/mnt/" not in text
    assert "held-v8" not in text


def test_workflow_runs_contracts_before_exact_compact_audit() -> None:
    text = _text()

    verify = text.index("Verify exact source run and compact artifact metadata")
    download = text.index("Download only the exact compact source artifact")
    validate = text.index("Validate and freeze the compact technical terminal")
    upload = text.index("Upload independent compact audit")
    enforce = text.index("Enforce the closed technical terminal")
    assert verify < download < validate < upload < enforce
    assert "needs: contracts" in text
    assert "github.event_name == 'push'" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "audit_deform360_prob4d_visible_source_v2.py" in text
    assert "validated-source-sample-materialization-failure" in text
    assert "source-pipeline-technical-terminal" in text
    assert ".source_gate_evaluated" in text
    assert ".confirmation_access_authorized" in text
    assert "retention-days: 180" in text


def test_workflow_push_trigger_is_limited_to_the_auditor_itself() -> None:
    document = yaml.load(_text(), Loader=yaml.BaseLoader)
    trigger = document["on"]

    assert trigger["push"] == {
        "branches": ["main"],
        "paths": [
            ".github/workflows/revalidate-deform360-prob4d-visible-source-v2.yml"
        ],
    }
