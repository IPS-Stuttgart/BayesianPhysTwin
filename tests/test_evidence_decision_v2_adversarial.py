from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_evidence_decision_v2 import _decision, _manifest, _metric

import bayesian_phystwin.evidence_decision_v2 as evidence_decision_v2_module
from bayesian_phystwin.evidence_decision import load_evidence_decision
from bayesian_phystwin.evidence_decision_v2 import (
    EVIDENCE_DECISION_SCHEMA,
    EvidenceDecisionV2,
    build_evidence_decision_v2,
    load_evidence_decision_v2,
    write_evidence_decision_v2,
)


def _build(manifest: Any, summary: Path, *, claim_id: str = "bpt.physical.guard"):
    return build_evidence_decision_v2(
        manifest=manifest,
        evidence_summary_path=summary,
        claim_id=claim_id,
        execution_status="completed",
        scientific_decision="negative",
        authorization="stop",
        evidence_level=3,
        metric=_metric(),
        created_utc="2026-08-12T12:00:00+00:00",
    )


def test_dispatcher_rejects_unknown_schema_and_version(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    path.write_text(
        json.dumps({"schema_name": "unknown", "schema_version": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported evidence-decision schema"):
        load_evidence_decision(path)

    path.write_text(
        json.dumps(
            {
                "schema_name": EVIDENCE_DECISION_SCHEMA,
                "schema_version": 3,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported evidence-decision schema version"):
        load_evidence_decision(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"execution_status": None}, "unsupported evidence execution status"),
        ({"scientific_decision": None}, "unsupported scientific decision"),
        ({"authorization": None}, "unsupported authorization decision"),
        ({"run_classification": None}, "unsupported run classification"),
        ({"evidence_level": 4}, "evidence_level must be one of"),
    ],
)
def test_decision_rejects_invalid_axis_values(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _decision(**changes)


def test_decision_rejects_malformed_repository_collections() -> None:
    repositories = _decision().repositories

    with pytest.raises(ValueError, match="must contain RepositoryState"):
        _decision(repositories=(object(),))

    without_primary = tuple(
        replace(state, role="downstream")
        if state.role == "primary"
        else state
        for state in repositories
    )
    with pytest.raises(ValueError, match="exactly one primary"):
        _decision(repositories=without_primary)

    duplicate_name = replace(
        repositories[1],
        repository=repositories[0].repository,
    )
    with pytest.raises(ValueError, match="repository names must be unique"):
        _decision(repositories=(repositories[0], duplicate_name, repositories[2]))


def test_decision_rejects_malformed_limitations_and_metric() -> None:
    with pytest.raises(ValueError, match="sequence of strings"):
        _decision(limitations="not-a-sequence")

    with pytest.raises(ValueError, match="limitations must be unique"):
        _decision(limitations=("same", "same"))

    with pytest.raises(ValueError, match="require a DecisionMetricV2"):
        _decision(metric=None)


def test_noncompleted_execution_requires_stop_and_a_limitation() -> None:
    common = {
        "execution_status": "infrastructure_failure",
        "scientific_decision": "not_evaluated",
        "metric": None,
    }
    with pytest.raises(ValueError, match="must stop authorization"):
        _decision(
            **common,
            authorization="advance",
            limitations=("runner failed",),
        )

    with pytest.raises(ValueError, match="at least one limitation"):
        _decision(
            **common,
            authorization="stop",
            limitations=(),
        )


def test_passing_result_may_stop_with_an_explicit_policy_limitation() -> None:
    decision = _decision(
        authorization="stop",
        limitations=("independent confirmation is still pending",),
    )

    assert decision.authorization == "stop"


def test_builder_rejects_wrong_manifest_claim_and_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    manifest = _manifest(summary)

    with pytest.raises(TypeError, match="manifest must be a RunManifestV2"):
        _build(object(), summary)

    with pytest.raises(ValueError, match="claim_id is not declared"):
        _build(manifest, summary, claim_id="bpt.undeclared")

    with pytest.raises(ValueError, match="regular non-symlink file"):
        _build(manifest, tmp_path / "missing.json")

    summary.write_text('{"different":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not an exact output artifact"):
        _build(manifest, summary)


def test_builder_detects_summary_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    manifest = _manifest(summary)
    original_sha256_file = evidence_decision_v2_module.sha256_file

    def hash_then_mutate(path: str | Path) -> str:
        digest = original_sha256_file(path)
        Path(path).write_text('{"changed":true}\n', encoding="utf-8")
        return digest

    monkeypatch.setattr(
        evidence_decision_v2_module,
        "sha256_file",
        hash_then_mutate,
    )
    with pytest.raises(ValueError, match="changed while it was being hashed"):
        _build(manifest, summary)


def test_writer_validates_type_and_supports_explicit_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decision.json"
    with pytest.raises(TypeError, match="decision must be an EvidenceDecisionV2"):
        write_evidence_decision_v2(path, object())  # type: ignore[arg-type]

    write_evidence_decision_v2(path, _decision())
    replacement = _decision(metadata={"replacement": True})
    write_evidence_decision_v2(path, replacement, overwrite=True)

    assert load_evidence_decision_v2(path) == replacement


def test_v2_loader_rejects_schema_and_version_drift(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    decision = _decision()
    payload = decision.as_dict()
    payload["schema_name"] = "unknown"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported evidence-decision schema"):
        load_evidence_decision_v2(path)

    payload = decision.as_dict()
    payload["schema_version"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported evidence-decision schema version"):
        load_evidence_decision_v2(path)


def test_adversarial_helpers_return_v2_records(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")

    assert isinstance(_build(_manifest(summary), summary), EvidenceDecisionV2)
