from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import pytest

from bayesian_phystwin.evidence_decision import load_evidence_decision
from bayesian_phystwin.evidence_decision_v1 import (
    DecisionMetricV1,
    EvidenceDecisionV1,
    write_evidence_decision,
)
from bayesian_phystwin.evidence_decision_v2 import (
    EVIDENCE_DECISION_SCHEMA,
    EVIDENCE_DECISION_SCHEMA_VERSION,
    DecisionMetricV2,
    EvidenceDecisionV2,
    build_evidence_decision_v2,
    load_evidence_decision_v2,
    write_evidence_decision_v2,
)
from bayesian_phystwin.repository_provenance import RepositoryState
from bayesian_phystwin.run_manifest import ArtifactDigest
from bayesian_phystwin.run_manifest_v2 import RunManifestV2


def _metric() -> DecisionMetricV2:
    return DecisionMetricV2(
        name="future_track_error_mm",
        comparison="action_discrepancy_vs_nominal",
        rule="relative_improvement_gt_0",
        observed_value=13.52,
        threshold_value=0.0,
        unit="percent",
    )


def _repositories() -> tuple[RepositoryState, ...]:
    return (
        RepositoryState(
            repository="IPS-Stuttgart/BayesianPhysTwin",
            revision="d" * 40,
            dirty=False,
            role="primary",
        ),
        RepositoryState(
            repository="IPS-Stuttgart/Prob4D",
            revision="e" * 40,
            dirty=False,
            role="observation",
        ),
        RepositoryState(
            repository="IPS-Stuttgart/Causal4D",
            revision="f" * 40,
            dirty=False,
            role="downstream",
        ),
    )


def _decision(**changes: object) -> EvidenceDecisionV2:
    values: dict[str, object] = {
        "claim_id": "bpt.physical.guard",
        "protocol_id": "deform360-independent-object-v1",
        "execution_status": "completed",
        "scientific_decision": "pass",
        "authorization": "advance",
        "run_classification": "confirmatory",
        "evidence_level": 3,
        "metric": _metric(),
        "run_manifest_id": "a" * 64,
        "evidence_fingerprint": "b" * 64,
        "evidence_summary_sha256": "c" * 64,
        "repositories": _repositories(),
        "limitations": (),
        "metadata": {"profile": "publication", "seed_count": 5},
        "created_utc": "2026-08-12T12:00:00+00:00",
    }
    values.update(changes)
    return EvidenceDecisionV2(**values)  # type: ignore[arg-type]


def _manifest(summary: Path, **changes: object) -> RunManifestV2:
    payload = summary.read_bytes()
    values: dict[str, object] = {
        "run_id": "physical-lane-v2",
        "repository": "IPS-Stuttgart/BayesianPhysTwin",
        "revision": "d" * 40,
        "dirty": False,
        "command": ("bpt", "experiment", "run", "physical-lane-v2"),
        "classification": "confirmatory",
        "statistical_unit": "episode",
        "information_boundary": {"held_out_object": True},
        "configuration": {"profile": "publication"},
        "outputs": (
            ArtifactDigest(
                name="evidence_summary",
                role="output",
                path="summary.json",
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            ),
        ),
        "related_repositories": _repositories()[1:],
        "claim_ids": ("bpt.physical.guard",),
        "method_freeze_id": "physical-lane-method-v2",
        "protocol_id": "deform360-independent-object-v1",
        "split_id": "leave-one-object-out-v1",
        "baseline_id": "nominal-physics-v1",
        "created_utc": "2026-08-12T11:00:00+00:00",
    }
    values.update(changes)
    return RunManifestV2(**values)  # type: ignore[arg-type]


def _v1_decision() -> EvidenceDecisionV1:
    return EvidenceDecisionV1(
        claim_id="bpt.legacy",
        protocol_id="legacy-v1",
        status="fail",
        run_classification="confirmatory",
        claim_authorized=False,
        evidence_level=2,
        metric=DecisionMetricV1(
            name="nll",
            comparison="candidate_vs_fallback",
            rule="delta_lt_0",
            observed_value=1.0,
            threshold_value=0.0,
            unit="nat",
        ),
        run_manifest_id="1" * 64,
        evidence_fingerprint="2" * 64,
        evidence_summary_sha256="3" * 64,
        repositories=(_repositories()[0],),
        created_utc="2026-08-12T10:00:00+00:00",
    )


def test_v2_round_trip_and_version_dispatch(tmp_path: Path) -> None:
    decision = _decision()
    path = tmp_path / "decision-v2.json"
    write_evidence_decision_v2(path, decision)

    assert load_evidence_decision_v2(path) == decision
    assert load_evidence_decision(path) == decision
    assert decision.as_dict()["schema_name"] == EVIDENCE_DECISION_SCHEMA
    assert decision.as_dict()["schema_version"] == EVIDENCE_DECISION_SCHEMA_VERSION


def test_version_dispatch_keeps_v1_loadable(tmp_path: Path) -> None:
    decision = _v1_decision()
    path = tmp_path / "decision-v1.json"
    write_evidence_decision(path, decision)

    assert load_evidence_decision(path) == decision


def test_completed_scientific_negative_is_not_an_execution_failure() -> None:
    decision = _decision(
        scientific_decision="negative",
        authorization="stop",
    )

    assert decision.execution_status == "completed"
    assert decision.scientific_decision == "negative"
    assert decision.authorization == "stop"
    assert decision.metric == _metric()


def test_completed_non_scientific_run_is_explicitly_not_evaluated() -> None:
    decision = _decision(
        scientific_decision="not_evaluated",
        authorization="stop",
        run_classification="infrastructure",
        metric=None,
        limitations=("environment preflight does not evaluate the claim",),
    )

    assert decision.execution_status == "completed"
    assert decision.scientific_decision == "not_evaluated"
    assert decision.metric is None


@pytest.mark.parametrize(
    "execution_status",
    ["infrastructure_failure", "protocol_invalid"],
)
def test_noncompleted_execution_is_not_a_scientific_negative(
    execution_status: str,
) -> None:
    decision = _decision(
        execution_status=execution_status,
        scientific_decision="not_evaluated",
        authorization="stop",
        metric=None,
        limitations=("registered endpoint was not validly evaluated",),
    )

    assert decision.scientific_decision == "not_evaluated"
    assert decision.authorization == "stop"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "execution_status": "infrastructure_failure",
                "scientific_decision": "pass",
                "authorization": "stop",
                "limitations": ("runner failed",),
            },
            "cannot record a scientific result",
        ),
        (
            {
                "execution_status": "protocol_invalid",
                "scientific_decision": "not_evaluated",
                "authorization": "stop",
                "limitations": ("split was invalid",),
            },
            "must not record a metric",
        ),
        (
            {
                "scientific_decision": "negative",
                "authorization": "advance",
            },
            "must stop authorization",
        ),
        (
            {
                "scientific_decision": "not_evaluated",
                "authorization": "stop",
                "metric": None,
            },
            "at least one limitation",
        ),
        (
            {
                "scientific_decision": "pass",
                "authorization": "stop",
            },
            "stopped by policy",
        ),
        (
            {
                "authorization": "advance",
                "run_classification": "exploratory",
            },
            "confirmatory",
        ),
    ],
)
def test_cross_field_inconsistencies_fail_closed(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _decision(**changes)


def test_advancement_rejects_dirty_repository() -> None:
    repositories = _repositories()
    dirty_primary = replace(repositories[0], dirty=True)

    with pytest.raises(ValueError, match="dirty repository"):
        _decision(repositories=(dirty_primary, *repositories[1:]))


def test_builder_binds_manifest_and_summary_for_scientific_negative(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text('{"scientific_decision":"negative"}\n', encoding="utf-8")
    manifest = _manifest(summary)

    decision = build_evidence_decision_v2(
        manifest=manifest,
        evidence_summary_path=summary,
        claim_id="bpt.physical.guard",
        execution_status="completed",
        scientific_decision="negative",
        authorization="stop",
        evidence_level=3,
        metric=_metric(),
        created_utc="2026-08-12T12:00:00+00:00",
    )

    assert decision.run_manifest_id == manifest.manifest_id
    assert decision.evidence_fingerprint == manifest.evidence_fingerprint
    assert decision.evidence_summary_sha256 == manifest.outputs[0].sha256
    assert [state.role for state in decision.repositories] == [
        "primary",
        "downstream",
        "observation",
    ]


def test_loader_rejects_digest_and_outcome_drift(tmp_path: Path) -> None:
    decision = _decision()
    path = tmp_path / "decision.json"
    write_evidence_decision_v2(path, decision)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["claim_id"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest does not match"):
        load_evidence_decision_v2(path)

    payload = decision.as_dict()
    payload["execution_status"] = "unknown"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported evidence execution status"):
        load_evidence_decision_v2(path)


def test_writer_refuses_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    write_evidence_decision_v2(path, _decision())

    with pytest.raises(FileExistsError, match="already exists"):
        write_evidence_decision_v2(path, _decision())


def test_packaged_v2_schema_matches_closed_python_wire_shape() -> None:
    schema_path = files("bayesian_phystwin").joinpath(
        "contract_data",
        "evidence_decision_v2",
        "evidence-decision-v2.schema.json",
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    python_fields = {
        "decision_id",
        "schema_name",
        "schema_version",
        *EvidenceDecisionV2.__dataclass_fields__,
    }

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == python_fields
    assert set(schema["properties"]) == python_fields
    assert schema["properties"]["schema_name"]["const"] == (
        "bayesian_phystwin.evidence_decision"
    )
    assert schema["properties"]["schema_version"]["const"] == 2
    assert len(schema["allOf"]) >= 6
