from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.evidence_decision_v1 import (
    EVIDENCE_DECISION_SCHEMA,
    EVIDENCE_DECISION_SCHEMA_VERSION,
    DecisionMetricV1,
    EvidenceDecisionV1,
    build_evidence_decision,
    load_evidence_decision,
    write_evidence_decision,
)
from bayesian_phystwin.repository_provenance import RepositoryState
from bayesian_phystwin.run_manifest import ArtifactDigest
from bayesian_phystwin.run_manifest_v2 import RunManifestV2


def _metric() -> DecisionMetricV1:
    return DecisionMetricV1(
        name="future_track_error_mm",
        comparison="action_discrepancy_vs_nominal",
        rule="relative_improvement_gt_0",
        observed_value=13.52,
        threshold_value=0.0,
        unit="percent",
    )


def _decision(**changes: object) -> EvidenceDecisionV1:
    values: dict[str, object] = {
        "claim_id": "bpt.physical.guard",
        "protocol_id": "deform360-independent-object-v1",
        "status": "pass",
        "run_classification": "confirmatory",
        "claim_authorized": True,
        "evidence_level": 3,
        "metric": _metric(),
        "run_manifest_id": "a" * 64,
        "evidence_fingerprint": "b" * 64,
        "evidence_summary_sha256": "c" * 64,
        "repositories": (
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
        ),
        "limitations": ("independent-object confirmation only",),
        "metadata": {"profile": "publication", "seed_count": 5},
        "created_utc": "2026-08-10T12:00:00+00:00",
    }
    values.update(changes)
    return EvidenceDecisionV1(**values)  # type: ignore[arg-type]


def _manifest(summary: Path, **changes: object) -> RunManifestV2:
    payload = summary.read_bytes()
    values: dict[str, object] = {
        "run_id": "physical-lane-v1",
        "repository": "IPS-Stuttgart/BayesianPhysTwin",
        "revision": "d" * 40,
        "dirty": False,
        "command": ("bpt", "experiment", "run", "physical-lane-v1"),
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
        "related_repositories": (
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
        ),
        "claim_ids": ("bpt.physical.guard",),
        "method_freeze_id": "physical-lane-method-v1",
        "protocol_id": "deform360-independent-object-v1",
        "split_id": "leave-one-object-out-v1",
        "baseline_id": "nominal-physics-v1",
        "created_utc": "2026-08-10T11:00:00+00:00",
    }
    values.update(changes)
    return RunManifestV2(**values)  # type: ignore[arg-type]


def test_decision_round_trip_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    decision = _decision()
    path = tmp_path / "decision.json"
    write_evidence_decision(path, decision)

    loaded = load_evidence_decision(path)

    assert loaded == decision
    assert loaded.decision_id == decision.decision_id
    assert loaded.as_dict()["schema_name"] == EVIDENCE_DECISION_SCHEMA
    assert loaded.as_dict()["schema_version"] == EVIDENCE_DECISION_SCHEMA_VERSION
    with pytest.raises(TypeError, match="immutable"):
        loaded.metadata["profile"] = "changed"  # type: ignore[index]


def test_builder_derives_manifest_and_summary_identities(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"pass"}\n', encoding="utf-8")
    manifest = _manifest(summary)

    decision = build_evidence_decision(
        manifest=manifest,
        evidence_summary_path=summary,
        claim_id="bpt.physical.guard",
        status="pass",
        claim_authorized=True,
        evidence_level=3,
        metric=_metric(),
        created_utc="2026-08-10T12:00:00+00:00",
    )

    assert decision.protocol_id == manifest.protocol_id
    assert decision.run_manifest_id == manifest.manifest_id
    assert decision.evidence_fingerprint == manifest.evidence_fingerprint
    assert decision.evidence_summary_sha256 == manifest.outputs[0].sha256
    assert [state.role for state in decision.repositories] == [
        "primary",
        "downstream",
        "observation",
    ]


def test_builder_fails_closed_for_unbound_or_nonconfirmatory_evidence(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"pass"}\n', encoding="utf-8")
    manifest = _manifest(summary)

    unbound = tmp_path / "unbound.json"
    unbound.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an exact output artifact"):
        build_evidence_decision(
            manifest=manifest,
            evidence_summary_path=unbound,
            claim_id="bpt.physical.guard",
            status="pass",
            claim_authorized=False,
            evidence_level=3,
            metric=_metric(),
        )

    with pytest.raises(ValueError, match="not declared"):
        build_evidence_decision(
            manifest=manifest,
            evidence_summary_path=summary,
            claim_id="bpt.other",
            status="pass",
            claim_authorized=False,
            evidence_level=3,
            metric=_metric(),
        )

    exploratory = _manifest(summary, classification="exploratory")
    with pytest.raises(ValueError, match="confirmatory"):
        build_evidence_decision(
            manifest=exploratory,
            evidence_summary_path=summary,
            claim_id="bpt.physical.guard",
            status="pass",
            claim_authorized=True,
            evidence_level=3,
            metric=_metric(),
        )


def test_loader_rejects_unknown_fields_duplicate_keys_and_digest_drift(
    tmp_path: Path,
) -> None:
    decision = _decision()
    path = tmp_path / "decision.json"
    write_evidence_decision(path, decision)
    payload = json.loads(path.read_text(encoding="utf-8"))

    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_evidence_decision(path)

    write_evidence_decision(path, decision, overwrite=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["claim_id"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        load_evidence_decision(path)

    path.write_text(
        (
            '{"decision_id":"'
            + decision.decision_id
            + '","decision_id":"'
            + decision.decision_id
            + '"}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_evidence_decision(path)


def test_decision_fails_closed_for_authorization_and_provenance() -> None:
    with pytest.raises(ValueError, match="passing decision"):
        _decision(status="fail")

    dirty = replace(_decision().repositories[0], dirty=True)
    with pytest.raises(ValueError, match="dirty repository"):
        _decision(repositories=(dirty, *_decision().repositories[1:]))

    with pytest.raises(ValueError, match="confirmatory"):
        _decision(run_classification="exploratory")

    with pytest.raises(ValueError, match="evidence_level"):
        _decision(evidence_level=4)

    with pytest.raises(ValueError, match="at least one limitation"):
        _decision(
            status="inconclusive",
            claim_authorized=False,
            limitations=(),
        )


def test_metric_rejects_nonfinite_values_and_schema_drift() -> None:
    with pytest.raises(ValueError, match="finite number"):
        DecisionMetricV1(
            name="metric",
            comparison="baseline",
            rule="smaller_is_better",
            observed_value=float("nan"),
            threshold_value=None,
            unit="mm",
        )

    metric = _decision().metric.as_dict()
    metric["direction"] = "minimize"
    with pytest.raises(ValueError, match="unknown"):
        DecisionMetricV1.from_mapping(metric)


def test_writer_refuses_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    write_evidence_decision(path, _decision())

    with pytest.raises(FileExistsError, match="already exists"):
        write_evidence_decision(path, _decision())

    replacement = _decision(claim_authorized=False)
    write_evidence_decision(path, replacement, overwrite=True)
    assert load_evidence_decision(path) == replacement
