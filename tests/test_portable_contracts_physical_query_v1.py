from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.physical_query_v1 import (
    COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
    MARGINAL_GAUGE_COVARIANCE,
    PHYSICAL_QUERY_CLAIM_BOUNDARY,
    PHYSICAL_QUERY_SCHEMA,
    PHYSICAL_QUERY_VERSION,
    PhysicalQueryBootstrapV1,
    PhysicalQueryDecisionMarginsV1,
    PhysicalQueryV1,
    load_physical_query,
    write_physical_query,
)
from bayesian_phystwin.probabilistic_scoring import GAUSSIAN_NLL_PER_DIMENSION
from bayesian_phystwin.repository_provenance import RepositoryState


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _revision(character: str) -> str:
    return character * 40


def _margins() -> PhysicalQueryDecisionMarginsV1:
    return PhysicalQueryDecisionMarginsV1(
        practical_equivalence_score=0.001,
        maximum_harmful_score_increase=0.005,
        minimum_accepted_coverage=0.85,
        maximum_mean_width=0.050,
        maximum_worst_group_score_regret=0.010,
        minimum_shared_covariance_relevance=0.05,
        width_unit="m",
    )


def _bootstrap() -> PhysicalQueryBootstrapV1:
    return PhysicalQueryBootstrapV1(
        independent_group_definition="complete physical object/session",
        method="paired-stratified-group-bootstrap",
        resamples=10_000,
        seed=1731,
        confidence_level=0.95,
        stratification_keys=("object-stratum", "action-family"),
    )


def _repositories() -> tuple[RepositoryState, ...]:
    return (
        RepositoryState(
            repository="IPS-Stuttgart/BayesianPhysTwin",
            revision=_revision("a"),
            dirty=False,
            role="primary",
        ),
        RepositoryState(
            repository="IPS-Stuttgart/Prob4D",
            revision=_revision("b"),
            dirty=False,
            role="observation",
        ),
        RepositoryState(
            repository="IPS-Stuttgart/Causal4D",
            revision=_revision("c"),
            dirty=False,
            role="downstream",
        ),
    )


def _query(**overrides: object) -> PhysicalQueryV1:
    values: dict[str, object] = {
        "query_name": "fresh-provider-endpoint-displacement",
        "dimension": 6,
        "component_order": (
            "early-x",
            "early-y",
            "early-z",
            "late-x",
            "late-y",
            "late-z",
        ),
        "physical_unit": "m",
        "coordinate_frame": "registered-world-frame",
        "horizon_values": (0.08, 0.20),
        "horizon_unit": "s",
        "jacobian_provider_id": _sha256("jacobian-provider"),
        "baseline_physical_belief_id": _sha256("physical-belief"),
        "exact_fallback_id": _sha256("fallback-bytes"),
        "covariance_treatments": (
            MARGINAL_GAUGE_COVARIANCE,
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
        ),
        "principal_covariance_treatment": (COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE),
        "primary_proper_score": GAUSSIAN_NLL_PER_DIMENSION,
        "decision_margins": _margins(),
        "shared_covariance_diagnostic": (
            "query trace fraction attributable to shared gauge covariance"
        ),
        "computational_selection_rule": (
            "use complete explicit joint gauge unless the frozen relevance "
            "diagnostic is below its source-only threshold"
        ),
        "bootstrap": _bootstrap(),
        "package_artifact_ids": {
            "bayesian-phystwin": _sha256("bpt-wheel"),
            "prob4d": _sha256("prob4d-wheel"),
            "causal4d": _sha256("causal4d-wheel"),
        },
        "provider_manifest_id": _sha256("provider-manifest"),
        "evidence_decision_ids": {
            "source-provider-gate": _sha256("source-decision"),
        },
        "repositories": _repositories(),
        "metadata": {"study": "fresh-provider-v1", "target_opened": False},
    }
    values.update(overrides)
    return PhysicalQueryV1(**values)  # type: ignore[arg-type]


def test_physical_query_roundtrip_and_no_clobber(tmp_path: Path) -> None:
    query = _query()
    path = tmp_path / "physical-query.json"

    write_physical_query(query, path)
    loaded = load_physical_query(path)

    assert loaded.query_id == query.query_id
    assert loaded.to_record() == query.to_record()
    assert loaded.to_record()["schema"] == PHYSICAL_QUERY_SCHEMA
    assert loaded.to_record()["schema_version"] == PHYSICAL_QUERY_VERSION
    assert loaded.to_record()["claim_boundary"] == PHYSICAL_QUERY_CLAIM_BOUNDARY
    assert loaded.repositories[0].repository == "IPS-Stuttgart/BayesianPhysTwin"
    assert loaded.covariance_treatments == tuple(
        sorted(
            (
                MARGINAL_GAUGE_COVARIANCE,
                COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
            )
        )
    )

    with pytest.raises(FileExistsError):
        write_physical_query(query, path)


def test_query_identity_covers_decision_margins() -> None:
    original = _query()
    changed = _query(
        decision_margins=replace(
            _margins(),
            maximum_mean_width=0.060,
        )
    )

    assert original.query_id != changed.query_id


@pytest.mark.parametrize(
    "treatments",
    (
        (MARGINAL_GAUGE_COVARIANCE,),
        (COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,),
    ),
)
def test_query_requires_both_registered_gauge_covariance_variants(
    treatments: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="missing required variants"):
        _query(
            covariance_treatments=treatments,
            principal_covariance_treatment=treatments[0],
        )


def test_query_requires_principal_treatment_and_registered_proper_score() -> None:
    with pytest.raises(ValueError, match="principal_covariance_treatment"):
        _query(principal_covariance_treatment="not-registered")

    with pytest.raises(ValueError, match="registered proper score"):
        _query(primary_proper_score="root-mean-square-error")


def test_query_requires_exact_package_and_repository_ownership() -> None:
    with pytest.raises(ValueError, match="missing required bindings"):
        _query(package_artifact_ids={"bayesian-phystwin": _sha256("wheel")})

    dirty = list(_repositories())
    dirty[1] = replace(dirty[1], dirty=True)
    with pytest.raises(ValueError, match="cannot bind dirty repositories"):
        _query(repositories=tuple(dirty))

    wrong_observation = list(_repositories())
    wrong_observation[1] = replace(wrong_observation[1], role="dependency")
    with pytest.raises(ValueError, match="bind Prob4D exactly once"):
        _query(repositories=tuple(wrong_observation))


def test_query_rejects_invalid_dimension_horizons_and_width_unit() -> None:
    with pytest.raises(ValueError, match="component_order length"):
        _query(dimension=5)

    with pytest.raises(ValueError, match="strictly increasing"):
        _query(horizon_values=(0.20, 0.08))

    with pytest.raises(ValueError, match="width unit"):
        _query(decision_margins=replace(_margins(), width_unit="mm"))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("minimum_accepted_coverage", 1.0),
        ("maximum_mean_width", 0.0),
        ("minimum_shared_covariance_relevance", 1.1),
        ("practical_equivalence_score", True),
    ),
)
def test_query_rejects_invalid_decision_margins(
    field_name: str,
    invalid_value: object,
) -> None:
    record = _margins().descriptor()
    record[field_name] = invalid_value

    with pytest.raises(ValueError):
        PhysicalQueryDecisionMarginsV1.from_mapping(record)


def test_bootstrap_rejects_non_group_or_adaptive_settings() -> None:
    with pytest.raises(ValueError, match="not registered"):
        replace(_bootstrap(), method="frame-bootstrap")

    with pytest.raises(ValueError, match="resamples must be an integer"):
        replace(_bootstrap(), resamples=True)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        replace(_bootstrap(), stratification_keys=("object", "object"))


def test_record_rejects_boundary_schema_and_content_id_tampering() -> None:
    query = _query()

    boundary_record = query.to_record()
    boundary = boundary_record["information_boundary"]
    assert isinstance(boundary, dict)
    boundary["target_outcomes_opened"] = 0
    with pytest.raises(ValueError, match="information boundary changed"):
        PhysicalQueryV1.from_mapping(boundary_record)

    version_record = query.to_record()
    version_record["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        PhysicalQueryV1.from_mapping(version_record)

    identity_record = query.to_record()
    identity_record["query_id"] = _sha256("different-query")
    with pytest.raises(ValueError, match="query_id does not match"):
        PhysicalQueryV1.from_mapping(identity_record)

    unknown_record = query.to_record()
    unknown_record["unknown"] = "field"
    with pytest.raises(ValueError, match="fields changed"):
        PhysicalQueryV1.from_mapping(unknown_record)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"query_id":"a","query_id":"b"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_physical_query(path)


def test_metadata_is_detached_and_immutable() -> None:
    metadata = {"nested": {"state": "source-only"}}
    query = _query(metadata=metadata)
    metadata["nested"]["state"] = "target-opened"

    assert query.metadata["nested"]["state"] == "source-only"
    with pytest.raises(TypeError, match="immutable"):
        query.metadata["new"] = "value"  # type: ignore[index]
