from __future__ import annotations

from typing import cast

import pytest

import bayesian_phystwin.backend_portfolio_v1 as portfolio
from bayesian_phystwin.backend_portfolio_v1 import BackendEvidenceStageV1


def test_current_portfolio_is_frozen_and_within_budget() -> None:
    report = portfolio.describe_backend_portfolio()

    assert report["schema"] == "bayesian-phystwin.backend-portfolio"
    assert report["schema_version"] == 1
    assert report["admission_frozen"] is True
    assert report["new_family_admission_allowed"] is False
    assert report["maximum_active_qualification_candidates"] == 2
    assert report["active_qualification_candidates"] == [
        "jax-fem-quasistatic-v1",
        "genesis-mpm-v1",
    ]
    assert report["source_value_qualified_profiles"] == []


def test_portfolio_separates_implementation_and_evidence_maturity() -> None:
    report = portfolio.describe_backend_portfolio()
    profiles = {
        item["profile_id"]: item
        for item in cast(list[dict[str, object]], report["profiles"])
    }

    assert profiles["jax-fem-quasistatic-v1"] == {
        "profile_id": "jax-fem-quasistatic-v1",
        "implementation_maturity": "preferred",
        "evidence_stage": "native-smoke-passed",
        "active_qualification_candidate": True,
        "recommendation_authorized": False,
    }
    assert profiles["warp-fem-v1"]["implementation_maturity"] == "supported"
    assert profiles["warp-fem-v1"]["evidence_stage"] == "registered-adapter"
    assert profiles["warp-fem-v1"]["recommendation_authorized"] is False
    assert profiles["genesis-mpm-v1"]["evidence_stage"] == (
        "source-physics-qualified"
    )
    assert profiles["genesis-mpm-v1"]["recommendation_authorized"] is False


def test_stage_roster_must_match_the_canonical_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = dict(portfolio._EVIDENCE_STAGE_BY_PROFILE)
    stages.pop("drake-fem-v1")
    monkeypatch.setattr(portfolio, "_EVIDENCE_STAGE_BY_PROFILE", stages)

    with pytest.raises(RuntimeError, match="exactly one evidence stage"):
        portfolio.validate_backend_portfolio()


def test_new_family_is_rejected_while_admission_freeze_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = dict(portfolio.MATERIAL_BACKEND_SPECS)
    specs["new-backend-v1"] = next(iter(specs.values()))
    stages = dict(portfolio._EVIDENCE_STAGE_BY_PROFILE)
    stages["new-backend-v1"] = "registered-adapter"
    monkeypatch.setattr(portfolio, "MATERIAL_BACKEND_SPECS", specs)
    monkeypatch.setattr(portfolio, "_EVIDENCE_STAGE_BY_PROFILE", stages)

    with pytest.raises(RuntimeError, match="new backend family admitted"):
        portfolio.validate_backend_portfolio()


def test_duplicate_active_candidates_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        portfolio,
        "ACTIVE_QUALIFICATION_CANDIDATES",
        ("jax-fem-quasistatic-v1", "jax-fem-quasistatic-v1"),
    )

    with pytest.raises(RuntimeError, match="must be unique"):
        portfolio.validate_backend_portfolio()


def test_active_qualification_budget_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        portfolio,
        "ACTIVE_QUALIFICATION_CANDIDATES",
        (
            "jax-fem-quasistatic-v1",
            "genesis-mpm-v1",
            "warp-fem-v1",
        ),
    )

    with pytest.raises(RuntimeError, match="budget exceeded"):
        portfolio.validate_backend_portfolio()


def test_unregistered_active_candidate_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        portfolio,
        "ACTIVE_QUALIFICATION_CANDIDATES",
        ("jax-fem-quasistatic-v1", "unknown-backend-v1"),
    )

    with pytest.raises(RuntimeError, match="is not registered"):
        portfolio.validate_backend_portfolio()


def test_active_candidate_requires_native_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        portfolio,
        "ACTIVE_QUALIFICATION_CANDIDATES",
        ("jax-fem-quasistatic-v1", "warp-fem-v1"),
    )

    with pytest.raises(RuntimeError, match="pass its native smoke"):
        portfolio.validate_backend_portfolio()


def test_source_value_backend_must_leave_active_funnel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = dict(portfolio._EVIDENCE_STAGE_BY_PROFILE)
    stages["jax-fem-quasistatic-v1"] = "source-value-qualified"
    monkeypatch.setattr(portfolio, "_EVIDENCE_STAGE_BY_PROFILE", stages)

    with pytest.raises(RuntimeError, match="must leave the active source funnel"):
        portfolio.validate_backend_portfolio()


def test_source_value_result_lifts_family_admission_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = dict(portfolio.MATERIAL_BACKEND_SPECS)
    specs["new-backend-v1"] = next(iter(specs.values()))
    stages = dict(portfolio._EVIDENCE_STAGE_BY_PROFILE)
    stages["jax-fem-quasistatic-v1"] = "source-value-qualified"
    stages["new-backend-v1"] = "registered-adapter"
    monkeypatch.setattr(portfolio, "MATERIAL_BACKEND_SPECS", specs)
    monkeypatch.setattr(portfolio, "_EVIDENCE_STAGE_BY_PROFILE", stages)
    monkeypatch.setattr(
        portfolio,
        "ACTIVE_QUALIFICATION_CANDIDATES",
        ("genesis-mpm-v1",),
    )

    report = portfolio.validate_backend_portfolio()

    assert report["admission_frozen"] is False
    assert report["new_family_admission_allowed"] is True
    assert report["source_value_qualified_profiles"] == ["jax-fem-quasistatic-v1"]


@pytest.mark.parametrize("profile_id", ["", "unknown-backend-v1"])
def test_evidence_stage_lookup_rejects_unknown_profiles(profile_id: str) -> None:
    with pytest.raises(ValueError):
        portfolio.backend_evidence_stage(profile_id)


def test_evidence_stage_lookup_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        portfolio.backend_evidence_stage(7)  # type: ignore[arg-type]


def test_evidence_stage_type_surface() -> None:
    stage: BackendEvidenceStageV1 = portfolio.backend_evidence_stage("genesis-mpm-v1")
    assert stage == "source-physics-qualified"
