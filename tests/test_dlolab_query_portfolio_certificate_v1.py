"""Evidence contracts for the DLO-Lab simultaneous query portfolio."""

from __future__ import annotations

from pathlib import Path

import pytest

from bayesian_phystwin.query_portfolio_certificate_v1 import (
    load_query_portfolio_certificate,
)
from scripts.build_dlolab_query_portfolio_certificate_v1 import build_certificate

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    ROOT / "results/source/dlolab_query_portfolio_certificate_v1/certificate.json"
)
WRAPPING_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-risk-certified-guard-source-v9"
)
SLINGSHOT_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v4"
)


def test_committed_portfolio_certificate_has_complete_accounting() -> None:
    certificate = load_query_portfolio_certificate(CERTIFICATE)
    assert certificate.artifact_id == (
        "4af4f4aa127fdd9c53f103502e4593f87f6a3c6ad36d49a93e0406502b58da8a"
    )
    assert len(certificate.members) == 6
    assert certificate.risk_evaluable_count == 3
    assert len(certificate.deployed_members) == 2
    assert len(certificate.fallback_members) == 4
    assert certificate.simultaneous_harm_passed
    assert certificate.simultaneous_positive_gain_passed
    assert certificate.maximum_deployed_harm_upper == pytest.approx(0.04706922523142958)
    assert all(
        member.familywise_gain_lower is not None and member.familywise_gain_lower > 0.0
        for member in certificate.deployed_members
    )
    assert certificate.to_record()["backend_wide_competence_claim"] is False
    assert certificate.to_record()["physical_safety_claim"] is False


@pytest.mark.skipif(
    not WRAPPING_ROOT.is_dir() or not SLINGSHOT_ROOT.is_dir(),
    reason="frozen public-simulator raw trees are not installed",
)
def test_certificate_rebuilds_from_frozen_public_simulator_trees() -> None:
    committed = load_query_portfolio_certificate(CERTIFICATE)
    rebuilt = build_certificate(
        wrapping_root=WRAPPING_ROOT,
        slingshot_root=SLINGSHOT_ROOT,
    )
    assert rebuilt.to_record() == committed.to_record()


def test_paper_synthesis_discloses_multiplicity_and_posthoc_boundary() -> None:
    result = (ROOT / "docs/dlolab_query_portfolio_certificate_v1_result.md").read_text()
    synthesis = (ROOT / "docs/query_conditional_simulator_competence_v3.md").read_text()
    assert "three queries that reached final risk evaluation" in result
    assert "post-hoc simultaneous synthesis" in result
    assert "Cross-task reward" in result
    assert "units are not pooled" in result
    assert "91.67%" in result
    assert "joint confidence lower bound is 90%" in synthesis
    assert "backend-wide competence" in synthesis
