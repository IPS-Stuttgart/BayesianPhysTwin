from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUPPORT = ROOT / "SUPPORT.md"
CHANGELOG = ROOT / "CHANGELOG.md"
CLAIM_CONTRACT = ROOT / "docs" / "phystwin_release_claim_v1.md"
PAPER_COMPACT_EVIDENCE = (
    "evidence/bayesian_phystwin/bpt-release-synthesis-v1/summary.json"
)


def _read(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_readme_retains_the_complete_bounded_release_summary() -> None:
    text = _read(README)

    for phrase in (
        "principal matched deterministic comparator",
        "Gaussian NLL by `-9.136`",
        "simultaneous 95% interval `[-13.961, -4.312]`",
        "`17/22`",
        "`22/22`",
        "`3.10×`",
        "retrospective mechanism evidence",
        "Independent real-provider and independent-object transfer remain unconfirmed",
        PAPER_COMPACT_EVIDENCE,
    ):
        assert phrase in text


def test_claim_contract_keeps_score_coverage_width_and_promotion_separate() -> None:
    text = _read(CLAIM_CONTRACT)

    for phrase in (
        "Exact-mean covariance-only retrospective evidence",
        "`-9.136`",
        "`[-13.961, -4.312]`",
        "`17 / 5 / 0`",
        "`22/22`",
        "`70.6%` to `91.0%`",
        "`16.45 mm` to `50.94 mm`",
        "`3.10×`",
        "`[8, 16, 16]`",
        "does not authorize independent calibration or deployment",
        PAPER_COMPACT_EVIDENCE,
    ):
        assert phrase in text


def test_support_policy_requires_the_scientific_companion_boundaries() -> None:
    text = _read(SUPPORT)

    for phrase in (
        "Scientific release boundary",
        "docs/phystwin_release_claim_v1.md",
        "last-residual comparator",
        "exact-mean covariance-only result",
        "`3.10×` interval-width cost",
        "raw posterior covariance remains severely undercalibrated",
        "independent real-provider and independent-object transfer remain unconfirmed",
    ):
        assert phrase in text


def test_changelog_records_release_evidence_and_wording_synchronization() -> None:
    text = _read(CHANGELOG)

    for phrase in (
        "accepted and rejected three-repository golden-path decision artifacts",
        "Python 3.10, 3.12, and 3.14",
        "NumPy `1.23.0`",
        "`NumericalEnvironmentV1`",
        "exact-mean covariance-only retrospective result",
        "`3.10×` interval-width cost",
    ):
        assert phrase in text
