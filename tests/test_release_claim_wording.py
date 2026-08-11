from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUPPORT = ROOT / "SUPPORT.md"
CLAIM = ROOT / "docs" / "phystwin_release_claim_v1.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_surfaces_retain_matched_comparator_and_raw_covariance_boundary() -> None:
    for path in (README, SUPPORT, CLAIM):
        text = _text(path)
        assert "last_residual" in text or "last-residual" in text
        assert "raw posterior covariance" in text
        assert "independent" in text

    claim = _text(CLAIM)
    assert "0.019156 m" in claim
    assert "0.019205 m" in claim
    assert "unique best deterministic predictor" in claim
    assert "calibrated raw posterior covariance" in claim


def test_release_surfaces_bind_covariance_only_effect_and_width_cost() -> None:
    for path in (README, SUPPORT, CLAIM):
        text = _text(path)
        assert "-9.136" in text
        assert "[-13.961, -4.312]" in text
        assert "0.01645 m" in text
        assert "0.05094 m" in text
        assert "3.10×" in text

    readme = _text(README)
    assert "exact `last_residual` point-prediction object" in readme
    assert "17/22" in readme
    assert "0.706" in readme
    assert "0.910" in readme


def test_release_surfaces_keep_terminal_and_registered_deform360_routes_separate() -> None:
    for path in (README, SUPPORT, CLAIM):
        text = _text(path)
        assert "313/324" in text
        assert "11" in text
        assert "100 sealed" in text
        assert "twelve" in text

    claim = _text(CLAIM)
    assert "Terminal complete-stream official-Hub provider version" in claim
    assert "Separate registered Deform360 v6 confirmation route" in claim
    assert "A source-negative result is complete evidence" in claim
    assert "No donor, scale, endpoint" in claim


def test_release_claim_forbids_overstatement() -> None:
    claim = _text(CLAIM)
    forbidden_boundaries = (
        "unique deterministic winner",
        "calibrated raw posterior covariance",
        "independent-object transfer",
        "deployment safety",
        "overall state of the art",
    )
    for boundary in forbidden_boundaries:
        assert boundary in claim
