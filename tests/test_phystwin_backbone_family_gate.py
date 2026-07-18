import pytest

from bayesian_phystwin.phystwin_backbone_family_gate import (
    choose_backbone_family,
    normalized_validation_score,
)


def _metrics(cd: float, track: float) -> dict[str, float]:
    return {"chamfer_distance_m": cd, "track_error_m": track}


def test_normalized_validation_score_balances_metrics() -> None:
    score = normalized_validation_score(_metrics(0.008, 0.024), _metrics(0.01, 0.02))

    assert score == pytest.approx(1.0)


def test_family_gate_uses_common_reference_and_preserves_tie_order() -> None:
    selected, scores = choose_backbone_family(
        {
            "released": _metrics(0.008, 0.016),
            "learned": _metrics(0.008, 0.016),
        },
        _metrics(0.01, 0.02),
    )

    assert selected == "released"
    assert scores == pytest.approx({"released": 0.8, "learned": 0.8})


def test_family_gate_can_accept_balanced_transfer() -> None:
    selected, scores = choose_backbone_family(
        {
            "released": _metrics(0.008, 0.018),
            "learned": _metrics(0.007, 0.017),
        },
        _metrics(0.01, 0.02),
    )

    assert selected == "learned"
    assert scores["learned"] < scores["released"]


def test_normalized_validation_score_rejects_invalid_reference() -> None:
    with pytest.raises(ValueError, match="positive references"):
        normalized_validation_score(_metrics(0.01, 0.02), _metrics(0.0, 0.02))
