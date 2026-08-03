from __future__ import annotations

from bayesian_phystwin.pokeflex_guard_postopen_audit import audit_sealed_guard_rows


def _row(
    object_name: str,
    baseline: float,
    candidate: float,
    *,
    accepted: bool,
    upper: float | None,
) -> dict[str, object]:
    return {
        "object_name": object_name,
        "baseline_error_mm": baseline,
        "candidate_error_mm": candidate,
        "accepted": accepted,
        "upper_regret_mm": upper,
    }


def test_postopen_audit_cannot_create_support_with_a_stricter_guard() -> None:
    rows = [
        _row("a", 2.0, 1.8, accepted=True, upper=-0.05),
        _row("a", 2.0, 2.1, accepted=True, upper=-0.01),
        _row("b", 3.0, 3.0, accepted=False, upper=None),
    ]

    result = audit_sealed_guard_rows(rows)

    assert result["accepted_scored_frame_count"] == 2
    assert result["accepted_harmful_frame_count"] == 1
    assert result["objects_without_sealed_candidate_effect"] == ["b"]
    assert result["current_policy"]["object_win_count"] == 1
    assert result["best_zero_loss_stricter_policy"]["object_win_count"] == 1
    assert result["frame_oracle_within_sealed_candidates"]["object_win_count"] == 1


def test_postopen_audit_reports_miscalibrated_accepted_bounds() -> None:
    rows = [
        _row("a", 1.0, 0.98, accepted=True, upper=-0.01),
        _row("a", 1.0, 1.02, accepted=True, upper=-0.01),
    ]

    result = audit_sealed_guard_rows(rows)

    assert result["accepted_upper_bound_coverage"] == 0.5
    assert result["accepted_false_safe_rate"] == 0.5
