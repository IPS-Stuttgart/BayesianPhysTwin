from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.deform_dlo45_decision_identifiability_v1.support_envelope_audit import (
    base_certificate_actions,
    policy_actions,
    retention_epsilon,
    summarize_actions,
    trajectory_score,
)


def _record(
    *,
    action: int,
    bounds: tuple[float, float, float],
    normalized_regret: tuple[float, float, float],
    physical_mse: tuple[float, float, float],
    trajectory: str = "1.pkl",
):
    decision = SimpleNamespace(
        decision=SimpleNamespace(
            certificate_action=action,
            worst_case_regret=np.asarray(bounds, dtype=float),
        )
    )
    return SimpleNamespace(
        stable_id=f"DLO4/{trajectory}/4",
        trajectory=trajectory,
        current_frame=4,
        decision=decision,
        normalized_regret=np.asarray(normalized_regret, dtype=float),
        physical_mse=np.asarray(physical_mse, dtype=float),
        fallback_mse=float(physical_mse[0]),
        certificate_source_regret_bound=float(bounds[action]),
        certificate_realized_regret=float(normalized_regret[action]),
        certificate_regret_excess=float(normalized_regret[action] - bounds[action]),
    )


def test_retention_epsilon_keeps_at_least_requested_source_fraction() -> None:
    records = [
        _record(
            action=1,
            bounds=(0.0, bound, 0.2),
            normalized_regret=(0.4, 0.1, 0.2),
            physical_mse=(4.0, 3.0, 5.0),
            trajectory=f"{index}.pkl",
        )
        for index, bound in enumerate((0.01, 0.02, 0.03, 0.04), start=1)
    ]

    epsilon = retention_epsilon(records, radius=0.2, retention_fraction=0.5)
    actions = policy_actions(records, radius=0.2, epsilon=epsilon)

    assert epsilon == pytest.approx(0.22)
    assert actions == [1, 1, 0, 0]


def test_trajectory_score_is_maximum_selected_regret_excess() -> None:
    records = [
        _record(
            action=1,
            bounds=(0.0, 0.04, 0.2),
            normalized_regret=(0.6, 0.14, 0.0),
            physical_mse=(6.0, 5.0, 4.0),
        ),
        _record(
            action=0,
            bounds=(0.0, 0.03, 0.2),
            normalized_regret=(0.8, 0.0, 0.1),
            physical_mse=(8.0, 4.0, 5.0),
        ),
        _record(
            action=2,
            bounds=(0.0, 0.03, 0.05),
            normalized_regret=(0.7, 0.1, 0.25),
            physical_mse=(7.0, 5.0, 6.0),
        ),
    ]

    score = trajectory_score(records)

    assert score["base_nonfallback_count"] == 2
    assert score["score"] == pytest.approx(0.20)
    assert score["maximum_selected_support_bound"] == pytest.approx(0.05)


def test_policy_and_summary_are_fail_closed_and_keep_fallback_identity() -> None:
    records = [
        _record(
            action=1,
            bounds=(0.0, 0.02, 0.2),
            normalized_regret=(0.5, 0.10, 0.0),
            physical_mse=(4.0, 3.0, 2.0),
        ),
        _record(
            action=2,
            bounds=(0.0, 0.2, 0.04),
            normalized_regret=(0.3, 0.0, 0.08),
            physical_mse=(9.0, 5.0, 7.0),
        ),
        _record(
            action=0,
            bounds=(0.0, 0.01, 0.01),
            normalized_regret=(0.0, 0.2, 0.2),
            physical_mse=(1.0, 2.0, 2.0),
        ),
    ]

    assert base_certificate_actions(records) == [1, 2, 0]
    actions = policy_actions(records, radius=0.10, epsilon=0.13)
    summary = summarize_actions(records, actions, epsilon=0.13)

    assert actions == [1, 0, 0]
    assert summary["nonfallback_count"] == 1
    assert summary["action_counts"] == [2, 1, 0]
    assert summary["harmful_nonfallback_count"] == 0
    assert summary["trajectory_any_regret_violation_count"] == 0


def test_infinite_radius_returns_fallback_for_every_base_nonfallback() -> None:
    records = [
        _record(
            action=1,
            bounds=(0.0, 0.01, 0.2),
            normalized_regret=(0.4, 0.1, 0.0),
            physical_mse=(4.0, 3.0, 2.0),
        ),
        _record(
            action=2,
            bounds=(0.0, 0.2, 0.01),
            normalized_regret=(0.4, 0.0, 0.1),
            physical_mse=(4.0, 2.0, 3.0),
        ),
    ]

    assert policy_actions(records, radius=float("inf"), epsilon=10.0) == [0, 0]
