from __future__ import annotations

import hashlib
import json

import numpy as np

from bayesian_phystwin import causal_continuation_diagnostic as causal
from bayesian_phystwin import deform360_corruption_diagnostic as corruption
from bayesian_phystwin import deform360_tail_gate_diagnostic as tail
from bayesian_phystwin import residual_velocity_diagnostic as velocity
from bayesian_phystwin.cli import online_belief_diagnostics as cli


def test_mismatch_corruption_is_deterministic_and_observation_only() -> None:
    frame_count = 4
    point_count = corruption.CENTER_COUNT + 2
    target = np.arange(frame_count * point_count * 3, dtype=float).reshape(
        frame_count, point_count, 3
    )
    original = target.copy()
    visible = np.ones((frame_count, point_count), dtype=bool)
    valid = np.ones_like(visible)
    centers = np.arange(corruption.CENTER_COUNT)

    first, first_available, first_info = corruption.corrupt_center_stream(
        target,
        visible,
        valid,
        centers,
        condition="mismatch_25pct",
        seed=3,
        case_name="fixture",
    )
    second, second_available, second_info = corruption.corrupt_center_stream(
        target,
        visible,
        valid,
        centers,
        condition="mismatch_25pct",
        seed=3,
        case_name="fixture",
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_available, second_available)
    np.testing.assert_array_equal(target, original)
    assert first_info == second_info
    assert first_info["realized_corruption_fraction"] == 0.25
    assert np.all(first_available)


def test_inlier_tail_gate_rejects_to_exact_physical_prior() -> None:
    frame_count = 60
    point_count = corruption.CENTER_COUNT + 1
    prior = np.zeros((frame_count, point_count, 3), dtype=np.float32)
    target = prior.copy()
    visible = np.ones((frame_count, point_count), dtype=bool)
    valid = np.ones_like(visible)
    centers = np.arange(corruption.CENTER_COUNT)
    observations = target[:, centers].astype(float)
    for frame in corruption.UPDATE_FRAMES:
        observations[frame, 12:, 0] = 1.0
    available = np.ones((frame_count, corruption.CENTER_COUNT), dtype=bool)

    result = tail.run_tail_filter(
        prior,
        target,
        visible,
        valid,
        centers,
        observations,
        available,
        target[:, centers],
        available,
        gate_rule="inlier13",
    )

    assert result["rejected_exact_fallback_count"] == 3
    assert [record["accepted"] for record in result["updates"]] == [
        False,
        False,
        False,
    ]
    assert [
        record["inlier_count_under_frozen_threshold"] for record in result["updates"]
    ] == [12, 12, 12]
    assert result["scores"]["recursive_rbf_risk_limited"] == {
        "hidden_identity_rmse_m": 0.0,
        "hidden_symmetric_chamfer_m": 0.0,
    }


def test_causal_projection_and_binary_selectors_are_deterministic() -> None:
    prior_delta = np.tile(np.array([[1.0, 0.0, 0.0]]), (8, 1))
    observed_delta = 0.75 * prior_delta
    intervals = [(prior_delta, observed_delta)]

    assert causal.huber_projection(intervals) == 0.75
    assert causal.median_projection(intervals) == 0.75
    assert causal.estimate_alpha("causal_last_huber_binary025", intervals) == 1.0
    assert causal.estimate_alpha("causal_last_huber_binary050", intervals) == 1.0


def test_residual_velocity_builder_preserves_rejected_intervals() -> None:
    prior = np.zeros((6, 4, 3), dtype=float)
    beta0 = prior.copy()
    beta0[3:] = 0.25
    target = prior.copy()
    visible = np.ones((6, 4), dtype=bool)
    valid = np.ones_like(visible)

    arms, records = velocity._build_arms(
        prior,
        target,
        visible,
        valid,
        np.array([0, 1, 2]),
        (2,),
        (False,),
        beta0,
        (0.01,),
        scale_points=prior[0],
    )

    assert records == [
        {
            "frame": 2,
            "previous_measurement_frame": 0,
            "accepted": False,
            "decision": "locked_gate_exact_physical_prior",
        }
    ]
    np.testing.assert_array_equal(arms["physical_prior"], prior)
    np.testing.assert_array_equal(arms["frozen_current_state"], prior)
    np.testing.assert_array_equal(arms["beta0p5"], prior)
    np.testing.assert_array_equal(arms["beta0_field"], beta0)


def test_unified_cli_writes_legacy_sorted_json(monkeypatch, tmp_path) -> None:
    result = {"z": [3, 2, 1], "a": {"claim": "development only"}}
    monkeypatch.setattr(
        cli.deform360_corruption_diagnostic,
        "run",
        lambda root: result,
    )
    output = tmp_path / "nested" / "results.json"

    cli.main(
        [
            "corruption-stress",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    expected = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    assert output.read_bytes() == expected
    assert (
        hashlib.sha256(output.read_bytes()).hexdigest()
        == hashlib.sha256(expected).hexdigest()
    )
