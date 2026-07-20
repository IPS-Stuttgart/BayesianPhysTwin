import json
import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_online_belief import RecursiveRbfBeliefConfig
from bayesian_phystwin.phystwin_online_belief_evaluation import (
    _score_trajectory,
    evaluate_online_belief_case,
    evaluate_online_belief_cohort,
    select_stable_geometry_centers,
)


def test_stable_center_selection_cannot_see_future_availability() -> None:
    rng = np.random.default_rng(4)
    points = rng.normal(size=(12, 24, 3))
    visibility = np.ones((12, 24), dtype=bool)
    motion_valid = np.ones_like(visibility)
    first, first_info = select_stable_geometry_centers(
        points,
        visibility,
        motion_valid,
        train_end_frame=6,
        center_count=8,
        minimum_training_availability_fraction=0.8,
        fallback_candidate_count=16,
    )
    changed_points = points.copy()
    changed_visibility = visibility.copy()
    changed_motion = motion_valid.copy()
    changed_points[6:] += 1000.0
    changed_visibility[6:, ::2] = False
    changed_motion[6:, 1::2] = False
    second, second_info = select_stable_geometry_centers(
        changed_points,
        changed_visibility,
        changed_motion,
        train_end_frame=6,
        center_count=8,
        minimum_training_availability_fraction=0.8,
        fallback_candidate_count=16,
    )
    np.testing.assert_array_equal(first, second)
    assert first_info == second_info


def test_assimilation_centres_are_excluded_from_phystwin_metrics() -> None:
    target = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    trajectory = target.copy()
    trajectory[1, 0] = [100.0, -100.0, 50.0]
    visible = np.ones((2, 3), dtype=bool)
    valid = np.ones_like(visible)

    score = _score_trajectory(
        trajectory,
        target,
        visible,
        valid,
        None,
        surface_point_count=3,
        center_ids=np.asarray([0]),
        scored_frames=(1,),
    )

    assert score["future_noncenter_point_error_m"] == 0.0
    assert score["future_chamfer_distance_m"] == 0.0


def _write_translation_case(root: Path) -> None:
    frame_count = 20
    point_count = 32
    initial = np.stack(
        (
            np.linspace(-0.2, 0.2, point_count),
            np.sin(np.linspace(0.0, 2.0 * np.pi, point_count)) * 0.03,
            np.zeros(point_count),
        ),
        axis=1,
    )
    baseline = np.repeat(initial[None], frame_count, axis=0)
    observed = baseline.copy()
    for frame in range(10, frame_count):
        observed[frame, :, 1] += 0.002 * (frame - 9)
    motion_valid = np.ones((frame_count, point_count), dtype=bool)
    motion_valid[-1] = False
    final_data = {
        "object_points": observed,
        "object_visibilities": np.ones((frame_count, point_count), dtype=bool),
        "object_motions_valid": motion_valid,
        "surface_points": np.empty((0, 3)),
    }
    with (root / "final_data.pkl").open("wb") as handle:
        pickle.dump(final_data, handle)
    with (root / "inference.pkl").open("wb") as handle:
        pickle.dump(baseline, handle)
    (root / "split.json").write_text(
        json.dumps({"frame_len": frame_count, "train": [0, 10], "test": [10, 20]}),
        encoding="utf-8",
    )


def test_case_evaluation_improves_future_noncenter_identities(tmp_path: Path) -> None:
    _write_translation_case(tmp_path)
    report, arrays = evaluate_online_belief_case(
        tmp_path,
        baseline_filename="inference.pkl",
        measurement_policy={
            "center_count": 8,
            "minimum_training_availability_fraction": 0.8,
            "fallback_candidate_count": 16,
            "update_fractions": [0.25, 0.5, 0.75],
        },
        belief_config=RecursiveRbfBeliefConfig(local_blend=0.25),
    )
    open_error = report["scores"]["open_loop"]["future_noncenter_point_error_m"]
    field_error = report["scores"]["recursive_rbf_belief"][
        "future_noncenter_point_error_m"
    ]
    assert field_error < open_error
    assert report["scores"]["open_loop"]["frame_count"] > 0
    assert (
        report["scores"]["open_loop"]["identity_frame_count"]
        == report["scores"]["open_loop"]["frame_count"] - 1
    )
    assert len(report["center_ids"]) == 8
    assert arrays["field_trajectory_m"].shape == (20, 32, 3)
    assert np.all(np.isnan(arrays["field_variance_m2"][:13]))


def test_cohort_gate_uses_declared_candidate_arm(tmp_path: Path) -> None:
    case_root = tmp_path / "cases"
    case_dir = case_root / "case"
    case_dir.mkdir(parents=True)
    _write_translation_case(case_dir)
    protocol = {
        "protocol_id": "test-causal-candidate",
        "confirmation_cohort": {
            "case_root": str(case_root),
            "baseline_filename": "inference.pkl",
            "physical_object_groups": {"case": "object"},
        },
        "measurement_policy": {
            "center_count": 8,
            "minimum_training_availability_fraction": 0.8,
            "fallback_candidate_count": 16,
            "update_fractions": [0.25, 0.5, 0.75],
        },
        "belief": {},
        "fixed_arms": [
            "open_loop",
            "recursive_global_translation",
            "recursive_rbf_belief",
            "recursive_rbf_causal_continuation",
            "risk_limited_frozen_current_state",
        ],
        "primary_metrics": [
            "future_noncenter_point_error_m",
            "future_chamfer_distance_m",
        ],
        "aggregation": {"bootstrap_draws": 10, "bootstrap_seed": 0},
        "confirmation_gate": {
            "candidate_arm": "recursive_rbf_causal_continuation",
            "minimum_relative_improvement_over_open_loop_each_primary_metric": 0.0,
            "minimum_two_metric_case_wins": 0,
            "maximum_case_regression_each_primary_metric": 100.0,
            "minimum_noncenter_improvement_over_global_translation": 0.0,
        },
        "claim_boundary": "unit test",
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    summary = evaluate_online_belief_cohort(
        protocol_path,
        tmp_path / "output",
    )

    assert summary["gate"]["candidate_arm"] == ("recursive_rbf_causal_continuation")


def test_support_gate_returns_exact_open_loop_and_skips_update(tmp_path: Path) -> None:
    _write_translation_case(tmp_path)
    data_path = tmp_path / "final_data.pkl"
    with data_path.open("rb") as handle:
        data = pickle.load(handle)
    # Selection uses the untouched training prefix. At every update, fewer than
    # the required eight centres can be visible, while future scoring frames
    # retain full support.
    for frame in (12, 15, 18):
        data["object_visibilities"][frame] = False
        data["object_visibilities"][frame, :7] = True
    with data_path.open("wb") as handle:
        pickle.dump(data, handle)

    report, arrays = evaluate_online_belief_case(
        tmp_path,
        baseline_filename="inference.pkl",
        measurement_policy={
            "center_count": 8,
            "minimum_training_availability_fraction": 0.8,
            "fallback_candidate_count": 16,
            "update_fractions": [0.25, 0.5, 0.75],
            "minimum_update_center_count": 8,
        },
        belief_config=RecursiveRbfBeliefConfig(local_blend=0.25),
    )
    with (tmp_path / "inference.pkl").open("rb") as handle:
        baseline = np.asarray(pickle.load(handle), dtype=np.float32)

    np.testing.assert_array_equal(arrays["field_trajectory_m"], baseline)
    np.testing.assert_array_equal(arrays["global_trajectory_m"], baseline)
    np.testing.assert_array_equal(arrays["causal_continuation_trajectory_m"], baseline)
    assert all(not update["accepted"] for update in report["updates"])
    assert all(
        update["decision"] == "insufficient_support_exact_fallback"
        for update in report["updates"]
    )
    assert all(update["mean_reliability"] is None for update in report["updates"])


def test_dispersion_gate_rejects_incoherent_high_support_update(
    tmp_path: Path,
) -> None:
    _write_translation_case(tmp_path)
    data_path = tmp_path / "final_data.pkl"
    with data_path.open("rb") as handle:
        data = pickle.load(handle)
    side = data["object_points"][0, :, 0] < 0.0
    for frame in (12, 15, 18):
        data["object_points"][frame, side, 1] += 0.10
        data["object_points"][frame, ~side, 1] -= 0.10
    with data_path.open("wb") as handle:
        pickle.dump(data, handle)

    report, arrays = evaluate_online_belief_case(
        tmp_path,
        baseline_filename="inference.pkl",
        measurement_policy={
            "center_count": 8,
            "minimum_training_availability_fraction": 0.8,
            "fallback_candidate_count": 16,
            "update_fractions": [0.25, 0.5, 0.75],
            "minimum_update_center_count": 5,
            "residual_dispersion_gate": {
                "minimum_history_center_count": 3,
                "history_quantile": 0.95,
                "history_multiplier": 1.5,
                "minimum_threshold_m": 0.01,
            },
        },
        belief_config=RecursiveRbfBeliefConfig(local_blend=0.25),
    )
    with (tmp_path / "inference.pkl").open("rb") as handle:
        baseline = np.asarray(pickle.load(handle), dtype=np.float32)

    np.testing.assert_array_equal(arrays["field_trajectory_m"], baseline)
    assert all(update["available_center_count"] == 8 for update in report["updates"])
    assert all(not update["accepted"] for update in report["updates"])
    assert all(
        update["decision"] == "incoherent_residual_exact_fallback"
        for update in report["updates"]
    )
    assert (
        report["risk_gate"]["residual_dispersion"]["maximum_update_dispersion_m"]
        == 0.01
    )
