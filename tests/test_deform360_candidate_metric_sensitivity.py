from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_candidate_metric_sensitivity import (
    METRIC_UNITS,
    PAIRWISE_ARM,
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    evaluate_candidate_metric_arrays,
    summarize_candidate_metric_sensitivity,
)


def test_array_evaluator_excludes_assimilation_centres() -> None:
    target = np.zeros((2, 4, 3), dtype=float)
    target[:, 1:, 0] = np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    prediction = target.copy()
    prediction[:, 0] = 1000.0
    prediction[:, 1:, 0] += 0.1
    visible = np.ones((2, 4), dtype=bool)
    valid = np.ones((2, 4), dtype=bool)

    metrics, support = evaluate_candidate_metric_arrays(
        {"candidate": prediction},
        target,
        visible,
        valid,
        center_ids=np.asarray([0]),
        scored_frames=(0, 1),
    )

    assert support["permanently_excluded_center_count"] == 1
    assert support["hidden_support_per_frame"]["minimum"] == 3
    assert np.allclose(metrics["candidate"]["track_mean_point_euclidean_m"], 0.1)
    assert np.allclose(
        metrics["candidate"]["chamfer_pred_to_target_mean_euclidean_m"],
        0.1,
    )


def test_array_evaluator_applies_visibility_and_validity() -> None:
    target = np.zeros((1, 4, 3), dtype=float)
    candidate = target.copy()
    candidate[0, 1, 0] = 1.0
    candidate[0, 2, 0] = 100.0
    candidate[0, 3, 0] = 100.0
    visible = np.asarray([[True, True, False, True]])
    valid = np.asarray([[True, True, True, False]])

    metrics, support = evaluate_candidate_metric_arrays(
        {"candidate": candidate},
        target,
        visible,
        valid,
        center_ids=np.asarray([0]),
        scored_frames=(0,),
    )

    assert support["hidden_support_per_frame"]["mean"] == 1.0
    assert metrics["candidate"]["track_mean_point_euclidean_m"][0] == 1.0


def _case_metric_fixture(
    candidate_scale: float,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, str]]:
    groups = {"a-1": "a", "a-2": "a", "b-1": "b"}
    methods = (PHYSICAL_ARM, PERSISTENCE_ARM, PAIRWISE_ARM)
    method_scales = {
        PHYSICAL_ARM: 2.0,
        PERSISTENCE_ARM: 1.5,
        PAIRWISE_ARM: candidate_scale,
    }
    result = {}
    for method in methods:
        result[method] = {}
        for metric in METRIC_UNITS:
            scale = method_scales[method]
            if METRIC_UNITS[metric] == "m^2":
                scale = scale**2
            result[method][metric] = {
                "a-1": np.asarray([scale, scale]),
                "a-2": np.asarray([scale]),
                "b-1": np.asarray([scale, scale, scale]),
            }
    return result, groups


def test_summary_passes_only_when_every_headline_convention_improves() -> None:
    metrics, groups = _case_metric_fixture(candidate_scale=1.0)

    summary = summarize_candidate_metric_sensitivity(metrics, groups)

    assert summary["metric_robustness_gate"]["passed"] is True
    for comparator in (PHYSICAL_ARM, PERSISTENCE_ARM):
        result = summary["comparisons"][comparator]["track_mean_point_euclidean_m"]
        assert result["episode_wins"] == 3
        assert result["object_wins"] == 2
        assert result["relative_change"]["object_balanced_mean"] < 0.0


def test_summary_fails_when_one_headline_convention_regresses() -> None:
    metrics, groups = _case_metric_fixture(candidate_scale=1.0)
    metrics[PAIRWISE_ARM]["chamfer_symmetric_mean_euclidean_m"]["b-1"] = np.asarray(
        [10.0]
    )

    summary = summarize_candidate_metric_sensitivity(metrics, groups)

    assert summary["metric_robustness_gate"]["passed"] is False
    failed = [
        check
        for check in summary["metric_robustness_gate"]["checks"]
        if not check["passed"]
    ]
    assert {check["metric"] for check in failed} == {
        "chamfer_symmetric_mean_euclidean_m"
    }
