import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_dlo_cross_object_transfer_v1 import (
    DLOS,
    evaluate_cross_object_transfer,
    feature_support_summary,
    load_cross_object_transfer_protocol,
)
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    build_deform_local_residual_features,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols" / "deform_dlo3_to_dlo45_no_refit_v1.json"
RUNNER = (
    REPOSITORY_ROOT
    / "scripts"
    / "remote"
    / "run_deform_dlo3_to_dlo45_no_refit_v1.py"
)


def _protocol(*, case_count: int = 4, horizon: int = 5) -> dict[str, object]:
    return {
        "data": {
            "trajectory_count_per_dlo": case_count,
            "prediction_horizon": horizon,
            "node_count": 12,
        },
        "evaluation": {
            "bootstrap_repetitions": 100,
            "bootstrap_seed": 7,
        },
        "promotion_gate": {
            "minimum_relative_improvement": 0.01,
            "minimum_case_wins": 3,
            "maximum_case_ratio": 1.10,
            "minimum_improving_seed_models": 2,
        },
    }


def _problem(
    *,
    case_count: int = 4,
    horizon: int = 5,
) -> tuple[
    dict[str, list[str]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    names = {
        dlo: [f"{dlo.lower()}-{index}" for index in range(case_count)]
        for dlo in DLOS
    }
    truth = {
        dlo: np.zeros((case_count, horizon, 12, 3), dtype=np.float64)
        for dlo in DLOS
    }
    physical = {dlo: np.full_like(truth[dlo], 0.10) for dlo in DLOS}
    object_specific = {dlo: np.full_like(truth[dlo], 0.04) for dlo in DLOS}
    return names, truth, physical, object_specific


def test_frozen_protocol_binds_pre_score_secondary_transfer(tmp_path: Path) -> None:
    protocol = load_cross_object_transfer_protocol(PROTOCOL)

    assert protocol["registration_boundary"]["target_scores_opened"] is False
    assert protocol["registration_boundary"]["observed_parent_stage"] == (
        "target-prediction-in-progress"
    )
    assert protocol["information_boundary"]["dlo4_or_dlo5_residual_refit"] is False
    assert protocol["promotion_gate"]["require_each_dlo"] is True
    assert [
        record["seed"] for record in protocol["dlo3_local_residual_models"]
    ] == [42, 43, 44]

    changed_payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    changed_payload["registration_boundary"]["target_scores_opened"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(changed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="registration boundary"):
        load_cross_object_transfer_protocol(changed)


def test_equal_seed_dlo3_transfer_passes_both_dlos() -> None:
    names, truth, physical, object_specific = _problem()
    transferred = {
        dlo: {
            42: np.full_like(truth[dlo], 0.06),
            43: np.full_like(truth[dlo], 0.07),
            44: np.full_like(truth[dlo], 0.08),
        }
        for dlo in DLOS
    }

    result = evaluate_cross_object_transfer(
        names_by_dlo=names,
        truth_by_dlo=truth,
        physical_by_dlo=physical,
        object_specific_by_dlo=object_specific,
        transferred_by_dlo=transferred,
        protocol=_protocol(),
    )

    assert result["decision"] == (
        "dlo3-residual-coefficients-transfer-to-both-fresh-dlos"
    )
    assert result["both_dlos_supported"] is True
    assert result["equal_dlo_summary"]["relative_improvement"] == pytest.approx(0.30)
    for dlo in DLOS:
        assert result["results"][dlo]["promotion_gate"]["supported"] is True
        assert result["results"][dlo]["promotion_gate"][
            "improving_seed_models"
        ] == 3
        assert result["results"][dlo]["primary_vs_matching_physical"][
            "wins"
        ] == 4
        assert result["results"][dlo][
            "matching_object_gain_retained_fraction"
        ] == pytest.approx(0.5)


def test_one_dlo_failure_cannot_be_overridden_by_pooled_gain() -> None:
    names, truth, physical, object_specific = _problem()
    transferred = {
        "DLO4": {
            seed: np.full_like(truth["DLO4"], 0.01) for seed in (42, 43, 44)
        },
        "DLO5": {
            seed: np.full_like(truth["DLO5"], 0.105) for seed in (42, 43, 44)
        },
    }

    result = evaluate_cross_object_transfer(
        names_by_dlo=names,
        truth_by_dlo=truth,
        physical_by_dlo=physical,
        object_specific_by_dlo=object_specific,
        transferred_by_dlo=transferred,
        protocol=_protocol(),
    )

    assert result["equal_dlo_summary"]["relative_improvement"] > 0.0
    assert result["results"]["DLO4"]["promotion_gate"]["supported"] is True
    assert result["results"]["DLO5"]["promotion_gate"]["supported"] is False
    assert result["both_dlos_supported"] is False
    assert result["decision"] == (
        "dlo3-residual-coefficients-do-not-transfer-to-both-fresh-dlos"
    )


def test_each_dlo_requires_independent_seed_stability() -> None:
    names, truth, physical, object_specific = _problem()
    transferred = {
        dlo: {
            42: np.full_like(truth[dlo], 0.01),
            43: np.full_like(truth[dlo], 0.11),
            44: np.full_like(truth[dlo], 0.11),
        }
        for dlo in DLOS
    }

    result = evaluate_cross_object_transfer(
        names_by_dlo=names,
        truth_by_dlo=truth,
        physical_by_dlo=physical,
        object_specific_by_dlo=object_specific,
        transferred_by_dlo=transferred,
        protocol=_protocol(),
    )

    for dlo in DLOS:
        gate = result["results"][dlo]["promotion_gate"]
        assert gate["passed"] is True
        assert gate["improving_seed_models"] == 1
        assert gate["seed_stability_passed"] is False
        assert gate["supported"] is False
    assert result["both_dlos_supported"] is False


def test_feature_support_is_descriptive_and_finite() -> None:
    case_count = 2
    horizon = 4
    nodes = 12
    initial = np.zeros((case_count, 2, nodes, 3), dtype=np.float64)
    initial[:, :, :, 0] = np.arange(nodes, dtype=np.float64)[None, None]
    initial[:, 1, :, 1] = 0.01
    clamped = np.asarray((0, 1, nodes - 2, nodes - 1))
    action = np.zeros((case_count, horizon, 4, 3), dtype=np.float64)
    for time in range(horizon):
        action[:, time] = initial[:, 1, clamped]
        action[:, time, :, 1] += 0.01 * (time + 1)
    baseline = np.zeros((case_count, horizon, nodes, 3), dtype=np.float64)
    baseline[:, :, :, 0] = np.arange(nodes, dtype=np.float64)[None, None]
    for time in range(horizon):
        baseline[:, time, :, 1] = 0.01 * (time + 1)
    features, _ = build_deform_local_residual_features(initial, action, baseline)
    model = {
        "feature_location": np.mean(features, axis=(0, 1)),
        "feature_scale": np.ones(features.shape[2:], dtype=np.float64),
    }

    summary = feature_support_summary(model, initial, action, baseline)

    assert summary["sample_count"] == features.size
    assert summary["affects_promotion_gate"] is False
    assert 0.0 <= summary["fraction_absolute_z_gt_3"] <= 1.0
    assert 0.0 <= summary["fraction_absolute_z_gt_5"] <= 1.0
    assert 0.0 <= summary["fraction_absolute_z_gt_10"] <= 1.0
    assert np.isfinite(summary["maximum_absolute_z"])


def test_runner_seals_before_any_parent_score_or_target_payload_read() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    seal = source.index('method_seal_path = output_root / "method_seal.json"')
    parent_score = source.index("parent_result = _validate_parent_result(")
    target_manifest = source.index("names, trajectories = _load_eval_panel(")
    prediction_archive = source.index(
        "physical, object_specific = _load_prediction_archive("
    )
    assert seal < parent_score < target_manifest
    assert seal < prediction_archive
    assert "allow_pickle=False" in source
    assert "target_dependent_selection" in source
    assert "target_retry_authorized" in source
