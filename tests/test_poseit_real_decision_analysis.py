from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from bayesian_phystwin_experiments.poseit_real_decision_analysis import (
    FIXED_ORDER_COUNT,
    PCA_COMPONENT_COUNT,
    PoseItDecisionMethod,
    PoseItFeatureFamily,
    PoseItGaussianTwin,
    PoseItLabeledFamily,
    calibrate_shared_stability_shortfall,
    evaluate_poseit_family,
    fixed_probe_order_roster_sha256,
    poseit_confirmation_gate,
    poseit_source_promotion_gate,
    registered_fixed_probe_orders,
    summarize_poseit_evaluation,
)


def _raw_feature(object_index: int, pose: int, family_index: int = 0) -> np.ndarray:
    coordinate = np.arange(1, 11, dtype=np.float64)
    return (
        np.sin((object_index + 1) * (pose + coordinate) * 0.071)
        + np.cos((family_index + 1) * (2 * pose + coordinate) * 0.113)
        + 0.01 * object_index * coordinate
    )


def _family(
    prefix: str,
    object_index: int,
    *,
    available_poses: tuple[int, ...] = tuple(range(1, 17)),
    family_index: int = 0,
) -> PoseItLabeledFamily:
    features = PoseItFeatureFamily(
        object_token=f"{prefix}-{object_index}",
        family_token=f"family-{family_index}",
        pre_shake_features={
            pose: _raw_feature(object_index, pose, family_index)
            for pose in available_poses
        },
    )
    labels = {
        pose: bool((object_index + family_index + pose) % 3 != 0)
        for pose in available_poses
    }
    return PoseItLabeledFamily(features=features, shake_stable=labels)


def _fit_families() -> tuple[PoseItLabeledFamily, ...]:
    return tuple(_family("fit", index) for index in range(10))


def test_feature_and_label_rosters_are_structural_and_immutable() -> None:
    raw = {1: np.arange(10, dtype=np.float64), 3: np.arange(10, dtype=np.float64)}
    features = PoseItFeatureFamily(" Object ", " family ", raw)
    labeled = PoseItLabeledFamily(features, {1: True, 3: False})

    raw[1][0] = 999.0

    assert features.object_token == "object"
    assert features.family_token == "family"
    assert features.available_poses == (1, 3)
    assert features.pre_shake_features[1][0] == 0.0
    assert labeled.shake_stable == {1: True, 3: False}
    with pytest.raises(ValueError, match="roster differs"):
        PoseItLabeledFamily(features, {1: True})


def test_twin_fit_is_deterministic_fit_only_and_round_trips() -> None:
    families = _fit_families()
    twin = PoseItGaussianTwin.fit(families)
    repeated = PoseItGaussianTwin.fit(families)

    assert twin.projector.components.shape == (PCA_COMPONENT_COUNT, 10)
    np.testing.assert_array_equal(twin.projector.center, repeated.projector.center)
    np.testing.assert_array_equal(
        twin.projector.components, repeated.projector.components
    )
    np.testing.assert_array_equal(twin.mean, repeated.mean)
    np.testing.assert_array_equal(twin.covariance, repeated.covariance)
    assert np.min(np.linalg.eigvalsh(twin.covariance)) > 0.0

    flipped = tuple(
        PoseItLabeledFamily(
            family.features,
            {pose: not stable for pose, stable in family.shake_stable.items()},
        )
        for family in families
    )
    flipped_twin = PoseItGaussianTwin.fit(flipped)
    np.testing.assert_array_equal(
        twin.projector.components, flipped_twin.projector.components
    )
    np.testing.assert_array_equal(twin.projector.center, flipped_twin.projector.center)

    restored = PoseItGaussianTwin.from_dict(twin.as_dict())
    np.testing.assert_array_equal(restored.mean, twin.mean)
    np.testing.assert_array_equal(restored.covariance, twin.covariance)


def test_twin_rejects_incomplete_fit_family_before_modeling() -> None:
    families = list(_fit_families())
    families[0] = _family("fit", 0, available_poses=(1, 2, 3))

    with pytest.raises(ValueError, match="structurally incomplete"):
        PoseItGaussianTwin.fit(families)


def test_hash_fixed_order_roster_is_exact_and_unique() -> None:
    orders = registered_fixed_probe_orders()

    assert len(orders) == FIXED_ORDER_COUNT
    assert len(set(orders)) == FIXED_ORDER_COUNT
    assert all(set(order) == set(range(2, 17)) for order in orders)
    assert (
        fixed_probe_order_roster_sha256()
        == "889f81c2ec6b1f33e3f55e7a2d9e6f4e879b9bf511ec8a5ead9933d45fc9bee3"
    )


def test_shared_certificate_uses_five_object_maximum_rank() -> None:
    twin = PoseItGaussianTwin.fit(_fit_families())
    calibration = tuple(
        _family("cal", index, available_poses=(1, 2, 3, 4)) for index in range(5)
    )

    shortfall, scores, rank = calibrate_shared_stability_shortfall(twin, calibration)

    assert rank == 5
    assert shortfall == max(scores)
    assert all(score >= 0.0 and np.isfinite(score) for score in scores)


def test_method_with_maximal_shortfall_abstains_without_false_safe() -> None:
    twin = PoseItGaussianTwin.fit(_fit_families())
    method = PoseItDecisionMethod(
        twin=twin,
        stability_multiplier=1_000_000.0,
        calibration_scores=(1_000_000.0,) * 5,
        calibration_rank=5,
    )
    family = _family("source", 0, available_poses=(1, 2, 3, 4))

    result = evaluate_poseit_family(method, family)
    restored = PoseItDecisionMethod.from_dict(method.as_dict())

    assert restored.stability_multiplier == method.stability_multiplier
    assert result["hash_fixed_order_mean"]["order_count"] == 256
    for selector in (
        "decision_directed",
        "system_identification",
        "lexicographic_fixed",
    ):
        assert len(result[selector]["budgets"]) == 4
        assert all(record["abstained"] for record in result[selector]["budgets"])
        assert not any(record["false_safe"] for record in result[selector]["budgets"])


def _budget(*, unsafe: bool = False, abstained: bool = False) -> dict[str, object]:
    return {
        "unsafe": unsafe,
        "false_safe": unsafe,
        "abstained": abstained,
    }


def _fake_family_result(
    object_token: str,
    family_token: str,
    *,
    decision_auc: float,
    identification_auc: float,
) -> dict[str, object]:
    def trace(auc: float, selected: list[int]) -> dict[str, object]:
        return {
            "regret_auc": auc,
            "selected_poses": selected,
            "budgets": [_budget() for _ in range(4)],
            "simultaneous_stability_covered_all_budgets": True,
        }

    return {
        "object_token": object_token,
        "family_token": family_token,
        "decision_directed": trace(decision_auc, [2, 3, 2]),
        "system_identification": trace(identification_auc, [4, 5, 6]),
        "lexicographic_fixed": trace(identification_auc, [2, 3, 4]),
        "hash_fixed_order_mean": {
            "simultaneous_stability_covered_all_orders_and_budgets": True,
        },
    }


def test_source_summary_aggregates_within_object_and_gate_stays_fail_closed() -> None:
    records: list[Mapping[str, object]] = []
    for object_index in range(5):
        records.extend(
            [
                _fake_family_result(
                    f"source-{object_index}",
                    "a",
                    decision_auc=0.4,
                    identification_auc=1.0,
                ),
                _fake_family_result(
                    f"source-{object_index}",
                    "b",
                    decision_auc=0.6,
                    identification_auc=1.0,
                ),
            ]
        )

    summary = summarize_poseit_evaluation(records, expected_object_count=5)
    gate = poseit_source_promotion_gate(summary)

    assert summary["object_count"] == 5
    assert summary["family_count"] == 10
    assert summary["decision_directed_mean_regret_auc"] == pytest.approx(0.5)
    assert summary["system_identification_mean_regret_auc"] == pytest.approx(1.0)
    assert summary["object_improvement_count"] == 5
    assert summary["selected_probe_count"] == 2
    assert gate["passed"] is True
    assert gate["target_authorized"] is False
    assert gate["confirmation_opened"] is False


def test_six_object_confirmation_uses_exact_sign_flip_and_no_retry() -> None:
    records = [
        _fake_family_result(
            f"confirm-{index}",
            "family",
            decision_auc=0.5,
            identification_auc=1.0,
        )
        for index in range(6)
    ]

    summary = summarize_poseit_evaluation(
        records,
        expected_object_count=6,
        confirmation=True,
    )
    gate = poseit_confirmation_gate(summary)

    assert summary["one_sided_exact_paired_sign_flip_p"] == pytest.approx(1 / 64)
    assert gate["passed"] is True
    assert gate["attempt_limit"] == 1
    assert gate["retry_authorized"] is False
