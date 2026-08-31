from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_deform360_dependence_query_v6.py"
spec = importlib.util.spec_from_file_location("deform360_dependence_query_v6", SCRIPT)
assert spec is not None and spec.loader is not None
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)


@dataclass
class FakeCovariance:
    mean_error: np.ndarray
    diagonal: np.ndarray
    factor: np.ndarray
    multiplier: float
    marginal_z: float
    source_marginal_coverage: float
    source_joint_nanees: float


class FakeBase:
    CovarianceModel = FakeCovariance


def make_covariance(dimension: int = 192) -> FakeCovariance:
    row = np.linspace(0.05, 0.25, dimension, dtype=np.float64)
    factor = np.column_stack((row, np.roll(row, 17)))
    return FakeCovariance(
        mean_error=np.zeros(dimension),
        diagonal=np.full(dimension, 0.07),
        factor=factor,
        multiplier=1.3,
        marginal_z=1.7,
        source_marginal_coverage=0.9,
        source_joint_nanees=1.0,
    )


def test_query_bank_is_complete_and_geometrically_balanced() -> None:
    bank = run.query_bank(384)
    assert tuple((name, event) for name, (_, event) in bank.items()) == run.QUERY_SPECS
    assert bank["total_load"][0].sum() == pytest.approx(1.0)
    for name in (
        "sensor_imbalance",
        "horizontal_balance",
        "vertical_balance",
        "center_periphery",
    ):
        assert bank[name][0].sum() == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(ValueError):
        run.query_bank(193)


def test_dependence_controls_preserve_every_coordinate_marginal() -> None:
    covariance = make_covariance()
    arms = run.covariance_arms(FakeBase, covariance, seed=91)
    reference = run.marginal_variance(covariance)
    for model in arms.values():
        np.testing.assert_allclose(run.marginal_variance(model), reference, atol=1e-12)
    assert arms["diagonal_marginal_matched"].factor.shape == (192, 0)
    assert not np.array_equal(
        arms["scrambled_marginal_matched"].factor,
        covariance.factor,
    )
    np.testing.assert_allclose(
        np.linalg.norm(arms["scrambled_marginal_matched"].factor, axis=1),
        np.linalg.norm(covariance.factor, axis=1),
        atol=1e-14,
    )


def test_one_full_arm_source_calibration_is_reused_without_control_refit() -> None:
    covariance = make_covariance()
    arms = run.covariance_arms(FakeBase, covariance, seed=7)
    weight = run.query_bank(192)["total_load"][0]
    raw = {
        name: run.covariance_query_variance(model, weight)
        for name, model in arms.items()
    }
    rng = np.random.default_rng(4)
    common = rng.normal(scale=0.3, size=(512, 1))
    source_errors = common + rng.normal(scale=0.05, size=(512, 192))
    centered = source_errors - source_errors.mean(axis=0, keepdims=True)
    source_truth = rng.normal(size=(512, 192))
    target_truth = rng.normal(size=(128, 192))
    target_errors = rng.normal(scale=0.2, size=(128, 1)) + rng.normal(
        scale=0.05, size=(128, 192)
    )
    calibration = run.source_query_calibration(
        centered,
        source_truth,
        weight,
        raw,
        event="upper",
        probability=0.9,
        event_quantile=0.9,
    )
    assert calibration["reference_raw_query_variance"] == pytest.approx(
        raw["full_low_rank"]
    )
    values: dict[str, dict[str, float]] = {}
    for name, model in arms.items():
        values[name] = run.query_metrics(
            centered_source_errors=centered,
            target_truth=target_truth,
            target_errors=target_errors,
            weight=weight,
            event="upper",
            model=model,
            calibration=calibration,
            fallback_cost=0.1,
            probability_clip=1e-9,
        )
        assert values[name]["shared_variance_scale"] == pytest.approx(
            calibration["shared_variance_scale"]
        )
        assert values[name]["shared_radius_multiplier"] == pytest.approx(
            calibration["shared_radius_multiplier"]
        )
    assert values["full_low_rank"]["source_query_nanees"] == pytest.approx(1.0)
    assert values["diagonal_marginal_matched"]["source_query_nanees"] != pytest.approx(
        1.0
    )


def test_event_probability_respects_registered_event_semantics() -> None:
    means = np.asarray([-1.0, 0.0, 1.0])
    upper = run.event_probability(means, 0.25, 0.5, "upper")
    assert np.all(np.diff(upper) > 0.0)
    absolute = run.event_probability(means, 0.25, 0.5, "absolute")
    assert absolute[0] == pytest.approx(absolute[2])
    assert absolute[1] < absolute[0]


def synthetic_arm(
    value: float,
    *,
    nanees: float = 1.0,
    coverage: float = 0.9,
) -> dict[str, float]:
    return {
        "target_query_nanees": nanees,
        "target_90_coverage": coverage,
        "mean_90_interval_width": 1.0,
        "query_nll": value,
        "event_brier": value,
        "event_log_loss": value,
        "decision_loss": value,
        "decision_regret": value,
        "acceptance_fraction": 0.5,
        "harmful_accept_fraction_all": value,
        "harmful_accept_rate_given_accept": value,
        "calibration_log_error": abs(float(np.log(nanees))),
        "coverage_absolute_error": abs(coverage - 0.9),
    }


def test_superior_target_gate_requires_same_mean_dependence_value_and_calibration() -> (
    None
):
    rows: list[dict[str, Any]] = []
    for index in range(92):
        arm_summary = {
            "full_low_rank": synthetic_arm(0.05),
            "diagonal_marginal_matched": synthetic_arm(0.08, nanees=3.0, coverage=0.6),
            "scrambled_marginal_matched": synthetic_arm(0.09, nanees=4.0, coverage=0.5),
        }
        queries = {
            name: {
                "arms": {
                    arm: {
                        "target_query_nanees": values["target_query_nanees"],
                        "target_90_coverage": values["target_90_coverage"],
                        "query_nll": values["query_nll"],
                        "event_brier": values["event_brier"],
                        "decision_loss": values["decision_loss"],
                        "acceptance_fraction": values["acceptance_fraction"],
                        "harmful_accept_fraction_all": values[
                            "harmful_accept_fraction_all"
                        ],
                    }
                    for arm, values in arm_summary.items()
                }
            }
            for name, _ in run.QUERY_SPECS
        }
        joint = {
            arm: {
                "joint_nanees": values["target_query_nanees"],
                "joint_90_ellipsoid_coverage": values["target_90_coverage"],
                "marginal_90_coverage": 0.9,
                "mean_marginal_90_width": 1.0,
                "nll_per_dimension": values["query_nll"],
            }
            for arm, values in arm_summary.items()
        }
        rows.append(
            {
                "object_id": f"o{index:03d}",
                "parent_point_result_exact": True,
                "same_mean_by_construction": True,
                "coordinate_marginal_parity_max_abs": 0.0,
                "arm_summary": arm_summary,
                "queries": queries,
                "joint_metrics": joint,
            }
        )
    protocol = {
        "evaluation": {
            "bootstrap_repetitions": 1000,
            "random_seed": 2,
            "marginal_parity_tolerance": 1e-12,
        },
        "success_gates": {
            "minimum_query_nanees": 0.5,
            "maximum_query_nanees": 2.0,
            "minimum_query_coverage": 0.8,
            "maximum_query_coverage": 0.98,
        },
    }
    _, decision = run.aggregate(rows, protocol)
    assert decision["superior_target_reached"]
    assert decision["dependence_value_supported"]
    assert decision["query_calibration_supported"]
    assert decision["paper_claim_authorized"] is False
