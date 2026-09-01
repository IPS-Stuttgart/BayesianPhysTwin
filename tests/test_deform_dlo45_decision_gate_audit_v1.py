from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.deform_dlo45_decision_identifiability_v1 import gate_audit
from experiments.deform_dlo45_decision_identifiability_v1._common import (
    Model,
    Protocol,
)
from experiments.deform_dlo45_decision_identifiability_v1._model import decide


def protocol() -> Protocol:
    return Protocol(
        prefix_frames=2,
        horizon_frames=2,
        stride_frames=1,
        action_scales=(0.0, 0.5, 1.0),
        neighbor_grid=(8,),
        cluster_grid=(4,),
        temperature_grid=(1.0,),
        regret_tolerance_grid=(0.05,),
        kmeans_iterations=5,
        source_fit_count=1,
        source_calibration_count=1,
        source_test_count=1,
        partition_domain="test",
        source_gate_mean_ratio=1.0,
        source_gate_worst_trajectory_ratio=1.0,
        source_gate_minimum_nonfallback_fraction=0.0,
        bootstrap_replicates=20,
        bootstrap_seed=7,
    )


def model() -> Model:
    rng = np.random.default_rng(17)
    residual_dimension = 2 * 8 * 3
    return Model(
        features=rng.normal(size=(16, 7)),
        residuals=rng.normal(scale=0.2, size=(16, residual_dimension)),
        class_labels=np.repeat(np.arange(4), 4).astype(np.int64),
        feature_mean=np.zeros(7),
        feature_scale=np.ones(7),
        loss_floor=1e-4,
        neighbors=8,
        temperature_scale=1.0,
        regret_tolerance=0.05,
        action_scales=np.asarray([0.0, 0.5, 1.0]),
    )


def window_records(count: int = 10) -> list[gate_audit.WindowRecord]:
    fitted = model()
    frozen = protocol()
    records: list[gate_audit.WindowRecord] = []
    for index in range(count):
        diagnostic = gate_audit.diagnose(fitted.features[index] + 0.01, fitted, frozen)
        scores = dict(diagnostic.scores)
        scores["deterministic_random"] = gate_audit.deterministic_score(
            f"DLO4/test/{index}", "deterministic_random"
        )
        diagnostic = diagnostic._replace(scores=scores)
        physical_mse = np.asarray([1.0, 0.8 + 0.01 * index, 0.7 + 0.02 * index])
        best = float(np.min(physical_mse))
        records.append(
            gate_audit.WindowRecord(
                stable_id=f"DLO4/test/{index}",
                dlo="DLO4",
                trajectory="test.pkl",
                current_frame=index,
                decision=diagnostic,
                physical_mse=physical_mse,
                normalized_regret=(physical_mse - best) / physical_mse[0],
                fallback_mse=float(physical_mse[0]),
                candidate_action=1 + index % 2,
                oracle_action=int(np.argmin(physical_mse)),
                nearest_selected_residual_rmse=0.1 + index * 0.01,
                nearest_global_residual_rmse=0.08 + index * 0.01,
                certificate_source_regret_bound=0.05,
                certificate_realized_regret=0.04,
                certificate_regret_excess=-0.01,
                certificate_harmful_vs_fallback=False,
            )
        )
    return records


def test_diagnostics_reproduce_registered_decision() -> None:
    fitted = model()
    frozen = protocol()
    feature = np.linspace(-0.3, 0.4, 7)

    diagnostic = gate_audit.diagnose(feature, fitted, frozen)
    reference = decide(feature, fitted, frozen)

    assert diagnostic.decision.certificate_action == reference.certificate_action
    assert diagnostic.decision.jeffrey_action == reference.jeffrey_action
    assert diagnostic.decision.kernel_action == reference.kernel_action
    assert diagnostic.decision.map_action == reference.map_action
    np.testing.assert_allclose(diagnostic.decision.correction, reference.correction)
    np.testing.assert_allclose(
        diagnostic.decision.worst_case_regret, reference.worst_case_regret
    )
    assert set(diagnostic.scores) == set(gate_audit.HEURISTICS) - {
        "deterministic_random"
    }
    assert all(np.isfinite(value) for value in diagnostic.scores.values())


def test_rank_threshold_matches_source_count_exactly() -> None:
    records = window_records()
    threshold = gate_audit.fit_rank_threshold(
        records, "maximum_kernel_weight", selected_count=4
    )

    actions = gate_audit.threshold_actions(records, "maximum_kernel_weight", threshold)

    assert np.count_nonzero(actions) == 4


def test_threshold_fit_does_not_depend_on_outcomes() -> None:
    records = window_records()
    threshold = gate_audit.fit_rank_threshold(
        records, "expected_action_gap", selected_count=3
    )
    changed = [
        replace(
            record,
            physical_mse=record.physical_mse[::-1].copy(),
            fallback_mse=999.0 + index,
            certificate_harmful_vs_fallback=True,
        )
        for index, record in enumerate(records)
    ]

    repeated = gate_audit.fit_rank_threshold(
        changed, "expected_action_gap", selected_count=3
    )

    assert repeated == threshold


def test_target_covariate_matching_uses_exact_requested_coverage() -> None:
    records = window_records()

    actions = gate_audit.exact_covariate_matched_actions(
        records, "quotient_concentration", selected_count=6
    )

    assert np.count_nonzero(actions) == 6


def test_wilson_interval_is_bounded_and_contains_rate() -> None:
    low, high = gate_audit._wilson_interval(3, 82)

    assert 0.0 <= low <= 3 / 82 <= high <= 1.0


def test_random_score_is_stable_and_keyed_by_decision() -> None:
    first = gate_audit.deterministic_score("DLO4/a/1", "deterministic_random")
    repeated = gate_audit.deterministic_score("DLO4/a/1", "deterministic_random")
    other = gate_audit.deterministic_score("DLO4/a/2", "deterministic_random")

    assert first == repeated
    assert first != other
    assert 0.0 <= first < 1.0
