from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.rct_real_decision import (
    COORDINATE_COUNT,
    GaussianState,
    RCTDecisionMethod,
    RCTGaussianTwin,
    RCTMaterialResponse,
    calibrate_simultaneous_force_multiplier,
    condition_gaussian,
    decision_value_of_probe,
    load_rct_force_responses,
    source_promotion_gate,
    summarize_evaluation,
    system_identification_value_of_probe,
    trace_policy,
)
from bayesian_phystwin.rct_real_decision_protocol import (
    SELECTABLE_PROBES,
)


def _write_force_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("material_id", "position", "sensor", "z_frame", "raw_fz"),
        )
        writer.writeheader()
        for position, sensor in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 3)):
            for z_frame, raw_fz in (
                (1.2, -0.2),
                (0.8, -1.2),
                (0.4, -2.2),
                (0.0, -3.2),
            ):
                writer.writerow(
                    {
                        "material_id": "material_0001",
                        "position": position,
                        "sensor": sensor,
                        "z_frame": z_frame,
                        "raw_fz": raw_fz,
                    }
                )
        writer.writerow(
            {
                "material_id": "material_0002",
                "position": "confirmation-position-must-not-be-parsed",
                "sensor": "confirmation-sensor-must-not-be-parsed",
                "z_frame": "confirmation-force-must-not-be-parsed",
                "raw_fz": "confirmation-force-must-not-be-parsed",
            }
        )


def _selector_covariance() -> np.ndarray:
    loadings = np.zeros((COORDINATE_COUNT, 6), dtype=np.float64)
    for coordinate in range(3):
        loadings[3 + coordinate, coordinate] = 0.6
        loadings[12 + coordinate, coordinate] = 0.6
        loadings[6 + coordinate, 3 + coordinate] = 2.0
        loadings[9 + coordinate, 3 + coordinate] = 2.0
    return loadings @ loadings.T + 0.1 * np.eye(COORDINATE_COUNT)


def test_adapter_skips_forbidden_rows_before_parsing_force_fields(tmp_path: Path) -> None:
    path = tmp_path / "force_metadata.csv"
    _write_force_csv(path)

    responses = load_rct_force_responses(
        path,
        allowed_material_ids=("0001",),
        forbidden_material_ids=("0002",),
    )

    assert len(responses) == 1
    assert responses[0].material_id == "0001"
    np.testing.assert_allclose(responses[0].force_n, np.tile((1.0, 2.0, 3.0), 5))


def test_adapter_rejects_allowed_and_forbidden_roster_overlap(tmp_path: Path) -> None:
    path = tmp_path / "force_metadata.csv"
    _write_force_csv(path)

    with pytest.raises(ValueError, match="overlap"):
        load_rct_force_responses(
            path,
            allowed_material_ids=("0001",),
            forbidden_material_ids=("0001",),
        )


def test_exact_conditioning_sets_observed_coordinates_and_covariance() -> None:
    state = GaussianState(np.zeros(COORDINATE_COUNT), np.eye(COORDINATE_COUNT))

    posterior = condition_gaussian(state, (0, 1, 2), (0.5, 1.0, 1.5))

    np.testing.assert_allclose(posterior.mean[:3], (0.5, 1.0, 1.5))
    np.testing.assert_allclose(posterior.covariance[:3, :], 0.0)
    np.testing.assert_allclose(posterior.covariance[:, :3], 0.0)
    assert posterior.observed_indices == (0, 1, 2)


def test_decision_and_system_identification_selectors_can_diverge() -> None:
    covariance = _selector_covariance()
    mean = np.zeros(COORDINATE_COUNT, dtype=np.float64)
    mean[12:15] = (0.4, 0.8, 1.2)
    state = GaussianState(mean, covariance)

    decision_values = {
        probe: decision_value_of_probe(state, probe, force_limit_n=0.8)
        for probe in SELECTABLE_PROBES
    }
    identification_values = {
        probe: system_identification_value_of_probe(state, probe)
        for probe in SELECTABLE_PROBES
    }

    assert max(decision_values, key=decision_values.get) == (1, 2)
    assert max(identification_values, key=identification_values.get) == (2, 1)


def test_adaptive_traces_use_only_anchor_and_selected_probe_values() -> None:
    covariance = _selector_covariance()
    mean = np.zeros(COORDINATE_COUNT, dtype=np.float64)
    mean[12:15] = (0.4, 0.8, 1.2)
    twin = RCTGaussianTwin(mean, covariance)
    response = RCTMaterialResponse("synthetic", np.maximum(mean, 0.0))

    decision = trace_policy(
        twin,
        response,
        selector="decision_directed",
        force_limit_n=0.8,
    )
    identification = trace_policy(
        twin,
        response,
        selector="system_identification",
        force_limit_n=0.8,
    )

    assert decision.probe_order[0] == (1, 2)
    assert identification.probe_order[0] == (2, 1)
    assert all(set(state.observed_indices).isdisjoint((12, 13, 14)) for state in decision.states)
    assert all(
        set(state.observed_indices).isdisjoint((12, 13, 14))
        for state in identification.states
    )


def test_universal_conformal_calibration_uses_registered_finite_sample_rank() -> None:
    generator = np.random.default_rng(20260902)
    twin = RCTGaussianTwin(
        np.ones(COORDINATE_COUNT),
        0.3 * np.eye(COORDINATE_COUNT),
    )
    calibration = tuple(
        RCTMaterialResponse(
            f"cal-{index:02d}",
            np.maximum(0.01, 1.0 + 0.1 * generator.standard_normal(COORDINATE_COUNT)),
        )
        for index in range(20)
    )

    multiplier, scores, rank = calibrate_simultaneous_force_multiplier(
        twin,
        calibration,
        force_limit_n=1.0,
    )

    assert rank == 19
    assert len(scores) == 20
    assert multiplier >= 0.0


def test_method_artifact_round_trip_preserves_frozen_numerics() -> None:
    twin = RCTGaussianTwin(
        np.arange(COORDINATE_COUNT, dtype=np.float64),
        np.eye(COORDINATE_COUNT),
    )
    method = RCTDecisionMethod(
        twin=twin,
        force_limit_n=2.5,
        conformal_multiplier=1.75,
        calibration_scores=tuple(float(index) / 10.0 for index in range(20)),
        conformal_rank=19,
    )

    restored = RCTDecisionMethod.from_dict(method.as_dict())

    np.testing.assert_array_equal(restored.twin.mean, method.twin.mean)
    np.testing.assert_array_equal(restored.twin.covariance, method.twin.covariance)
    assert restored.force_limit_n == method.force_limit_n
    assert restored.conformal_multiplier == method.conformal_multiplier
    assert restored.calibration_scores == method.calibration_scores


def test_source_gate_is_fail_closed_and_never_authorizes_confirmation() -> None:
    summary = {
        "relative_auc_improvement": 0.06,
        "material_improvement_count": 12,
        "decision_directed_simultaneous_force_coverage": 0.9,
        "decision_directed_false_safe_rate": 0.1,
        "decision_directed_unsafe_action_rate": 0.1,
        "system_identification_unsafe_action_rate": 0.05,
        "selected_probe_count": 2,
    }

    passed = source_promotion_gate(summary)
    failed = source_promotion_gate({**summary, "relative_auc_improvement": 0.049})

    assert passed["passed"] is True
    assert passed["target_authorized"] is False
    assert passed["confirmation_opened"] is False
    assert failed["passed"] is False


def test_reported_simultaneous_coverage_uses_material_level_all_budget_events() -> None:
    def policy(covered: tuple[bool, bool, bool, bool]) -> dict[str, object]:
        return {
            "regret_auc": 1.0,
            "probe_order": [
                {"position": 1, "sensor": 2},
                {"position": 2, "sensor": 1},
                {"position": 2, "sensor": 2},
            ],
            "simultaneous_force_covered_all_budgets": all(covered),
            "budgets": [
                {
                    "false_safe": False,
                    "unsafe": False,
                    "abstained": False,
                    "simultaneous_force_covered": value,
                }
                for value in covered
            ],
        }

    records = [
        {
            "material_id": "covered",
            "decision_directed": policy((True, True, True, True)),
            "system_identification": policy((True, True, True, True)),
        },
        {
            "material_id": "not-simultaneously-covered",
            "decision_directed": policy((True, True, True, False)),
            "system_identification": policy((True, True, True, True)),
        },
    ]

    summary = summarize_evaluation(records, require_confirmation_count=False)

    assert summary["decision_directed_simultaneous_force_coverage"] == 0.5
