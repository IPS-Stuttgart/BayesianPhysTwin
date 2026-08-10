from __future__ import annotations

from bayesian_phystwin.discrepancy_candidate_tournament import (
    DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
    analyze_discrepancy_candidate_tournament,
)


def _candidate(
    candidate_id: str, family: str, state_dimension: int
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "family": family,
        "state_dimension": state_dimension,
        "parameter_count": state_dimension,
        "runtime_milliseconds": float(state_dimension),
        "covariance_bytes": 8 * state_dimension * state_dimension,
        "source_revision": "1" * 40,
        "configuration_sha256": "2" * 64,
        "prediction_artifact_sha256": candidate_id.encode().hex().ljust(64, "0")[:64],
    }


def _record(
    candidate_id: str,
    group_id: str,
    point_loss: float,
    proper_score: float,
    *,
    accepted: bool,
) -> dict[str, object]:
    fallback_point_loss = 10.0
    fallback_proper_score = 5.0
    return {
        "candidate_id": candidate_id,
        "unit_id": f"{group_id}-endpoint",
        "group_id": group_id,
        "horizon": "endpoint",
        "accepted": accepted,
        "point_loss": point_loss,
        "fallback_point_loss": fallback_point_loss,
        "deployed_point_loss": point_loss if accepted else fallback_point_loss,
        "proper_score": proper_score,
        "fallback_proper_score": fallback_proper_score,
        "deployed_proper_score": (proper_score if accepted else fallback_proper_score),
        "interval_covered": True,
        "interval_width": 2.0,
    }


def _crossfit_unstable_payload() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for index in range(6):
        group_id = f"group-{index}"
        structured_point = 3.0 if index == 0 else 7.7
        structured_proper = -2.0 if index == 0 else 3.9
        records.extend(
            [
                _record(
                    "physical_fallback",
                    group_id,
                    10.0,
                    5.0,
                    accepted=False,
                ),
                _record(
                    "last_residual",
                    group_id,
                    8.0,
                    4.0,
                    accepted=True,
                ),
                _record(
                    "dynamic",
                    group_id,
                    7.6,
                    3.8,
                    accepted=True,
                ),
                _record(
                    "structured",
                    group_id,
                    structured_point,
                    structured_proper,
                    accepted=True,
                ),
            ]
        )
    return {
        "contract": DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
        "schema_version": 1,
        "protocol_id": "discrepancy-source-crossfit-v1",
        "statistical_unit": "physical-object-or-session",
        "split": "source-only",
        "reference_candidate": "last_residual",
        "physical_fallback_candidate": "physical_fallback",
        "information_boundary": {
            "candidate_predictions_sealed_before_scoring": True,
            "candidate_generation_used_scored_targets": False,
            "future_observations_used": False,
            "confirmation_payloads_opened": False,
            "replacement_allowed": False,
        },
        "evaluation": {
            "evaluator_revision": "3" * 40,
            "scoring_policy_sha256": "4" * 64,
            "scored_unit_roster_sha256": "5" * 64,
            "physical_fallback_artifact_sha256": "6" * 64,
            "prediction_barrier_sha256": "7" * 64,
            "point_loss_id": "endpoint-rmse-v1",
            "proper_score_id": "gaussian-nll-v1",
            "interval_semantics_id": "marginal-90-v1",
        },
        "selection": {
            "minimum_group_count": 5,
            "minimum_relative_point_improvement": 0.01,
            "maximum_worst_group_relative_regression": 0.0,
            "maximum_harmful_accepted_count": 0,
            "maximum_mean_proper_score_regression": 0.0,
            "require_paired_point_upper_bound_nonpositive": True,
            "bootstrap_samples": 500,
            "bootstrap_seed": 20260810,
            "require_crossfit_stability": True,
            "nominal_interval_coverage": 0.9,
            "maximum_interval_coverage_shortfall": 0.0,
            "numerical_tolerance": 1e-12,
        },
        "candidates": [
            _candidate("physical_fallback", "physical", 0),
            _candidate("last_residual", "persistence", 0),
            _candidate("dynamic", "dynamic", 6),
            _candidate("structured", "structured", 4),
        ],
        "records": records,
    }


def test_crossfit_instability_retains_reference_as_final_candidate() -> None:
    report = analyze_discrepancy_candidate_tournament(_crossfit_unstable_payload())

    assert report["provisional_selected_candidate"] == "structured"
    assert report["cross_fitted"]["stable_selection"] is False
    assert report["cross_fitted"]["held_group_nonregression"] is True
    assert report["source_gate_passed"] is False
    assert report["decision"] == "retain-reference-candidate"
    assert report["selected_candidate"] == "last_residual"
    assert report["selected_candidate_summary"]["candidate_id"] == "last_residual"
    assert (
        report["provisional_selected_candidate_summary"]["candidate_id"] == "structured"
    )
