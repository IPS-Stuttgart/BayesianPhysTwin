from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.target_directed_intervention_design_v1 import (
    TARGET_DIRECTED_INTERVENTION_CLAIM_BOUNDARY,
    InterventionDesignStatus,
    TargetDirectedInterventionDesignV1,
)

SHA = "a" * 64


def _design(
    source: object,
    target: object,
    candidates: dict[str, object],
    costs: dict[str, float],
    **kwargs: object,
) -> TargetDirectedInterventionDesignV1:
    return TargetDirectedInterventionDesignV1(
        source_design_id=SHA,
        target_query_id=SHA,
        candidate_roster_id=SHA,
        source_design=np.asarray(source, dtype=float),
        target_map=np.asarray(target, dtype=float),
        candidate_intervention_ids={item: SHA for item in candidates},
        candidate_designs={
            item: np.asarray(value, dtype=float) for item, value in candidates.items()
        },
        intervention_costs=costs,
        **kwargs,
    )


def test_target_already_identifiable_requires_no_intervention() -> None:
    design = _design(
        [[1.0, 1.0, 0.0]],
        [[1.0, 1.0, 0.0]],
        {
            "material-probe": [[0.0, 0.0, 1.0]],
            "state-gauge-probe": [[1.0, -1.0, 0.0]],
        },
        {"material-probe": 1.0, "state-gauge-probe": 1.0},
    )

    assert design.status is InterventionDesignStatus.ALREADY_IDENTIFIABLE
    assert design.target_identified
    assert design.selected_interventions == ()
    assert design.selected_total_cost == 0.0
    assert design.minimum_full_cause_identification_cost == 2.0
    assert design.cost_saving_vs_full_cause_identification == 2.0


def test_target_specific_probe_is_cheaper_than_full_cause_identification() -> None:
    design = _design(
        [[1.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]],
        {
            "material-probe": [[0.0, 0.0, 1.0]],
            "state-gauge-probe": [[1.0, -1.0, 0.0]],
        },
        {"material-probe": 1.0, "state-gauge-probe": 1.0},
    )

    assert design.status is InterventionDesignStatus.TARGET_IDENTIFIED
    assert design.selected_interventions == ("material-probe",)
    assert design.selected_total_cost == 1.0
    assert design.minimum_full_cause_interventions == (
        "material-probe",
        "state-gauge-probe",
    )
    assert design.minimum_full_cause_identification_cost == 2.0
    assert design.cost_saving_vs_full_cause_identification == 1.0


def test_two_interventions_can_be_jointly_required() -> None:
    design = _design(
        np.empty((0, 2)),
        [[1.0, 1.0]],
        {
            "observe-first": [[1.0, 0.0]],
            "observe-second": [[0.0, 1.0]],
        },
        {"observe-first": 1.0, "observe-second": 1.0},
    )

    assert design.status is InterventionDesignStatus.TARGET_IDENTIFIED
    assert design.selected_interventions == (
        "observe-first",
        "observe-second",
    )
    assert design.selected_total_cost == 2.0


def test_equal_optima_are_retained_before_canonical_choice() -> None:
    design = _design(
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        {
            "probe-a": [[0.0, 1.0]],
            "probe-b": [[0.0, 2.0]],
        },
        {"probe-a": 1.0, "probe-b": 1.0},
    )

    assert design.status is InterventionDesignStatus.TARGET_IDENTIFIED
    assert design.selected_interventions == ("probe-b",)
    assert design.equally_optimal_subsets == (("probe-b",),)
    assert design.selected_record.target_stability_gain == pytest.approx(0.5)


def test_unresolvable_portfolio_fails_closed() -> None:
    design = _design(
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        {"redundant": [[2.0, 0.0]]},
        {"redundant": 1.0},
    )

    assert design.status is InterventionDesignStatus.UNRESOLVABLE
    assert not design.target_identified
    assert design.selected_total_cost is None
    assert design.source_target_identifiable_dimension == 0
    assert design.maximum_target_identifiable_dimension == 0


def test_partial_improvement_is_reported_without_full_promotion() -> None:
    design = _design(
        np.empty((0, 3)),
        np.eye(3),
        {"one-axis": [[1.0, 0.0, 0.0]]},
        {"one-axis": 1.0},
    )

    assert design.status is InterventionDesignStatus.PARTIAL_IMPROVEMENT
    assert not design.target_identified
    assert design.selected_interventions == ("one-axis",)
    assert design.selected_total_cost is None
    assert design.maximum_target_identifiable_dimension == 1


def test_arrays_are_immutable_and_content_addressed() -> None:
    source = np.asarray([[1.0, 0.0]])
    design = _design(
        source,
        [[0.0, 1.0]],
        {"probe": [[0.0, 1.0]]},
        {"probe": 1.0},
        metadata={"protocol": "source-frozen"},
    )
    artifact_id = design.artifact_id
    source[:] = 9.0

    for value in design.arrays().values():
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 0.0
    np.testing.assert_allclose(design.source_design, [[1.0, 0.0]])
    assert design.artifact_id == artifact_id
    assert design.to_record()["claim_boundary"] == (
        TARGET_DIRECTED_INTERVENTION_CLAIM_BOUNDARY
    )

    roundtrip = _design(
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        {"probe": [[0.0, 1.0]]},
        {"probe": 1.0},
        metadata={"protocol": "source-frozen"},
        artifact_id=artifact_id,
    )
    assert roundtrip.artifact_id == artifact_id
    with pytest.raises(ValueError, match="artifact_id"):
        _design(
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            {"probe": [[0.0, 1.0]]},
            {"probe": 1.0},
            artifact_id="b" * 64,
        )


def test_invalid_candidate_rosters_fail_closed() -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        TargetDirectedInterventionDesignV1(
            source_design_id=SHA,
            target_query_id=SHA,
            candidate_roster_id=SHA,
            source_design=np.asarray([[1.0]]),
            target_map=np.asarray([[1.0]]),
            candidate_intervention_ids={"other": SHA},
            candidate_designs={"probe": np.asarray([[1.0]])},
            intervention_costs={"probe": 1.0},
        )
    with pytest.raises(ValueError, match="nonnegative"):
        _design(
            [[1.0]],
            [[1.0]],
            {"probe": [[1.0]]},
            {"probe": -1.0},
        )
    with pytest.raises(ValueError, match="latent dimension"):
        _design(
            [[1.0]],
            [[1.0]],
            {"probe": [[1.0, 0.0]]},
            {"probe": 1.0},
        )
