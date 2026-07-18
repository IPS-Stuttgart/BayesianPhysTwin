import numpy as np

from bayesian_phystwin.phystwin_zero_order_topology import (
    ZeroOrderTopologySearchConfig,
    generate_topology_field_candidates,
    select_topology_field_candidate,
)


def test_zero_order_candidate_bank_is_deterministic_and_bounded() -> None:
    config = ZeroOrderTopologySearchConfig(
        region_count=3,
        candidates_per_family=4,
        seed=17,
    )

    first = generate_topology_field_candidates(config)
    second = generate_topology_field_candidates(config)

    assert first == second
    assert len(first) == 13
    assert len({candidate.candidate_id for candidate in first}) == len(first)
    assert first[0].candidate_id == "exact_teacher"
    assert {candidate.family for candidate in first} == {
        "identity",
        "topology_only",
        "field_only",
        "joint",
    }
    for candidate in first[1:]:
        assert len(candidate.radius_multipliers) == 3
        assert len(candidate.neighbour_multipliers) == 3
        assert len(candidate.region_object_log_scales) == 3
        assert np.all(
            np.asarray(candidate.radius_multipliers) >= config.radius_bounds[0]
        )
        assert np.all(
            np.asarray(candidate.radius_multipliers) <= config.radius_bounds[1]
        )
        assert np.all(
            np.asarray(candidate.neighbour_multipliers)
            >= config.neighbour_bounds[0]
        )
        assert np.all(
            np.asarray(candidate.neighbour_multipliers)
            <= config.neighbour_bounds[1]
        )


def test_fit_selector_skips_lower_score_with_metric_regression() -> None:
    config = ZeroOrderTopologySearchConfig(
        region_count=2,
        candidates_per_family=1,
        minimum_fit_improvement=0.01,
        maximum_fit_metric_ratio=1.02,
    )
    candidates = generate_topology_field_candidates(config)
    metrics = {
        "exact_teacher": {
            "chamfer_distance_m": 0.010,
            "track_error_m": 0.020,
        },
        "topology_000": {
            "chamfer_distance_m": 0.009,
            "track_error_m": 0.019,
        },
        "field_000": {
            "chamfer_distance_m": 0.007,
            "track_error_m": 0.0206,
        },
        "joint_000": {
            "chamfer_distance_m": 0.0101,
            "track_error_m": 0.0199,
        },
    }

    selection = select_topology_field_candidate(metrics, candidates, config)

    assert selection["best_raw_candidate_id"] == "field_000"
    assert selection["selected_candidate_id"] == "topology_000"
    assert selection["candidate_accepted"] is True
    assert selection["fallback"] is None


def test_fit_selector_falls_back_to_exact_teacher() -> None:
    config = ZeroOrderTopologySearchConfig(
        region_count=2,
        candidates_per_family=1,
    )
    candidates = generate_topology_field_candidates(config)
    metrics = {
        candidate.candidate_id: {
            "chamfer_distance_m": 0.0101,
            "track_error_m": 0.0201,
        }
        for candidate in candidates
    }
    metrics["exact_teacher"] = {
        "chamfer_distance_m": 0.010,
        "track_error_m": 0.020,
    }

    selection = select_topology_field_candidate(metrics, candidates, config)

    assert selection["selected_candidate_id"] == "exact_teacher"
    assert selection["candidate_accepted"] is False
    assert selection["fallback"] == "exact_teacher"
