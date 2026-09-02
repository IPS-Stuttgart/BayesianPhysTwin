from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.deform_dlo45_decision_identifiability_v1 import (
    support_envelope as envelope,
)
from experiments.deform_dlo45_decision_identifiability_v1 import (
    support_envelope_crossfit as crossfit,
)


def protocol() -> crossfit.CrossfitProtocol:
    return crossfit.CrossfitProtocol(
        dataset_repository="roahmlab/DEFORM",
        dataset_commit="a" * 40,
        route_count=2,
        training_count_per_route=28,
        calibration_count_per_route=28,
        nested_fit_count=21,
        nested_tune_count=7,
        source_split_domain="source",
        nested_split_domain="nested",
        target_route_domain="target",
        candidate_action_mask=(False, True, True),
        miscoverage_levels=(0.05, 0.1, 0.2),
        primary_miscoverage=0.1,
        regret_budget_grid=(0.3, 0.5),
        primary_regret_budget=0.5,
        bootstrap_replicates=10,
        bootstrap_seed=7,
        claim_boundary="test",
    )


def test_complementary_routes_use_every_trajectory_once_per_role() -> None:
    names = tuple(f"{index}.pkl" for index in range(56))
    routes = crossfit.complementary_source_routes(names, "DLO4", protocol())
    assert len(routes) == 2
    for route in routes:
        assert len(route["training"]) == 28
        assert len(route["calibration"]) == 28
        assert set(route["training"]).isdisjoint(route["calibration"])
        assert set(route["training"]) | set(route["calibration"]) == set(names)
    assert routes[0]["training"] == routes[1]["calibration"]
    assert routes[0]["calibration"] == routes[1]["training"]


def test_nested_model_selection_never_uses_route_calibration() -> None:
    names = tuple(f"{index}.pkl" for index in range(28))
    split = crossfit.nested_training_split(names, "DLO5", 1, protocol())
    assert len(split["fit"]) == 21
    assert len(split["tune"]) == 7
    assert set(split["fit"]).isdisjoint(split["tune"])
    assert set(split["fit"]) | set(split["tune"]) == set(names)


def test_target_route_is_deterministic_and_metadata_only() -> None:
    first = [
        crossfit.target_route(f"{index}.pkl", "DLO4", protocol())
        for index in range(50)
    ]
    second = [
        crossfit.target_route(f"{index}.pkl", "DLO4", protocol())
        for index in range(50)
    ]
    assert first == second
    assert set(first) == {0, 1}
    changed = replace(protocol(), target_route_domain="other")
    third = [
        crossfit.target_route(f"{index}.pkl", "DLO4", changed)
        for index in range(50)
    ]
    assert first != third


def test_trajectory_score_is_simultaneous_over_nonfallback_actions() -> None:
    records = []
    for index, (realized, registered) in enumerate(
        (
            ([0.0, 0.3, 0.4], [0.2, 0.2, 0.3]),
            ([0.0, 0.5, 0.1], [0.1, 0.2, 0.2]),
        )
    ):
        records.append(
            envelope.WindowMeasurement(
                stable_id=f"DLO4/1.pkl/{index}",
                dlo="DLO4",
                trajectory="1.pkl",
                current_frame=index,
                registered_regret=np.asarray(registered),
                realized_regret=np.asarray(realized),
                physical_mse=np.ones(3),
                fallback_mse=1.0,
                base_certificate_action=0,
            )
        )
    assert crossfit._trajectory_score(records, (False, True, True)) == 0.3
