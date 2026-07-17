from __future__ import annotations

import numpy as np

from bayesian_phystwin.phystwin_graph import PhysTwinSpringGraphConfig
from causal4d_public.deform360_partial_graph_state import (
    PartialGraphStateConfig,
    fit_partial_graph_state,
)
from causal4d_public.deform360_reusable_graph import (
    ReusableGraphRegistrationConfig,
    build_canonical_deform360_graph,
)


def test_partial_state_completion_preserves_graph_and_explains_observations() -> None:
    coordinate = np.linspace(0.0, 1.0, 20)
    source = np.column_stack((0.10 * coordinate, np.zeros(20), np.zeros(20)))
    colors = np.column_stack((coordinate, 1.0 - coordinate, np.full(20, 0.3)))
    canonical = build_canonical_deform360_graph(
        source,
        colors,
        registration_config=ReusableGraphRegistrationConfig(
            canonical_node_count=20,
        ),
        spring_config=PhysTwinSpringGraphConfig(
            object_radius=0.012,
            object_max_neighbours=4,
            controller_radius=0.02,
            controller_max_neighbours=1,
        ),
    )
    angle = np.linspace(0.1, 1.2, 15)
    target = np.column_stack(
        (
            0.04 + 0.055 * np.cos(angle),
            0.055 * np.sin(angle),
            np.zeros(15),
        )
    )
    target_colors = colors[3:18]
    result = fit_partial_graph_state(
        canonical,
        target,
        target_colors,
        config=PartialGraphStateConfig(
            start_count=3,
            anchor_count=5,
            iterations=250,
            learning_rate=0.003,
            observation_scale_m=0.01,
            hidden_node_distance_cap_m=0.02,
            hidden_node_fit_weight=0.25,
            edge_strain_weight=20.0,
            readout_neighbour_count=3,
            maximum_supported_distance_m=0.02,
            minimum_observed_target_fraction=0.90,
            minimum_effective_target_reliability=0.50,
            maximum_p99_relative_edge_strain=0.50,
        ),
        device="cpu",
    )
    assert result.metrics["passed"] is True
    assert result.metrics["observed_target_fraction"] >= 0.90
    assert result.metrics["p99_absolute_relative_edge_strain"] <= 0.50
    assert result.readout_weights.shape == (15, 20)
    np.testing.assert_allclose(result.readout_weights.sum(axis=1), 1.0)
    assert result.readout_covariance_m2.shape == (15, 3, 3)
    assert result.state_covariance_m2.shape == (20, 3, 3)


def test_duplicate_target_observations_do_not_remove_assignment_uncertainty() -> None:
    source = np.column_stack((np.linspace(0.0, 0.08, 10), np.zeros(10), np.zeros(10)))
    colors = np.tile(np.asarray([[0.2, 0.5, 0.7]]), (10, 1))
    canonical = build_canonical_deform360_graph(
        source,
        colors,
        registration_config=ReusableGraphRegistrationConfig(
            canonical_node_count=10,
        ),
        spring_config=PhysTwinSpringGraphConfig(
            object_radius=0.02,
            object_max_neighbours=4,
            controller_radius=0.02,
            controller_max_neighbours=1,
        ),
    )
    target = np.repeat(source[2:8], 2, axis=0)
    target_colors = np.repeat(colors[2:8], 2, axis=0)
    result = fit_partial_graph_state(
        canonical,
        target,
        target_colors,
        config=PartialGraphStateConfig(
            start_count=2,
            anchor_count=3,
            iterations=80,
            edge_strain_weight=20.0,
            minimum_observed_target_fraction=0.80,
            minimum_effective_target_reliability=0.30,
        ),
        device="cpu",
    )
    assert np.all(result.target_prior_reliability < 1.0)
    assert np.all(np.trace(result.readout_covariance_m2, axis1=1, axis2=2) >= 12e-6)
