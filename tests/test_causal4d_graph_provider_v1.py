from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.causal4d_graph_provider_v1 import (
    CAUSAL4D_GRAPH_PROVIDER_API_VERSION,
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
    causal4d_graph_provider_manifest,
    controller_hand_count,
    infer_controller_groups,
)


def test_graph_provider_manifest_is_versioned() -> None:
    manifest = causal4d_graph_provider_manifest(provider_revision="abc123")
    assert manifest == {
        "provider_name": "bayesian-phystwin",
        "provider_version": manifest["provider_version"],
        "provider_revision": "abc123",
        "schema_version": CAUSAL4D_GRAPH_PROVIDER_API_VERSION,
        "capabilities": ["controller_grouping", "phystwin_spring_graph"],
        "artifact_schema_versions": {"PhysTwinSpringGraph": 1},
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_graph_provider_v1",
            "provider_api_version": 1,
        },
    }


@pytest.mark.parametrize(
    ("case_name", "expected"),
    [
        ("single_lift_sloth", 1),
        ("double_stretch_sloth", 2),
        ("rope_double_hand", 2),
    ],
)
def test_controller_hand_count_matches_released_contract(
    case_name: str,
    expected: int,
) -> None:
    assert controller_hand_count(case_name) == expected


def test_controller_grouping_is_deterministic_and_ordered() -> None:
    points = np.asarray(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.1, 0.0],
            [1.0, -0.1, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    labels = infer_controller_groups(points, group_count=2)
    np.testing.assert_array_equal(labels, np.asarray([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        infer_controller_groups(points, group_count=1),
        np.zeros(4, dtype=np.int32),
    )


def test_graph_builder_is_exposed_without_experiment_imports() -> None:
    structure = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    controls = np.asarray([[0.0, 0.05, 0.0]], dtype=np.float32)
    graph = build_phystwin_spring_graph(
        structure,
        controls,
        config=PhysTwinSpringGraphConfig(
            object_radius=0.15,
            object_max_neighbours=3,
            controller_radius=0.1,
            controller_max_neighbours=2,
        ),
    )
    assert isinstance(graph, PhysTwinSpringGraph)
    assert graph.vertices.shape == (4, 3)
    assert graph.springs.ndim == 2 and graph.springs.shape[1] == 2
    assert graph.rest_lengths.shape == (len(graph.springs),)
    assert graph.num_object_points == 3


def test_graph_provider_rejects_invalid_controller_inputs() -> None:
    with pytest.raises(ValueError, match="case_name"):
        controller_hand_count("")
    with pytest.raises(ValueError, match="finite shape"):
        infer_controller_groups(np.asarray([[np.nan, 0.0, 0.0]]), group_count=1)
    with pytest.raises(ValueError, match="one or two hands"):
        infer_controller_groups(np.zeros((3, 3)), group_count=3)
