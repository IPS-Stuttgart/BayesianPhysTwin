from __future__ import annotations

import numpy as np
import pytest

import bayesian_phystwin.causal4d_graph_provider_v1 as provider
from bayesian_phystwin.causal4d_graph_provider_v1 import (
    CAUSAL4D_GRAPH_PROVIDER_API_VERSION,
    CAUSAL4D_GRAPH_PROVIDER_PACKAGE_VERSION,
    PARENT_PROVIDER_API_VERSION,
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
    causal4d_graph_provider_manifest,
    controller_hand_count,
    infer_controller_groups,
)


def test_graph_provider_manifest_is_versioned_and_parented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version_calls: list[tuple[str, str]] = []

    def installed_version(name: str, *, fallback: str) -> str:
        version_calls.append((name, fallback))
        return "0.4.9"

    monkeypatch.setattr(provider, "installed_distribution_version", installed_version)
    manifest = causal4d_graph_provider_manifest(provider_revision="abc123")
    assert manifest == {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.9",
        "provider_revision": "abc123",
        "schema_version": CAUSAL4D_GRAPH_PROVIDER_API_VERSION,
        "capabilities": ["controller_grouping", "phystwin_spring_graph"],
        "artifact_schema_versions": {"PhysTwinSpringGraph": 1},
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_graph_provider_v1",
            "provider_api_version": 1,
            "parent_provider_api": "bayesian_phystwin.causal4d_provider_v2",
            "parent_provider_api_version": PARENT_PROVIDER_API_VERSION,
        },
    }
    assert version_calls == [
        ("bayesian-phystwin", CAUSAL4D_GRAPH_PROVIDER_PACKAGE_VERSION)
    ]


def test_graph_provider_manifest_revision_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provider,
        "installed_distribution_version",
        lambda name, *, fallback: fallback,
    )
    revision_calls: list[str] = []

    def installed_revision(name: str) -> str | None:
        revision_calls.append(name)
        return "installed"

    monkeypatch.setattr(provider, "installed_distribution_revision", installed_revision)
    monkeypatch.setenv("BAYESIAN_PHYSTWIN_REVISION", "environment")
    assert causal4d_graph_provider_manifest()["provider_revision"] == "environment"
    assert revision_calls == []

    monkeypatch.delenv("BAYESIAN_PHYSTWIN_REVISION")
    assert causal4d_graph_provider_manifest()["provider_revision"] == "installed"
    assert revision_calls == ["bayesian-phystwin"]

    monkeypatch.setattr(
        provider,
        "installed_distribution_revision",
        lambda name: None,
    )
    assert (
        causal4d_graph_provider_manifest()["provider_revision"]
        == "unversioned-install"
    )


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

    reversed_points = points[::-1].copy()
    np.testing.assert_array_equal(
        infer_controller_groups(reversed_points, group_count=2),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )

    identical = np.zeros((4, 3), dtype=float)
    np.testing.assert_array_equal(
        infer_controller_groups(identical, group_count=2),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
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
        controller_hand_count(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="case_name"):
        controller_hand_count("")
    with pytest.raises(ValueError, match="finite shape"):
        infer_controller_groups(np.zeros(3), group_count=1)
    with pytest.raises(ValueError, match="finite shape"):
        infer_controller_groups(np.zeros((2, 2)), group_count=1)
    with pytest.raises(ValueError, match="finite shape"):
        infer_controller_groups(np.zeros((1, 3)), group_count=2)
    with pytest.raises(ValueError, match="finite shape"):
        infer_controller_groups(np.asarray([[np.nan, 0.0, 0.0]]), group_count=1)
    with pytest.raises(ValueError, match="one or two hands"):
        infer_controller_groups(np.zeros((3, 3)), group_count=3)
