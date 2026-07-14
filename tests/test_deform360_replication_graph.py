from __future__ import annotations

import numpy as np

from causal4d_public.deform360_replication_graph import (
    build_filament_sparse_graph,
    build_sheet_sparse_graph,
    build_sparse_graph_for_stratum,
    build_volumetric_sparse_graph,
)


def test_filament_graph_is_an_ordered_chain_with_bend_springs() -> None:
    x = np.linspace(-0.2, 0.2, 400)
    points = np.column_stack((x, 0.01 * np.sin(12.0 * x), np.zeros_like(x)))
    graph = build_filament_sparse_graph(points)

    assert graph.positions_m.shape == (21, 3)
    assert np.count_nonzero(graph.spring_families == 0) == 20
    assert np.count_nonzero(graph.spring_families == 1) == 19
    assert graph.stratum == "filament"


def test_sheet_graph_has_fixed_lattice_and_two_spring_families() -> None:
    u, v = np.meshgrid(np.linspace(-0.3, 0.3, 40), np.linspace(-0.2, 0.2, 30))
    points = np.column_stack((u.ravel(), v.ravel(), 0.02 * u.ravel() ** 2))
    graph = build_sheet_sparse_graph(points)

    assert graph.positions_m.shape == (25, 3)
    assert set(graph.spring_families.tolist()) == {0, 1}
    assert len(np.unique(np.sort(graph.spring_edges, axis=1), axis=0)) == len(
        graph.spring_edges
    )
    assert graph.diagnostics["lattice_shape"] == [5, 5]


def test_volumetric_graph_is_deterministic_and_connected() -> None:
    rng = np.random.default_rng(17)
    directions = rng.normal(size=(1000, 3))
    points = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    first = build_volumetric_sparse_graph(points)
    second = build_volumetric_sparse_graph(points)

    np.testing.assert_array_equal(first.positions_m, second.positions_m)
    np.testing.assert_array_equal(first.spring_edges, second.spring_edges)
    visited = {0}
    while True:
        expanded = visited | {
            int(right if left in visited else left)
            for left, right in first.spring_edges[first.spring_families == 0]
            if int(left) in visited or int(right) in visited
        }
        if expanded == visited:
            break
        visited = expanded
    assert len(visited) == len(first.positions_m)


def test_stratum_dispatch_rejects_unknown_family() -> None:
    points = np.eye(3)
    try:
        build_sparse_graph_for_stratum(points, "unknown")
    except ValueError as error:
        assert "unsupported Deform360 stratum" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown stratum was accepted")
