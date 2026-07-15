from __future__ import annotations

import numpy as np
import pytest

from causal4d_public.deform360_replication_graph import Deform360SparseGraph
from causal4d_public.deform360_reusable_twin import (
    fit_reusable_filament_twin,
    load_reusable_twin_artifact,
    write_reusable_twin_artifact,
)


def _filament(length_m: float) -> Deform360SparseGraph:
    positions = np.column_stack(
        (np.linspace(0.0, length_m, 5), np.zeros(5), np.zeros(5))
    )
    return Deform360SparseGraph(
        positions_m=positions,
        spring_edges=np.asarray(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 2],
                [1, 3],
                [2, 4],
            ],
            dtype=np.int32,
        ),
        spring_families=np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int8),
        masses=np.ones(5),
        stratum="filament",
        diagnostics={},
    )


def test_reusable_filament_twin_fits_source_only_lower_envelope(tmp_path) -> None:
    graphs = [_filament(length) for length in (0.4, 0.5, 0.6)]
    hashes = ("1" * 64, "2" * 64, "3" * 64)
    twin = fit_reusable_filament_twin(
        "rope",
        graphs,
        ("rope/episode_0000", "rope/episode_0001", "rope/episode_0002"),
        hashes,
        rest_length_quantile=0.10,
    )

    assert twin.fit_policy["future_outcomes_read"] is False
    assert twin.diagnostics["selected_total_rest_length_m"] == pytest.approx(0.42)
    np.testing.assert_allclose(twin.object_rest_lengths_m[:4], 0.105)
    np.testing.assert_allclose(twin.object_rest_lengths_m[4:], 0.210)

    path = write_reusable_twin_artifact(tmp_path / "twin.json", twin)
    restored = load_reusable_twin_artifact(path)
    np.testing.assert_array_equal(
        restored.rest_lengths_for_graph(graphs[1]), twin.object_rest_lengths_m
    )
    assert restored.as_artifact() == twin.as_artifact()


def test_reusable_twin_rejects_changed_episode_topology() -> None:
    graphs = [_filament(length) for length in (0.4, 0.5, 0.6)]
    twin = fit_reusable_filament_twin(
        "rope",
        graphs,
        ("e0", "e1", "e2"),
        ("a" * 64, "b" * 64, "c" * 64),
    )
    changed = _filament(0.5)
    changed = Deform360SparseGraph(
        positions_m=changed.positions_m,
        spring_edges=changed.spring_edges[:-1],
        spring_families=changed.spring_families[:-1],
        masses=changed.masses,
        stratum=changed.stratum,
        diagnostics={},
    )
    with pytest.raises(ValueError, match="topology differs"):
        twin.rest_lengths_for_graph(changed)
