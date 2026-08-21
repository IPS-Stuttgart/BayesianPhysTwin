from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.deform360_matphys_part_features_v1 import (
    aggregate_direct_node_features,
    build_part_arrays,
    material_distribution_for_stratum,
)


def test_material_prior_is_frozen_by_stratum() -> None:
    sheet = material_distribution_for_stratum("sheet", part_count=3)
    volumetric = material_distribution_for_stratum("volumetric", part_count=2)

    assert sheet.shape == (3, 10)
    np.testing.assert_array_equal(sheet[:, 2], np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(np.sum(sheet, axis=1), np.ones(3))
    np.testing.assert_allclose(volumetric[:, [0, 1, 4]], 1.0 / 3.0)
    np.testing.assert_allclose(np.sum(volumetric, axis=1), 1.0)
    with pytest.raises(ValueError, match="stratum"):
        material_distribution_for_stratum("unknown", part_count=1)


def test_multiview_aggregation_uses_only_supported_rows() -> None:
    sampled = {
        "camera-b": np.array([[0.0, 2.0], [0.0, 0.0], [1.0, 1.0]]),
        "camera-a": np.array([[2.0, 0.0], [3.0, 0.0], [0.0, 0.0]]),
    }
    support = {
        "camera-a": np.array([True, True, False]),
        "camera-b": np.array([True, False, True]),
    }

    features, counts = aggregate_direct_node_features(sampled, support)

    np.testing.assert_array_equal(counts, [2, 1, 1])
    np.testing.assert_allclose(features[0], [0.5, 0.5])
    np.testing.assert_allclose(features[1], [1.0, 0.0])
    np.testing.assert_allclose(features[2], [2**-0.5, 2**-0.5])


def test_graph_parts_fill_unseen_nodes_without_future_evidence() -> None:
    points = np.column_stack(
        (
            np.arange(6, dtype=np.float32),
            np.zeros(6, dtype=np.float32),
            np.zeros(6, dtype=np.float32),
        )
    )
    edges = np.column_stack((np.arange(5), np.arange(1, 6)))
    features = np.zeros((6, 4), dtype=np.float32)
    features[0] = [1.0, 0.0, 0.0, 0.0]
    features[5] = [0.0, 1.0, 0.0, 0.0]
    counts = np.array([2, 0, 0, 0, 0, 3], dtype=np.int32)

    result = build_part_arrays(
        points,
        edges,
        features,
        counts,
        stratum="sheet",
        part_count=2,
    )

    assert result.point_part.shape == (6,)
    assert set(result.point_part.tolist()) == {0, 1}
    assert result.part_features.shape == (2, 4)
    np.testing.assert_array_equal(result.contributor_count, counts)
    np.testing.assert_array_equal(result.material_distribution[:, 2], [1.0, 1.0])
    np.testing.assert_allclose(np.linalg.norm(result.node_features, axis=1), 1.0)
