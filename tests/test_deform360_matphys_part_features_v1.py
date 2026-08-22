from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.deform360_matphys_part_features_v1 import (
    aggregate_direct_node_features,
    array_sha256,
    build_part_arrays,
    material_distribution_for_stratum,
    ordinary_file,
)


def test_array_hash_binds_dtype_shape_and_bytes() -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)

    assert array_sha256(values) == array_sha256(values[:, ::-1][:, ::-1])
    assert array_sha256(values) != array_sha256(values.astype(np.float64))
    assert array_sha256(values) != array_sha256(values.reshape(3, 2))


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
    with pytest.raises(ValueError, match="part_count"):
        material_distribution_for_stratum("sheet", part_count=0)
    with pytest.raises(ValueError, match="material_class_count"):
        material_distribution_for_stratum(
            "sheet",
            part_count=1,
            material_class_count=4,
        )


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


@pytest.mark.parametrize(
    "invalid_support",
    [
        np.array([1.0, 0.0]),
        np.array([1.0, np.nan]),
        np.array([1, 0], dtype=np.int64),
    ],
)
def test_multiview_aggregation_rejects_nonboolean_support(
    invalid_support: np.ndarray,
) -> None:
    sampled = {"camera-a": np.array([[1.0, 0.0], [0.0, 1.0]])}

    with pytest.raises(ValueError, match="boolean dtype"):
        aggregate_direct_node_features(
            sampled,
            {"camera-a": invalid_support},
        )


def test_multiview_aggregation_rejects_invalid_camera_evidence() -> None:
    with pytest.raises(ValueError, match="same cameras"):
        aggregate_direct_node_features({}, {})
    with pytest.raises(ValueError, match="same cameras"):
        aggregate_direct_node_features(
            {"camera-a": np.ones((2, 2))},
            {"camera-b": np.ones(2, dtype=np.bool_)},
        )
    with pytest.raises(ValueError, match="feature shape"):
        aggregate_direct_node_features(
            {"camera-a": np.ones((1, 2))},
            {"camera-a": np.ones(2, dtype=np.bool_)},
        )
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_direct_node_features(
            {"camera-a": np.array([[np.nan, 0.0]])},
            {"camera-a": np.ones(1, dtype=np.bool_)},
        )
    with pytest.raises(ValueError, match="zero supported feature"):
        aggregate_direct_node_features(
            {"camera-a": np.zeros((1, 2))},
            {"camera-a": np.ones(1, dtype=np.bool_)},
        )
    with pytest.raises(ValueError, match="shapes disagree"):
        aggregate_direct_node_features(
            {
                "camera-a": np.ones((2, 2)),
                "camera-b": np.ones((2, 3)),
            },
            {
                "camera-a": np.ones(2, dtype=np.bool_),
                "camera-b": np.ones(2, dtype=np.bool_),
            },
        )
    with pytest.raises(ValueError, match="no node"):
        aggregate_direct_node_features(
            {"camera-a": np.zeros((2, 2))},
            {"camera-a": np.zeros(2, dtype=np.bool_)},
        )


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


def test_graph_parts_reject_malformed_graph_arrays() -> None:
    with pytest.raises(ValueError, match="points_m"):
        build_part_arrays(
            np.zeros((2, 2)),
            np.array([[0, 1]]),
            np.ones((2, 2)),
            np.ones(2),
            stratum="sheet",
        )
    with pytest.raises(ValueError, match="cover every graph node"):
        build_part_arrays(
            np.zeros((2, 3)),
            np.array([[0, 1]]),
            np.ones((1, 2)),
            np.ones(1),
            stratum="sheet",
        )


def test_ordinary_file_rejects_missing_and_symlinked_inputs(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"registered")

    assert ordinary_file(source, name="source") == source.resolve()
    with pytest.raises(ValueError, match="ordinary non-symlink"):
        ordinary_file(tmp_path / "missing.bin", name="missing")

    direct_link = tmp_path / "direct-link.bin"
    direct_link.symlink_to(source)
    with pytest.raises(ValueError, match="ordinary non-symlink"):
        ordinary_file(direct_link, name="direct link")

    real_directory = tmp_path / "real-directory"
    real_directory.mkdir()
    nested = real_directory / "nested.bin"
    nested.write_bytes(b"registered")
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="ordinary non-symlink"):
        ordinary_file(linked_directory / "nested.bin", name="nested link")
