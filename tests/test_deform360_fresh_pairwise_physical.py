from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_fresh_pairwise_physical import (
    CANONICAL_NODE_COUNT,
    build_warp_backbone_arrays,
    load_frame_zero_ply,
)


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(points), dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, axis]
    path.write_bytes(header + vertices.tobytes())


def test_frame_zero_ply_loader_preserves_xyz_and_rgb(tmp_path: Path) -> None:
    points = np.arange(384, dtype=np.float32).reshape(128, 3) / 1000.0
    colors = np.mod(np.arange(384).reshape(128, 3), 256).astype(np.uint8)
    path = tmp_path / "start_obj_pcd.ply"
    _write_binary_ply(path, points, colors)

    loaded_points, loaded_colors = load_frame_zero_ply(path)

    np.testing.assert_array_equal(loaded_points, points)
    np.testing.assert_allclose(loaded_colors, colors.astype(np.float32) / 255.0)
    assert CANONICAL_NODE_COUNT == 384


def test_driven_minus_zero_backbone_uses_frozen_action_support() -> None:
    point_count = 128
    node_count = 2
    initial = np.zeros((point_count, 3), dtype=np.float64)
    initial[:, 0] = np.linspace(0.0, 0.1, point_count)
    vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    springs = np.array([[0, 1]], dtype=np.int64)
    rest_lengths = np.array([0.1])
    weights = np.zeros((point_count, node_count))
    weights[:, 0] = 1.0
    driven = np.repeat(vertices[None], 76, axis=0)
    zero = driven.copy()
    driven[:, 0, 1] += np.linspace(0.0, 0.05, 76)

    arrays = build_warp_backbone_arrays(
        initial,
        vertices=vertices,
        springs=springs,
        rest_lengths=rest_lengths,
        contact_anchor_indices=np.array([0]),
        readout_weights=weights,
        driven_vertices_m=driven,
        zero_action_vertices_m=zero,
    )

    np.testing.assert_array_equal(arrays["prediction_m"][0], initial.astype(np.float32))
    np.testing.assert_allclose(
        arrays["prediction_m"][-1, :, 1],
        np.full(point_count, 0.9 * 0.05),
        atol=1e-7,
    )
    assert np.all(arrays["action_support"] == 1.0)
