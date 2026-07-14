from pathlib import Path

import numpy as np

from causal4d_public.deform360_replication_geometry import (
    ReplicationGeometryConfig,
    _geometry_quality,
    _initial_hull,
    load_replication_hull_archive,
    pack_multiview_masks,
    replication_geometry_frame_indices,
    unpack_multiview_masks,
)


def test_geometry_frame_indices_include_endpoint_and_final_frame() -> None:
    config = ReplicationGeometryConfig(prefix_frame_count=6, score_frame_stride=6)
    assert replication_geometry_frame_indices(30, 4, config) == (9, 10, 16, 22, 28, 29)
    assert replication_geometry_frame_indices(30, 4, config, prefix_only=True) == (9,)


def test_prefix_only_geometry_has_vacuous_future_availability() -> None:
    quality = _geometry_quality(
        [np.zeros((64, 3))], ReplicationGeometryConfig()
    )
    assert quality == {
        "available_frame_count": 1,
        "total_frame_count": 1,
        "available_future_frame_fraction": 1.0,
    }


def test_multiview_mask_pack_roundtrip() -> None:
    rng = np.random.default_rng(3)
    masks = rng.random((3, 4, 7, 13)) > 0.7
    packed, shape = pack_multiview_masks(masks)
    restored = unpack_multiview_masks(packed, shape)
    assert np.array_equal(restored, masks)


def test_initial_hull_never_relaxes_below_locked_consensus() -> None:
    cameras = [f"camera-{index}" for index in range(8)]
    masks = {camera: np.ones((32, 32), dtype=bool) for camera in cameras}
    intrinsics = {
        camera: np.array([[10.0, 0.0, 16.0], [0.0, 10.0, 16.0], [0.0, 0.0, 1.0]])
        for camera in cameras
    }
    extrinsics = {camera: np.eye(4) for camera in cameras}
    config = ReplicationGeometryConfig(
        initial_cube_half_extent_m=0.5,
        initial_voxel_resolution=16,
        initial_minimum_hull_points=16,
        minimum_consensus_votes=8,
    )
    hull, diagnostics = _initial_hull(masks, intrinsics, extrinsics, config)
    assert len(hull) >= 16
    assert diagnostics["carving"]["required_vote_count"] == 8
    assert diagnostics["selected_mask_dilation_radius_pixels"] == 0


def test_load_ragged_hull_archive(tmp_path: Path) -> None:
    frames = np.array([5, 11], dtype=np.int32)
    offsets = np.array([0, 2, 5], dtype=np.int64)
    points = np.arange(15, dtype=np.float64).reshape(5, 3) / 10.0
    archive = tmp_path / "hulls.npz"
    np.savez_compressed(
        archive,
        frame_indices=frames,
        point_offsets=offsets,
        points_world_m=points,
    )
    import hashlib
    import json

    def array_sha(value: np.ndarray) -> str:
        array = np.ascontiguousarray(value)
        descriptor = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(descriptor + b"\0" + array.view(np.uint8).tobytes()).hexdigest()

    payload = {
        "schema_version": 2,
        "artifact_kind": "Deform360ReplicationSampledVisualHulls",
        "archive": {
            "path": str(archive),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "points_sha256": array_sha(points),
            "offsets_sha256": array_sha(offsets),
        },
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    loaded_frames, hulls = load_replication_hull_archive(payload)
    assert np.array_equal(loaded_frames, frames)
    assert [len(hull) for hull in hulls] == [2, 3]
