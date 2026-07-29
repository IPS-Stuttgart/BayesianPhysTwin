from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_assets import (
    canonical_sha256,
)
from bayesian_phystwin.deform360_causal_response_prefix_geometry import (
    GEOMETRY_CONTRACT,
    GEOMETRY_MANIFEST_KIND,
    GEOMETRY_PROTOCOL_ID,
    load_v14_prefix_geometry_protocol,
    mask_binding_for_rank,
    projected_seed_support,
    validate_v14_prefix_geometry_manifest,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14_assets.json"
)
GEOMETRY = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14_prefix_geometry.json"
)
GEOMETRY_MODULE = ROOT / "src" / "bayesian_phystwin" / (
    "deform360_causal_response_prefix_geometry.py"
)
GEOMETRY_BUILDER = ROOT / "scripts" / "remote" / (
    "build_deform360_causal_response_direct_depth_v14_prefix_geometry.py"
)


def test_v14_prefix_geometry_protocol_binds_existing_mask_dispositions() -> None:
    protocol = load_v14_prefix_geometry_protocol(GEOMETRY)

    assert protocol["parent_prefix_assets"]["file_sha256"] == file_sha256(ASSETS)
    assert [row["queue_rank"] for row in protocol["mask_inputs"]] == list(
        range(3, 15)
    )
    assert sum(row["successful_camera_count"] for row in protocol["mask_inputs"]) == 140
    assert protocol["implementation_file_sha256"] == {
        "geometry_module": file_sha256(GEOMETRY_MODULE),
        "geometry_builder": file_sha256(GEOMETRY_BUILDER),
    }
    assert protocol["geometry"]["projected_support_role"].endswith(
        "not an additional admission gate"
    )
    assert mask_binding_for_rank(protocol, 8)["successful_camera_count"] == 11
    with pytest.raises(ValueError, match="lacks one mask binding"):
        mask_binding_for_rank(protocol, 2)


def test_projected_seed_support_counts_nodes_not_correlated_cameras() -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    intrinsics = np.array(
        [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    depth = np.zeros((5, 5), dtype=np.uint16)
    depth[2, 2] = 1000
    depth[2, 3] = 1000

    count, per_camera = projected_seed_support(
        points,
        intrinsics_by_camera={"a": intrinsics, "b": intrinsics},
        camera_to_world_by_camera={"a": np.eye(4), "b": np.eye(4)},
        depth_mm_by_camera={"a": depth, "b": depth.copy()},
        depth_tolerance_m=0.03,
    )

    assert count == 2
    assert per_camera == {"a": 2, "b": 2}


def _write_geometry_manifest(
    tmp_path: Path,
    protocol: dict[str, object],
    *,
    physical_node_count: int = 128,
) -> tuple[Path, Path]:
    episode = tmp_path / "geometry" / "episode_0000"
    camera = "camera-a"
    paths = {
        "intrinsics": episode / "undistorted_intrinsics.npy",
        "extrinsics": episode / "extrinsics.npy",
        "robot": episode / "robot" / "robot.npz",
        "frame_zero_splat": episode / "splatfacto" / "splat_000000.ply",
        "frame_zero_points": episode / "start_obj_pcd.ply",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))
    depth = episode / camera / "rendered_depth.h5"
    depth.parent.mkdir(parents=True)
    depth.write_bytes(b"depth")
    manifest = {
        "schema_version": 1,
        "artifact_kind": GEOMETRY_MANIFEST_KIND,
        "contract": GEOMETRY_CONTRACT,
        "protocol_id": GEOMETRY_PROTOCOL_ID,
        "geometry_protocol_config_sha256": protocol["config_sha256"],
        "status": "ready_for_physical_preflight",
        "physical_node_count": physical_node_count,
        "cameras": [camera],
        "camera_records": [
            {
                "camera": camera,
                "rgb_frame_count": 58,
                "mask_frame_count": 58,
                "depth_frame_count": 58,
                "gripper_mask_frame_count": 58,
            }
        ],
        "calibration_valid": True,
        "outputs_sha256": {
            **{name: file_sha256(path) for name, path in paths.items()},
            "depth_by_camera": {camera: file_sha256(depth)},
        },
    }
    manifest["artifact_sha256"] = canonical_sha256(
        manifest,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-v14\0"
        ),
        digest_key="artifact_sha256",
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return path, episode


def test_v14_prefix_geometry_manifest_is_hash_only_and_admissible(
    tmp_path: Path,
) -> None:
    protocol = load_v14_prefix_geometry_protocol(GEOMETRY)
    manifest, episode = _write_geometry_manifest(tmp_path, protocol)

    loaded = validate_v14_prefix_geometry_manifest(
        manifest,
        protocol=protocol,
        geometry_episode=episode,
        forbidden_plaintext="secret-object-id",
    )

    assert loaded["physical_node_count"] == 128


def test_v14_prefix_geometry_manifest_rejects_backend_inadmissibility(
    tmp_path: Path,
) -> None:
    protocol = load_v14_prefix_geometry_protocol(GEOMETRY)
    manifest, episode = _write_geometry_manifest(
        tmp_path,
        protocol,
        physical_node_count=127,
    )

    with pytest.raises(ValueError, match="node count is inadmissible"):
        validate_v14_prefix_geometry_manifest(
            manifest,
            protocol=protocol,
            geometry_episode=episode,
        )
