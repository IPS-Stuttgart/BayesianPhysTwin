from __future__ import annotations

import json

import numpy as np

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    AdaptiveCausalResponseQueryConfig,
    build_adaptive_causal_response_query_schedule,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    AdaptiveDirectDepthSourcePreflightConfigV14,
    evaluate_adaptive_direct_depth_source_preflight_v14,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    CausalResponseSourceCameraRecord,
)


def _carrier(*, masks_available: bool = True):
    camera_count = 8
    node_count = 20
    height = width = 96
    coordinate = np.linspace(-1.0, 1.0, node_count)
    frame_zero = np.column_stack(
        (
            0.18 * coordinate,
            0.04 * np.sin(np.pi * coordinate),
            np.full(node_count, 2.0),
        )
    )
    graph_basis = np.zeros((node_count, 3, 8))
    for mode in range(8):
        graph_basis[:, mode % 3, mode] = coordinate ** (mode % 4 + 1)
    support = np.ones(node_count)
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 60.0
    intrinsics[:, 1, 1] = 60.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    poses = np.repeat(np.eye(4)[None], camera_count, axis=0)
    depth = np.full((camera_count, height, width), 2.0)
    masks = np.full_like(depth, masks_available, dtype=bool)
    camera_ids = REGISTERED_CAMERA_IDS[:camera_count]
    return build_adaptive_causal_response_query_schedule(
        frame_zero,
        graph_basis,
        support,
        intrinsics,
        poses,
        depth,
        masks,
        camera_ids=camera_ids,
        config=AdaptiveCausalResponseQueryConfig(
            query_count=8,
            graph_basis_rank=8,
        ),
    )


def _records() -> tuple[CausalResponseSourceCameraRecord, ...]:
    return tuple(
        CausalResponseSourceCameraRecord(
            camera_id=camera,
            depth_frame_count=76 if index < 8 else 0,
            mask_frame_count=76 if index < 8 else 0,
            calibration_valid=index < 8,
            frame_zero_projected_support_count=20 if index < 8 else 0,
        )
        for index, camera in enumerate(REGISTERED_CAMERA_IDS)
    )


def _sources() -> dict[str, str]:
    sources = {
        "metadata": "1" * 64,
        "robot": "2" * 64,
        "physical_geometry": "3" * 64,
        "tactile": "4" * 64,
    }
    for camera in REGISTERED_CAMERA_IDS[:8]:
        for modality, digest in (
            ("depth", "5" * 64),
            ("mask", "6" * 64),
            ("calibration", "7" * 64),
        ):
            sources[f"{modality}/{camera}"] = digest
    return sources


def _evaluate(**overrides):
    arguments = {
        "object_id": "fresh-object",
        "episode_id": 0,
        "category": "cloth",
        "bimanual_value": "no",
        "episode_frame_count": 76,
        "robot_frame_count": 76,
        "tactile_frame_count": 76,
        "physical_node_count": 256,
        "camera_records": _records(),
        "carrier": _carrier(),
        "source_sha256": _sources(),
        "config": AdaptiveDirectDepthSourcePreflightConfigV14(),
    }
    arguments.update(overrides)
    return evaluate_adaptive_direct_depth_source_preflight_v14(**arguments)


def test_v14_preflight_accepts_eight_complete_cameras_and_carrier() -> None:
    result = _evaluate()
    descriptor = result.descriptor()

    assert result.admitted
    assert len(result.complete_camera_ids) == 8
    assert descriptor["complete_camera_count"] == 8
    serialized = json.dumps(descriptor, sort_keys=True)
    assert "fresh-object" not in serialized
    assert descriptor["information_boundary"]["future_identity_or_metric_read"] is False


def test_v14_preflight_rejects_malformed_metadata_and_backend_geometry() -> None:
    result = _evaluate(
        bimanual_value="yess",
        physical_node_count=54,
    )

    assert not result.admitted
    assert "invalid-bimanual-enum" in result.rejection_reasons
    assert "physical-backend-node-count" in result.rejection_reasons


def test_v14_preflight_rejects_missing_complete_camera_source_provenance() -> None:
    sources = _sources()
    del sources[f"depth/{REGISTERED_CAMERA_IDS[0]}"]
    result = _evaluate(source_sha256=sources)

    assert not result.admitted
    assert "insufficient-complete-camera-count" in result.rejection_reasons
    assert "carrier-available-camera-set-mismatch" in result.rejection_reasons


def test_v14_preflight_rejects_an_abstained_adaptive_carrier() -> None:
    result = _evaluate(carrier=_carrier(masks_available=False))

    assert not result.admitted
    assert "adaptive-carrier-abstained" in result.rejection_reasons
