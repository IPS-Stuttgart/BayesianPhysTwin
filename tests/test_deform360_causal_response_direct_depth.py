from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    INFLATED_FALLBACK_ARM,
    STRICT_ARM,
    AdaptiveCausalResponseQueryConfig,
    build_adaptive_causal_response_query_schedule,
)
from bayesian_phystwin.deform360_causal_response_direct_depth import (
    ARCHIVE_FILENAME,
    REPORT_FILENAME,
    predict_adaptive_direct_depth_v14,
    scan_adaptive_direct_depth_v14,
    validate_adaptive_direct_depth_v14_artifacts,
    write_adaptive_direct_depth_v14_artifacts,
)
from bayesian_phystwin.deform360_causal_response_event import (
    CausalResponseEventConfig,
)
from bayesian_phystwin.deform360_causal_response_update import (
    BASELINE_ARM,
    CANDIDATE_ARM,
)


def _scene(
    *,
    fallback: bool = False,
    common_depth_bias_m: float = 0.0,
    nonrigid_response: bool = True,
) -> tuple[np.ndarray, ...]:
    frame_count = 14
    prefix_count = 11
    node_count = 16
    camera_count = 8
    height = width = 256
    x = np.linspace(-0.12, 0.12, node_count)
    y = np.where(np.arange(node_count) % 2, 0.045, -0.045)
    frame_zero = np.column_stack((x, y, np.full(node_count, 2.0)))
    physical_mode = np.column_stack(
        (
            np.linspace(-0.0007, 0.0007, node_count),
            np.linspace(0.0005, -0.0005, node_count),
            np.zeros(node_count),
        )
    )
    physical = np.stack(
        [frame_zero + frame * physical_mode for frame in range(frame_count)]
    )
    local_scale = np.asarray(
        [
            0.1,
            1.8,
            0.4,
            2.0,
            0.7,
            1.5,
            0.2,
            1.9,
            0.6,
            1.3,
            0.3,
            1.7,
            0.5,
            1.6,
            0.8,
            1.4,
        ]
    )
    actual = physical.copy()
    extra_mode = 2.5 * physical_mode * local_scale[:, None]
    if nonrigid_response:
        for frame in range(7, prefix_count):
            actual[frame] += (frame - 6) * extra_mode

    coordinate = np.linspace(-1.0, 1.0, node_count)
    graph_basis = np.zeros((node_count, 3, 8), dtype=np.float64)
    for mode in range(8):
        graph_basis[:, mode % 3, mode] = coordinate ** (mode % 4 + 1)
    action_support = np.full(node_count, 0.8)
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 1600.0
    intrinsics[:, 1, 1] = 1600.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    poses = np.repeat(np.eye(4)[None], camera_count, axis=0)
    angles = np.linspace(0.0, 2.0 * np.pi, camera_count, endpoint=False)
    poses[:, 0, 3] = 0.01 * np.cos(angles)
    poses[:, 1, 3] = 0.01 * np.sin(angles)
    depths = np.zeros(
        (camera_count, prefix_count, height, width),
        dtype=np.float64,
    )
    masks = np.zeros_like(depths, dtype=bool)
    unavailable = {2, 3, 6, 7} if fallback else set()
    for camera in range(camera_count):
        if camera in unavailable:
            continue
        world_to_camera = np.linalg.inv(poses[camera])
        for frame in range(prefix_count):
            homogeneous = np.column_stack((actual[frame], np.ones(node_count)))
            camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
            pixels = (intrinsics[camera] @ camera_points.T).T
            pixels = pixels[:, :2] / pixels[:, 2:]
            for node, pixel in enumerate(pixels):
                column, row = np.rint(pixel).astype(int)
                masks[camera, frame, row, column] = True
                depths[camera, frame, row, column] = (
                    camera_points[node, 2]
                    + (common_depth_bias_m if frame >= 7 else 0.0)
                )
    camera_ids = tuple(f"camera-{index}" for index in range(camera_count))
    tactile = np.zeros(prefix_count)
    tactile[7:] = 1.0
    actuator = np.zeros((prefix_count, 1, 3))
    actuator[:, 0, 0] = 0.002 * np.arange(prefix_count)
    return (
        physical,
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depths,
        masks,
        camera_ids,
        tactile,
        actuator,
    )


def _carrier(scene: tuple[np.ndarray, ...]):
    (
        physical,
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depths,
        masks,
        camera_ids,
        _,
        _,
    ) = scene
    return build_adaptive_causal_response_query_schedule(
        physical[0],
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depths[:, 0],
        masks[:, 0],
        camera_ids=camera_ids,
        config=AdaptiveCausalResponseQueryConfig(
            prefix_frame_count=11,
            query_count=8,
            graph_basis_rank=8,
        ),
    )


def _scan(
    scene: tuple[np.ndarray, ...],
    *,
    tactile: np.ndarray | None = None,
):
    (
        physical,
        _,
        action_support,
        intrinsics,
        poses,
        depths,
        masks,
        _,
        scene_tactile,
        actuator,
    ) = scene
    return scan_adaptive_direct_depth_v14(
        "fresh-source",
        physical,
        _carrier(scene),
        intrinsics,
        poses,
        depths,
        masks,
        action_support,
        scene_tactile if tactile is None else tactile,
        actuator,
        event_config=CausalResponseEventConfig(
            endpoint_lag_frames=6,
            first_candidate_update_frame=8,
            last_candidate_update_frame=10,
        ),
    )


def test_strict_carrier_runs_causal_direct_depth_update() -> None:
    scene = _scene()
    carrier = _carrier(scene)
    scan = _scan(scene)
    report, arrays = predict_adaptive_direct_depth_v14(scene[0], scan)

    assert carrier.arm == STRICT_ARM
    assert scan.depth_config.minimum_camera_support == 3
    assert scan.depth_config.correlation_covariance_inflation == 1.0
    assert scan.scan.admitted
    assert report["candidate_applied"]
    update = scan.scan.selected_admission.update_frame
    np.testing.assert_array_equal(
        arrays[CANDIDATE_ARM][: update + 1],
        arrays[BASELINE_ARM][: update + 1],
    )
    assert not np.array_equal(
        arrays[CANDIDATE_ARM][update + 1 :],
        arrays[BASELINE_ARM][update + 1 :],
    )


def test_two_view_carrier_applies_registered_covariance_inflation() -> None:
    scene = _scene(fallback=True)
    carrier = _carrier(scene)
    scan = _scan(scene)

    assert carrier.arm == INFLATED_FALLBACK_ARM
    assert scan.depth_config.minimum_camera_support == 2
    assert scan.depth_config.correlation_covariance_inflation == 4.0
    if scan.scan.selected_proposal is not None:
        assert scan.scan.selected_proposal.config == scan.depth_config
        assert np.all(
            scan.scan.selected_proposal.support_count[
                scan.scan.selected_proposal.accepted_support
            ]
            >= 2
        )


def test_missing_tactile_support_preserves_baseline_bit_exactly() -> None:
    scene = _scene()
    scan = _scan(scene, tactile=np.zeros(11))
    report, arrays = predict_adaptive_direct_depth_v14(scene[0], scan)

    assert not scan.scan.admitted
    assert report["bit_exact_baseline_fallback"]
    assert arrays[CANDIDATE_ARM].tobytes() == arrays[BASELINE_ARM].tobytes()


def test_common_mode_depth_bias_cannot_create_a_confident_update() -> None:
    scene = _scene(
        common_depth_bias_m=0.02,
        nonrigid_response=False,
    )
    scan = _scan(scene)
    report, arrays = predict_adaptive_direct_depth_v14(scene[0], scan)

    assert not report["candidate_applied"]
    assert arrays[CANDIDATE_ARM].tobytes() == arrays[BASELINE_ARM].tobytes()


def test_v14_prediction_artifact_round_trips_without_future_outcome(
    tmp_path: Path,
) -> None:
    scene = _scene()
    carrier = _carrier(scene)
    scan = _scan(scene, tactile=np.zeros(11))
    candidate, arrays = predict_adaptive_direct_depth_v14(scene[0], scan)
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "prediction"

    report = write_adaptive_direct_depth_v14_artifacts(
        output,
        carrier,
        scan,
        candidate,
        arrays,
        case_id="fresh-source",
        repository_revision="a" * 40,
        protocol_path=protocol,
        input_sha256={"causal_prefix": "b" * 64},
    )
    validated, loaded = validate_adaptive_direct_depth_v14_artifacts(output)

    assert validated["result_sha256"] == report["result_sha256"]
    assert validated["status"] == "exact_baseline_fallback_sealed"
    assert loaded[CANDIDATE_ARM].tobytes() == loaded[BASELINE_ARM].tobytes()
    assert (output / REPORT_FILENAME).is_file()
    assert (output / ARCHIVE_FILENAME).is_file()


def test_v14_validator_rejects_archive_tampering(tmp_path: Path) -> None:
    scene = _scene()
    carrier = _carrier(scene)
    scan = _scan(scene, tactile=np.zeros(11))
    candidate, arrays = predict_adaptive_direct_depth_v14(scene[0], scan)
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "prediction"
    write_adaptive_direct_depth_v14_artifacts(
        output,
        carrier,
        scan,
        candidate,
        arrays,
        case_id="fresh-source",
        repository_revision="a" * 40,
        protocol_path=protocol,
        input_sha256={"causal_prefix": "b" * 64},
    )
    with (output / ARCHIVE_FILENAME).open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="archive checksum"):
        validate_adaptive_direct_depth_v14_artifacts(output)
