from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_artifacts import (
    ARCHIVE_FILENAME,
    validate_causal_response_prediction_artifacts,
    write_causal_response_prediction_artifacts,
)
from bayesian_phystwin.deform360_causal_response_event import (
    PERSISTENCE_BACKBONE,
    PHYSICAL_BACKBONE,
    CausalResponseEventConfig,
    predict_scanned_causal_response,
    scan_causal_response_event,
)
from bayesian_phystwin.deform360_causal_response_query import (
    CausalResponseQueryConfig,
    build_causal_response_query_schedule,
)
from bayesian_phystwin.deform360_causal_response_update import (
    BASELINE_ARM,
    CANDIDATE_ARM,
)


def _scene() -> tuple[np.ndarray, ...]:
    frame_count = 14
    prefix_count = 11
    node_count = 12
    camera_count = 6
    height = width = 256
    x = np.linspace(-0.15, 0.15, node_count)
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
        [0.1, 1.8, 0.4, 2.0, 0.7, 1.5, 0.2, 1.9, 0.6, 1.3, 0.3, 1.7]
    )
    actual = physical.copy()
    extra_mode = 2.5 * physical_mode * local_scale[:, None]
    for frame in range(7, prefix_count):
        actual[frame] += (frame - 6) * extra_mode

    graph_basis = np.zeros((node_count, 3, 8), dtype=np.float64)
    coordinate = np.linspace(-1.0, 1.0, node_count)
    for mode in range(8):
        graph_basis[:, mode % 3, mode] = coordinate ** (mode % 4 + 1)
    action_support = np.full(node_count, 0.8)
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 500.0
    intrinsics[:, 1, 1] = 500.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    poses = np.repeat(np.eye(4)[None], camera_count, axis=0)
    angles = np.linspace(0.0, 2.0 * np.pi, camera_count, endpoint=False)
    poses[:, 0, 3] = 0.04 * np.cos(angles)
    poses[:, 1, 3] = 0.04 * np.sin(angles)
    depths = np.zeros(
        (camera_count, prefix_count, height, width),
        dtype=np.float64,
    )
    masks = np.zeros_like(depths, dtype=bool)
    for camera in range(camera_count):
        world_to_camera = np.linalg.inv(poses[camera])
        for frame in range(prefix_count):
            homogeneous = np.column_stack((actual[frame], np.ones(node_count)))
            camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
            pixels = (intrinsics[camera] @ camera_points.T).T
            pixels = pixels[:, :2] / pixels[:, 2:]
            for node, pixel in enumerate(pixels):
                column, row = np.rint(pixel).astype(int)
                masks[camera, frame, row, column] = True
                depths[camera, frame, row, column] = camera_points[node, 2]
    camera_ids = tuple(f"camera-{index}" for index in range(camera_count))
    proposal = np.asarray([0, 2, 4])
    validation = np.asarray([1, 3, 5])
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
        np.asarray(camera_ids),
        proposal,
        validation,
        tactile,
        actuator,
    )


def _schedule(scene: tuple[np.ndarray, ...]):
    (
        physical,
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depths,
        masks,
        camera_ids,
        proposal,
        validation,
        _,
        _,
    ) = scene
    return build_causal_response_query_schedule(
        physical[0],
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depths[:, 0],
        masks[:, 0],
        camera_ids=tuple(camera_ids),
        proposal_camera_indices=proposal,
        validation_camera_indices=validation,
        config=CausalResponseQueryConfig(
            prefix_frame_count=11,
            query_count=8,
            graph_basis_rank=8,
        ),
    )


def _scan(
    scene: tuple[np.ndarray, ...],
    *,
    tactile: np.ndarray | None = None,
    physical_prediction: np.ndarray | None = None,
    persistence_prediction: np.ndarray | None = None,
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
        _,
        _,
        scene_tactile,
        actuator,
    ) = scene
    selected_physical = physical if physical_prediction is None else physical_prediction
    return scan_causal_response_event(
        "fresh-source",
        selected_physical,
        _schedule(scene),
        intrinsics,
        poses,
        depths,
        masks,
        action_support,
        scene_tactile if tactile is None else tactile,
        actuator,
        persistence_prediction_m=persistence_prediction,
        event_config=CausalResponseEventConfig(
            endpoint_lag_frames=6,
            first_candidate_update_frame=8,
            last_candidate_update_frame=10,
        ),
    )


def _actual_prediction(physical: np.ndarray) -> np.ndarray:
    actual = physical.copy()
    local_scale = np.asarray(
        [0.1, 1.8, 0.4, 2.0, 0.7, 1.5, 0.2, 1.9, 0.6, 1.3, 0.3, 1.7]
    )
    physical_mode = physical[1] - physical[0]
    extra_mode = 2.5 * physical_mode * local_scale[:, None]
    for frame in range(7, 11):
        actual[frame] += (frame - 6) * extra_mode
    return actual


def test_earliest_causal_response_is_selected_and_updates_only_future() -> None:
    scene = _scene()
    scan = _scan(scene)

    assert scan.admitted
    assert scan.selected_admission is not None
    assert scan.selected_admission.update_frame == 8
    assert len(scan.attempts) == 1
    report, arrays = predict_scanned_causal_response(scene[0], scan)
    update = scan.selected_admission.update_frame
    assert report["candidate_applied"]
    assert report["validation_panel_formed_update"] is False
    assert np.array_equal(
        arrays[CANDIDATE_ARM][: update + 1],
        arrays[BASELINE_ARM][: update + 1],
    )
    assert not np.array_equal(
        arrays[CANDIDATE_ARM][update + 1 :],
        arrays[BASELINE_ARM][update + 1 :],
    )


def test_proposal_pairwise_selector_chooses_physical_only_with_margin() -> None:
    scene = _scene()
    physical = scene[0]
    persistence = np.repeat(physical[:1], len(physical), axis=0)

    scan = _scan(scene, persistence_prediction=persistence)

    assert scan.selected_backbone == PHYSICAL_BACKBONE
    assert scan.baseline_selections[-1].relative_physical_improvement >= 0.05


def test_persistence_is_the_default_when_it_fits_the_prefix_better() -> None:
    scene = _scene()
    actual = _actual_prediction(scene[0])
    physical = scene[0].copy()
    physical[7:] += 2.0 * (actual[7:] - physical[7:])

    scan = _scan(
        scene,
        physical_prediction=physical,
        persistence_prediction=actual,
    )
    report, arrays = predict_scanned_causal_response(
        physical,
        scan,
        persistence_prediction_m=actual,
    )

    assert scan.selected_backbone == PERSISTENCE_BACKBONE
    assert scan.baseline_selections[-1].relative_physical_improvement < 0.05
    assert report["selected_backbone"] == PERSISTENCE_BACKBONE
    assert arrays[BASELINE_ARM].tobytes() == actual.tobytes()
    update = scan.selected_admission.update_frame
    assert (
        arrays[CANDIDATE_ARM][: update + 1].tobytes() == actual[: update + 1].tobytes()
    )


def test_missing_tactile_contact_scans_prefix_then_falls_back_exactly() -> None:
    scene = _scene()
    scan = _scan(scene, tactile=np.zeros(11))

    assert not scan.admitted
    assert len(scan.attempts) == 3
    assert all(
        attempt.reason == "insufficient-tactile-contact" for attempt in scan.attempts
    )
    report, arrays = predict_scanned_causal_response(scene[0], scan)
    assert report["bit_exact_baseline_fallback"]
    assert np.array_equal(arrays[CANDIDATE_ARM], scene[0])


def test_query_abstention_has_no_event_attempt_and_exact_fallback() -> None:
    scene = list(_scene())
    scene[6] = np.zeros_like(scene[6])
    scene = tuple(scene)
    schedule = _schedule(scene)
    assert not schedule.admitted
    (
        physical,
        _,
        action_support,
        intrinsics,
        poses,
        depths,
        masks,
        _,
        _,
        _,
        tactile,
        actuator,
    ) = scene

    scan = scan_causal_response_event(
        "fresh-source",
        physical,
        schedule,
        intrinsics,
        poses,
        depths,
        masks,
        action_support,
        tactile,
        actuator,
        event_config=CausalResponseEventConfig(
            endpoint_lag_frames=6,
            first_candidate_update_frame=8,
            last_candidate_update_frame=10,
        ),
    )
    report, arrays = predict_scanned_causal_response(physical, scan)

    assert not scan.attempts
    assert not scan.admitted
    assert report["bit_exact_baseline_fallback"]
    assert np.array_equal(arrays[CANDIDATE_ARM], physical)


def test_future_tactile_carrier_is_rejected_even_when_prefix_matches() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="tactile contact probability"):
        _scan(scene, tactile=np.pad(scene[10], (0, 3)))


def test_prediction_artifact_seals_and_validates_without_outcomes(
    tmp_path: Path,
) -> None:
    scene = _scene()
    schedule = _schedule(scene)
    scan = _scan(scene)
    candidate_report, candidate_arrays = predict_scanned_causal_response(
        scene[0],
        scan,
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "prediction"

    report = write_causal_response_prediction_artifacts(
        output,
        schedule,
        scan,
        candidate_report,
        candidate_arrays,
        case_id="fresh-source",
        repository_revision="a" * 40,
        protocol_path=protocol,
        input_sha256={"staged_prefix": "b" * 64},
    )
    validated, arrays = validate_causal_response_prediction_artifacts(output)

    assert validated["result_sha256"] == report["result_sha256"]
    assert validated["status"] == "candidate_prediction_sealed"
    assert validated["information_boundary"]["future_identity_or_metric_read"] is False
    assert np.array_equal(arrays[CANDIDATE_ARM], candidate_arrays[CANDIDATE_ARM])


def test_fallback_artifact_is_bit_exact_and_detects_archive_tampering(
    tmp_path: Path,
) -> None:
    scene = _scene()
    schedule = _schedule(scene)
    scan = _scan(scene, tactile=np.zeros(11))
    candidate_report, candidate_arrays = predict_scanned_causal_response(
        scene[0],
        scan,
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "prediction"
    write_causal_response_prediction_artifacts(
        output,
        schedule,
        scan,
        candidate_report,
        candidate_arrays,
        case_id="fresh-source",
        repository_revision="a" * 40,
        protocol_path=protocol,
        input_sha256={"staged_prefix": "b" * 64},
    )
    validated, arrays = validate_causal_response_prediction_artifacts(output)

    assert validated["status"] == "exact_baseline_fallback_sealed"
    assert arrays[CANDIDATE_ARM].tobytes() == arrays[BASELINE_ARM].tobytes()
    with (output / ARCHIVE_FILENAME).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="archive checksum"):
        validate_causal_response_prediction_artifacts(output)
