from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    AdaptiveCausalResponseQueryConfig,
    build_adaptive_causal_response_query_schedule,
    write_adaptive_causal_response_query_artifacts,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_admission_v14 import (
    CARRIER_DIRECTORY,
    PREFLIGHT_FILENAME,
    write_v14_admission_report,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_prediction_v14 import (
    PREFIX_FRAME_COUNT,
    aggregate_tactile_contact_confidence,
    build_v14_prefix_inputs,
    load_v14_admitted_carrier,
    measured_actuator_origins,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    evaluate_adaptive_direct_depth_source_preflight_v14,
    write_adaptive_direct_depth_source_preflight_v14,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    CausalResponseSourceCameraRecord,
)


def _carrier():
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
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 60.0
    intrinsics[:, 1, 1] = 60.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    return build_adaptive_causal_response_query_schedule(
        frame_zero,
        graph_basis,
        np.ones(node_count),
        intrinsics,
        np.repeat(np.eye(4)[None], camera_count, axis=0),
        np.full((camera_count, height, width), 2.0),
        np.ones((camera_count, height, width), dtype=bool),
        camera_ids=REGISTERED_CAMERA_IDS[:camera_count],
        config=AdaptiveCausalResponseQueryConfig(
            query_count=8,
            graph_basis_rank=8,
        ),
    )


def _preflight(carrier):
    records = tuple(
        CausalResponseSourceCameraRecord(
            camera_id=camera,
            depth_frame_count=58 if index < 8 else 0,
            mask_frame_count=58 if index < 8 else 0,
            calibration_valid=index < 8,
            frame_zero_projected_support_count=20 if index < 8 else 0,
        )
        for index, camera in enumerate(REGISTERED_CAMERA_IDS)
    )
    sources = {
        "metadata": "1" * 64,
        "robot": "2" * 64,
        "physical_geometry": "3" * 64,
        "tactile": "4" * 64,
    }
    for camera in REGISTERED_CAMERA_IDS[:8]:
        sources[f"depth/{camera}"] = "5" * 64
        sources[f"mask/{camera}"] = "6" * 64
        sources[f"calibration/{camera}"] = "7" * 64
    return evaluate_adaptive_direct_depth_source_preflight_v14(
        object_id="fresh-runtime-object",
        episode_id=0,
        category="cloth",
        bimanual_value="no",
        episode_frame_count=76,
        robot_frame_count=76,
        tactile_frame_count=76,
        physical_node_count=256,
        camera_records=records,
        carrier=carrier,
        source_sha256=sources,
    )


def test_tactile_aggregation_is_duplicate_invariant() -> None:
    values = np.zeros((PREFIX_FRAME_COUNT, 16, 32), dtype=np.float32)
    values[:, 3, 4] = np.linspace(0.0, 1.0, PREFIX_FRAME_COUNT)

    single = aggregate_tactile_contact_confidence([values])
    duplicate = aggregate_tactile_contact_confidence([values, values.copy()])

    np.testing.assert_array_equal(single, duplicate)
    np.testing.assert_allclose(single, values[:, 3, 4])


def test_tactile_aggregation_rejects_non_normalized_values() -> None:
    values = np.zeros((PREFIX_FRAME_COUNT, 16, 32), dtype=np.float32)
    values[4, 2, 3] = 1.1

    with pytest.raises(ValueError, match="normalized contract"):
        aggregate_tactile_contact_confidence([values])


def test_measured_actuator_uses_pose_origin_only() -> None:
    actions = np.zeros((PREFIX_FRAME_COUNT + 4, 5, 3), dtype=np.float64)
    expected = np.column_stack(
        (
            np.linspace(0.0, 0.2, len(actions)),
            np.linspace(-0.1, 0.1, len(actions)),
            np.linspace(0.3, 0.4, len(actions)),
        )
    )
    actions[:, 0] = expected
    actions[:, 1:] = 100.0

    observed = measured_actuator_origins(actions)

    assert observed.shape == (PREFIX_FRAME_COUNT, 1, 3)
    np.testing.assert_array_equal(observed[:, 0], expected[:PREFIX_FRAME_COUNT])


def test_prefix_builder_uses_correlation_safe_tactile_and_measured_origin() -> None:
    camera_count = 8
    frame_shape = (camera_count, PREFIX_FRAME_COUNT, 4, 5)
    tactile = np.zeros((PREFIX_FRAME_COUNT, 16, 32), dtype=np.float32)
    tactile[7:, 1, 2] = 0.8
    actions = np.zeros((PREFIX_FRAME_COUNT, 5, 3), dtype=np.float64)
    actions[:, 0, 0] = np.arange(PREFIX_FRAME_COUNT) * 0.001

    inputs = build_v14_prefix_inputs(
        camera_ids=REGISTERED_CAMERA_IDS[:camera_count],
        intrinsics=np.repeat(np.eye(3)[None], camera_count, axis=0),
        camera_to_world=np.repeat(np.eye(4)[None], camera_count, axis=0),
        depths_m=np.ones(frame_shape, dtype=np.float32),
        object_masks=np.ones(frame_shape, dtype=bool),
        tactile_sensor_arrays=[tactile, tactile.copy()],
        robot_actions=actions,
    )

    np.testing.assert_array_equal(
        inputs.tactile_contact_probability,
        np.max(tactile, axis=(1, 2)),
    )
    np.testing.assert_array_equal(
        inputs.measured_actuator_positions_m[:, 0],
        actions[:, 0],
    )


def test_admitted_carrier_reconstructs_exactly(tmp_path: Path) -> None:
    carrier = _carrier()
    preflight = _preflight(carrier)
    root = tmp_path / "admission"
    root.mkdir()
    method = tmp_path / "method.json"
    method.write_text("{}\n", encoding="utf-8")
    physical_manifest = tmp_path / "physical.json"
    physical_manifest.write_text("{}\n", encoding="utf-8")
    physical_archive = tmp_path / "physical.npz"
    physical_archive.write_bytes(b"physical")
    carrier_report = write_adaptive_causal_response_query_artifacts(
        root / CARRIER_DIRECTORY,
        carrier,
        case_id=preflight.case_hash,
        repository_revision="a" * 40,
        protocol_path=method,
        physical_manifest_path=physical_manifest,
        physical_archive_path=physical_archive,
        camera_certificate_sha256="b" * 64,
    )
    write_adaptive_direct_depth_source_preflight_v14(
        root / PREFLIGHT_FILENAME,
        preflight,
    )
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(b"bound input")
    write_v14_admission_report(
        root,
        queue_rank=3,
        object_hash=preflight.object_hash,
        case_hash=preflight.case_hash,
        repository_revision="c" * 40,
        admission_protocol={"config_sha256": "d" * 64},
        physical_artifact_sha256="e" * 64,
        geometry_artifact_sha256="f" * 64,
        carrier_result_sha256=carrier_report["result_sha256"],
        carrier_artifact_sha256=carrier.artifact_sha256,
        preflight_artifact_sha256=preflight.artifact_sha256,
        admitted=True,
        input_files={"input": input_file},
    )

    admission, loaded = load_v14_admitted_carrier(root)

    assert admission["status"] == "admitted"
    assert loaded.descriptor() == carrier.descriptor()
