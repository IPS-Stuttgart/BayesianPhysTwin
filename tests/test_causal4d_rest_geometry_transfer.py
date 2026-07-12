import json

import numpy as np
import pytest

from bayesian_phystwin.phystwin_graph import PhysTwinSpringGraphConfig
from causal4d.real_protocol import build_same_object_real_protocol
from causal4d.rest_geometry import RigidFrameCorrection
from causal4d.rest_geometry_cross_action import (
    canonical_rest_geometry_hyperparameters,
    rest_geometry_hyperparameter_id,
)
from causal4d.rest_geometry_transfer import (
    SourceRestGeometryCorrection,
    attach_target_controller_to_canonical_graph,
    canonical_material_graph_sha256,
    load_canonical_material_graph,
    load_source_rest_geometry_correction,
    prepare_target_rest_geometry_configuration,
    write_canonical_material_graph,
    write_source_rest_geometry_correction,
)


def _source(controller_mode="preserve"):
    hyperparameters = canonical_rest_geometry_hyperparameters(
        {
            "frame_mode": "translation",
            "frame_scale": 1.0,
            "rest_geometry_scale": 1.0,
            "controller_rest_mode": controller_mode,
            "graph_prior_strength": 0.1,
            "rest_length_ratio_bound": 1.15,
        }
    )
    frame = RigidFrameCorrection(
        linear=np.eye(3),
        translation=np.array([0.5, 0.0, 0.0]),
        mode="translation",
        rotation_angle_rad=0.0,
        fitted_point_count=2,
    )
    return SourceRestGeometryCorrection(
        protocol_id="protocol",
        protocol_design_sha256="a" * 64,
        source_execution_id="source",
        selected_candidate_id=rest_geometry_hyperparameter_id(hyperparameters),
        hyperparameters=hyperparameters,
        frame=frame,
        nonrigid_field=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        corrected_reference_vertices=np.array(
            [[0.5, 0.0, 0.0], [1.6, 0.0, 0.0]]
        ),
        corrected_object_rest_lengths=np.array([1.1]),
        canonical_material_graph_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )


def test_material_graph_hash_excludes_controller_attachment_tail() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    springs = np.array([[0, 1], [2, 0]], dtype=np.int32)

    first = canonical_material_graph_sha256(
        vertices,
        springs,
        np.array([1.0, 0.4]),
        num_object_springs=1,
    )
    second = canonical_material_graph_sha256(
        vertices,
        np.array([[0, 1], [3, 1]], dtype=np.int32),
        np.array([1.0, 9.0]),
        num_object_springs=1,
    )

    assert first == second


def test_canonical_graph_round_trip_rebuilds_only_controller_springs(tmp_path) -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    springs = np.array([[0, 1]], dtype=np.int32)
    path = tmp_path / "canonical.npz"
    written = write_canonical_material_graph(
        path,
        vertices,
        springs,
        np.array([1.0]),
    )
    canonical = load_canonical_material_graph(path)

    attached = attach_target_controller_to_canonical_graph(
        canonical,
        np.array([[0.0, 1.0, 0.0]]),
        config=PhysTwinSpringGraphConfig(
            object_radius=2.0,
            object_max_neighbours=2,
            controller_radius=2.0,
            controller_max_neighbours=2,
        ),
    )

    assert canonical.sha256 == written["canonical_material_graph_sha256"]
    np.testing.assert_array_equal(attached.springs[:1], springs)
    np.testing.assert_allclose(attached.rest_lengths[:1], [1.0])
    assert attached.num_object_springs == 1
    assert len(attached.springs) == 3


def test_same_grasp_transfers_object_geometry_but_preserves_attachment() -> None:
    source = _source("preserve")
    springs = np.array([[0, 1], [2, 0], [2, 1]], dtype=np.int32)
    released = np.array([1.0, 1.0, np.sqrt(2.0)])
    controller = np.array([[[0.0, 1.0, 0.0]], [[0.1, 1.0, 0.0]]])

    target = prepare_target_rest_geometry_configuration(
        source,
        target_material_graph_sha256="b" * 64,
        target_position=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        target_velocity=np.zeros((2, 3)),
        target_controller_points=controller,
        target_springs=springs,
        target_released_rest_lengths=released,
        num_object_springs=1,
        contact_policy="same_grasp",
    )

    np.testing.assert_allclose(
        target.position,
        [[0.5, 0.0, 0.0], [1.6, 0.0, 0.0]],
    )
    np.testing.assert_allclose(target.controller_points, controller + [0.5, 0.0, 0.0])
    np.testing.assert_allclose(target.rest_lengths, [1.1, 1.0, np.sqrt(2.0)])
    assert target.controller_attachment_policy == "preserve_registered_attachment"


def test_new_contact_rebuilds_attachment_and_rejects_a_different_graph() -> None:
    source = _source("preserve")
    springs = np.array([[0, 1], [2, 0], [2, 1]], dtype=np.int32)
    released = np.array([1.0, 1.0, np.sqrt(2.0)])
    arguments = {
        "target_position": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "target_velocity": np.zeros((2, 3)),
        "target_controller_points": np.array([[[0.0, 1.0, 0.0]]]),
        "target_springs": springs,
        "target_released_rest_lengths": released,
        "num_object_springs": 1,
        "contact_policy": "new_contact",
    }

    target = prepare_target_rest_geometry_configuration(
        source,
        target_material_graph_sha256="b" * 64,
        **arguments,
    )

    assert target.controller_attachment_policy == "rebuild_on_corrected_target_contact"
    assert target.rest_lengths[0] == pytest.approx(1.1)
    assert target.rest_lengths[1] == pytest.approx(1.0)
    assert target.rest_lengths[2] > released[2]
    with pytest.raises(ValueError, match="canonical material graph"):
        prepare_target_rest_geometry_configuration(
            source,
            target_material_graph_sha256="d" * 64,
            **arguments,
        )


def test_source_correction_round_trip_preserves_hashes(tmp_path) -> None:
    protocol = build_same_object_real_protocol()
    execution_id = protocol["executions"][0]["execution_id"]
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    springs = np.array([[0, 1]], dtype=np.int32)
    released = np.array([1.0])
    material_digest = canonical_material_graph_sha256(
        reference,
        springs,
        released,
        num_object_springs=1,
    )
    summary = {
        "config": {
            "frame_mode": "translation",
            "graph_prior_strength": 0.1,
            "maximum_rest_log_ratio": np.log(1.15),
        },
        "selection": {
            "selected_frame_scale": 1.0,
            "selected_rest_geometry_scale": 1.0,
            "selected_controller_rest_mode": "preserve",
        },
        "information_boundary": {
            "holdout_frames_used_for_inference": False,
            "holdout_frames_used_for_hyperparameter_selection": False,
            "manual_gt_track_used_for_hyperparameter_selection": False,
        },
        "graph": {
            "object_vertex_count": 2,
            "object_spring_count": 1,
            "canonical_material_graph_sha256": material_digest,
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    archive_path = tmp_path / "case.npz"
    np.savez_compressed(
        archive_path,
        frame_linear=np.eye(3),
        frame_translation=np.array([0.5, 0.0, 0.0]),
        nonrigid_field=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        canonical_reference_vertices=reference,
        corrected_reference_vertices=np.array(
            [[0.5, 0.0, 0.0], [1.6, 0.0, 0.0]]
        ),
        corrected_rest_lengths=np.array([1.1]),
        released_rest_lengths=released,
        object_springs=springs,
    )

    exported = write_source_rest_geometry_correction(
        protocol,
        execution_id,
        summary_path,
        archive_path,
        tmp_path / "export",
    )
    loaded = load_source_rest_geometry_correction(exported["manifest_path"])

    assert loaded.source_execution_id == execution_id
    assert loaded.canonical_material_graph_sha256 == material_digest
    np.testing.assert_allclose(loaded.corrected_object_rest_lengths, [1.1])
