from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_graph import PhysTwinSpringGraphConfig
from bayesian_phystwin.phystwin_piecewise_topology import (
    PIECEWISE_TOPOLOGY_CONTRACT,
    build_piecewise_topology_candidate,
    load_piecewise_topology_artifact,
    write_piecewise_topology_artifact,
)


def _inputs():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    controls = np.array([[0.3, 0.0, 0.0]], dtype=np.float32)
    assignments = np.array([0, 0, 1, 1], dtype=np.int32)
    config = PhysTwinSpringGraphConfig(
        object_radius=1.0,
        object_max_neighbours=2,
        controller_radius=0.5,
        controller_max_neighbours=2,
    )
    # Three object springs followed by two controller springs.
    spring_y = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    return points, controls, assignments, config, spring_y


def test_identity_piecewise_candidate_preserves_teacher_field_and_connectivity():
    points, controls, assignments, config, spring_y = _inputs()

    artifact = build_piecewise_topology_candidate(
        points,
        controls,
        assignments,
        spring_y,
        teacher_config=config,
        radius_multipliers=(1.0, 1.0),
        neighbour_multipliers=(1.0, 1.0),
    )

    np.testing.assert_array_equal(artifact.reference_spring_y, spring_y)
    assert artifact.applied_object_log_scale == 0.0
    assert artifact.transfer.exact_edge_count == 5
    assert artifact.transfer.interpolated_edge_count == 0
    assert artifact.transfer.removed_teacher_edge_count == 0
    assert artifact.diagnostics["object_component_count"] == 1
    assert artifact.diagnostics["isolated_object_point_count"] == 0


def test_topology_artifact_round_trip_preserves_new_edges_and_scales(tmp_path: Path):
    points, controls, assignments, config, spring_y = _inputs()
    artifact = build_piecewise_topology_candidate(
        points,
        controls,
        assignments,
        spring_y,
        teacher_config=config,
        radius_multipliers=(1.0, 1.2),
        neighbour_multipliers=(1.0, 2.0),
        object_log_scale=np.log(2.0),
        controller_log_scale=np.log(0.5),
    )
    path = tmp_path / "topology.npz"

    identity = write_piecewise_topology_artifact(path, artifact)
    loaded = load_piecewise_topology_artifact(path)

    assert identity["path"] == str(path.resolve())
    assert len(identity["sha256"]) == 64
    with np.load(path, allow_pickle=False) as archive:
        assert archive["contract"].item() == PIECEWISE_TOPOLOGY_CONTRACT
    np.testing.assert_array_equal(loaded.graph.springs, artifact.graph.springs)
    np.testing.assert_array_equal(
        loaded.graph.rest_lengths, artifact.graph.rest_lengths
    )
    np.testing.assert_allclose(
        loaded.reference_spring_y,
        artifact.reference_spring_y,
        rtol=0.0,
        atol=0.0,
    )
    assert loaded.transfer.interpolated_edge_count > 0
    np.testing.assert_allclose(
        loaded.reference_spring_y[: loaded.graph.num_object_springs]
        / loaded.transfer.spring_y[: loaded.graph.num_object_springs],
        2.0,
    )
    np.testing.assert_allclose(
        loaded.reference_spring_y[loaded.graph.num_object_springs :]
        / loaded.transfer.spring_y[loaded.graph.num_object_springs :],
        0.5,
    )
    assert loaded.diagnostics == artifact.diagnostics
    assert loaded.applied_object_log_scale == artifact.applied_object_log_scale
    assert (
        loaded.applied_controller_log_scale
        == artifact.applied_controller_log_scale
    )
    assert loaded.object_scale_normalization == "none"


def test_density_normalization_preserves_total_object_stiffness():
    points, controls, assignments, _, _ = _inputs()
    config = PhysTwinSpringGraphConfig(
        object_radius=1.0,
        object_max_neighbours=3,
        controller_radius=0.5,
        controller_max_neighbours=2,
    )
    teacher = build_piecewise_topology_candidate(
        points,
        controls,
        assignments,
        np.ones(7, dtype=np.float32),
        teacher_config=config,
        radius_multipliers=(1.0, 1.0),
        neighbour_multipliers=(1.0, 1.0),
    )
    spring_y = np.arange(1, len(teacher.graph.springs) + 1, dtype=np.float32)
    normalized_identity = build_piecewise_topology_candidate(
        points,
        controls,
        assignments,
        spring_y,
        teacher_config=config,
        radius_multipliers=(1.0, 1.0),
        neighbour_multipliers=(1.0, 1.0),
        preserve_total_object_stiffness=True,
    )

    candidate = build_piecewise_topology_candidate(
        points,
        controls,
        assignments,
        spring_y,
        teacher_config=config,
        radius_multipliers=(0.45, 0.45),
        neighbour_multipliers=(2.0 / 3.0, 2.0 / 3.0),
        preserve_total_object_stiffness=True,
    )

    assert candidate.graph.num_object_springs < teacher.graph.num_object_springs
    assert normalized_identity.applied_object_log_scale == 0.0
    np.testing.assert_array_equal(normalized_identity.reference_spring_y, spring_y)
    np.testing.assert_allclose(
        np.sum(candidate.reference_spring_y[: candidate.graph.num_object_springs]),
        np.sum(spring_y[: teacher.graph.num_object_springs]),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        candidate.reference_spring_y[candidate.graph.num_object_springs :],
        candidate.transfer.spring_y[candidate.graph.num_object_springs :],
    )
    assert (
        candidate.object_scale_normalization
        == "preserve_total_object_stiffness"
    )
    assert candidate.applied_object_log_scale > 0.0


def test_region_spring_field_averages_endpoint_log_scales(tmp_path: Path):
    points, controls, assignments, config, spring_y = _inputs()
    region_scales = np.log(np.array([2.0, 0.5]))

    artifact = build_piecewise_topology_candidate(
        points,
        controls,
        assignments,
        spring_y,
        teacher_config=config,
        radius_multipliers=(1.0, 1.0),
        neighbour_multipliers=(1.0, 1.0),
        region_object_log_scales=region_scales,
    )
    object_edges = artifact.graph.springs[: artifact.graph.num_object_springs]
    expected = np.exp(
        0.5
        * (
            region_scales[assignments[object_edges[:, 0]]]
            + region_scales[assignments[object_edges[:, 1]]]
        )
    )

    np.testing.assert_allclose(
        artifact.reference_spring_y[: artifact.graph.num_object_springs]
        / artifact.transfer.spring_y[: artifact.graph.num_object_springs],
        expected,
    )
    np.testing.assert_array_equal(
        artifact.reference_spring_y[artifact.graph.num_object_springs :],
        artifact.transfer.spring_y[artifact.graph.num_object_springs :],
    )
    path = tmp_path / "regional.npz"
    write_piecewise_topology_artifact(path, artifact)
    loaded = load_piecewise_topology_artifact(path)
    np.testing.assert_array_equal(loaded.region_object_log_scales, region_scales)
