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
