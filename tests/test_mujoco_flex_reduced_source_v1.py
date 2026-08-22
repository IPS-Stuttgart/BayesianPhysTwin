from __future__ import annotations

from xml.etree import ElementTree

import numpy as np

from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
)
from bayesian_phystwin.mujoco_flex_reduced_source_v1 import (
    ATTACHMENT_MODEL,
    BACKEND_VARIANT,
    EMPTY_CELL_STIFFNESS_SENTINEL,
    GRID_FRAME_POLICY,
    INTERPOLATION_MODEL,
    MASS_MODEL,
    build_mujoco_trilinear_grid_v1,
    build_mujoco_trilinear_scene_v1,
)
from bayesian_phystwin.native_tet_fem_source_v1 import (
    NativeTetSourceGeometryV1,
    prepare_native_tet_source_geometry_v1,
)


def _fixture() -> tuple[NativeTetSourceGeometryV1, RigidContactProjectionV1]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    cells = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    attachments = np.arange(4, dtype=np.int64)
    translation = np.asarray([0.0005, 0.0, 0.0])
    contact = RigidContactProjectionV1(
        projected_targets_m=np.stack(
            (points[attachments], points[attachments] + translation)
        ),
        rotations=np.repeat(np.eye(3)[None, None], 2, axis=0),
        translations_m=np.asarray(
            [[[0.0, 0.0, 0.0]], [translation]],
            dtype=np.float64,
        ),
        patch_local_indices=(np.arange(4, dtype=np.int64),),
        patch_ranks=(3,),
    )
    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points,
        cells=cells,
        attachment_indices=attachments,
        contact=contact,
    )
    return geometry, contact


def _scene_kwargs() -> dict[str, object]:
    return {
        "cellcount": (4, 4, 4),
        "integrator": "implicitfast",
        "integrator_time_step_s": 1.0e-4,
        "young_modulus_pa": 1000.0,
        "poisson_ratio": 0.3,
        "density_kg_m3": 1000.0,
        "edge_damping": 1.0,
        "elasticity_damping": 1.0,
        "joint_damping": 0.1,
        "solver_iterations": 100,
        "solver_tolerance": 1.0e-12,
    }


def test_grid_is_exact_mass_conserving_and_reduced() -> None:
    geometry, contact = _fixture()
    grid = build_mujoco_trilinear_grid_v1(
        geometry,
        contact=contact,
        cellcount=(4, 4, 4),
        density_kg_m3=1000.0,
    )
    assert BACKEND_VARIANT == "mujoco-volumetric-trilinear-flex-v1"
    assert ATTACHMENT_MODEL == "disjoint-rigid-patch-trilinear-node-Dirichlet-v1"
    assert INTERPOLATION_MODEL == "axis-aligned-multicell-trilinear-v1"
    assert GRID_FRAME_POLICY == "identity-world-frame-with-empty-cell-sentinel-v1"
    assert MASS_MODEL == "reference-mass-conserving-occupied-cell-lumping-v1"
    assert EMPTY_CELL_STIFFNESS_SENTINEL == np.nextafter(0.0, 1.0)
    assert len(grid.node_positions_backend_m) == 125
    assert len(grid.occupied_cell_indices) == 64
    assert len(grid.active_node_indices) == 125
    assert len(grid.patch_node_indices) == 1
    assert len(grid.patch_node_indices[0]) == 4
    assert len(grid.free_node_indices) == 121
    assert len(grid.inactive_node_indices) == 0
    np.testing.assert_array_equal(grid.grid_from_world_rotation, np.eye(3))
    assert grid.cell_zero_tetrahedron_count == 2
    assert np.isclose(grid.total_reference_mass_kg, 0.5)
    np.testing.assert_allclose(
        np.sum(
            grid.node_positions_world_m[grid.vertex_node_indices]
            * grid.vertex_node_weights[:, :, None],
            axis=1,
        ),
        geometry.points_m,
        atol=1.0e-15,
        rtol=0.0,
    )
    assert np.isclose(
        np.sum(grid.node_masses_kg),
        grid.total_reference_mass_kg,
    )


def test_scene_exposes_native_trilinear_ownership_without_constraints() -> None:
    geometry, contact = _fixture()
    scene = build_mujoco_trilinear_scene_v1(
        geometry,
        contact=contact,
        **_scene_kwargs(),
    )
    root = ElementTree.fromstring(scene.xml)
    option = root.find("./option")
    assert option is not None and option.attrib["integrator"] == "implicitfast"
    flex = root.find("./deformable/flex")
    assert flex is not None
    assert flex.attrib["dof"] == "trilinear"
    assert flex.attrib["cellcount"] == "4 4 4"
    assert len(flex.attrib["body"].split()) == len(geometry.points_m)
    assert len(flex.attrib["node"].split()) == 125
    assert root.find(".//equality") is None
    assert root.find(".//pin") is None
    for names in scene.patch_body_names:
        for name in names:
            body = root.find(f"./worldbody/body[@name='{name}']")
            assert body is not None and body.attrib["mocap"] == "true"
    assert len(scene.free_body_names) == 121
    free = root.find(f"./worldbody/body[@name='{scene.free_body_names[0]}']")
    assert free is not None and len(free.findall("joint")) == 3


def test_scene_xml_and_grid_roster_are_deterministic() -> None:
    geometry, contact = _fixture()
    left = build_mujoco_trilinear_scene_v1(
        geometry,
        contact=contact,
        **_scene_kwargs(),
    )
    right = build_mujoco_trilinear_scene_v1(
        geometry,
        contact=contact,
        **_scene_kwargs(),
    )
    assert left.xml == right.xml
    assert left.xml_sha256 == right.xml_sha256
    assert left.node_body_names == right.node_body_names
    np.testing.assert_array_equal(
        left.grid.vertex_node_indices,
        right.grid.vertex_node_indices,
    )
    np.testing.assert_array_equal(
        left.grid.vertex_node_weights,
        right.grid.vertex_node_weights,
    )
