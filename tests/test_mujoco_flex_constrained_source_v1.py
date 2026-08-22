from __future__ import annotations

from xml.etree import ElementTree

import numpy as np

from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
)
from bayesian_phystwin.mujoco_flex_constrained_source_v1 import (
    ATTACHMENT_MODEL,
    BACKEND_VARIANT,
    CONSTRAINT_POLICY,
    MASS_CONDITION_NUMBER_CAP,
    MASS_POLICY,
    build_mujoco_constrained_flex_scene_v1,
    condition_lumped_vertex_masses_v1,
)
from bayesian_phystwin.native_tet_fem_source_v1 import (
    prepare_native_tet_source_geometry_v1,
)


def test_scene_keeps_every_vertex_dynamic_and_uses_native_connects() -> None:
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
    scene = build_mujoco_constrained_flex_scene_v1(
        geometry,
        integrator="implicitfast",
        integrator_time_step_s=1.0e-4,
        young_modulus_pa=1000.0,
        poisson_ratio=0.3,
        density_kg_m3=1000.0,
        edge_damping=1.0,
        elasticity_damping=1.0,
        joint_damping=0.1,
        solver_iterations=100,
        solver_tolerance=1.0e-12,
    )
    assert BACKEND_VARIANT == "mujoco-volumetric-flex-point-constraint-v1"
    assert ATTACHMENT_MODEL == "dynamic-vertex-native-connect-to-rigid-target-v1"
    assert CONSTRAINT_POLICY == "critical-native-connect-minimum-safe-timeconstant-v1"
    assert MASS_POLICY == "minimal-conservative-uniform-blend-condition-cap-v1"
    assert MASS_CONDITION_NUMBER_CAP == 100.0
    assert scene.constraint_time_constant_s == 2.0e-4
    assert scene.conditioned_mass_condition_number <= 100.0 * (1.0 + 1.0e-12)
    root = ElementTree.fromstring(scene.xml)
    flex = root.find("./deformable/flex")
    assert flex is not None
    assert len(flex.attrib["body"].split()) == len(points)
    assert set(flex.attrib["body"].split()) == set(scene.vertex_body_names)
    connects = root.findall("./equality/connect")
    assert len(connects) == len(attachments)
    assert [item.attrib["name"] for item in connects] == list(scene.equality_names)
    for vertex_name, target_name, constraint in zip(
        scene.attachment_vertex_body_names,
        scene.target_body_names,
        connects,
        strict=True,
    ):
        vertex = root.find(f"./worldbody/body[@name='{vertex_name}']")
        target = root.find(f"./worldbody/body[@name='{target_name}']")
        assert vertex is not None and "mocap" not in vertex.attrib
        assert vertex.find("inertial") is not None
        assert len(vertex.findall("joint")) == 3
        assert target is not None and target.attrib["mocap"] == "true"
        assert constraint.attrib["body1"] == vertex_name
        assert constraint.attrib["body2"] == target_name
        assert constraint.attrib["anchor"] == "0 0 0"
        assert constraint.attrib["solref"] == "0.00020000000000000001 1"


def test_scene_xml_is_deterministic() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
        dtype=np.float64,
    )
    contact = RigidContactProjectionV1(
        projected_targets_m=np.stack((points, points)),
        rotations=np.repeat(np.eye(3)[None, None], 2, axis=0),
        translations_m=np.zeros((2, 1, 3), dtype=np.float64),
        patch_local_indices=(np.arange(4, dtype=np.int64),),
        patch_ranks=(3,),
    )
    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points,
        cells=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        attachment_indices=np.arange(4, dtype=np.int64),
        contact=contact,
    )
    kwargs = {
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
    left = build_mujoco_constrained_flex_scene_v1(geometry, **kwargs)
    right = build_mujoco_constrained_flex_scene_v1(geometry, **kwargs)
    assert left.xml == right.xml
    assert left.xml_sha256 == right.xml_sha256


def test_mass_conditioning_is_minimal_conservative_and_capped() -> None:
    raw = np.asarray([1.0, 2.0, 20.0, 10000.0], dtype=np.float64)
    conditioned, blend = condition_lumped_vertex_masses_v1(raw)
    assert 0.0 < blend < 1.0
    assert np.isclose(np.sum(conditioned), np.sum(raw), atol=1.0e-12, rtol=0.0)
    assert np.isclose(
        np.max(conditioned) / np.min(conditioned),
        100.0,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    unchanged, zero_blend = condition_lumped_vertex_masses_v1(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    )
    assert zero_blend == 0.0
    np.testing.assert_array_equal(unchanged, [1.0, 2.0, 3.0])
