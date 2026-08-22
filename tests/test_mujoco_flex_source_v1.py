from __future__ import annotations

from xml.etree import ElementTree

import numpy as np

from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
)
from bayesian_phystwin.mujoco_flex_source_v1 import (
    ATTACHMENT_MODEL,
    CONSTITUTIVE_MODEL,
    MUJOCO_REVISION,
    MUJOCO_VERSION,
    MUJOCO_WHEEL_SHA256,
    build_mujoco_flex_scene_v1,
)
from bayesian_phystwin.native_tet_fem_source_v1 import (
    NativeTetSourceGeometryV1,
    prepare_native_tet_source_geometry_v1,
)


def _geometry() -> NativeTetSourceGeometryV1:
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
    contact = RigidContactProjectionV1(
        projected_targets_m=np.stack((points[attachments], points[attachments])),
        rotations=np.repeat(np.eye(3)[None, None], 2, axis=0),
        translations_m=np.zeros((2, 1, 3)),
        patch_local_indices=(np.arange(4, dtype=np.int64),),
        patch_ranks=(3,),
    )
    return prepare_native_tet_source_geometry_v1(
        points_m=points,
        cells=cells,
        attachment_indices=attachments,
        contact=contact,
    )


def test_module_import_freezes_runtime_and_native_models() -> None:
    assert MUJOCO_VERSION == "3.9.0"
    assert MUJOCO_REVISION == "237c17e48539b6c90bf90d3161547cbdcbfaa1e0"
    assert MUJOCO_WHEEL_SHA256 == (
        "c148824d73487fe5ee29c371eff981645f372ccada1f20ea331288323e37c65e"
    )
    assert ATTACHMENT_MODEL == "direct-rigid-patch-mocap-body-Dirichlet-v1"
    assert "Saint-Venant-Kirchhoff" in CONSTITUTIVE_MODEL


def test_scene_assigns_patch_vertices_directly_and_free_vertices_to_slides() -> None:
    scene = build_mujoco_flex_scene_v1(
        _geometry(),
        integrator_time_step_s=1e-4,
        young_modulus_pa=1000.0,
        poisson_ratio=0.3,
        density_kg_m3=1000.0,
        edge_damping=1.0,
        elasticity_damping=1.0,
        joint_damping=0.1,
        solver_iterations=100,
        solver_tolerance=1e-12,
    )
    root = ElementTree.fromstring(scene.xml)
    flex = root.find("./deformable/flex")
    assert flex is not None
    assert flex.attrib["dim"] == "3"
    assert flex.attrib["body"].split() == [
        "contact_patch_0",
        "contact_patch_0",
        "contact_patch_0",
        "contact_patch_0",
        "free_vertex_4",
    ]
    assert scene.patch_body_names == ("contact_patch_0",)
    assert scene.free_body_names == ("free_vertex_4",)
    patch = root.find("./worldbody/body[@name='contact_patch_0']")
    assert patch is not None and patch.attrib["mocap"] == "true"
    free = root.find("./worldbody/body[@name='free_vertex_4']")
    assert free is not None
    assert len(free.findall("joint")) == 3
    assert root.find(".//equality") is None
    assert root.find(".//pin") is None


def test_scene_xml_and_mass_are_deterministic() -> None:
    kwargs = {
        "integrator_time_step_s": 1e-4,
        "young_modulus_pa": 1000.0,
        "poisson_ratio": 0.3,
        "density_kg_m3": 1000.0,
        "edge_damping": 1.0,
        "elasticity_damping": 1.0,
        "joint_damping": 0.1,
        "solver_iterations": 100,
        "solver_tolerance": 1e-12,
    }
    left = build_mujoco_flex_scene_v1(_geometry(), **kwargs)
    right = build_mujoco_flex_scene_v1(_geometry(), **kwargs)
    assert left.xml == right.xml
    assert left.xml_sha256 == right.xml_sha256
    assert left.total_reference_mass_kg > 0.0
