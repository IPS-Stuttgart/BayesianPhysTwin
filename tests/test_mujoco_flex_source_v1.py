from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

import bayesian_phystwin.mujoco_flex_source_v1 as module
from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
)
from bayesian_phystwin.mujoco_flex_source_v1 import (
    ATTACHMENT_MODEL,
    CONSTITUTIVE_MODEL,
    MUJOCO_REVISION,
    MUJOCO_VERSION,
    MUJOCO_WHEEL_FILENAME,
    MUJOCO_WHEEL_SHA256,
    build_mujoco_flex_scene_v1,
    load_native_mujoco_flex_modules_v1,
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
    assert ATTACHMENT_MODEL == ("direct-rigid-projected-vertex-mocap-Dirichlet-v2")
    assert "Saint-Venant-Kirchhoff" in CONSTITUTIVE_MODEL


def test_scene_assigns_patch_vertices_directly_and_free_vertices_to_slides() -> None:
    scene = build_mujoco_flex_scene_v1(
        _geometry(),
        integrator="implicitfast",
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
    option = root.find("./option")
    assert option is not None and option.attrib["integrator"] == "implicitfast"
    flex = root.find("./deformable/flex")
    assert flex is not None
    assert flex.attrib["dim"] == "3"
    assert flex.attrib["body"].split() == [
        "contact_vertex_0",
        "contact_vertex_1",
        "contact_vertex_2",
        "contact_vertex_3",
        "free_vertex_4",
    ]
    assert scene.attachment_body_names == (
        "contact_vertex_0",
        "contact_vertex_1",
        "contact_vertex_2",
        "contact_vertex_3",
    )
    assert scene.free_body_names == ("free_vertex_4",)
    contact = root.find("./worldbody/body[@name='contact_vertex_0']")
    assert contact is not None and contact.attrib["mocap"] == "true"
    free = root.find("./worldbody/body[@name='free_vertex_4']")
    assert free is not None
    assert len(free.findall("joint")) == 3
    assert root.find(".//equality") is None
    assert root.find(".//pin") is None


def test_scene_xml_and_mass_are_deterministic() -> None:
    kwargs = {
        "integrator": "implicitfast",
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


def test_native_loader_verifies_wheel_abi_and_installed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / MUJOCO_WHEEL_FILENAME
    wheel.write_bytes(b"pinned wheel")
    package_root = tmp_path / "site"
    for relative in module.MUJOCO_INSTALLED_FILE_SHA256:
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    module_path = package_root / "mujoco" / "__init__.py"
    mujoco = SimpleNamespace(__file__=str(module_path), __version__=MUJOCO_VERSION)
    distribution = SimpleNamespace(locate_file=lambda relative: package_root / relative)

    monkeypatch.setattr(module.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(
        module.platform, "python_version_tuple", lambda: ("3", "10", "0")
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: MUJOCO_VERSION)
    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda _: distribution
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda _: mujoco)

    def pinned_digest(path: Path) -> str:
        if path == wheel:
            return MUJOCO_WHEEL_SHA256
        return module.MUJOCO_INSTALLED_FILE_SHA256[
            path.relative_to(package_root).as_posix()
        ]

    monkeypatch.setattr(module, "_sha256_file", pinned_digest)
    native = load_native_mujoco_flex_modules_v1(wheel)

    assert native.mujoco is mujoco
    assert native.package_root == package_root
    assert native.installed_records == {
        relative: {"sha256": digest}
        for relative, digest in module.MUJOCO_INSTALLED_FILE_SHA256.items()
    }


def test_runtime_helpers_cover_hash_format_validation_and_warnings(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    assert module._sha256_file(payload) == hashlib.sha256(b"payload").hexdigest()
    assert module._numbers(np.asarray([1, 2], dtype=np.int64)) == "1 2"
    assert module._numbers(np.asarray([0.5], dtype=np.float64)) == "0.5"
    warnings = SimpleNamespace(
        warning=[SimpleNamespace(number=2), SimpleNamespace(number=3)]
    )
    assert module._warning_count(warnings) == 5

    with pytest.raises(ValueError, match="value must be finite"):
        module._finite(True, name="value")
    with pytest.raises(ValueError, match="positive and finite"):
        module._finite(0.0, name="value", positive=True)
    with pytest.raises(ValueError, match="explicit failure"):
        module._require(False, "explicit failure")
