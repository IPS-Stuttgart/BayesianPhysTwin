from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.rgbench_isotropic_mesh import (
    RGBenchIsotropicMeshConfig,
    build_isotropic_mesh_artifact,
    build_isotropic_mesh_manifest,
    load_isotropic_mesh_artifact,
    load_isotropic_mesh_manifest,
    write_json_once,
    write_obj_triangles,
)
from bayesian_phystwin.rgbench_online_belief import (
    load_obj_triangles,
    sha256_file,
)


def _grid_mesh(rows: int, columns: int, scale: float = 0.01) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    vertices = np.asarray(
        [
            (
                scale * column / (columns - 1),
                scale * row / (rows - 1),
                0.0,
            )
            for row in range(rows)
            for column in range(columns)
        ],
        dtype=np.float64,
    )
    faces: list[tuple[int, int, int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower = row * columns + column
            faces.append((lower, lower + 1, lower + columns + 1))
            faces.append((lower, lower + columns + 1, lower + columns))
    return vertices, np.asarray(faces, dtype=np.int64)


def _write_parameters(path: Path) -> None:
    path.write_text(
        "cloth_model_file_name: garment.obj\nshoulder_index: [0, 3]\n",
        encoding="utf-8",
    )


def test_identity_artifact_is_byte_compatible_and_round_trips(
    tmp_path: Path,
) -> None:
    vertices, faces = _grid_mesh(8, 16)
    source = tmp_path / "source.obj"
    derived = tmp_path / "derived.obj"
    parameters = tmp_path / "cloth.yaml"
    write_obj_triangles(source, vertices, faces)
    _write_parameters(parameters)
    artifact = build_isotropic_mesh_artifact(
        garment="synthetic",
        source_mesh=source,
        source_mesh_relative_path="source.obj",
        cloth_parameters=parameters,
        cloth_parameters_relative_path="cloth.yaml",
        source_fling_pin_indices=(0, 15),
        derived_mesh=derived,
        derived_mesh_relative_path="meshes/synthetic.obj",
        config=RGBenchIsotropicMeshConfig(identity_max_vertices=128),
        self_intersection_counter=lambda _vertices, _faces: 0,
    )
    assert artifact.mode == "identity"
    assert artifact.derived_fling_pin_indices == (0, 15)
    assert sha256_file(derived) == sha256_file(source)

    artifact_path = tmp_path / "artifact.json"
    write_json_once(artifact_path, artifact.descriptor())
    assert load_isotropic_mesh_artifact(artifact_path) == artifact


def test_remesh_selects_first_admissible_candidate_and_preserves_pins(
    tmp_path: Path,
) -> None:
    source_vertices, source_faces = _grid_mesh(12, 12)
    valid_vertices, valid_faces = _grid_mesh(8, 16)
    invalid_faces = np.vstack((valid_faces, valid_faces[0]))
    source = tmp_path / "source.obj"
    derived = tmp_path / "derived.obj"
    parameters = tmp_path / "cloth.yaml"
    write_obj_triangles(source, source_vertices, source_faces)
    _write_parameters(parameters)

    def remesher(
        _vertices: np.ndarray,
        _faces: np.ndarray,
        target_edge_length_m: float,
        _config: RGBenchIsotropicMeshConfig,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if target_edge_length_m == pytest.approx(0.001):
            return valid_vertices, invalid_faces, "test"
        return valid_vertices, valid_faces, "test"

    config = RGBenchIsotropicMeshConfig(
        identity_max_vertices=128,
        target_edge_lengths_um=(1_000, 2_000),
    )
    artifact = build_isotropic_mesh_artifact(
        garment="synthetic",
        source_mesh=source,
        source_mesh_relative_path="source.obj",
        cloth_parameters=parameters,
        cloth_parameters_relative_path="cloth.yaml",
        source_fling_pin_indices=(0, len(source_vertices) - 1),
        derived_mesh=derived,
        derived_mesh_relative_path="meshes/synthetic.obj",
        config=config,
        remesher=remesher,
        self_intersection_counter=lambda _vertices, _faces: 0,
    )
    assert artifact.mode == "isotropic_remesh"
    assert artifact.selected_target_edge_length_um == 2_000
    assert len(artifact.attempts) == 2
    assert "non_manifold_edges" in artifact.attempts[0].rejection_reasons
    assert artifact.attempts[1].accepted
    reloaded, _ = load_obj_triangles(derived)
    source_reloaded, _ = load_obj_triangles(source)
    np.testing.assert_array_equal(
        reloaded[np.asarray(artifact.derived_fling_pin_indices)],
        source_reloaded[[0, len(source_vertices) - 1]],
    )


def test_self_intersection_gate_rejects_every_candidate(tmp_path: Path) -> None:
    vertices, faces = _grid_mesh(8, 16)
    source = tmp_path / "source.obj"
    parameters = tmp_path / "cloth.yaml"
    write_obj_triangles(source, vertices, faces)
    _write_parameters(parameters)

    def unchanged_remesher(
        _vertices: np.ndarray,
        _faces: np.ndarray,
        _target_edge_length_m: float,
        _config: RGBenchIsotropicMeshConfig,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        return vertices, faces, "test"

    with pytest.raises(ValueError, match="no admissible mesh candidate"):
        build_isotropic_mesh_artifact(
            garment="synthetic",
            source_mesh=source,
            source_mesh_relative_path="source.obj",
            cloth_parameters=parameters,
            cloth_parameters_relative_path="cloth.yaml",
            source_fling_pin_indices=(0, 15),
            derived_mesh=tmp_path / "derived.obj",
            derived_mesh_relative_path="meshes/synthetic.obj",
            config=RGBenchIsotropicMeshConfig(
                identity_max_vertices=128,
                target_edge_lengths_um=(1_000,),
            ),
            remesher=unchanged_remesher,
            self_intersection_counter=lambda _vertices, _faces: 2,
        )


def test_artifact_digest_rejects_tampering(tmp_path: Path) -> None:
    vertices, faces = _grid_mesh(8, 16)
    source = tmp_path / "source.obj"
    parameters = tmp_path / "cloth.yaml"
    write_obj_triangles(source, vertices, faces)
    _write_parameters(parameters)
    artifact = build_isotropic_mesh_artifact(
        garment="synthetic",
        source_mesh=source,
        source_mesh_relative_path="source.obj",
        cloth_parameters=parameters,
        cloth_parameters_relative_path="cloth.yaml",
        source_fling_pin_indices=(0, 15),
        derived_mesh=tmp_path / "derived.obj",
        derived_mesh_relative_path="meshes/synthetic.obj",
        config=RGBenchIsotropicMeshConfig(identity_max_vertices=128),
        self_intersection_counter=lambda _vertices, _faces: 0,
    )
    payload = artifact.descriptor()
    payload["derived_vertex_count"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact digest changed"):
        load_isotropic_mesh_artifact(path)


def test_manifest_binds_artifact_and_mesh_files(tmp_path: Path) -> None:
    vertices, faces = _grid_mesh(8, 16)
    source = tmp_path / "source.obj"
    parameters = tmp_path / "cloth.yaml"
    derived = tmp_path / "meshes/synthetic.obj"
    write_obj_triangles(source, vertices, faces)
    _write_parameters(parameters)
    artifact = build_isotropic_mesh_artifact(
        garment="synthetic",
        source_mesh=source,
        source_mesh_relative_path="source.obj",
        cloth_parameters=parameters,
        cloth_parameters_relative_path="cloth.yaml",
        source_fling_pin_indices=(0, 15),
        derived_mesh=derived,
        derived_mesh_relative_path="meshes/synthetic.obj",
        config=RGBenchIsotropicMeshConfig(identity_max_vertices=128),
        self_intersection_counter=lambda _vertices, _faces: 0,
    )
    artifact_path = tmp_path / "artifacts/synthetic.json"
    write_json_once(artifact_path, artifact.descriptor())
    manifest = build_isotropic_mesh_manifest(
        (artifact_path,),
        root=tmp_path,
        rgbbench_commit="test",
        dataset_revision="test",
        dataset_manifest_artifact_sha256="a" * 64,
        dataset_manifest_file_sha256="b" * 64,
    )
    manifest_path = tmp_path / "manifest.json"
    write_json_once(manifest_path, manifest.descriptor())
    assert load_isotropic_mesh_manifest(manifest_path) == manifest

    derived.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bound derived mesh changed"):
        load_isotropic_mesh_manifest(manifest_path)


def test_registered_protocol_matches_runtime_mesh_config() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/sota/rgbbench_isotropic_dynamic_v2.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    registered = payload["mesh_admission"]
    runtime = asdict(RGBenchIsotropicMeshConfig())
    assert registered["identity_max_vertices"] == runtime["identity_max_vertices"]
    assert registered["physical_min_vertices"] == runtime["physical_min_vertices"]
    assert registered["remesh_iterations"] == runtime["remesh_iterations"]
    assert registered["maximum_surface_distance_um"] == runtime[
        "maximum_surface_distance_um"
    ]
    assert registered["maximum_source_mean_distance_um"] == runtime[
        "maximum_source_mean_distance_um"
    ]
    assert registered["maximum_source_p99_distance_um"] == runtime[
        "maximum_source_p99_distance_um"
    ]
    assert registered["maximum_source_distance_um"] == runtime[
        "maximum_source_distance_um"
    ]
    assert registered["feature_angle_degrees"] == runtime[
        "feature_angle_degrees"
    ]
    edge_grid = registered["target_edge_lengths_um"]
    assert tuple(
        range(
            edge_grid["start"],
            edge_grid["stop_inclusive"] + edge_grid["step"],
            edge_grid["step"],
        )
    ) == runtime["target_edge_lengths_um"]
