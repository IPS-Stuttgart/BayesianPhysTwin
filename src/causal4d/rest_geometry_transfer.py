"""Canonical source-to-target transfer of inferred PhysTwin rest geometry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from causal4d.real_protocol import validate_protocol
from causal4d.rest_geometry import (
    RigidFrameCorrection,
    apply_frame_correction,
    reattach_controller_rest_lengths,
    rotate_vectors,
    rotation_angle,
)
from causal4d.rest_geometry_cross_action import (
    canonical_rest_geometry_hyperparameters,
    rest_geometry_hyperparameter_id,
)


SOURCE_CORRECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceRestGeometryCorrection:
    """Validated persistent correction inferred from one source execution."""

    protocol_id: str
    protocol_design_sha256: str
    source_execution_id: str
    selected_candidate_id: str
    hyperparameters: dict[str, Any]
    frame: RigidFrameCorrection
    nonrigid_field: np.ndarray
    corrected_reference_vertices: np.ndarray
    corrected_object_rest_lengths: np.ndarray
    canonical_material_graph_sha256: str
    source_manifest_sha256: str


@dataclass(frozen=True)
class CanonicalMaterialGraph:
    """Object-only graph shared by every execution in the real protocol."""

    vertices: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    masses: np.ndarray
    sha256: str


@dataclass(frozen=True)
class TargetRestGeometryConfiguration:
    """State, controls, and rest lengths ready for one target Warp restart."""

    position: np.ndarray
    velocity: np.ndarray
    controller_points: np.ndarray
    rest_lengths: np.ndarray
    contact_policy: str
    controller_attachment_policy: str
    source_execution_id: str
    selected_candidate_id: str
    canonical_material_graph_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array(digest, name: str, values: np.ndarray, dtype) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest.update(name.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def canonical_material_graph_sha256(
    reference_vertices: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    num_object_springs: int,
) -> str:
    """Hash the canonical object graph while excluding contact springs."""

    vertices = np.asarray(reference_vertices, dtype=np.float32)
    edges = np.asarray(springs, dtype=np.int32)
    lengths = np.asarray(rest_lengths, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(
        np.isfinite(vertices)
    ):
        raise ValueError("reference_vertices must have finite shape (N, 3)")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) != len(lengths):
        raise ValueError("springs and rest_lengths must agree")
    if not 0 < num_object_springs <= len(edges):
        raise ValueError("num_object_springs must lie in (0, S]")
    object_edges = edges[:num_object_springs]
    if np.any(object_edges < 0) or np.any(object_edges >= len(vertices)):
        raise ValueError("object spring endpoint exceeds reference vertices")
    if np.any(lengths[:num_object_springs] <= 0.0) or not np.all(
        np.isfinite(lengths[:num_object_springs])
    ):
        raise ValueError("object rest lengths must be positive and finite")
    digest = hashlib.sha256()
    _array(digest, "vertices", vertices, np.float32)
    _array(digest, "springs", object_edges, np.int32)
    _array(
        digest,
        "rest_lengths",
        lengths[:num_object_springs],
        np.float32,
    )
    return digest.hexdigest()


def write_canonical_material_graph(
    path: str | Path,
    vertices: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    masses: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write the immutable object-only graph used by every protocol execution."""

    points = np.asarray(vertices, dtype=np.float32)
    edges = np.asarray(springs, dtype=np.int32)
    lengths = np.asarray(rest_lengths, dtype=np.float32)
    weights = (
        np.ones(len(points), dtype=np.float32)
        if masses is None
        else np.asarray(masses, dtype=np.float32)
    )
    if weights.shape != (len(points),) or np.any(weights <= 0.0):
        raise ValueError("canonical material masses must be positive per vertex")
    digest = canonical_material_graph_sha256(
        points,
        edges,
        lengths,
        num_object_springs=len(edges),
    )
    output = Path(path)
    if output.suffix != ".npz":
        raise ValueError("canonical material graph path must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices=points,
        springs=edges,
        rest_lengths=lengths,
        masses=weights,
        canonical_material_graph_sha256=np.asarray(digest),
    )
    return {
        "path": str(output.resolve()),
        "file_sha256": _sha256_file(output),
        "canonical_material_graph_sha256": digest,
        "object_vertex_count": len(points),
        "object_spring_count": len(edges),
    }


def load_canonical_material_graph(path: str | Path) -> CanonicalMaterialGraph:
    """Load and validate an immutable object-only graph artifact."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "vertices",
            "springs",
            "rest_lengths",
            "masses",
            "canonical_material_graph_sha256",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                "canonical material graph is missing: " + ", ".join(sorted(missing))
            )
        vertices = np.asarray(archive["vertices"], dtype=np.float32)
        springs = np.asarray(archive["springs"], dtype=np.int32)
        rest_lengths = np.asarray(archive["rest_lengths"], dtype=np.float32)
        masses = np.asarray(archive["masses"], dtype=np.float32)
        stored_digest = str(np.asarray(archive["canonical_material_graph_sha256"]).item())
    if masses.shape != (len(vertices),) or np.any(masses <= 0.0):
        raise ValueError("canonical material graph masses are invalid")
    digest = canonical_material_graph_sha256(
        vertices,
        springs,
        rest_lengths,
        num_object_springs=len(springs),
    )
    if stored_digest != digest:
        raise ValueError("canonical material graph digest mismatch")
    return CanonicalMaterialGraph(
        vertices=vertices,
        springs=springs,
        rest_lengths=rest_lengths,
        masses=masses,
        sha256=digest,
    )


def attach_target_controller_to_canonical_graph(
    canonical: CanonicalMaterialGraph,
    controller_reference: np.ndarray,
    *,
    config: PhysTwinSpringGraphConfig,
) -> PhysTwinSpringGraph:
    """Preserve object topology and rebuild only target controller springs."""

    candidate = build_phystwin_spring_graph(
        canonical.vertices,
        controller_reference,
        config=config,
    )
    if candidate.num_object_springs != len(canonical.springs):
        raise ValueError("canonical object spring count changed during attachment")
    if not np.array_equal(
        candidate.springs[: candidate.num_object_springs], canonical.springs
    ) or not np.allclose(
        candidate.rest_lengths[: candidate.num_object_springs],
        canonical.rest_lengths,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("controller attachment rebuilt the canonical object graph")
    object_count = len(canonical.vertices)
    controller_count = len(candidate.vertices) - object_count
    springs = np.concatenate(
        (
            canonical.springs,
            candidate.springs[candidate.num_object_springs :],
        ),
        axis=0,
    )
    rest_lengths = np.concatenate(
        (
            canonical.rest_lengths,
            candidate.rest_lengths[candidate.num_object_springs :],
        )
    )
    return PhysTwinSpringGraph(
        vertices=np.concatenate(
            (canonical.vertices, candidate.vertices[object_count:]), axis=0
        ).astype(np.float32),
        springs=springs.astype(np.int32),
        rest_lengths=rest_lengths.astype(np.float32),
        masses=np.concatenate(
            (
                canonical.masses,
                np.ones(controller_count, dtype=np.float32),
            )
        ),
        num_object_springs=len(canonical.springs),
        num_object_points=object_count,
    )


def source_correction_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _summary_hyperparameters(summary: Mapping[str, Any]) -> dict[str, Any]:
    config = summary["config"]
    selection = summary["selection"]
    return canonical_rest_geometry_hyperparameters(
        {
            "frame_mode": config["frame_mode"],
            "frame_scale": selection["selected_frame_scale"],
            "rest_geometry_scale": selection["selected_rest_geometry_scale"],
            "controller_rest_mode": selection[
                "selected_controller_rest_mode"
            ],
            "graph_prior_strength": config["graph_prior_strength"],
            "rest_length_ratio_bound": float(
                np.exp(config["maximum_rest_log_ratio"])
            ),
        }
    )


def write_source_rest_geometry_correction(
    protocol: Mapping[str, Any],
    source_execution_id: str,
    summary_path: str | Path,
    archive_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export only the persistent, pre-holdout correction needed for transfer."""

    validate_protocol(protocol)
    execution_ids = {
        execution["execution_id"] for execution in protocol["executions"]
    }
    if source_execution_id not in execution_ids:
        raise ValueError("source correction execution is not preregistered")
    summary_source = Path(summary_path)
    archive_source = Path(archive_path)
    summary = json.loads(summary_source.read_text(encoding="utf-8"))
    boundary = summary.get("information_boundary", {})
    if (
        boundary.get("holdout_frames_used_for_inference") is not False
        or boundary.get("holdout_frames_used_for_hyperparameter_selection")
        is not False
        or boundary.get("manual_gt_track_used_for_hyperparameter_selection")
        is not False
    ):
        raise ValueError("source correction crossed the holdout boundary")
    hyperparameters = _summary_hyperparameters(summary)
    selected_candidate_id = rest_geometry_hyperparameter_id(hyperparameters)
    with np.load(archive_source, allow_pickle=False) as archive:
        required = (
            "frame_linear",
            "frame_translation",
            "nonrigid_field",
            "canonical_reference_vertices",
            "corrected_reference_vertices",
            "corrected_rest_lengths",
            "released_rest_lengths",
            "object_springs",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(
                "source correction archive is missing: " + ", ".join(missing)
            )
        arrays = {name: np.asarray(archive[name]) for name in required}
    object_spring_count = int(summary["graph"]["object_spring_count"])
    object_vertex_count = int(summary["graph"]["object_vertex_count"])
    if arrays["nonrigid_field"].shape != (object_vertex_count, 3):
        raise ValueError("source nonrigid field does not match the object graph")
    if arrays["corrected_reference_vertices"].shape != (object_vertex_count, 3):
        raise ValueError("source corrected reference does not match the object graph")
    if len(arrays["object_springs"]) != object_spring_count:
        raise ValueError("source object spring count changed")
    material_digest = canonical_material_graph_sha256(
        arrays["canonical_reference_vertices"],
        arrays["object_springs"],
        arrays["released_rest_lengths"][:object_spring_count],
        num_object_springs=object_spring_count,
    )
    expected_digest = summary["graph"].get("canonical_material_graph_sha256")
    if expected_digest != material_digest:
        raise ValueError("source summary and archive canonical graph disagree")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    correction_archive = output / "source_rest_geometry_correction.npz"
    np.savez_compressed(
        correction_archive,
        frame_linear=np.asarray(arrays["frame_linear"], dtype=np.float64),
        frame_translation=np.asarray(arrays["frame_translation"], dtype=np.float64),
        nonrigid_field=np.asarray(arrays["nonrigid_field"], dtype=np.float64),
        corrected_reference_vertices=np.asarray(
            arrays["corrected_reference_vertices"], dtype=np.float64
        ),
        corrected_object_rest_lengths=np.asarray(
            arrays["corrected_rest_lengths"][:object_spring_count],
            dtype=np.float64,
        ),
    )
    correction_archive_sha256 = _sha256_file(correction_archive)
    manifest = {
        "schema_version": SOURCE_CORRECTION_SCHEMA_VERSION,
        "artifact_kind": "source_rest_geometry_correction",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "source_execution_id": source_execution_id,
        "selected_candidate_id": selected_candidate_id,
        "hyperparameters": hyperparameters,
        "canonical_material_graph_sha256": material_digest,
        "object_vertex_count": object_vertex_count,
        "object_spring_count": object_spring_count,
        "information_boundary": {
            "source_evidence_frames": "pre_holdout_only",
            "source_holdout_frames_used": False,
            "target_frames_used": False,
        },
        "inputs": {
            "summary_sha256": _sha256_file(summary_source),
            "source_archive_sha256": _sha256_file(archive_source),
        },
        "correction_archive": {
            "path": correction_archive.name,
            "sha256": correction_archive_sha256,
        },
    }
    manifest["manifest_sha256"] = source_correction_manifest_sha256(manifest)
    manifest_path = output / "source_rest_geometry_correction.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def load_source_rest_geometry_correction(
    manifest_path: str | Path,
) -> SourceRestGeometryCorrection:
    """Load and validate a persistent source correction artifact."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SOURCE_CORRECTION_SCHEMA_VERSION:
        raise ValueError("unsupported source correction schema")
    if manifest.get("artifact_kind") != "source_rest_geometry_correction":
        raise ValueError("unexpected source correction artifact kind")
    if manifest.get("manifest_sha256") != source_correction_manifest_sha256(
        manifest
    ):
        raise ValueError("source correction manifest SHA-256 mismatch")
    if manifest.get("information_boundary") != {
        "source_evidence_frames": "pre_holdout_only",
        "source_holdout_frames_used": False,
        "target_frames_used": False,
    }:
        raise ValueError("source correction crossed its information boundary")
    if not _is_sha256(manifest.get("canonical_material_graph_sha256")):
        raise ValueError("source correction canonical graph digest is invalid")
    hyperparameters = canonical_rest_geometry_hyperparameters(
        manifest.get("hyperparameters", {})
    )
    candidate_id = rest_geometry_hyperparameter_id(hyperparameters)
    if manifest.get("selected_candidate_id") != candidate_id:
        raise ValueError("source correction candidate digest changed")
    descriptor = manifest.get("correction_archive", {})
    archive_name = descriptor.get("path")
    if (
        not isinstance(archive_name, str)
        or not archive_name
        or Path(archive_name).is_absolute()
        or ".." in Path(archive_name).parts
    ):
        raise ValueError("source correction archive path is unsafe")
    archive_path = path.parent / archive_name
    if not archive_path.is_file() or _sha256_file(archive_path) != descriptor.get(
        "sha256"
    ):
        raise ValueError("source correction archive SHA-256 mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        linear = np.asarray(archive["frame_linear"], dtype=float)
        translation = np.asarray(archive["frame_translation"], dtype=float)
        nonrigid = np.asarray(archive["nonrigid_field"], dtype=float)
        corrected_reference = np.asarray(
            archive["corrected_reference_vertices"], dtype=float
        )
        object_rest = np.asarray(
            archive["corrected_object_rest_lengths"], dtype=float
        )
    vertex_count = int(manifest["object_vertex_count"])
    spring_count = int(manifest["object_spring_count"])
    if linear.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("source correction frame has invalid shape")
    if not np.allclose(linear.T @ linear, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(linear), 1.0, atol=1e-7
    ):
        raise ValueError("source correction frame is not a proper rotation")
    if nonrigid.shape != (vertex_count, 3) or corrected_reference.shape != (
        vertex_count,
        3,
    ):
        raise ValueError("source correction object arrays have invalid shape")
    if object_rest.shape != (spring_count,) or np.any(object_rest <= 0.0):
        raise ValueError("source correction object rest lengths are invalid")
    if not all(
        np.all(np.isfinite(value))
        for value in (linear, translation, nonrigid, corrected_reference, object_rest)
    ):
        raise ValueError("source correction arrays must be finite")
    frame = RigidFrameCorrection(
        linear=linear,
        translation=translation,
        mode=hyperparameters["frame_mode"],
        rotation_angle_rad=rotation_angle(linear),
        fitted_point_count=vertex_count,
    )
    return SourceRestGeometryCorrection(
        protocol_id=manifest["protocol_id"],
        protocol_design_sha256=manifest["protocol_design_sha256"],
        source_execution_id=manifest["source_execution_id"],
        selected_candidate_id=candidate_id,
        hyperparameters=hyperparameters,
        frame=frame,
        nonrigid_field=nonrigid,
        corrected_reference_vertices=corrected_reference,
        corrected_object_rest_lengths=object_rest,
        canonical_material_graph_sha256=manifest[
            "canonical_material_graph_sha256"
        ],
        source_manifest_sha256=manifest["manifest_sha256"],
    )


def prepare_target_rest_geometry_configuration(
    source: SourceRestGeometryCorrection,
    *,
    target_material_graph_sha256: str,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    target_controller_points: np.ndarray,
    target_springs: np.ndarray,
    target_released_rest_lengths: np.ndarray,
    num_object_springs: int,
    contact_policy: str,
) -> TargetRestGeometryConfiguration:
    """Transfer persistent source geometry into a target state and contact graph."""

    if target_material_graph_sha256 != source.canonical_material_graph_sha256:
        raise ValueError("source and target do not share the canonical material graph")
    if contact_policy not in {
        "factual_same_execution",
        "same_grasp",
        "new_contact",
    }:
        raise ValueError("unsupported rest-geometry transfer contact policy")
    position = np.asarray(target_position, dtype=float)
    velocity = np.asarray(target_velocity, dtype=float)
    controller = np.asarray(target_controller_points, dtype=float)
    springs = np.asarray(target_springs, dtype=np.int64)
    released = np.asarray(target_released_rest_lengths, dtype=float)
    object_count = len(source.nonrigid_field)
    if position.shape != (object_count, 3) or velocity.shape != position.shape:
        raise ValueError("target state does not match the canonical object")
    if controller.ndim != 3 or controller.shape[2] != 3:
        raise ValueError("target_controller_points must have shape (T, C, 3)")
    if springs.ndim != 2 or springs.shape[1] != 2 or len(springs) != len(released):
        raise ValueError("target springs and rest lengths must agree")
    if num_object_springs != len(source.corrected_object_rest_lengths):
        raise ValueError("target object spring count differs from the source")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("target state must be finite")
    if not np.all(np.isfinite(controller)):
        raise ValueError("target controller trajectory must be finite")
    if np.any(released <= 0.0) or not np.all(np.isfinite(released)):
        raise ValueError("target released rest lengths must be positive and finite")
    corrected_position = apply_frame_correction(position, source.frame)
    corrected_position += (
        source.hyperparameters["rest_geometry_scale"] * source.nonrigid_field
    )
    corrected_velocity = rotate_vectors(velocity, source.frame)
    corrected_controller = apply_frame_correction(controller, source.frame)
    corrected_rest = released.copy()
    corrected_rest[:num_object_springs] = source.corrected_object_rest_lengths

    recompute_attachment = (
        contact_policy == "new_contact"
        or source.hyperparameters["controller_rest_mode"] == "recompute"
    )
    attachment_policy = "preserve_registered_attachment"
    if recompute_attachment:
        corrected_rest, _, _ = reattach_controller_rest_lengths(
            source.corrected_reference_vertices,
            corrected_controller[0],
            springs,
            corrected_rest,
            num_object_springs=num_object_springs,
            maximum_log_ratio=float(
                np.log(source.hyperparameters["rest_length_ratio_bound"])
            ),
        )
        attachment_policy = "rebuild_on_corrected_target_contact"
    return TargetRestGeometryConfiguration(
        position=corrected_position,
        velocity=corrected_velocity,
        controller_points=corrected_controller,
        rest_lengths=corrected_rest,
        contact_policy=contact_policy,
        controller_attachment_policy=attachment_policy,
        source_execution_id=source.source_execution_id,
        selected_candidate_id=source.selected_candidate_id,
        canonical_material_graph_sha256=target_material_graph_sha256,
    )
