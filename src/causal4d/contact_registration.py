"""Typed multiview contact-registration artifact for physical acquisition."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.real_protocol import validate_protocol


CONTACT_REGISTRATION_SCHEMA_VERSION = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return result


def _covariance(value: Any, dimension: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (dimension, dimension) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape ({dimension}, {dimension})")
    if not np.allclose(result, result.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    if np.min(np.linalg.eigvalsh(result)) < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite")
    return result


def _validate_transform(transform: Mapping[str, Any], name: str) -> None:
    matrix = np.asarray(transform.get("matrix"), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name}.matrix must be a finite 4x4 transform")
    _require(
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0.0),
        f"{name}.matrix has an invalid homogeneous row",
    )
    rotation = matrix[:3, :3]
    _require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=1e-6)
        and np.linalg.det(rotation) > 0.0,
        f"{name}.matrix rotation must lie in SO(3)",
    )
    _covariance(transform.get("covariance_se3"), 6, f"{name}.covariance_se3")


def _validate_descriptor(descriptor: Mapping[str, Any], name: str) -> None:
    path = descriptor.get("path")
    _require(isinstance(path, str) and path, f"{name}.path is missing")
    parsed = Path(path)
    _require(
        not parsed.is_absolute() and ".." not in parsed.parts, f"{name}.path is unsafe"
    )
    _require(_is_sha256(descriptor.get("sha256")), f"{name}.sha256 is invalid")
    byte_count = descriptor.get("bytes")
    _require(
        isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count >= 0,
        f"{name}.bytes is invalid",
    )


def build_contact_registration_template(
    protocol: Mapping[str, Any],
    *,
    camera_ids: Sequence[str],
    object_node_count: int,
) -> dict[str, Any]:
    """Build an explicitly incomplete version-2 registration template."""

    validate_protocol(protocol)
    cameras = [str(value) for value in camera_ids]
    if (
        len(cameras) < 3
        or len(set(cameras)) != len(cameras)
        or any(not value for value in cameras)
    ):
        raise ValueError("at least three unique camera ids are required")
    if object_node_count < 1:
        raise ValueError("object_node_count must be positive")
    transform = {"matrix": None, "covariance_se3": None}
    descriptor = {"path": None, "sha256": None, "bytes": None}
    return {
        "schema_version": CONTACT_REGISTRATION_SCHEMA_VERSION,
        "artifact_kind": "PhysicalContactRegistration",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "status": "template",
        "object": {
            "object_id": protocol["object"]["object_id"],
            "physical_instance_serial": None,
            "twin_geometry_sha256": None,
            "object_node_count": int(object_node_count),
            "geometry_artifact": deepcopy(descriptor),
        },
        "frames": {
            "cameras": {
                camera_id: {
                    "camera_to_world": deepcopy(transform),
                    "calibration_artifact": deepcopy(descriptor),
                }
                for camera_id in cameras
            },
            "controller_to_world": deepcopy(transform),
            "support_to_world": deepcopy(transform),
            "gravity_direction_world": None,
            "closure": {
                camera_id: {
                    "translation_error_m": None,
                    "rotation_error_deg": None,
                }
                for camera_id in cameras
            },
        },
        "support_geometry": {
            "surface_id": None,
            "kind": None,
            "origin_world_m": None,
            "normal_world": None,
            "uncertainty_m": None,
            "contact_state_changes_recorded": None,
            "artifact": deepcopy(descriptor),
        },
        "contact_regions": {
            region["id"]: {
                "physical_centroid_world_m": None,
                "physical_normal_world": None,
                "tangent_basis_world": None,
                "attachment": {
                    "representation": "weighted_node_patch",
                    "node_indices": None,
                    "weights": None,
                    "centroid_covariance_world_m2": None,
                },
                "per_view_overlays": {
                    camera_id: {
                        "centroid_px": None,
                        "artifact": deepcopy(descriptor),
                    }
                    for camera_id in cameras
                },
                "independent_reviews": [],
                "interreview_rms_m": None,
                "multiview_reprojection_rmse_px": None,
            }
            for region in protocol["contact_regions"]
        },
        "acceptance": {
            "multiview_agreement_passed": None,
            "independent_review_passed": None,
            "attachment_uncertainty_separates_regions": None,
            "frame_closure_recorded": None,
            "target_outcomes_used": False,
        },
        "approval": {
            "approved": False,
            "approver_id": None,
            "approved_at_utc": None,
        },
        "source_checksums": {},
    }


def validate_contact_registration(
    artifact: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a completed physical-to-graph registration before acquisition."""

    validate_protocol(protocol)
    _require(
        artifact.get("schema_version") == 2, "unsupported contact registration schema"
    )
    _require(
        artifact.get("artifact_kind") == "PhysicalContactRegistration",
        "unexpected contact registration kind",
    )
    _require(
        artifact.get("protocol_id") == protocol["protocol_id"], "protocol id changed"
    )
    _require(
        artifact.get("protocol_design_sha256") == protocol["design_sha256"],
        "protocol digest changed",
    )
    _require(
        artifact.get("status") == "approved", "contact registration is not approved"
    )
    object_record = artifact["object"]
    _require(
        object_record["object_id"] == protocol["object"]["object_id"],
        "object id changed",
    )
    _require(
        isinstance(object_record.get("physical_instance_serial"), str)
        and bool(object_record["physical_instance_serial"]),
        "physical object serial is missing",
    )
    _require(
        _is_sha256(object_record.get("twin_geometry_sha256")),
        "twin geometry hash is invalid",
    )
    node_count = object_record.get("object_node_count")
    _require(
        isinstance(node_count, int)
        and not isinstance(node_count, bool)
        and node_count > 0,
        "object node count is invalid",
    )
    _validate_descriptor(object_record["geometry_artifact"], "object.geometry_artifact")

    frames = artifact["frames"]
    cameras = dict(frames["cameras"])
    _require(len(cameras) >= 3, "at least three calibrated cameras are required")
    for camera_id, camera in cameras.items():
        _require(bool(camera_id), "camera id is empty")
        _validate_transform(
            camera["camera_to_world"], f"frames.cameras[{camera_id}].camera_to_world"
        )
        _validate_descriptor(
            camera["calibration_artifact"],
            f"frames.cameras[{camera_id}].calibration_artifact",
        )
    _validate_transform(frames["controller_to_world"], "frames.controller_to_world")
    _validate_transform(frames["support_to_world"], "frames.support_to_world")
    gravity = _vector(
        frames["gravity_direction_world"], 3, "frames.gravity_direction_world"
    )
    _require(
        np.isclose(np.linalg.norm(gravity), 1.0, atol=1e-6),
        "gravity direction must be unit length",
    )
    _require(
        set(frames["closure"]) == set(cameras), "frame closure must cover every camera"
    )
    for camera_id, closure in frames["closure"].items():
        for key in ("translation_error_m", "rotation_error_deg"):
            value = float(closure[key])
            _require(
                np.isfinite(value) and value >= 0.0,
                f"closure {camera_id} {key} is invalid",
            )

    support = artifact["support_geometry"]
    _require(
        isinstance(support.get("surface_id"), str) and support["surface_id"],
        "support id missing",
    )
    _require(
        support.get("kind") in {"plane", "mesh", "suspended"}, "support kind is invalid"
    )
    _vector(support["origin_world_m"], 3, "support origin")
    support_normal = _vector(support["normal_world"], 3, "support normal")
    _require(
        np.isclose(np.linalg.norm(support_normal), 1.0, atol=1e-6),
        "support normal must be unit length",
    )
    uncertainty = float(support["uncertainty_m"])
    _require(
        np.isfinite(uncertainty) and uncertainty >= 0.0,
        "support uncertainty is invalid",
    )
    _require(
        isinstance(support.get("contact_state_changes_recorded"), bool),
        "support contact-state recording flag is missing",
    )
    _validate_descriptor(support["artifact"], "support_geometry.artifact")

    expected_regions = {region["id"] for region in protocol["contact_regions"]}
    regions = dict(artifact["contact_regions"])
    _require(set(regions) == expected_regions, "contact region set changed")
    centroids = {}
    uncertainty_radius = {}
    for region_id, region in regions.items():
        centroid = _vector(
            region["physical_centroid_world_m"], 3, f"{region_id} centroid"
        )
        normal = _vector(region["physical_normal_world"], 3, f"{region_id} normal")
        tangent = np.asarray(region["tangent_basis_world"], dtype=float)
        _require(
            tangent.shape == (2, 3) and np.all(np.isfinite(tangent)),
            f"{region_id} tangent basis invalid",
        )
        _require(
            np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6)
            and np.allclose(tangent @ tangent.T, np.eye(2), atol=1e-6, rtol=1e-6)
            and np.allclose(tangent @ normal, 0.0, atol=1e-6, rtol=0.0),
            f"{region_id} contact frame is not orthonormal",
        )
        attachment = region["attachment"]
        _require(
            attachment.get("representation") == "weighted_node_patch",
            f"{region_id} must use a weighted node patch",
        )
        indices = np.asarray(attachment["node_indices"], dtype=int).reshape(-1)
        weights = np.asarray(attachment["weights"], dtype=float).reshape(-1)
        _require(
            len(indices) >= 2
            and len(indices) == len(weights)
            and len(np.unique(indices)) == len(indices)
            and np.all((0 <= indices) & (indices < node_count)),
            f"{region_id} node patch is invalid",
        )
        _require(
            np.all(np.isfinite(weights))
            and np.all(weights > 0.0)
            and np.isclose(np.sum(weights), 1.0, atol=1e-8),
            f"{region_id} attachment weights are invalid",
        )
        covariance = _covariance(
            attachment["centroid_covariance_world_m2"],
            3,
            f"{region_id} centroid covariance",
        )
        overlays = dict(region["per_view_overlays"])
        _require(
            set(overlays) == set(cameras),
            f"{region_id} overlays must cover every camera",
        )
        for camera_id, overlay in overlays.items():
            _vector(overlay["centroid_px"], 2, f"{region_id} {camera_id} centroid_px")
            _validate_descriptor(
                overlay["artifact"], f"{region_id} {camera_id} overlay"
            )
        reviews = list(region["independent_reviews"])
        _require(len(reviews) >= 2, f"{region_id} needs two independent reviews")
        for review in reviews:
            _require(
                isinstance(review.get("reviewer_id"), str)
                and review["reviewer_id"]
                and isinstance(review.get("reviewed_at_utc"), str)
                and review["reviewed_at_utc"],
                f"{region_id} review provenance is missing",
            )
            _vector(review["centroid_world_m"], 3, f"{region_id} review centroid")
        for key in ("interreview_rms_m", "multiview_reprojection_rmse_px"):
            value = float(region[key])
            _require(
                np.isfinite(value) and value >= 0.0, f"{region_id} {key} is invalid"
            )
        centroids[region_id] = centroid
        uncertainty_radius[region_id] = float(
            np.sqrt(np.max(np.linalg.eigvalsh(covariance)))
        )

    for region_id, centroid in centroids.items():
        separation = min(
            np.linalg.norm(centroid - other)
            for other_id, other in centroids.items()
            if other_id != region_id
        )
        _require(
            uncertainty_radius[region_id] < 0.5 * separation,
            f"{region_id} uncertainty does not separate contact regions",
        )

    acceptance = artifact["acceptance"]
    for key in (
        "multiview_agreement_passed",
        "independent_review_passed",
        "attachment_uncertainty_separates_regions",
        "frame_closure_recorded",
    ):
        _require(acceptance.get(key) is True, f"acceptance gate {key} failed")
    _require(
        acceptance.get("target_outcomes_used") is False,
        "target outcomes entered registration",
    )
    approval = artifact["approval"]
    _require(approval.get("approved") is True, "registration approval is missing")
    _require(
        isinstance(approval.get("approver_id"), str)
        and approval["approver_id"]
        and isinstance(approval.get("approved_at_utc"), str)
        and approval["approved_at_utc"],
        "approval provenance is missing",
    )
    checksums = artifact.get("source_checksums", {})
    _require(
        bool(checksums)
        and all(key and _is_sha256(value) for key, value in checksums.items()),
        "source checksums are missing or invalid",
    )
    return {
        "passed": True,
        "schema_version": 2,
        "camera_count": len(cameras),
        "contact_region_count": len(regions),
        "object_node_count": node_count,
        "approved": True,
    }


def write_contact_registration(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
