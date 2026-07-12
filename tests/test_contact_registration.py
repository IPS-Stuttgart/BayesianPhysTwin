from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from causal4d.contact_registration import (
    build_contact_registration_template,
    validate_contact_registration,
)
from causal4d.real_protocol import build_same_object_real_protocol


def _descriptor(name: str) -> dict[str, object]:
    return {"path": f"artifacts/{name}.json", "sha256": "a" * 64, "bytes": 10}


def _transform() -> dict[str, object]:
    return {"matrix": np.eye(4).tolist(), "covariance_se3": (np.eye(6) * 1e-6).tolist()}


def _approved_artifact() -> tuple[dict, dict]:
    protocol = build_same_object_real_protocol()
    artifact = build_contact_registration_template(
        protocol,
        camera_ids=["camera_0", "camera_1", "camera_2"],
        object_node_count=100,
    )
    artifact["status"] = "approved"
    artifact["object"].update(
        {
            "physical_instance_serial": "sloth-physical-001",
            "twin_geometry_sha256": "b" * 64,
            "geometry_artifact": _descriptor("geometry"),
        }
    )
    for camera_id, camera in artifact["frames"]["cameras"].items():
        camera["camera_to_world"] = _transform()
        camera["calibration_artifact"] = _descriptor(f"calibration-{camera_id}")
        artifact["frames"]["closure"][camera_id] = {
            "translation_error_m": 0.001,
            "rotation_error_deg": 0.1,
        }
    artifact["frames"]["controller_to_world"] = _transform()
    artifact["frames"]["support_to_world"] = _transform()
    artifact["frames"]["gravity_direction_world"] = [0.0, 0.0, -1.0]
    artifact["support_geometry"].update(
        {
            "surface_id": "support-plane-1",
            "kind": "plane",
            "origin_world_m": [0.0, 0.0, 0.0],
            "normal_world": [0.0, 0.0, 1.0],
            "uncertainty_m": 0.001,
            "contact_state_changes_recorded": True,
            "artifact": _descriptor("support"),
        }
    )
    centroids = {
        "left_forepaw": [-0.10, 0.0, 0.0],
        "right_forepaw": [0.10, 0.0, 0.0],
        "upper_torso": [0.0, 0.12, 0.0],
    }
    node_offsets = {"left_forepaw": 0, "right_forepaw": 10, "upper_torso": 20}
    for region_id, region in artifact["contact_regions"].items():
        region.update(
            {
                "physical_centroid_world_m": centroids[region_id],
                "physical_normal_world": [0.0, 0.0, 1.0],
                "tangent_basis_world": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "attachment": {
                    "representation": "weighted_node_patch",
                    "node_indices": [
                        node_offsets[region_id],
                        node_offsets[region_id] + 1,
                    ],
                    "weights": [0.6, 0.4],
                    "centroid_covariance_world_m2": (np.eye(3) * 1e-6).tolist(),
                },
                "independent_reviews": [
                    {
                        "reviewer_id": "reviewer-a",
                        "reviewed_at_utc": "2026-07-12T12:00:00Z",
                        "centroid_world_m": centroids[region_id],
                    },
                    {
                        "reviewer_id": "reviewer-b",
                        "reviewed_at_utc": "2026-07-12T12:05:00Z",
                        "centroid_world_m": centroids[region_id],
                    },
                ],
                "interreview_rms_m": 0.001,
                "multiview_reprojection_rmse_px": 0.5,
            }
        )
        for camera_id, overlay in region["per_view_overlays"].items():
            overlay["centroid_px"] = [100.0, 120.0]
            overlay["artifact"] = _descriptor(f"overlay-{region_id}-{camera_id}")
    artifact["acceptance"] = {
        "multiview_agreement_passed": True,
        "independent_review_passed": True,
        "attachment_uncertainty_separates_regions": True,
        "frame_closure_recorded": True,
        "target_outcomes_used": False,
    }
    artifact["approval"] = {
        "approved": True,
        "approver_id": "principal-investigator",
        "approved_at_utc": "2026-07-12T12:10:00Z",
    }
    artifact["source_checksums"] = {"calibration_bundle": "c" * 64}
    return protocol, artifact


def test_contact_registration_accepts_weighted_multiview_patch() -> None:
    protocol, artifact = _approved_artifact()
    result = validate_contact_registration(artifact, protocol)
    assert result["passed"] is True
    assert result["camera_count"] == 3
    assert result["contact_region_count"] == 3


def test_contact_registration_rejects_exact_node_or_bad_weights() -> None:
    protocol, artifact = _approved_artifact()
    mutated = deepcopy(artifact)
    mutated["contact_regions"]["left_forepaw"]["attachment"]["node_indices"] = [0]
    mutated["contact_regions"]["left_forepaw"]["attachment"]["weights"] = [1.0]
    with pytest.raises(ValueError, match="node patch is invalid"):
        validate_contact_registration(mutated, protocol)
