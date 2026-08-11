from __future__ import annotations

import hashlib
import json

from bayesian_phystwin.deform360_covariance_residual_history_v1 import (
    ResidualHistoryDryRunPolicyV1,
    camera_hardware_family,
    deterministic_disjoint_camera_partition,
)

SOURCE_GATE_RESULT_ID = (
    "f246394c84fd643b6ec8961dbcb2101a73c34e46d5eaf43961f28429aeb197eb"
)
SOURCE_GATE_ARTIFACT_SHA256 = (
    "7247a2a260509c4c226e7ca437aff09d090abf6d2ca08f471a2143ea7d4bf7de"
)
SOURCE_OBJECT_ID = "026-sock-cloth"
SOURCE_CAMERA_IDS = (
    "brics-odroid-001_cam0",
    "brics-odroid-001_cam1",
    "brics-odroid-006_cam0",
    "brics-odroid-007_cam0",
    "brics-odroid-007_cam1",
    "brics-odroid-008_cam0",
    "brics-odroid-008_cam1",
    "brics-odroid-009_cam0",
    "brics-odroid-009_cam1",
    "brics-odroid-010_cam0",
    "brics-odroid-010_cam1",
    "brics-odroid-011_cam0",
    "brics-odroid-012_cam0",
    "brics-odroid-012_cam1",
    "brics-odroid-013_cam0",
    "brics-odroid-013_cam1",
    "brics-odroid-014_cam1",
    "brics-odroid-015_cam0",
    "brics-odroid-015_cam1",
    "brics-odroid-016_cam0",
    "brics-odroid-017_cam0",
    "brics-odroid-017_cam1",
    "brics-odroid-019_cam1",
    "brics-odroid-021_cam0",
    "brics-odroid-021_cam1",
    "brics-odroid-022_cam0",
    "brics-odroid-022_cam1",
    "brics-odroid-023_cam0",
    "brics-odroid-024_cam0",
    "brics-odroid-024_cam1",
    "brics-odroid-025_cam0",
    "brics-odroid-025_cam1",
    "brics-odroid-027_cam0",
    "brics-odroid-027_cam1",
    "brics-odroid-028_cam0",
)
ROSTER_BINDING_SHA256 = (
    "6ea1db0ffe3f7be563cbb4e55b56bb8c627554f7a806ca053c903a6f1bf4c879"
)
EXPECTED_PROVIDER_FAMILIES = (
    "brics-odroid-001",
    "brics-odroid-006",
    "brics-odroid-012",
    "brics-odroid-015",
    "brics-odroid-016",
    "brics-odroid-017",
    "brics-odroid-019",
    "brics-odroid-022",
    "brics-odroid-023",
    "brics-odroid-025",
    "brics-odroid-027",
)
EXPECTED_SCORING_FAMILIES = (
    "brics-odroid-007",
    "brics-odroid-008",
    "brics-odroid-009",
    "brics-odroid-010",
    "brics-odroid-011",
    "brics-odroid-013",
    "brics-odroid-014",
    "brics-odroid-021",
    "brics-odroid-024",
    "brics-odroid-028",
)


def _roster_binding_sha256() -> str:
    payload = {
        "artifact_digest": SOURCE_GATE_ARTIFACT_SHA256,
        "camera_ids": list(SOURCE_CAMERA_IDS),
        "object_id": SOURCE_OBJECT_ID,
        "source_gate_result_id": SOURCE_GATE_RESULT_ID,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_real_deform360_camera_suffix_keeps_physical_recorder_serial() -> None:
    assert camera_hardware_family("brics-odroid-001_cam0") == ("brics-odroid-001")
    assert camera_hardware_family("brics-odroid-001_cam1") == ("brics-odroid-001")
    assert camera_hardware_family("brics-odroid-001-camera-3-left") == (
        "brics-odroid-001"
    )


def test_verified_026_source_roster_meets_the_locked_disjoint_role_gate() -> None:
    assert _roster_binding_sha256() == ROSTER_BINDING_SHA256
    assert len(SOURCE_CAMERA_IDS) == 35
    assert len({camera_hardware_family(value) for value in SOURCE_CAMERA_IDS}) == 21

    policy = ResidualHistoryDryRunPolicyV1(
        minimum_prefix_frames=2,
        minimum_final_observed_count=9,
        minimum_final_observed_fraction=0.5,
        minimum_cameras_per_role=8,
        minimum_camera_families_per_role=4,
        covariance_scales=(8.0, 16.0, 16.0),
    )
    partition = deterministic_disjoint_camera_partition(
        SOURCE_CAMERA_IDS,
        policy=policy,
    )

    assert partition.provider_family_ids == EXPECTED_PROVIDER_FAMILIES
    assert partition.scoring_family_ids == EXPECTED_SCORING_FAMILIES
    assert len(partition.provider_camera_ids) == 18
    assert len(partition.scoring_camera_ids) == 17
    assert set(partition.provider_camera_ids).isdisjoint(partition.scoring_camera_ids)
    assert set(partition.provider_camera_ids) | set(
        partition.scoring_camera_ids
    ) == set(SOURCE_CAMERA_IDS)
