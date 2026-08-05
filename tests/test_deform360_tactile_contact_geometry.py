from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_tactile_contact_geometry import (
    build_assignment_mixture_geometry,
    evaluate_tactile_contact_geometry_quality,
    extract_active_tactile_rows,
    load_deform360_tactile_contact_geometry_lock,
    parse_tactile_sensor_name,
    validate_deform360_tactile_contact_geometry_lock,
    verify_tactile_contact_geometry_artifact,
    write_tactile_contact_geometry_artifact,
)


def _lock() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "bayesian-phystwin.deform360-tactile-contact-geometry-lock",
        "schema_version": 1,
        "status": "locked-source-only-pre-geometry",
        "source": {
            "object_id": "026-sock-cloth",
            "bimanual": True,
            "robot_prefix_artifact_id": "1" * 64,
            "robot_prefix_manifest_sha256": "2" * 64,
            "robot_prefix_archive_sha256": "3" * 64,
            "robot_prefix_anchor_authorized": True,
            "contact_frame_start": 144,
            "causal_frame_stop": 150,
            "tactile_files": {
                sensor: {
                    "relative_path": f"{sensor}/synced_tactile.npy",
                    "sha256": str(index) * 64,
                }
                for index, sensor in enumerate(
                    (
                        "brics-odroid_tactilel_left",
                        "brics-odroid_tactilel_right",
                        "brics-odroid_tactiler_left",
                        "brics-odroid_tactiler_right",
                    ),
                    start=4,
                )
            },
        },
        "geometry": {
            "processing_revision": "d8522a4403b766aeb387510c04e89032a56fdf35",
            "implementation_revision": "2" * 40,
            "active_threshold": 0.0,
            "taxel_rows_used": 12,
            "taxel_columns": 32,
            "assignments": [
                {"name": "direct", "tactilel_gripper": 0, "tactiler_gripper": 1},
                {"name": "swapped", "tactilel_gripper": 1, "tactiler_gripper": 0},
            ],
            "assignment_prior_probability": [0.5, 0.5],
        },
        "quality_gate": {
            "minimum_active_frames": 3,
            "minimum_active_taxels": 6,
            "minimum_assignment_separation_m": 0.05,
        },
        "information_boundary": {
            "calibration_scores_opened": False,
            "confirmation_payloads_opened": False,
            "future_tactile_values_used": False,
            "held_v8_accessed": False,
            "metric_covariance_calibrated": False,
            "object_association_fitted": False,
            "target_outcomes_used": False,
        },
    }
    value["artifact_id"] = content_id(value)
    return value


def _streams(*, future_value: float = 0.0) -> dict[str, np.ndarray]:
    names = (
        "brics-odroid_tactilel_left",
        "brics-odroid_tactilel_right",
        "brics-odroid_tactiler_left",
        "brics-odroid_tactiler_right",
    )
    result = {name: np.zeros((10, 16, 32), dtype=np.float32) for name in names}
    for frame in (4, 5, 6):
        result["brics-odroid_tactiler_right"][frame, 1, frame] = 0.5
        result["brics-odroid_tactiler_right"][frame, 2, frame] = 0.4
    for values in result.values():
        values[7:] = future_value
    return result


def _geometry() -> dict[str, np.ndarray]:
    active = extract_active_tactile_rows(
        _streams(), frame_start=4, frame_stop=7, active_threshold=0.0
    )
    frame_ids = np.arange(4, 7)
    transforms = np.tile(np.eye(4), (3, 2, 1, 1))
    transforms[:, 1, 0, 3] = 0.2
    openings = np.full((3, 2), 0.08)

    def points(_opening: float, transform: np.ndarray) -> np.ndarray:
        result = np.zeros((768, 3))
        result[:, 0] = transform[0, 3]
        result[:, 1] = np.arange(768)
        return result

    return build_assignment_mixture_geometry(
        active,
        robot_source_frame_ids=frame_ids,
        robot_transforms=transforms,
        robot_openings_m=openings,
        taxel_points=points,
    )


def test_sensor_names_do_not_imply_gripper_indices() -> None:
    assert parse_tactile_sensor_name("brics-odroid_tactilel_left") == (0, 0)
    assert parse_tactile_sensor_name("brics-odroid_tactilel_right") == (0, 1)
    assert parse_tactile_sensor_name("brics-odroid_tactiler_left") == (1, 0)
    assert parse_tactile_sensor_name("brics-odroid_tactiler_right") == (1, 1)
    with pytest.raises(ValueError, match="unsupported tactile sensor"):
        parse_tactile_sensor_name("brics-odroid_tactile0_left")


def test_lock_preserves_equal_assignment_mixture_and_fails_closed() -> None:
    lock = _lock()
    assert validate_deform360_tactile_contact_geometry_lock(lock) == lock["artifact_id"]

    changed = copy.deepcopy(lock)
    changed["geometry"]["assignment_prior_probability"] = [1.0, 0.0]
    descriptor = dict(changed)
    descriptor.pop("artifact_id")
    changed["artifact_id"] = content_id(descriptor)
    with pytest.raises(ValueError, match="assignment prior changed"):
        validate_deform360_tactile_contact_geometry_lock(changed)


def test_active_extraction_ignores_future_tactile_mutation() -> None:
    first = extract_active_tactile_rows(
        _streams(future_value=0.0), frame_start=4, frame_stop=7
    )
    second = extract_active_tactile_rows(
        _streams(future_value=99.0), frame_start=4, frame_stop=7
    )

    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert len(first["source_frame_ids"]) == 6


def test_assignment_geometry_retains_both_distinct_gripper_hypotheses() -> None:
    geometry = _geometry()

    np.testing.assert_array_equal(
        geometry["assignment_prior_probability"], [0.5, 0.5]
    )
    np.testing.assert_array_equal(
        geometry["gripper_indices_hypotheses"],
        np.tile([1, 0], (6, 1)),
    )
    np.testing.assert_allclose(geometry["world_points_hypotheses_m"][:, 0, 0], 0.2)
    np.testing.assert_allclose(geometry["world_points_hypotheses_m"][:, 1, 0], 0.0)


def test_geometry_quality_requires_repeated_distinct_contact() -> None:
    geometry = _geometry()
    quality = evaluate_tactile_contact_geometry_quality(
        geometry, quality_gate=_lock()["quality_gate"]
    )

    assert quality.admitted
    assert quality.reason_codes == ()
    assert quality.summary["active_frame_count"] == 3
    assert quality.summary["active_taxel_count"] == 6
    assert quality.summary["median_assignment_separation_m"] == pytest.approx(0.2)


def test_geometry_quality_rejects_collapsed_assignment() -> None:
    geometry = _geometry()
    geometry["world_points_hypotheses_m"][:, 1] = geometry[
        "world_points_hypotheses_m"
    ][:, 0]

    quality = evaluate_tactile_contact_geometry_quality(
        geometry, quality_gate=_lock()["quality_gate"]
    )

    assert not quality.admitted
    assert quality.reason_codes == ("assignment-hypotheses-not-distinct",)


def test_geometry_artifact_stays_unauthorized_as_measurement(tmp_path: Path) -> None:
    geometry = _geometry()
    lock = _lock()
    quality = evaluate_tactile_contact_geometry_quality(
        geometry, quality_gate=lock["quality_gate"]
    )
    manifest_path = tmp_path / "geometry.json"
    manifest = write_tactile_contact_geometry_artifact(
        arrays=geometry,
        quality=quality,
        lock=lock,
        output_npz=tmp_path / "geometry.npz",
        output_manifest=manifest_path,
        implementation_revision="3" * 40,
        source_artifacts={"robot-prefix.json": "4" * 64},
    )

    assert manifest["quality"]["admitted"] is True
    assert manifest["assignment_marginalized"] is True
    assert manifest["metric_covariance_calibrated"] is False
    assert manifest["object_association_fitted"] is False
    assert manifest["contact_anchor_authorized"] is False
    assert verify_tactile_contact_geometry_artifact(manifest_path)["artifact_id"] == (
        manifest["artifact_id"]
    )


def test_lock_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    write_atomic_json(_lock(), path, overwrite=False)
    assert load_deform360_tactile_contact_geometry_lock(path)["source"][
        "object_id"
    ] == "026-sock-cloth"


def test_committed_geometry_smoke_lock_remains_valid() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "protocols/locks/"
        "deform360_official_hub_tactile_contact_geometry_smoke_v1.json"
    )

    lock = load_deform360_tactile_contact_geometry_lock(path)

    assert lock["artifact_id"] == (
        "9f3fb26568d4bf9269ad35ce792ebd8739cd397d82f806b63b265f54f42879f9"
    )
    assert lock["claim_boundary"]["contact_anchor_authorized"] is False
