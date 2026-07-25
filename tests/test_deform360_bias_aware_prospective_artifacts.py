import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from bayesian_phystwin.deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
)
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    BACKBONE_SEAL_FILENAME,
    authorize_prospective_outcome_case,
    build_prospective_backbone_seal,
    build_prospective_calibration_support_rejection,
    build_prospective_prediction_cohort_seal,
    load_physical_archive,
    prospective_case_records,
    record_prospective_quality_failure,
    select_raw_backbone_arrays,
    source_reliability_and_variance,
    validate_prospective_backbone_seal,
    validate_prospective_calibration_support_rejection,
    validate_prospective_prediction_cohort_seal,
)
from bayesian_phystwin.deform360_bias_aware_prospective_staging import (
    select_action_only_window,
)
from bayesian_phystwin.deform360_bias_aware_prospective_physical import (
    ACTION_RESPONSE,
    FRAME_ZERO_PERSISTENCE_FALLBACK_SOURCE_CONFIG_SHA256,
    LENGTH_SCALE_M,
    build_persistence_backbone_arrays,
    build_prediction_only_bundle,
    build_warp_backbone_arrays,
    frame_zero_physical_policy,
    load_controller_trajectory,
)
from bayesian_phystwin.deform360_bias_aware_prospective_uncertainty import (
    inflate_covariance_from_cycle,
    jacobian_measurement_covariance,
    leave_one_camera_out_covariance,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "deform360_bias_aware_guarded_belief_prospective_v1.json"
)


def _physical_arrays(point_count: int = 20) -> dict[str, np.ndarray]:
    frame_zero = np.column_stack(
        (
            np.linspace(0.0, 0.1, point_count),
            np.zeros(point_count),
            np.ones(point_count),
        )
    ).astype(np.float32)
    persistence = np.repeat(frame_zero[None], 76, axis=0)
    response = np.linspace(0.0, 0.02, 76, dtype=np.float32)[:, None, None]
    driven = persistence + response * np.array([1.0, 0.0, 0.0], dtype=np.float32)
    zero = persistence.copy()
    prediction = persistence + 0.9 * response * np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    prediction[0] = frame_zero
    driven[0] = frame_zero
    return {
        "prediction_m": prediction,
        "persistence_m": persistence,
        "driven_readout_m": driven,
        "zero_action_readout_m": zero,
        "action_support": np.ones(point_count, dtype=np.float32),
        "frame_zero_points_m": frame_zero,
    }


def test_case_records_preserve_roles_and_locked_order() -> None:
    calibration = prospective_case_records(PROTOCOL, role="calibration")
    target = prospective_case_records(PROTOCOL, role="target")

    assert len(calibration) == 9
    assert len(target) == 24
    assert {row["role"] for row in calibration} == {"calibration"}
    assert {row["role"] for row in target} == {"target"}
    assert calibration[0]["case"] == "160-hose-ep0001"
    assert target[-1]["case"] == "164-sheep-ep0001"


def test_backbone_seal_round_trip_and_mutation_rejection(tmp_path: Path) -> None:
    physical = tmp_path / "input.npz"
    np.savez_compressed(physical, **_physical_arrays())
    physical_manifest = tmp_path / "physical.json"
    physical_manifest.write_text(
        json.dumps(
            {
                "information_boundary": {
                    "future_object_rgb_read": False,
                    "future_object_geometry_read": False,
                    "outcome_read": False,
                }
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "sealed"
    seal = build_prospective_backbone_seal(
        PROTOCOL,
        output,
        object_id="160-hose",
        episode_id=1,
        physical_archive=physical,
        physical_manifest=physical_manifest,
    )
    validate_prospective_backbone_seal(seal, protocol_path=PROTOCOL, case_dir=output)
    loaded = load_physical_archive(output / "physical_prediction.npz")
    assert np.array_equal(loaded["prediction_m"], _physical_arrays()["prediction_m"])

    archive = output / "physical_prediction.npz"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive checksum changed"):
        validate_prospective_backbone_seal(
            json.loads((output / BACKBONE_SEAL_FILENAME).read_text()),
            protocol_path=PROTOCOL,
            case_dir=output,
        )


def test_current_observation_selector_never_reads_a_target() -> None:
    arrays = _physical_arrays(point_count=6)
    physical = arrays["prediction_m"]
    persistence = arrays["persistence_m"]
    measurement = np.full_like(physical, np.nan)
    visibility = np.zeros(physical.shape[:2], dtype=bool)
    validity = np.zeros_like(visibility)
    centers = np.array([0, 1, 2, 3], dtype=np.int64)
    for frame in (19, 38, 57):
        measurement[frame, centers] = persistence[frame, centers]
        visibility[frame, centers] = True
        validity[frame, centers] = True

    report, selected = select_raw_backbone_arrays(
        physical,
        persistence,
        measurement,
        visibility,
        validity,
        center_ids=centers,
    )

    assert [row["selected_backbone"] for row in report["updates"]] == [
        "persistence",
        "persistence",
        "persistence",
    ]
    assert np.array_equal(selected[20:38], persistence[20:38])
    assert report["information_boundary"]["target_argument_accepted"] is False


def test_reliability_is_residual_independent_and_covariance_metric() -> None:
    config = Deform360BiasAwareDevelopmentConfig()
    centers = np.arange(4, dtype=np.int64)
    measurement = {
        "selected_cameras": np.asarray([f"cam{index}" for index in range(8)]),
        "triangulation_inlier_view_count": np.full((3, 4), 4),
        "triangulation_median_reprojection_px": np.full((3, 4), 1.0),
    }
    covariance = np.full((76, 4, 3, 3), np.nan)
    valid = np.zeros((76, 4), dtype=bool)
    for frame in config.update_frames:
        covariance[frame] = np.eye(3)[None] * 1.0e-6
        valid[frame] = True
    cycle = {
        "measurement_covariance_m2": covariance,
        "measurement_covariance_valid": valid,
    }

    reliability, variance = source_reliability_and_variance(
        measurement, cycle, center_ids=centers, config=config
    )

    expected_reliability = (4.0 - 1.0) / (8.0 - 1.0) * np.exp(-0.5 / 9.0)
    assert np.allclose(reliability, expected_reliability)
    assert np.all(variance == config.observation_variance_floor_m2)


def test_action_window_uses_only_robot_fields_and_earliest_tie() -> None:
    actions = np.zeros((100, 1, 4, 3), dtype=np.float64)
    openings = np.ones((100, 1), dtype=np.float64)

    result = select_action_only_window(actions, openings)

    assert result["selected_raw_frame_range_half_open"] == [8, 89]
    assert result["prediction_raw_frame_range_half_open"] == [8, 84]
    assert result["prefix_raw_frame_range_half_open"] == [8, 66]
    assert result["object_geometry_read"] is False
    assert result["tactile_read"] is False


def test_covariance_helpers_match_geometry_and_cycle_contract() -> None:
    point = np.array([0.0, 0.0, 1.0])
    first = np.array(
        [[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    second = first.copy()
    second[0, 3] = -10.0
    covariance, diagnostic = jacobian_measurement_covariance(
        point, [first, second], 0.5, maximum_condition_number=1.0e8
    )
    assert covariance is not None
    assert diagnostic["decision"] == "accepted"
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)

    observations = {
        "a": np.array([0.0, 0.0]),
        "b": np.array([-10.0, 0.0]),
    }
    jackknife, samples = leave_one_camera_out_covariance(
        observations, {"a": first, "b": second}
    )
    assert not np.any(jackknife)
    assert samples.shape == (0, 3)

    inflated, cycle = inflate_covariance_from_cycle(
        covariance,
        np.zeros((3, 3)),
        0.5,
        np.array([2.0, 2.5, 3.0]),
        pixel_noise_floor_px=0.5,
    )
    assert cycle["jacobian_covariance_scale"] > 1.0
    assert np.all(np.linalg.eigvalsh(inflated) > np.linalg.eigvalsh(covariance))


def _write_robot(path: Path, *, bimanual: bool) -> None:
    gripper_count = 2 if bimanual else 1
    poses = np.repeat(np.eye(4)[None, None], 76 * gripper_count, axis=0).reshape(
        76, gripper_count, 4, 4
    )
    poses[:, :, 0, 3] = np.arange(gripper_count)[None] * 0.2
    openings = np.full((76, gripper_count), 0.08)
    actions = np.zeros((76, gripper_count, 4, 3))
    if not bimanual:
        poses = poses[:, 0]
        openings = openings[:, 0]
        actions = actions[:, 0]
    np.savez_compressed(
        path,
        format_version=np.asarray(1),
        actions=actions,
        T_worlds=poses,
        openings=openings,
        bimanual=np.asarray(bimanual, dtype=np.bool_),
    )


def test_controller_taxel_cloud_and_prediction_bundle_are_prediction_only(
    tmp_path: Path,
) -> None:
    robot = tmp_path / "robot.npz"
    _write_robot(robot, bimanual=True)
    controllers, metadata = load_controller_trajectory(robot)
    assert controllers.shape == (76, 1536, 3)
    assert metadata["bimanual"] is True

    points = np.column_stack(
        (np.linspace(0.0, 0.2, 128), np.zeros(128), np.ones(128))
    ).astype(np.float32)
    colors = np.full_like(points, 0.5)
    geometry = tmp_path / "geometry.npz"
    np.savez_compressed(geometry, points_m=points, colors=colors)
    output = tmp_path / "prediction.pkl"
    summary = build_prediction_only_bundle(
        geometry,
        robot,
        output,
        object_id="160-hose",
        episode_id=1,
        case="160-hose-ep0001",
    )
    with output.open("rb") as stream:
        payload = pickle.load(stream)
    assert summary["frame_count"] == 76
    assert np.array_equal(payload["object_points"], np.repeat(points[None], 76, axis=0))
    assert payload["prediction_only_input"]["object_observation_frames_used"] == [0]
    assert (
        payload["prediction_only_input"]["future_object_observations_present"] is False
    )


def test_warp_backbone_uses_driven_minus_zero_graph_support() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    springs = np.array([[0, 1], [1, 2]], dtype=np.int64)
    rest_lengths = np.array([0.1, 0.1])
    weights = np.eye(3)
    zero = np.repeat(vertices[None], 76, axis=0)
    driven = zero.copy()
    driven[1:, :, 1] += 0.01

    arrays = build_warp_backbone_arrays(
        vertices,
        vertices=vertices,
        springs=springs,
        rest_lengths=rest_lengths,
        contact_anchor_indices=np.array([0]),
        readout_weights=weights,
        driven_vertices_m=driven,
        zero_action_vertices_m=zero,
    )

    expected_support = np.exp(-np.array([0.0, 0.1, 0.2]) / LENGTH_SCALE_M)
    assert np.allclose(arrays["action_support"], expected_support)
    assert np.allclose(
        arrays["prediction_m"][1, :, 1],
        ACTION_RESPONSE * expected_support * 0.01,
    )
    assert np.array_equal(arrays["prediction_m"][0], arrays["frame_zero_points_m"])


def test_inadmissible_twin_fallback_is_exact_persistence() -> None:
    points = np.column_stack(
        (np.linspace(0.0, 0.2, 128), np.zeros(128), np.ones(128))
    ).astype(np.float32)
    arrays = build_persistence_backbone_arrays(points)

    assert np.array_equal(arrays["prediction_m"], arrays["persistence_m"])
    assert np.array_equal(arrays["driven_readout_m"], arrays["persistence_m"])
    assert np.array_equal(arrays["zero_action_readout_m"], arrays["persistence_m"])
    assert np.count_nonzero(arrays["action_support"]) == 0


def test_frame_zero_policy_preserves_legacy_and_validates_opt_in() -> None:
    assert frame_zero_physical_policy({}) == "automatic_twin"
    assert (
        frame_zero_physical_policy(
            {
                "physical_policy": "persistence_only",
                "material_point_source": "strict-multiview-visual-hull-surface",
                "fallback_source_config_sha256": (
                    FRAME_ZERO_PERSISTENCE_FALLBACK_SOURCE_CONFIG_SHA256
                ),
            }
        )
        == "persistence_only"
    )
    with pytest.raises(ValueError, match="requires the frozen visual-hull"):
        frame_zero_physical_policy(
            {
                "physical_policy": "persistence_only",
                "material_point_source": "original-splat",
                "fallback_source_config_sha256": (
                    FRAME_ZERO_PERSISTENCE_FALLBACK_SOURCE_CONFIG_SHA256
                ),
            }
        )
    with pytest.raises(ValueError, match="policy changed"):
        frame_zero_physical_policy({"physical_policy": "warp_if_convenient"})
    with pytest.raises(ValueError, match="expected source-frozen config"):
        frame_zero_physical_policy(
            {
                "physical_policy": "persistence_only",
                "material_point_source": "strict-multiview-visual-hull-surface",
                "fallback_source_config_sha256": "wrong-source",
            }
        )


def test_prediction_cohort_seal_requires_every_locked_case(tmp_path: Path) -> None:
    cases = prospective_case_records(PROTOCOL, role="calibration")
    for case in cases:
        record_prospective_quality_failure(
            PROTOCOL,
            tmp_path / case["case"],
            object_id=case["object_id"],
            episode_id=case["episode_id"],
            stage="frame-zero-reconstruction",
            error_type="SyntheticQualityFailure",
            error_message="pre-outcome synthetic contract test",
        )

    seal = build_prospective_prediction_cohort_seal(
        PROTOCOL,
        "calibration",
        tmp_path,
        tmp_path / "calibration_prediction_cohort_seal.json",
    )

    assert seal["complete"] is True
    assert seal["prediction_count"] == 0
    assert seal["quality_failure_count"] == 9
    assert seal["replacement_count"] == 0
    validate_prospective_prediction_cohort_seal(
        seal,
        protocol_path=PROTOCOL,
        role="calibration",
        artifact_root=tmp_path,
    )
    with pytest.raises(ValueError, match="quality failure has no authorized future"):
        authorize_prospective_outcome_case(
            seal,
            protocol_path=PROTOCOL,
            role="calibration",
            artifact_root=tmp_path,
            object_id="160-hose",
            episode_id=1,
        )

    rejection = build_prospective_calibration_support_rejection(
        PROTOCOL,
        seal,
        tmp_path,
        tmp_path / "calibration_support_rejection.json",
    )
    assert rejection["decision_stage"] == "pre-outcome-support"
    assert rejection["evaluable_object_count"] == 0
    assert rejection["quality_failure_count"] == 9
    assert rejection["calibration_gate_passed"] is False
    assert rejection["target_access_authorized"] is False
    assert rejection["information_boundary"]["calibration_future_read"] is False
    validate_prospective_calibration_support_rejection(
        rejection,
        protocol_path=PROTOCOL,
        cohort_seal=seal,
        artifact_root=tmp_path,
    )
    forged = {**rejection, "target_access_authorized": True}
    with pytest.raises(ValueError, match="support rejection changed"):
        validate_prospective_calibration_support_rejection(
            forged,
            protocol_path=PROTOCOL,
            cohort_seal=seal,
            artifact_root=tmp_path,
        )
