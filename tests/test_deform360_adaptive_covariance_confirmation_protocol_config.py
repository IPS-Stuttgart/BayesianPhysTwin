from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from bayesian_phystwin.deform360_adaptive_covariance_confirmation_evaluation import (
    BOOTSTRAP_DOMAIN,
    BOOTSTRAP_REPLICATE_COUNT,
    BOOTSTRAP_UPPER_INDEX,
    BOOTSTRAP_UPPER_QUANTILE,
    HARMFUL_OBJECT_RATIO,
    MAXIMUM_HARMFUL_OBJECT_COUNT,
    MAXIMUM_MEAN_CHARGED_CAMERAS,
    MAXIMUM_PHYSICAL_FALLBACK_COUNT,
    MAXIMUM_RETAINED_TECHNICAL_FAILURE_CASE_COUNT,
    MAXIMUM_SEVERE_CASE_COUNT,
    METRICS,
    MINIMUM_JOINT_SIGN_SUCCESSES,
    NONINFERIORITY_RATIO_MARGIN,
    SEVERE_CASE_RATIO,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime import (
    COHORT_LOCK_REPOSITORY_PATH,
    DEFORM360_EXECUTION_COMMIT,
    EXTERNAL_EXECUTION_COMMIT,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
    DATASET_REVISION,
    EXCLUSION_UNION_SHA256,
    OBJECT_QUOTAS,
    PROTOCOL_ID,
)
from bayesian_phystwin.deform360_adaptive_covariance_rbf import (
    FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
)
from bayesian_phystwin.deform360_held_online_prefix import (
    FRAME_COUNT,
    HELD_RBF_CONFIG,
    UPDATE_FRAMES,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_MOLMOMOTION_REVISION,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    ALLTRACKER_SOURCE_TREE,
    RawCameraObservationConfig,
)
from bayesian_phystwin.deform360_raw_camera_uncertainty import (
    RawCameraUncertaintyConfig,
)


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "deform360_adaptive_covariance_confirmation_v1.json"
)


def test_h1_protocol_config_is_canonical_and_matches_compiled_contract() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = payload["config"]
    assert (
        payload["config_sha256"]
        == hashlib.sha256(
            json.dumps(
                config,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert config["protocol_id"] == PROTOCOL_ID
    assert config["dataset"]["revision"] == DATASET_REVISION
    assert config["dataset"]["deform360_code_revision"] == DEFORM360_EXECUTION_COMMIT
    assert config["physical_backbone"]["execution_commit"] == EXTERNAL_EXECUTION_COMMIT
    assert config["cohort"]["excluded_identity_union_sha256"] == (
        EXCLUSION_UNION_SHA256
    )
    assert config["cohort"]["object_quotas"] == dict(OBJECT_QUOTAS)
    assert (
        config["two_commit_freeze"]["h2"].split("only ", 1)[1].rstrip(".")
        == COHORT_LOCK_REPOSITORY_PATH
    )

    observation = config["observation"]
    expected_observation = asdict(RawCameraObservationConfig())
    assert observation["frame_count"] == FRAME_COUNT
    assert tuple(observation["update_frames"]) == tuple(UPDATE_FRAMES)
    assert observation["center_count"] == expected_observation["center_count"]
    assert observation["triangulation"] == {
        "minimum_initial_view_count": expected_observation[
            "minimum_initial_view_count"
        ],
        "minimum_triangulation_view_count": expected_observation[
            "minimum_triangulation_view_count"
        ],
        "minimum_ray_angle_degrees": expected_observation["minimum_ray_angle_degrees"],
        "frame_zero_depth_tolerance_m": expected_observation[
            "frame_zero_depth_tolerance_m"
        ],
        "reprojection_inlier_threshold_px": expected_observation[
            "reprojection_inlier_threshold_px"
        ],
        "maximum_reprojection_median_px": expected_observation[
            "maximum_reprojection_median_px"
        ],
        "maximum_displacement_from_initial_m": expected_observation[
            "maximum_displacement_from_initial_m"
        ],
    }
    assert observation["alltracker"] == {
        "repository_revision": ALLTRACKER_MOLMOMOTION_REVISION,
        "source_tree": ALLTRACKER_SOURCE_TREE,
        "runtime_source_sha256": ALLTRACKER_RUNTIME_SOURCE_SHA256,
        "checkpoint_sha256": ALLTRACKER_CHECKPOINT_SHA256,
        "maximum_side": expected_observation["alltracker_max_side"],
        "inference_iterations": expected_observation["alltracker_inference_iterations"],
        "window_length": expected_observation["alltracker_window_length"],
        "visibility_threshold": expected_observation["visibility_threshold"],
    }
    expected_uncertainty = asdict(RawCameraUncertaintyConfig())
    assert all(
        observation["uncertainty"][key] == value
        for key, value in expected_uncertainty.items()
    )
    routing = config["predictor"]["routing"]
    assert routing["route_order"] == [
        "4_view_rbf",
        "8_view_rbf",
        "physical_prior_fallback",
    ]
    assert set(
        config["development_evidence"]["adaptive_routes_over_81_updates"]
    ) == set(routing["route_order"])
    assert all(
        routing[key] == value
        for key, value in asdict(FROZEN_ADAPTIVE_COVARIANCE_CONFIG).items()
        if key != "camera_budgets"
    )
    assert config["observation"]["camera_budgets"] == list(
        FROZEN_ADAPTIVE_COVARIANCE_CONFIG.camera_budgets
    )
    assert config["predictor"]["recursive_rbf"] == asdict(HELD_RBF_CONFIG)

    primary = config["evaluation"]["primary_pass_requires_all"]
    assert primary["both_point_estimate_ratios_below"] == (NONINFERIORITY_RATIO_MARGIN)
    assert primary["both_one_sided_bootstrap_upper_bounds_below"] == (
        NONINFERIORITY_RATIO_MARGIN
    )
    assert primary["joint_object_noninferiority_successes_at_least"] == (
        MINIMUM_JOINT_SIGN_SUCCESSES
    )
    assert (
        "less than or equal to 1.05"
        in primary["joint_object_noninferiority_tie_convention"]
    )
    assert primary["object_balanced_mean_charged_cameras_at_most"] == (
        MAXIMUM_MEAN_CHARGED_CAMERAS
    )
    assert primary[
        "fallback_route_updates_including_retained_technical_failures_at_most"
    ] == (MAXIMUM_PHYSICAL_FALLBACK_COUNT)
    assert primary["retained_technical_failure_cases_at_most"] == (
        MAXIMUM_RETAINED_TECHNICAL_FAILURE_CASE_COUNT
    )
    assert MAXIMUM_RETAINED_TECHNICAL_FAILURE_CASE_COUNT == 0
    assert primary["objects_harmful_over_1.10_on_either_metric_at_most"] == (
        MAXIMUM_HARMFUL_OBJECT_COUNT
    )
    assert primary["cases_severe_over_1.25_on_either_metric_at_most"] == (
        MAXIMUM_SEVERE_CASE_COUNT
    )
    evaluation = config["evaluation"]
    assert evaluation["metrics"] == list(METRICS)
    assert evaluation["scored_frame_intervals_half_open"] == [
        [20, 38],
        [39, 57],
        [58, 76],
    ]
    assert evaluation["assimilation_centers_permanently_excluded_from_scores"] is True
    assert "operator-declared" in evaluation["retained_failure_evidence_policy"]
    bootstrap = evaluation["bootstrap"]
    assert bootstrap["replicates"] == BOOTSTRAP_REPLICATE_COUNT
    assert bootstrap["resampling_unit"] == "physical object"
    assert bootstrap["upper_quantile"] == BOOTSTRAP_UPPER_QUANTILE
    assert bootstrap["upper_order_statistic_zero_based_index"] == (
        BOOTSTRAP_UPPER_INDEX
    )
    assert bootstrap["upper_quantile_rule"] == (
        "ceil(0.95 * 200000) - 1 after ascending sort"
    )
    assert "joined in that order by single NUL bytes" in bootstrap["seed"]
    assert bootstrap["index_generator"] == (
        f"SHA256 rejection sampling with domain {BOOTSTRAP_DOMAIN.decode('ascii')}"
    )
    assert HARMFUL_OBJECT_RATIO == 1.10
    assert SEVERE_CASE_RATIO == 1.25
    secondary = config["evaluation"][
        "secondary_adaptive_superiority_over_fixed_four_requires_all"
    ]
    assert secondary == {
        "primary_confirmation_passed": True,
        "both_point_estimate_ratios_below": 1.0,
        "both_one_sided_bootstrap_upper_bounds_below": 1.0,
        "joint_object_strict_improvements_at_least": MINIMUM_JOINT_SIGN_SUCCESSES,
    }
