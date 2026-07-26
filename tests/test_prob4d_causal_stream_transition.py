from __future__ import annotations

import numpy as np

from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.prob4d_causal_lineage import (
    validate_prob4d_causal_observation_belief,
)


def test_prob4d_020_joint_artifact_without_explicit_version_is_inferred() -> None:
    belief = ObservationBeliefV1(
        case_id="case",
        stream_id="prob4d:causal-overlap-window-points",
        causal_frame_stop=2,
        view_names=("camera-0",),
        window_names=("window-0",),
        factor_names=(
            "joint_gauge_latent_0000",
            "joint_gauge_latent_0001",
        ),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="d" * 40,
        source_artifact_sha256="c" * 64,
        declared_frame_ids=np.asarray([1]),
        mean_xyz_m=np.asarray([[0.0, 0.0, 1.0]]),
        frame_ids=np.asarray([1]),
        entity_ids=np.asarray([0]),
        view_indices=np.asarray([0]),
        window_indices=np.asarray([0]),
        correlation_group_ids=np.asarray([0]),
        factor_group_ids=np.asarray([0]),
        prior_reliability=np.asarray([0.8]),
        association_probability=np.asarray([1.0]),
        local_covariance_m2=np.eye(3)[None] * 1e-5,
        low_rank_factor_m=np.zeros((1, 3, 2)),
        group_ids=np.asarray([0]),
        group_prior_nominal_probability=np.asarray([1.0]),
        group_composite_weight=np.asarray([1.0]),
        metadata={
            "metric_coordinates": True,
            "metric_units": "m",
            "coordinate_frame": "phystwin-world",
            "metric_gauge_anchor": {
                "schema_name": "prob4d.metric-gauge-anchor",
                "schema_version": 1,
                "artifact_id": "a" * 64,
                "window_id": "window-0",
                "coordinate_frame": "phystwin-world",
                "metric_units": "m",
                "source_kind": "prefix_registration",
                "source_artifact_sha256": "1" * 64,
                "covariance_treatment": "fixed_external_calibration",
            },
            "gauge_mode": "sequential",
            "joint_cross_window_gauge_covariance_represented": True,
            "gauge_posterior": {
                "model": "sequential_joint_spanning_tree_v1",
                "window_count": 1,
                "full_dimension": 7,
                "exported_factor_rank": 2,
                "retained_covariance_trace_fraction": 1.0,
                "minimum_retained_gauge_trace": 0.999,
                "cross_window_covariance_preserved": True,
                "fixed_lag_boundary_covariance_is_approximate": False,
                "parent_window_ids": [None],
            },
            "causal_source_lineage": {
                "schema_version": 1,
                "producer": "Prob4D",
                "motioncrafter_lineage_schema_version": 1,
                "motioncrafter_windowing_model": (
                    "motioncrafter_sliding_window_v1"
                ),
                "source_product": (
                    "independently_decoded_overlap_windows"
                ),
                "causal_frame_stop_exclusive": 2,
                "admissibility_rule": (
                    "source_frame_max < causal_frame_stop_exclusive"
                ),
                "future_prediction_payloads_opened": 0,
                "source_artifact_sha256": "c" * 64,
                "selected_windows": [
                    {
                        "window_id": "window-0",
                        "source_frame_start": 0,
                        "source_frame_stop_exclusive": 2,
                        "source_frame_max": 1,
                        "frame_indices_sha256": "2" * 64,
                        "payload_sha256": "1" * 64,
                    }
                ],
            },
        },
    )

    validation = validate_prob4d_causal_observation_belief(belief)

    assert validation["stream_contract_version"] == 2
    assert validation["stream_contract_version_inferred"] is True
    assert validation["gauge_covariance_semantics"] == (
        "joint_cross_window_sim3_gauge_covariance"
    )
