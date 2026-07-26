from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.observation_belief_gauge_adapter import (
    build_gauge_aware_batch_from_observation_belief,
)
from bayesian_phystwin.prob4d_causal_lineage import (
    validate_prob4d_causal_observation_belief,
)


def _metadata() -> dict[str, object]:
    return {
        "metric_coordinates": True,
        "metric_units": "m",
        "coordinate_frame": "phystwin-world",
        "metric_gauge_anchor": {
            "artifact_id": "a" * 64,
            "window_id": "window-0",
            "world_frame_id": "phystwin-world",
            "source_artifact_sha256": "1" * 64,
            "calibration_artifact_sha256": "b" * 64,
            "covariance_treatment": "fixed_external_calibration",
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
            "causal_frame_stop_exclusive": 6,
            "admissibility_rule": (
                "source_frame_max < causal_frame_stop_exclusive"
            ),
            "future_prediction_payloads_opened": 0,
            "source_artifact_sha256": "c" * 64,
            "selected_windows": [
                {
                    "window_id": "window-0",
                    "source_frame_start": 0,
                    "source_frame_stop_exclusive": 3,
                    "source_frame_max": 2,
                    "frame_indices_sha256": "2" * 64,
                    "payload_sha256": "1" * 64,
                },
                {
                    "window_id": "window-1",
                    "source_frame_start": 2,
                    "source_frame_stop_exclusive": 5,
                    "source_frame_max": 4,
                    "frame_indices_sha256": "3" * 64,
                    "payload_sha256": "4" * 64,
                },
            ],
        },
    }


def _joint_metadata() -> dict[str, object]:
    metadata = deepcopy(_metadata())
    anchor = metadata["metric_gauge_anchor"]
    anchor.pop("world_frame_id")
    anchor.pop("calibration_artifact_sha256")
    anchor.update(
        {
            "schema_name": "prob4d.metric-gauge-anchor",
            "schema_version": 1,
            "coordinate_frame": "phystwin-world",
            "metric_units": "m",
            "source_kind": "prefix_registration",
        }
    )
    metadata.update(
        {
            "prob4d_causal_stream_contract_version": 2,
            "gauge_mode": "sequential",
            "joint_cross_window_gauge_covariance_represented": True,
            "gauge_posterior": {
                "model": "sequential_joint_spanning_tree_v1",
                "window_count": 2,
                "full_dimension": 14,
                "exported_factor_rank": 5,
                "retained_covariance_trace_fraction": 1.0,
                "minimum_retained_gauge_trace": 0.999,
                "cross_window_covariance_preserved": True,
                "fixed_lag_boundary_covariance_is_approximate": False,
                "parent_window_ids": [None, "window-0"],
            },
        }
    )
    return metadata


def _belief() -> ObservationBeliefV1:
    local = np.repeat(np.eye(3)[None], 4, axis=0) * 1e-5
    factors = np.zeros((4, 3, 7))
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    return ObservationBeliefV1(
        case_id="case",
        stream_id="prob4d:causal-overlap-window-points",
        causal_frame_stop=6,
        view_names=("camera-0",),
        window_names=("window-0", "window-1"),
        factor_names=tuple(
            f"gauge_latent_{index}" for index in range(7)
        ),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="d" * 40,
        source_artifact_sha256="c" * 64,
        declared_frame_ids=np.asarray([1, 2, 3, 4]),
        mean_xyz_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, 0.1, 1.0],
            ]
        ),
        frame_ids=np.asarray([1, 2, 3, 4]),
        entity_ids=np.asarray([0, 1, 0, 1]),
        view_indices=np.zeros(4, dtype=np.int64),
        window_indices=np.asarray([0, 0, 1, 1]),
        correlation_group_ids=np.asarray([0, 0, 1, 1]),
        factor_group_ids=np.asarray([0, 0, 1, 1]),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6]),
        association_probability=np.ones(4),
        local_covariance_m2=local,
        low_rank_factor_m=factors,
        group_ids=np.asarray([0, 1]),
        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata=_metadata(),
    )


def _joint_belief() -> ObservationBeliefV1:
    factors = np.zeros((4, 3, 5))
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    return replace(
        _belief(),
        factor_names=tuple(
            f"joint_gauge_latent_{index:04d}" for index in range(5)
        ),
        factor_group_ids=np.zeros(4, dtype=np.int64),
        low_rank_factor_m=factors,
        metadata=_joint_metadata(),
    )


def _adapt(belief: ObservationBeliefV1):
    state = np.zeros((belief.observation_count, 3, 1))
    state[:, 0, 0] = 1.0
    return build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=np.zeros_like(belief.mean_xyz_m),
        state_jacobian=state,
        query_state_jacobian=state[:1],
        physical_response_scale_m=0.05,
    )


def test_valid_prob4d_causal_lineage_is_bound_before_adaptation() -> None:
    belief = _belief()
    validation = validate_prob4d_causal_observation_belief(belief)
    adapted = _adapt(belief)

    assert validation["validated"] is True
    assert validation["window_count"] == 2
    assert adapted.summary()["prob4d_causal_lineage_validated"] is True
    assert adapted.batch.metadata["prob4d_causal_lineage"] == validation


def test_joint_prob4d_stream_contract_is_validated_before_adaptation() -> None:
    belief = _joint_belief()
    validation = validate_prob4d_causal_observation_belief(belief)
    adapted = _adapt(belief)

    assert validation["stream_contract_version"] == 2
    assert validation["gauge_covariance_semantics"] == (
        "joint_cross_window_sim3_gauge_covariance"
    )
    assert adapted.summary()["gauge_parameter_count"] == 5
    assert adapted.batch.metadata["prob4d_causal_lineage"] == validation


def test_joint_prob4d_stream_rejects_noncanonical_factor_names() -> None:
    belief = _joint_belief()
    names = list(belief.factor_names)
    names[-1] = "joint_gauge_latent_wrong"

    with pytest.raises(ValueError, match="factor names are not canonical"):
        _adapt(replace(belief, factor_names=tuple(names)))


def test_joint_prob4d_stream_rejects_approximate_fixed_lag_covariance() -> None:
    belief = _joint_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["gauge_posterior"][
        "fixed_lag_boundary_covariance_is_approximate"
    ] = True

    with pytest.raises(ValueError, match="rejects approximate fixed-lag"):
        _adapt(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_changed_cutoff() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["causal_source_lineage"][
        "causal_frame_stop_exclusive"
    ] = 7

    with pytest.raises(ValueError, match="cutoff differs"):
        _adapt(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_future_payload_access() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["causal_source_lineage"][
        "future_prediction_payloads_opened"
    ] = 1

    with pytest.raises(ValueError, match="opening future payloads"):
        _adapt(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_window_mismatch() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["causal_source_lineage"]["selected_windows"][1][
        "window_id"
    ] = "another-window"

    with pytest.raises(ValueError, match="window order differs"):
        _adapt(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_uncertain_anchor_claim() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["metric_gauge_anchor"]["covariance_treatment"] = (
        "marginalized_global_anchor"
    )

    with pytest.raises(ValueError, match="requires a fixed metric anchor"):
        _adapt(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_source_digest_mismatch() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["causal_source_lineage"][
        "source_artifact_sha256"
    ] = "e" * 64

    with pytest.raises(ValueError, match="differs from the descriptor"):
        _adapt(replace(belief, metadata=metadata))
