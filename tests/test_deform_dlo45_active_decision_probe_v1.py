from __future__ import annotations

import numpy as np

from experiments.deform_dlo45_active_decision_probe_v1._core import (
    LocalSupport,
    PilotProtocol,
    assign_outcome,
    build_probe_bundle,
    probe_signature,
)
from experiments.deform_dlo45_decision_identifiability_v1._common import (
    Model,
    Protocol,
)


def passive_protocol() -> Protocol:
    return Protocol(
        prefix_frames=2,
        horizon_frames=4,
        stride_frames=4,
        action_scales=(0.0, 0.5, 1.0),
        neighbor_grid=(4,),
        cluster_grid=(2,),
        temperature_grid=(1.0,),
        regret_tolerance_grid=(0.05,),
        kmeans_iterations=20,
        source_fit_count=2,
        source_calibration_count=1,
        source_test_count=1,
        partition_domain="active-probe-test",
        source_gate_mean_ratio=1.2,
        source_gate_worst_trajectory_ratio=1.5,
        source_gate_minimum_nonfallback_fraction=0.0,
        bootstrap_replicates=20,
        bootstrap_seed=7,
    )


def pilot_protocol() -> PilotProtocol:
    return PilotProtocol(
        probe_frames=(0, 1, 2),
        outcome_count=2,
        cluster_count=2,
        neighbors=4,
        temperature_scale=1.0,
        regret_tolerance=0.05,
        target_support_multiplier=2.0,
        bootstrap_replicates=20,
        bootstrap_seed=7,
    )


def test_probe_signature_uses_mean_endpoint_and_rms() -> None:
    residual = np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3)
    signature = probe_signature(residual)

    assert signature.shape == (7,)
    np.testing.assert_allclose(signature[:3], np.mean(residual, axis=(0, 1)))
    np.testing.assert_allclose(signature[3:6], np.mean(residual[-1], axis=0))
    assert signature[6] == np.sqrt(np.mean(np.square(residual)))


def test_source_probe_outcomes_are_supported_and_keep_state_ambiguity() -> None:
    passive = passive_protocol()
    pilot = pilot_protocol()
    shaped = np.zeros((4, passive.horizon_frames, 2, 3), dtype=np.float64)
    shaped[:2, :2, :, 0] = -1.0
    shaped[2:, :2, :, 0] = 1.0
    shaped[:2, -1, :, 0] = -2.0
    shaped[2:, -1, :, 0] = 2.0
    residuals = shaped.reshape(4, -1)
    support = LocalSupport(
        selected=np.arange(4, dtype=np.int64),
        kernel_weights=np.full(4, 0.25),
        class_index=np.zeros(4, dtype=np.int64),
        quotient_weights=np.asarray([1.0]),
        jeffrey_weights=np.full(4, 0.25),
        residuals=residuals,
    )
    model = Model(
        features=np.zeros((4, 1)),
        residuals=residuals,
        class_labels=np.zeros(4, dtype=np.int64),
        feature_mean=np.zeros(1),
        feature_scale=np.ones(1),
        loss_floor=1e-12,
        neighbors=4,
        temperature_scale=1.0,
        regret_tolerance=0.05,
        action_scales=np.asarray([0.0, 0.5, 1.0]),
    )

    bundle = build_probe_bundle(support, model, passive, pilot, frames=2)

    assert bundle.candidate.outcome_likelihood.shape == (4, 2)
    for hypothesis in range(4):
        assignment = assign_outcome(shaped[hypothesis, :2], bundle, pilot)
        assert assignment.supported
        assert assignment.outcome == bundle.outcome_labels[hypothesis]
        assert assignment.compatible_hypothesis_count == 2


def test_zero_frame_bundle_is_a_no_probe_action() -> None:
    passive = passive_protocol()
    pilot = pilot_protocol()
    residuals = np.zeros((4, passive.horizon_frames * 2 * 3))
    support = LocalSupport(
        selected=np.arange(4, dtype=np.int64),
        kernel_weights=np.full(4, 0.25),
        class_index=np.zeros(4, dtype=np.int64),
        quotient_weights=np.asarray([1.0]),
        jeffrey_weights=np.full(4, 0.25),
        residuals=residuals,
    )
    model = Model(
        features=np.zeros((4, 1)),
        residuals=residuals,
        class_labels=np.zeros(4, dtype=np.int64),
        feature_mean=np.zeros(1),
        feature_scale=np.ones(1),
        loss_floor=1e-12,
        neighbors=4,
        temperature_scale=1.0,
        regret_tolerance=0.05,
        action_scales=np.asarray([0.0, 0.5, 1.0]),
    )

    bundle = build_probe_bundle(support, model, passive, pilot, frames=0)
    assignment = assign_outcome(np.zeros((0, 2, 3)), bundle, pilot)

    assert bundle.candidate.outcome_likelihood.shape == (4, 1)
    assert assignment.outcome == 0
    assert assignment.supported
    assert assignment.compatible_hypothesis_count == 4
