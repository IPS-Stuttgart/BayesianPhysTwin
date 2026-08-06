from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import bayesian_phystwin.prob4d_visual_bias_update as update_module
from bayesian_phystwin._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
)
from bayesian_phystwin._prob4d_stream_binding import (
    prob4d_observation_identity_summary,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.prob4d_visual_bias_update import (
    PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
    ClaimBearingProb4DVisualBiasUpdateV2,
    Prob4DVisualBiasBindingV1,
    update_claim_bearing_prob4d_with_visual_bias_from_artifacts,
    validate_prob4d_visual_bias_nuisance,
)

LINEARIZATION_ID = "b" * 64
PROVIDER_ID = "c" * 64
CALIBRATION_IDS = {"gauge": "d" * 64, "point": "e" * 64}


def _observation() -> ObservationBeliefV1:
    return ObservationBeliefV1(
        case_id="case-a",
        stream_id="stream-a",
        causal_frame_stop=2,
        view_names=("camera-0",),
        window_names=("window-0",),
        factor_names=(),
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="f" * 64,
        declared_frame_ids=np.asarray([0, 1], dtype=np.int64),
        mean_xyz_m=np.asarray(
            [[0.01, 0.00, 0.00], [0.00, 0.02, 0.00]],
            dtype=np.float64,
        ),
        frame_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=np.asarray([7, 7], dtype=np.int64),
        view_indices=np.asarray([0, 0], dtype=np.int64),
        window_indices=np.asarray([0, 0], dtype=np.int64),
        correlation_group_ids=np.asarray([0, 0], dtype=np.int64),
        factor_group_ids=np.asarray([0, 0], dtype=np.int64),
        prior_reliability=np.ones(2, dtype=np.float64),
        association_probability=np.ones(2, dtype=np.float64),
        local_covariance_m2=np.repeat(
            (1e-4 * np.eye(3, dtype=np.float64))[None, :, :],
            2,
            axis=0,
        ),
        low_rank_factor_m=np.zeros((2, 3, 0), dtype=np.float64),
        group_ids=np.asarray([0], dtype=np.int64),
        group_prior_nominal_probability=np.asarray([0.95], dtype=np.float64),
        group_composite_weight=np.asarray([1.0], dtype=np.float64),
        metadata={},
    )


def _binding(
    observation: ObservationBeliefV1,
    *,
    orthogonalized: bool = True,
) -> Prob4DVisualBiasBindingV1:
    _, _, identity_sha = prob4d_observation_identity_summary(observation)
    return Prob4DVisualBiasBindingV1(
        observation_artifact_id=observation.artifact_id,
        observation_identity_sha256=identity_sha,
        bias_ids=("camera-0", "camera-1"),
        basis_names=("ray-depth",),
        row_bias_indices=np.asarray([0, 1], dtype=np.int64),
        bias_jacobian=np.asarray(
            [
                [[1.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0]],
            ],
            dtype=np.float64,
        ),
        joint_bias_covariance=np.asarray(
            [[4e-6, 1e-6], [1e-6, 9e-6]],
            dtype=np.float64,
        ),
        orthogonalization_semantics=(
            PROB4D_VISUAL_BIAS_ORTHOGONALIZATION
            if orthogonalized
            else "not-orthogonalized"
        ),
        maximum_gauge_projection=0.0,
        gauge_projection_tolerance=1e-8,
        metadata={"uses_truth": False},
    )


def _claim_lineage(observation: ObservationBeliefV1) -> dict[str, object]:
    return {
        "observation_artifact_id": observation.artifact_id,
        "linearization_artifact_id": LINEARIZATION_ID,
        "prob4d_claim_bearing_provider_manifest_id": PROVIDER_ID,
        "prob4d_claim_bearing_calibration_artifact_ids": CALIBRATION_IDS,
        "prob4d_claim_bearing_runtime_revision_source": "independent-vcs-check",
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
    }


def _adapted_batch(
    observation: ObservationBeliefV1,
    shared_design: np.ndarray,
    view_design: np.ndarray,
    *,
    gauge_design: np.ndarray | None = None,
) -> SimpleNamespace:
    gauge = (
        np.zeros((observation.observation_count, 3, 0), dtype=np.float64)
        if gauge_design is None
        else np.asarray(gauge_design, dtype=np.float64)
    )
    batch = GaugeAwareObservationBatch(
        innovation_m=np.zeros((2, 3), dtype=np.float64),
        observation_covariance_m2=observation.local_covariance_m2,
        state_jacobian=np.asarray(
            [
                [[1.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0]],
            ],
            dtype=np.float64,
        ),
        gauge_jacobian=gauge,
        shared_bias_jacobian=shared_design,
        view_bias_jacobian=view_design,
        query_state_jacobian=np.asarray(
            [[[1.0], [0.0], [0.0]]],
            dtype=np.float64,
        ),
        gauge_prior_covariance=np.eye(gauge.shape[2], dtype=np.float64),
        correlation_group_ids=("group-0", "group-0"),
        prior_reliability=np.ones(2, dtype=np.float64),
        prior_nominal_probability=np.full(2, 0.95, dtype=np.float64),
        composite_weight=np.ones(2, dtype=np.float64),
        physical_response_scale_m=0.01,
        metadata=_claim_lineage(observation),
    )
    return SimpleNamespace(
        batch=batch,
        observation_artifact_id=observation.artifact_id,
    )


def _solver_result(batch: GaugeAwareObservationBatch) -> GaugeAwareBeliefResult:
    shared_count = batch.shared_bias_jacobian.shape[2]
    dimension = 1 + batch.gauge_jacobian.shape[2] + shared_count
    covariance = np.eye(dimension, dtype=np.float64)
    covariance[0, 0] = 0.5
    if shared_count:
        start = 1 + batch.gauge_jacobian.shape[2]
        covariance[start:, start:] = np.diag(
            np.linspace(0.6, 0.7, shared_count)
        )
    return GaugeAwareBeliefResult(
        inference_admissible=True,
        reason="accepted",
        state_coefficients=np.asarray([0.2], dtype=np.float64),
        gauge_delta=np.zeros(batch.gauge_jacobian.shape[2], dtype=np.float64),
        shared_bias_coefficients=np.linspace(
            0.3,
            -0.4,
            shared_count,
            dtype=np.float64,
        ),
        view_bias_coefficients=np.zeros(0, dtype=np.float64),
        anchor_bias_coefficients=np.zeros(0, dtype=np.float64),
        posterior_covariance=covariance,
        identifiable_state_transform=np.asarray([[1.0]], dtype=np.float64),
        identifiable_fractions=np.asarray([1.0], dtype=np.float64),
        query_sensitivity_fractions=np.asarray([1.0], dtype=np.float64),
        robust_weights=np.ones(2, dtype=np.float64),
        anchor_robust_weights=np.zeros(0, dtype=np.float64),
        diagnostics={"solver": "test"},
        input_lineage=batch.metadata or {},
    )


def test_complete_covariance_reparameterization_and_true_immutability() -> None:
    observation = _observation()
    binding = _binding(observation)
    scale = 0.02
    design = binding.global_design().reshape(6, -1)
    transformed = binding.reparameterized_design(
        shared_bias_prior_std_m=scale
    ).reshape(6, -1)
    expected = design @ binding.joint_bias_covariance @ design.T
    actual = transformed @ (scale**2 * np.eye(binding.latent_dimension)) @ (
        transformed.T
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-12)

    for array in (
        binding.row_bias_indices,
        binding.bias_jacobian,
        binding.joint_bias_covariance,
        binding.global_design(),
        binding.symmetric_covariance_root(),
        binding.reparameterized_design(shared_bias_prior_std_m=scale),
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_validation_binds_observation_rows_and_orthogonalization() -> None:
    observation = _observation()
    binding = _binding(observation)
    validated = validate_prob4d_visual_bias_nuisance(observation, binding)
    assert validated.artifact_id == binding.artifact_id

    with pytest.raises(ValueError, match="different observation"):
        validate_prob4d_visual_bias_nuisance(
            observation,
            replace(
                binding,
                observation_artifact_id="1" * 64,
                artifact_id=None,
            ),
        )
    with pytest.raises(ValueError, match="row identity"):
        validate_prob4d_visual_bias_nuisance(
            observation,
            replace(
                binding,
                observation_identity_sha256="2" * 64,
                artifact_id=None,
            ),
        )
    with pytest.raises(ValueError, match="gauge-orthogonalized"):
        validate_prob4d_visual_bias_nuisance(
            observation,
            _binding(observation, orthogonalized=False),
        )


def test_one_call_v2_preserves_joint_prior_and_binds_lineage(monkeypatch) -> None:
    observation = _observation()
    binding = _binding(observation)
    captured: dict[str, np.ndarray] = {}

    def build(*args, **kwargs):
        shared = np.asarray(kwargs["shared_bias_jacobian"])
        view = np.asarray(kwargs["view_bias_jacobian"])
        captured["shared"] = shared
        captured["view"] = view
        return _adapted_batch(observation, shared, view)

    def solve(batch, *, config=None):
        assert config is not None
        return _solver_result(batch)

    monkeypatch.setattr(
        update_module,
        "build_claim_bearing_gauge_aware_batch_from_artifacts",
        build,
    )
    monkeypatch.setattr(
        update_module,
        "update_prior_aware_gauge_belief",
        solve,
    )

    update = update_claim_bearing_prob4d_with_visual_bias_from_artifacts(
        observation,
        SimpleNamespace(artifact_id=LINEARIZATION_ID),
        visual_bias_nuisance=binding,
        physical_prediction_xyz_m=np.zeros((2, 3), dtype=np.float64),
    )
    assert isinstance(update, ClaimBearingProb4DVisualBiasUpdateV2)
    assert update.inference_admissible
    assert captured["view"].shape == (2, 3, 0)

    scale = update.shared_bias_prior_std_m
    transformed = captured["shared"].reshape(6, -1)
    original = binding.global_design().reshape(6, -1)
    np.testing.assert_allclose(
        transformed @ (scale**2 * np.eye(binding.latent_dimension)) @ transformed.T,
        original @ binding.joint_bias_covariance @ original.T,
        atol=1e-14,
        rtol=1e-12,
    )
    expected_coefficients = (
        binding.symmetric_covariance_root()
        @ update.result.shared_bias_coefficients
        / scale
    )
    np.testing.assert_allclose(
        update.provider_bias_coefficients,
        expected_coefficients,
    )
    assert len(update.update_id) == 64
    assert (
        update.result.input_lineage["prob4d_visual_bias_artifact_id"]
        == binding.artifact_id
    )
    assert (
        update.result.input_lineage[
            "prob4d_visual_bias_marginal_covariance_added"
        ]
        is False
    )
    for array in (
        update.provider_bias_coefficients,
        update.provider_bias_covariance,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_recomputed_gauge_overlap_fails_before_solver(monkeypatch) -> None:
    observation = _observation()
    binding = _binding(observation)
    events: list[str] = []

    def build(*args, **kwargs):
        events.append("build")
        shared = np.asarray(kwargs["shared_bias_jacobian"])
        view = np.asarray(kwargs["view_bias_jacobian"])
        gauge = binding.global_design()[:, :, :1]
        return _adapted_batch(
            observation,
            shared,
            view,
            gauge_design=gauge,
        )

    def solve(*args, **kwargs):
        events.append("solve")
        raise AssertionError("solver must not run")

    monkeypatch.setattr(
        update_module,
        "build_claim_bearing_gauge_aware_batch_from_artifacts",
        build,
    )
    monkeypatch.setattr(
        update_module,
        "update_prior_aware_gauge_belief",
        solve,
    )

    with pytest.raises(ValueError, match="admitted BayesianPhysTwin gauge design"):
        update_claim_bearing_prob4d_with_visual_bias_from_artifacts(
            observation,
            SimpleNamespace(artifact_id=LINEARIZATION_ID),
            visual_bias_nuisance=binding,
            physical_prediction_xyz_m=np.zeros((2, 3), dtype=np.float64),
        )
    assert events == ["build"]
