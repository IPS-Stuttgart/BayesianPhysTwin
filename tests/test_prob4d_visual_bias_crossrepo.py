from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._prob4d_stream_binding import (
    prob4d_observation_identity_summary,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.prob4d_visual_bias_update import (
    PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
    validate_prob4d_visual_bias_nuisance,
)

visual_bias_module = pytest.importorskip("prob4d.visual_bias")
VisualBiasNuisanceV1 = visual_bias_module.VisualBiasNuisanceV1
load_visual_bias_nuisance = visual_bias_module.load_visual_bias_nuisance
write_visual_bias_nuisance = visual_bias_module.write_visual_bias_nuisance


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


def test_strict_prob4d_roundtrip_matches_independent_bpt_identity(
    tmp_path: Path,
) -> None:
    observation = _observation()
    _, _, identity_sha = prob4d_observation_identity_summary(observation)
    nuisance = VisualBiasNuisanceV1(
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
        orthogonalization_semantics=(PROB4D_VISUAL_BIAS_ORTHOGONALIZATION),
        maximum_gauge_projection=0.0,
        gauge_projection_tolerance=1e-8,
        metadata={"uses_truth": False},
    )
    manifest = tmp_path / "visual-bias.json"
    write_visual_bias_nuisance(nuisance, manifest)
    loaded = load_visual_bias_nuisance(manifest)
    binding = validate_prob4d_visual_bias_nuisance(observation, loaded)

    assert binding.artifact_id == nuisance.artifact_id
    assert binding.observation_identity_sha256 == identity_sha
    original = binding.global_design().reshape(6, -1)
    transformed = binding.reparameterized_design(shared_bias_prior_std_m=0.02).reshape(
        6, -1
    )
    np.testing.assert_allclose(
        transformed @ (0.02**2 * np.eye(binding.latent_dimension)) @ transformed.T,
        original @ nuisance.joint_bias_covariance @ original.T,
        atol=1e-14,
        rtol=1e-12,
    )
    with pytest.raises(ValueError):
        binding.joint_bias_covariance.setflags(write=True)
