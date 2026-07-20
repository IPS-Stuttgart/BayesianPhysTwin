import numpy as np

from bayesian_phystwin.cpd_registration import NonrigidCpdConfig
from bayesian_phystwin.deform360_robust_correspondence_diagnostic import (
    ASSOCIATION_ADAPTIVE_ARM,
    CPD_ARM,
    LEGACY_MIXED_UNGATED_RBF_ARM,
    ROBUST_RBF_ARM,
    SELECTED_RAW_ARM,
    UNGATED_RBF_ARM,
    MatchedObservationStress,
    corrupt_matched_current_observation,
    evaluate_robust_correspondence_arrays,
)
from bayesian_phystwin.phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
    detect_pairwise_consensus_correspondences,
    pairwise_distance_strain_m,
)


def _strict_config() -> PairwiseCorrespondenceGateConfig:
    return PairwiseCorrespondenceGateConfig(
        absolute_pair_strain_m=0.005,
        relative_pair_strain=0.01,
        minimum_inlier_count=9,
        minimum_inlier_fraction=0.70,
    )


def test_pairwise_gate_accepts_common_rigid_motion() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(16, 3))
    angle = 0.7
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    observed = source @ rotation.T + np.asarray([0.5, -0.2, 0.8])

    result = detect_pairwise_consensus_correspondences(
        source,
        observed,
        np.ones(16, dtype=bool),
        config=_strict_config(),
    )

    assert result.accepted
    assert result.decision == "accepted"
    assert result.inlier_count == 16
    np.testing.assert_array_equal(result.inlier_mask, np.ones(16, dtype=bool))


def test_pairwise_gate_removes_swapped_endpoints() -> None:
    source = np.column_stack((np.linspace(0.0, 1.5, 16), np.zeros(16), np.zeros(16)))
    observed = source.copy()
    observed[[0, 15]] = observed[[15, 0]]

    result = detect_pairwise_consensus_correspondences(
        source,
        observed,
        np.ones(16, dtype=bool),
        material_ids=np.arange(100, 116),
        config=_strict_config(),
    )

    assert result.accepted
    assert result.inlier_count == 14
    assert not result.inlier_mask[0]
    assert not result.inlier_mask[15]
    assert np.all(result.inlier_mask[1:15])


def test_pairwise_gate_abstains_at_half_mismatches() -> None:
    rng = np.random.default_rng(15)
    source = rng.uniform(-0.5, 0.5, size=(16, 3))
    observed = source.copy()
    corrupted = np.arange(8)
    observed[corrupted] = source[np.roll(corrupted, 1)]

    result = detect_pairwise_consensus_correspondences(
        source,
        observed,
        np.ones(16, dtype=bool),
        config=_strict_config(),
    )

    assert not result.accepted
    assert result.inlier_count <= 8
    assert result.decision in {
        "insufficient_consensus_count",
        "insufficient_consensus_fraction",
    }


def test_distance_strain_is_zero_for_reflection_and_translation() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.3, 1.2, 0.5]])
    observed = source * np.asarray([-1.0, 1.0, 1.0]) + 2.0

    strain, source_distance = pairwise_distance_strain_m(source, observed)

    np.testing.assert_allclose(strain, 0.0, atol=1e-12)
    assert np.all(source_distance >= 0.0)


def test_mismatch_corruption_is_deterministic_and_set_preserving() -> None:
    clean = np.arange(48, dtype=float).reshape(16, 3)
    available = np.ones(16, dtype=bool)
    stress = MatchedObservationStress(name="mismatch_25pct", mismatch_fraction=0.25)

    first, first_bad, first_info = corrupt_matched_current_observation(
        clean,
        available,
        case_name="open-case",
        frame=19,
        stress=stress,
        seed=3,
    )
    second, second_bad, second_info = corrupt_matched_current_observation(
        clean,
        available,
        case_name="open-case",
        frame=19,
        stress=stress,
        seed=3,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_bad, second_bad)
    assert first_info == second_info
    assert int(np.sum(first_bad)) == 4
    np.testing.assert_array_equal(
        first[np.lexsort(first.T[::-1])], clean[np.lexsort(clean.T[::-1])]
    )


def test_rejected_update_is_bit_exact_selected_backbone_fallback() -> None:
    rng = np.random.default_rng(21)
    point_count = 24
    frame_count = 76
    frame_zero = rng.uniform(-0.4, 0.4, size=(point_count, 3)).astype(np.float32)
    prior = np.repeat(frame_zero[None], frame_count, axis=0)
    for frame in range(frame_count):
        prior[frame, :, 1] += np.float32(frame * 0.001)
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    target = prior.copy()
    visible = np.ones((frame_count, point_count), dtype=bool)
    validity = visible.copy()
    centers = np.arange(16, dtype=np.int64)

    report, trajectories = evaluate_robust_correspondence_arrays(
        prior,
        persistence,
        target,
        visible,
        validity,
        center_ids=centers,
        scored_frames=tuple(range(20, frame_count)),
        case_name="synthetic-open-case",
        stress=MatchedObservationStress(name="mismatch_50pct", mismatch_fraction=0.50),
        seed=0,
        gate_config=_strict_config(),
        cpd_config=NonrigidCpdConfig(maximum_iterations=5),
    )

    assert all(not update["pairwise_gate"]["accepted"] for update in report["updates"])
    assert all(
        update["pairwise_gate"]["rejected_exact_selected_backbone_fallback"]
        for update in report["updates"]
    )
    np.testing.assert_array_equal(
        trajectories[ROBUST_RBF_ARM], trajectories[SELECTED_RAW_ARM]
    )
    np.testing.assert_array_equal(
        trajectories[ASSOCIATION_ADAPTIVE_ARM], trajectories[CPD_ARM]
    )


def test_association_adaptive_raw_fallback_is_exact_without_cpd_support() -> None:
    rng = np.random.default_rng(25)
    point_count = 24
    frame_count = 76
    frame_zero = rng.uniform(-0.4, 0.4, size=(point_count, 3)).astype(np.float32)
    prior = np.repeat(frame_zero[None], frame_count, axis=0)
    persistence = prior.copy()
    target = prior.copy()
    visible = np.ones((frame_count, point_count), dtype=bool)
    validity = visible.copy()
    centers = np.arange(16, dtype=np.int64)
    for frame in (19, 38, 57):
        visible[frame, centers[2:]] = False

    report, trajectories = evaluate_robust_correspondence_arrays(
        prior,
        persistence,
        target,
        visible,
        validity,
        center_ids=centers,
        scored_frames=tuple(range(20, frame_count)),
        case_name="synthetic-insufficient-support",
        stress=MatchedObservationStress(name="clean"),
        seed=0,
        cpd_config=NonrigidCpdConfig(maximum_iterations=5),
    )

    assert all(
        update["association_adaptive"]["route"] == "selected_raw_backbone"
        for update in report["updates"]
    )
    assert all(
        update["association_adaptive"]["bit_exact_selected_route"]
        for update in report["updates"]
    )
    np.testing.assert_array_equal(
        trajectories[ASSOCIATION_ADAPTIVE_ARM], trajectories[SELECTED_RAW_ARM]
    )


def test_selector_switch_uses_separate_backbone_relative_states() -> None:
    rng = np.random.default_rng(32)
    point_count = 24
    frame_count = 76
    frame_zero = rng.uniform(-0.3, 0.3, size=(point_count, 3)).astype(np.float32)
    prior = np.repeat(frame_zero[None], frame_count, axis=0)
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    for frame in range(frame_count):
        prior[frame, :, 0] += np.float32(frame * 0.003)
    target = prior.copy()
    target[:29, :, 1] += np.float32(0.010)
    target[29:48] = persistence[29:48]
    target[29:48, :, 2] -= np.float32(0.010)
    target[48:, :, 1] += np.float32(0.015)
    target[0] = frame_zero
    visible = np.ones((frame_count, point_count), dtype=bool)
    validity = visible.copy()

    report, trajectories = evaluate_robust_correspondence_arrays(
        prior,
        persistence,
        target,
        visible,
        validity,
        center_ids=np.arange(16, dtype=np.int64),
        scored_frames=tuple(range(20, frame_count)),
        case_name="synthetic-switch-case",
        stress=MatchedObservationStress(name="clean"),
        seed=0,
        cpd_config=NonrigidCpdConfig(maximum_iterations=5),
    )

    assert [
        update["selected_backbone"]["selected"] for update in report["updates"]
    ] == [
        "physical_prior",
        "persistence",
        "physical_prior",
    ]
    assert not np.array_equal(
        trajectories[UNGATED_RBF_ARM],
        trajectories[LEGACY_MIXED_UNGATED_RBF_ARM],
    )
