import numpy as np

from bayesian_phystwin.cpd_registration import NonrigidCpdConfig
from bayesian_phystwin.deform360_raw_pairwise_correspondence_diagnostic import (
    ASSOCIATION_ADAPTIVE_ARM,
    CLIQUE_RBF_ARM,
    CPD_ARM,
    LEGACY_PHYSICAL_DEFAULT_ARM,
    SELECTED_RAW_ARM,
    evaluate_raw_pairwise_correspondence_arrays,
)
from bayesian_phystwin.phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
)


def _arrays(seed: int = 8) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    point_count = 24
    frame_count = 76
    frame_zero = rng.uniform(-0.3, 0.3, size=(point_count, 3)).astype(np.float32)
    prior = np.repeat(frame_zero[None], frame_count, axis=0)
    for frame in range(frame_count):
        prior[frame, :, 0] += np.float32(frame * 0.002)
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    target = prior.copy()
    visible = np.ones((frame_count, point_count), dtype=bool)
    validity = visible.copy()
    measurement = np.full_like(prior, np.nan)
    measurement_visibility = np.zeros((frame_count, point_count), dtype=bool)
    measurement_validity = measurement_visibility.copy()
    return (
        prior,
        persistence,
        target,
        visible,
        validity,
        measurement,
        measurement_visibility,
        measurement_validity,
    )


def test_raw_insufficient_support_uses_persistence_and_preserves_legacy_ablation() -> (
    None
):
    arrays = list(_arrays())
    prior, persistence, target = arrays[:3]
    measurement = arrays[5]
    measurement_visibility = arrays[6]
    measurement_validity = arrays[7]
    centers = np.arange(16, dtype=np.int64)
    for frame in (19, 38, 57):
        ids = centers[:2]
        measurement[frame, ids] = target[frame, ids]
        measurement_visibility[frame, ids] = True
        measurement_validity[frame, ids] = True

    report, trajectories = evaluate_raw_pairwise_correspondence_arrays(
        *arrays,
        center_ids=centers,
        scored_frames=tuple(range(20, 76)),
        cpd_config=NonrigidCpdConfig(maximum_iterations=5),
    )

    assert all(
        not update["selector_support_sufficient"] for update in report["updates"]
    )
    assert all(
        update["selected_backbone"] == "persistence" for update in report["updates"]
    )
    post_update = np.asarray(
        [*range(20, 38), *range(39, 57), *range(58, 76)], dtype=np.int64
    )
    np.testing.assert_array_equal(
        trajectories[SELECTED_RAW_ARM][post_update], persistence[post_update]
    )
    np.testing.assert_array_equal(
        trajectories[CLIQUE_RBF_ARM][post_update], persistence[post_update]
    )
    np.testing.assert_array_equal(
        trajectories[CPD_ARM][post_update], persistence[post_update]
    )
    np.testing.assert_array_equal(
        trajectories[ASSOCIATION_ADAPTIVE_ARM][post_update], persistence[post_update]
    )
    np.testing.assert_array_equal(
        trajectories[LEGACY_PHYSICAL_DEFAULT_ARM][post_update], prior[post_update]
    )


def test_raw_association_adaptive_routes_rejected_identity_swaps_to_cpd() -> None:
    arrays = list(_arrays(seed=12))
    target = arrays[2]
    measurement = arrays[5]
    measurement_visibility = arrays[6]
    measurement_validity = arrays[7]
    centers = np.arange(16, dtype=np.int64)
    corrupted = np.arange(8)
    for frame in (19, 38, 57):
        measurement[frame, centers] = target[frame, centers]
        measurement[frame, corrupted] = target[frame, np.roll(corrupted, 1)]
        measurement_visibility[frame, centers] = True
        measurement_validity[frame, centers] = True

    report, trajectories = evaluate_raw_pairwise_correspondence_arrays(
        *arrays,
        center_ids=centers,
        scored_frames=tuple(range(20, 76)),
        gate_config=PairwiseCorrespondenceGateConfig(
            absolute_pair_strain_m=0.005,
            relative_pair_strain=0.01,
            minimum_inlier_count=9,
            minimum_inlier_fraction=0.70,
        ),
        cpd_config=NonrigidCpdConfig(maximum_iterations=5),
    )

    assert all(
        not update["selected_pairwise_gate"]["accepted"] for update in report["updates"]
    )
    assert all(
        update["association_adaptive"]["route"] == "unordered_cpd"
        for update in report["updates"]
    )
    np.testing.assert_array_equal(
        trajectories[ASSOCIATION_ADAPTIVE_ARM], trajectories[CPD_ARM]
    )
