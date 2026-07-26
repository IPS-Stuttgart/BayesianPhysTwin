import numpy as np
import pytest

from bayesian_phystwin.phystwin_sparse_identity_split import (
    split_sparse_identity_tracks,
)


def _tracks() -> np.ndarray:
    frame_zero = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
        ]
    )
    return np.stack([frame_zero + [0.0, frame, 0.0] for frame in range(6)])


def test_split_uses_geometry_spanning_frame_zero_identities() -> None:
    split = split_sparse_identity_tracks(
        _tracks(),
        observed_count=2,
        future_start_frame=4,
    )

    np.testing.assert_array_equal(split.observed_indices, [0, 3])
    np.testing.assert_array_equal(split.hidden_indices, [1, 2])


def test_split_keeps_observation_and_future_score_identities_disjoint() -> None:
    tracks = _tracks()
    split = split_sparse_identity_tracks(
        tracks,
        observed_count=2,
        future_start_frame=4,
    )

    assert np.all(np.isfinite(split.observation_tracks_m[:4, [0, 3]]))
    assert np.all(np.isnan(split.observation_tracks_m[:, [1, 2, 4]]))
    assert np.all(np.isnan(split.observation_tracks_m[4:, [0, 3]]))

    assert np.all(np.isfinite(split.scoring_tracks_m[0, :4]))
    assert np.all(np.isnan(split.scoring_tracks_m[1:4, [1, 2]]))
    assert np.all(np.isfinite(split.scoring_tracks_m[1:4, [0, 3]]))
    assert np.all(np.isnan(split.scoring_tracks_m[4:, [0, 3]]))
    assert np.all(np.isfinite(split.scoring_tracks_m[4:, [1, 2]]))
    assert np.all(np.isnan(split.scoring_tracks_m[:, 4]))


def test_future_mutation_cannot_change_prefix_observation_carrier() -> None:
    original = _tracks()
    mutated = original.copy()
    mutated[4:] += 1000.0

    first = split_sparse_identity_tracks(
        original,
        observed_count=2,
        future_start_frame=4,
    )
    second = split_sparse_identity_tracks(
        mutated,
        observed_count=2,
        future_start_frame=4,
    )

    np.testing.assert_array_equal(first.observed_indices, second.observed_indices)
    np.testing.assert_array_equal(first.hidden_indices, second.hidden_indices)
    np.testing.assert_array_equal(
        first.observation_tracks_m,
        second.observation_tracks_m,
    )


def test_split_can_require_complete_declared_prefix_support() -> None:
    tracks = _tracks()
    tracks[1:4, 3] = np.nan

    split = split_sparse_identity_tracks(
        tracks,
        observed_count=2,
        future_start_frame=4,
        observed_support_frame_range=(1, 4),
    )

    np.testing.assert_array_equal(split.observed_indices, [0, 2])
    np.testing.assert_array_equal(split.hidden_indices, [1, 3])
    assert np.all(np.isfinite(split.observation_tracks_m[1:4, [0, 2]]))


def test_support_qualified_split_is_invariant_to_future_mutation() -> None:
    original = _tracks()
    mutated = original.copy()
    mutated[4:, 0] = np.nan
    mutated[4:, 3] += 1000.0

    first = split_sparse_identity_tracks(
        original,
        observed_count=2,
        future_start_frame=4,
        observed_support_frame_range=(1, 4),
    )
    second = split_sparse_identity_tracks(
        mutated,
        observed_count=2,
        future_start_frame=4,
        observed_support_frame_range=(1, 4),
    )

    np.testing.assert_array_equal(first.observed_indices, second.observed_indices)
    np.testing.assert_array_equal(first.hidden_indices, second.hidden_indices)


@pytest.mark.parametrize("frame_range", [(-1, 2), (2, 2), (1, 5)])
def test_split_rejects_invalid_support_ranges(
    frame_range: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="support range"):
        split_sparse_identity_tracks(
            _tracks(),
            observed_count=2,
            future_start_frame=4,
            observed_support_frame_range=frame_range,
        )


@pytest.mark.parametrize("observed_count", [0, 4, 5])
def test_split_rejects_invalid_observed_counts(observed_count: int) -> None:
    with pytest.raises(ValueError):
        split_sparse_identity_tracks(
            _tracks(),
            observed_count=observed_count,
            future_start_frame=4,
        )
