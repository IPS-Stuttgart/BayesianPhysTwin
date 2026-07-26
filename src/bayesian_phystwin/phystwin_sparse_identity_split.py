"""Causal observed/hidden splits for sparse material-identity tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phystwin_online_belief import deterministic_farthest_point_ids


@dataclass(frozen=True)
class DisjointSparseIdentityTracks:
    """Prefix observations and disjoint future-scoring tracks.

    ``observation_tracks_m`` contains only the selected material identities and
    only before ``future_start_frame``. ``scoring_tracks_m`` exposes the
    selected identities during the prefix and the disjoint hidden identities in
    the future. Frame zero retains every eligible identity solely to define the
    fixed official nearest-node correspondence.
    """

    observed_indices: np.ndarray
    hidden_indices: np.ndarray
    observation_tracks_m: np.ndarray
    scoring_tracks_m: np.ndarray
    future_start_frame: int

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed_indices, dtype=np.int64).copy()
        hidden = np.asarray(self.hidden_indices, dtype=np.int64).copy()
        observation = np.asarray(self.observation_tracks_m, dtype=float).copy()
        scoring = np.asarray(self.scoring_tracks_m, dtype=float).copy()
        if observed.ndim != 1 or hidden.ndim != 1:
            raise ValueError("identity indices must be vectors")
        if len(observed) == 0 or len(hidden) == 0:
            raise ValueError("observed and hidden identity sets must be nonempty")
        if np.intersect1d(observed, hidden).size:
            raise ValueError("observed and hidden identities must be disjoint")
        if observation.ndim != 3 or observation.shape[2] != 3:
            raise ValueError("observation tracks must have shape (T, K, 3)")
        if scoring.shape != observation.shape:
            raise ValueError("scoring tracks must match observation tracks")
        if not 1 <= self.future_start_frame < len(observation):
            raise ValueError("future_start_frame must lie inside the trajectory")
        identity_count = observation.shape[1]
        if (
            np.any(observed < 0)
            or np.any(hidden < 0)
            or np.any(observed >= identity_count)
            or np.any(hidden >= identity_count)
        ):
            raise ValueError("identity index exceeds the track array")
        for value in (observed, hidden, observation, scoring):
            value.setflags(write=False)
        object.__setattr__(self, "observed_indices", observed)
        object.__setattr__(self, "hidden_indices", hidden)
        object.__setattr__(self, "observation_tracks_m", observation)
        object.__setattr__(self, "scoring_tracks_m", scoring)


def split_sparse_identity_tracks(
    tracks_m: np.ndarray,
    *,
    observed_count: int,
    future_start_frame: int,
    observed_support_frame_range: tuple[int, int] | None = None,
) -> DisjointSparseIdentityTracks:
    """Select prefix sensors from frame zero and reserve all others for scoring.

    Selection uses deterministic farthest-point sampling on finite frame-zero
    locations. An optional support range can restrict candidates to identities
    observed in every declared prefix frame. No future visibility,
    displacement, or error influences the identity split.
    """

    tracks = np.asarray(tracks_m, dtype=float)
    if tracks.ndim != 3 or tracks.shape[2] != 3:
        raise ValueError("tracks_m must have shape (T, K, 3)")
    if not 1 <= future_start_frame < len(tracks):
        raise ValueError("future_start_frame must lie inside the trajectory")
    if observed_count < 1:
        raise ValueError("observed_count must be positive")
    frame_zero_finite = np.all(np.isfinite(tracks[0]), axis=1)
    if observed_support_frame_range is None:
        support_finite = np.ones(tracks.shape[1], dtype=bool)
    else:
        support_start, support_stop = observed_support_frame_range
        if not 0 <= support_start < support_stop <= future_start_frame:
            raise ValueError(
                "observed support range must be nonempty and end by "
                "future_start_frame"
            )
        support_finite = np.all(
            np.all(np.isfinite(tracks[support_start:support_stop]), axis=2),
            axis=0,
        )
    eligible = np.flatnonzero(frame_zero_finite & support_finite)
    if observed_count > len(eligible):
        raise ValueError("observed_count exceeds support-eligible identities")
    frame_zero_indices = np.flatnonzero(frame_zero_finite)
    if observed_count >= len(frame_zero_indices):
        raise ValueError(
            "observed_count must leave at least one frame-zero identity hidden"
        )
    observed = deterministic_farthest_point_ids(
        tracks[0],
        eligible,
        observed_count,
    )
    hidden = np.setdiff1d(frame_zero_indices, observed, assume_unique=True)

    observation = np.full_like(tracks, np.nan)
    observation[:future_start_frame, observed] = tracks[:future_start_frame, observed]

    scoring = np.full_like(tracks, np.nan)
    scoring[0, frame_zero_indices] = tracks[0, frame_zero_indices]
    scoring[1:future_start_frame, observed] = tracks[1:future_start_frame, observed]
    scoring[future_start_frame:, hidden] = tracks[future_start_frame:, hidden]
    return DisjointSparseIdentityTracks(
        observed_indices=observed,
        hidden_indices=hidden,
        observation_tracks_m=observation,
        scoring_tracks_m=scoring,
        future_start_frame=future_start_frame,
    )
