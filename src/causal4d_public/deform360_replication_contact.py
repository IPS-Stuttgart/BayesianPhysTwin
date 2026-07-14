"""Source-fitted visual and tactile contact schedules for the replication."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


TACTILE_ROWS_USED = 12
CONTACT_PATIENCE_FRAMES = 5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def causal_confirmed(signal: np.ndarray, confirmation_frames: int) -> np.ndarray:
    """Apply an online symmetric debounce to a binary contact signal."""

    raw = np.asarray(signal, dtype=bool)
    _require(raw.ndim == 1, "contact signal must be one-dimensional")
    _require(confirmation_frames >= 1, "confirmation count must be positive")
    output = np.zeros_like(raw)
    state = False
    run = 0
    for index, value in enumerate(raw):
        if bool(value) == state:
            run = 0
        else:
            run += 1
            if run >= confirmation_frames:
                state = not state
                run = 0
        output[index] = state
    return output


def official_contact_window(active: np.ndarray) -> np.ndarray:
    """Match Deform360's first-event tactile window with five-frame patience."""

    signal = np.asarray(active, dtype=bool)
    _require(signal.ndim == 1, "contact signal must be one-dimensional")
    output = np.zeros_like(signal)
    start: int | None = None
    end: int | None = None
    missing = 0
    for frame, is_active in enumerate(signal):
        if start is None:
            if is_active:
                start = frame
            continue
        if is_active:
            missing = 0
        else:
            missing += 1
            if missing > CONTACT_PATIENCE_FRAMES:
                end = frame - missing
                break
    if start is None:
        return output
    if end is None:
        end = len(signal) - 1
    output[start : end + 1] = True
    return output


def _gripper_group(sensor_name: str) -> str:
    for suffix in ("_left", "_right"):
        if sensor_name.endswith(suffix):
            return sensor_name[: -len(suffix)]
    return sensor_name


def _mono_event_group(groups: Mapping[str, np.ndarray]) -> str:
    ranked = []
    for name, counts in sorted(groups.items()):
        active = counts > 1
        guard = max(10, int(round(0.1 * len(active))))
        initial_fraction = float(np.mean(active[:guard]))
        active_fraction = float(np.mean(active))
        salience = (1.0 - initial_fraction) * active_fraction
        ranked.append((-salience, name))
    _require(ranked, "no tactile groups are available")
    return min(ranked)[1]


@dataclass(frozen=True)
class ReplicationContactEpisode:
    episode_id: str
    openings_m: np.ndarray
    tactile_by_group: Mapping[str, np.ndarray]
    bimanual: bool
    nonprehensile: bool

    def __post_init__(self) -> None:
        openings = np.asarray(self.openings_m, dtype=np.float64)
        _require(openings.ndim == 2 and len(openings) >= 2, "openings must be (T,C)")
        _require(
            openings.shape[1] == (2 if self.bimanual else 1),
            "opening count differs from bimanual metadata",
        )
        _require(np.all(np.isfinite(openings)), "openings are nonfinite")
        groups = {
            str(name): np.asarray(values, dtype=bool)
            for name, values in self.tactile_by_group.items()
        }
        _require(groups, "episode has no tactile contact group")
        _require(
            all(values.shape == (len(openings),) for values in groups.values()),
            "tactile contact length differs from robot trajectory",
        )
        copied = openings.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "openings_m", copied)
        object.__setattr__(self, "tactile_by_group", groups)


@dataclass(frozen=True)
class ReplicationOpeningContactModel:
    opening_threshold_m: float
    confirmation_frames: int
    tactile_group_to_robot_axis: Mapping[str, int]
    source_balanced_accuracy: float
    calibration_balanced_accuracy: float


def load_replication_contact_episode(
    episode_dir: str | Path,
    *,
    episode_id: str,
    bimanual: bool,
    nonprehensile: bool,
    tactile_threshold: float = 0.0,
) -> ReplicationContactEpisode:
    """Load one aligned robot/tactile episode without object-specific naming."""

    directory = Path(episode_dir).resolve()
    robot_path = directory / "robot" / "robot.npz"
    _require(robot_path.is_file(), f"robot trajectory is missing: {robot_path}")
    with np.load(robot_path, allow_pickle=False) as stored:
        openings = np.asarray(stored["openings"], dtype=np.float64)
        stored_bimanual = bool(np.asarray(stored["bimanual"]).item())
    if openings.ndim == 1:
        openings = openings[:, None]
    _require(stored_bimanual == bimanual, "robot and metadata bimanual flags differ")
    groups: dict[str, np.ndarray] = {}
    tactile_paths = sorted(directory.glob("*/synced_tactile.npy"))
    _require(tactile_paths, f"tactile streams are missing: {directory}")
    for path in tactile_paths:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        _require(values.ndim == 3 and len(values) == len(openings), "invalid tactile stream")
        counts = np.count_nonzero(
            values[:, :TACTILE_ROWS_USED, :] > tactile_threshold,
            axis=(1, 2),
        )
        group = _gripper_group(path.parent.name)
        groups[group] = groups.get(group, np.zeros_like(counts)) + counts
    windows = {
        group: official_contact_window(counts > 1)
        for group, counts in groups.items()
    }
    if not bimanual:
        selected = _mono_event_group(groups)
        windows = {selected: windows[selected]}
    return ReplicationContactEpisode(
        episode_id=episode_id,
        openings_m=openings,
        tactile_by_group=windows,
        bimanual=bimanual,
        nonprehensile=nonprehensile,
    )


def _mapping_candidates(
    episodes: Sequence[ReplicationContactEpisode],
) -> list[dict[str, int]]:
    groups = sorted(
        {
            group
            for episode in episodes
            if episode.bimanual and not episode.nonprehensile
            for group in episode.tactile_by_group
        }
    )
    _require(len(groups) == 2, "expected two bimanual tactile groups")
    return [dict(zip(groups, permutation)) for permutation in itertools.permutations((0, 1))]


def contact_state_by_robot_axis(
    episode: ReplicationContactEpisode,
    mapping: Mapping[str, int],
) -> np.ndarray:
    """Map tactile groups into a ``(T,C)`` robot-axis schedule."""

    output = np.zeros_like(episode.openings_m, dtype=bool)
    if episode.bimanual:
        _require(set(episode.tactile_by_group) == set(mapping), "tactile groups changed")
        for group, values in episode.tactile_by_group.items():
            output[:, int(mapping[group])] = values
    else:
        _require(len(episode.tactile_by_group) == 1, "mono episode has multiple groups")
        output[:, 0] = next(iter(episode.tactile_by_group.values()))
    return output


def _balanced_accuracy(reference: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(reference, dtype=bool)
    guess = np.asarray(prediction, dtype=bool)
    positive = float(np.mean(guess[truth])) if np.any(truth) else 1.0
    negative = float(np.mean(~guess[~truth])) if np.any(~truth) else 1.0
    return 0.5 * (positive + negative)


def _evaluate_model(
    episodes: Sequence[ReplicationContactEpisode],
    mapping: Mapping[str, int],
    threshold: float,
    confirmation_frames: int,
) -> float:
    values = []
    for episode in episodes:
        reference = contact_state_by_robot_axis(episode, mapping)
        for axis in range(episode.openings_m.shape[1]):
            prediction = causal_confirmed(
                episode.openings_m[:, axis] <= threshold,
                confirmation_frames,
            )
            values.append(_balanced_accuracy(reference[:, axis], prediction))
    return float(np.mean(values))


def fit_replication_opening_contact_model(
    source_episodes: Sequence[ReplicationContactEpisode],
    calibration_episodes: Sequence[ReplicationContactEpisode],
    *,
    confirmation_frames: int = 3,
) -> ReplicationOpeningContactModel:
    """Fit the opening trigger and tactile-axis map on source episodes only."""

    source = [episode for episode in source_episodes if not episode.nonprehensile]
    calibration = [
        episode for episode in calibration_episodes if not episode.nonprehensile
    ]
    _require(source, "no prehensile source episode is available")
    _require(calibration, "no prehensile calibration episode is available")
    openings = np.concatenate([episode.openings_m.reshape(-1) for episode in source])
    thresholds = np.unique(np.quantile(openings, np.linspace(0.01, 0.99, 199)))
    best: tuple[float, float, tuple[tuple[str, int], ...]] | None = None
    selected_mapping: dict[str, int] | None = None
    selected_threshold = 0.0
    for mapping in _mapping_candidates(source):
        mapping_key = tuple(sorted(mapping.items()))
        for threshold in thresholds:
            score = _evaluate_model(
                source, mapping, float(threshold), confirmation_frames
            )
            candidate = (score, -float(threshold), mapping_key)
            if best is None or candidate > best:
                best = candidate
                selected_mapping = mapping
                selected_threshold = float(threshold)
    _require(selected_mapping is not None, "opening contact-model search failed")
    return ReplicationOpeningContactModel(
        opening_threshold_m=selected_threshold,
        confirmation_frames=confirmation_frames,
        tactile_group_to_robot_axis=selected_mapping,
        source_balanced_accuracy=_evaluate_model(
            source, selected_mapping, selected_threshold, confirmation_frames
        ),
        calibration_balanced_accuracy=_evaluate_model(
            calibration, selected_mapping, selected_threshold, confirmation_frames
        ),
    )


def visual_contact_schedule(
    episode: ReplicationContactEpisode,
    model: ReplicationOpeningContactModel,
) -> np.ndarray:
    """Predict each gripper's contact causally from released openings."""

    return np.column_stack(
        [
            causal_confirmed(
                episode.openings_m[:, axis] <= model.opening_threshold_m,
                model.confirmation_frames,
            )
            for axis in range(episode.openings_m.shape[1])
        ]
    )


def prefix_window_from_visual_contact(
    visual_contact: np.ndarray,
    *,
    prefix_frame_count: int = 6,
) -> tuple[int, int]:
    """Return the first all-gripper trigger and exclusive prefix end."""

    schedule = np.asarray(visual_contact, dtype=bool)
    _require(schedule.ndim == 2, "visual contact schedule must be (T,C)")
    trigger = np.flatnonzero(np.all(schedule, axis=1))
    _require(len(trigger) > 0, "visual all-gripper contact trigger never activates")
    start = int(trigger[0])
    stop = start + prefix_frame_count
    _require(stop <= len(schedule), "contact prefix exceeds the episode")
    return start, stop


__all__ = [
    "CONTACT_PATIENCE_FRAMES",
    "ReplicationContactEpisode",
    "ReplicationOpeningContactModel",
    "causal_confirmed",
    "contact_state_by_robot_axis",
    "fit_replication_opening_contact_model",
    "load_replication_contact_episode",
    "official_contact_window",
    "prefix_window_from_visual_contact",
    "visual_contact_schedule",
]
