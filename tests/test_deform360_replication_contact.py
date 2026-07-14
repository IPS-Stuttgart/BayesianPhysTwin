import numpy as np

from causal4d_public.deform360_replication_contact import (
    ReplicationContactEpisode,
    causal_confirmed,
    contact_state_by_robot_axis,
    fit_replication_opening_contact_model,
    prefix_window_from_visual_contact,
)


def _episode(name: str, *, swapped: bool = False) -> ReplicationContactEpisode:
    first = np.array([False, False, True, True, True, False, False])
    second = np.array([False, False, False, True, True, True, False])
    openings = np.column_stack(
        (
            np.where(first, 0.01, 0.05),
            np.where(second, 0.012, 0.055),
        )
    )
    groups = {"left": second, "right": first} if swapped else {"left": first, "right": second}
    return ReplicationContactEpisode(name, openings, groups, True, False)


def test_causal_confirmed_debounces_online() -> None:
    signal = np.array([False, True, False, True, True, False, False])
    assert causal_confirmed(signal, 2).tolist() == [False, False, False, False, True, True, False]


def test_source_fit_recovers_tactile_axis_mapping() -> None:
    source = [_episode("a"), _episode("b")]
    calibration = [_episode("c")]
    model = fit_replication_opening_contact_model(
        source, calibration, confirmation_frames=1
    )
    assert model.tactile_group_to_robot_axis == {"left": 0, "right": 1}
    assert model.source_balanced_accuracy == 1.0
    reference = contact_state_by_robot_axis(source[0], model.tactile_group_to_robot_axis)
    assert reference.shape == source[0].openings_m.shape


def test_prefix_window_uses_all_grippers() -> None:
    schedule = np.array(
        [[False, False], [True, False], [True, True], [True, True], [True, True]]
    )
    assert prefix_window_from_visual_contact(schedule, prefix_frame_count=3) == (2, 5)
